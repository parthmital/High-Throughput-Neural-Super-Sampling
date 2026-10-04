"""Perceptual near-duplicate detection across the combined dataset and leakage-safe splitting.

Pipeline: 64-bit DCT perceptual hash (pHash) and difference hash (dHash) on thumbnails, DINOv2 image embeddings,
exhaustive GPU nearest-neighbour search (chunked matrix products, both GPUs), thresholds calibrated on synthetic
positive pairs (crops, rescales, JPEG, colour shifts, flips, text overlays) against negative pairs, union-find
clustering of near-duplicate edges together with structural edges (same sequence / scene / environment), and a
split assigned per connected component so no component spans train, validation and test.
"""

import hashlib
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------------------------------------------
_DCT32 = None


def _dct_matrix(n=32):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


def phash_bits(thumbs):
    """thumbs: uint8 (N, h, w, 3) -> bool (N, 64). DCT of a 32x32 grey image, 8x8 low frequencies (DC excluded
    from the median), each bit = coefficient above the median."""
    global _DCT32
    if _DCT32 is None:
        _DCT32 = _dct_matrix(32)
    grey = np.stack(
        [
            np.asarray(
                Image.fromarray(t).convert("L").resize((32, 32), Image.BILINEAR),
                np.float32,
            )
            for t in thumbs
        ]
    )
    coeffs = _DCT32 @ grey @ _DCT32.T
    low = coeffs[:, :8, :8].reshape(len(thumbs), 64)
    med = np.median(low[:, 1:], axis=1, keepdims=True)
    return low > med


def dhash_bits(thumbs):
    """Horizontal gradient-sign hash on a 9x8 grey image -> bool (N, 64)."""
    grey = np.stack(
        [
            np.asarray(
                Image.fromarray(t).convert("L").resize((9, 8), Image.BILINEAR),
                np.float32,
            )
            for t in thumbs
        ]
    )
    return (grey[:, :, 1:] > grey[:, :, :-1]).reshape(len(thumbs), 64)


def bits_to_hex(bits):
    return [
        "".join(
            f"{int(''.join('1' if b else '0' for b in row[i:i + 4]), 2):x}"
            for i in range(0, 64, 4)
        )
        for row in bits
    ]


# ---------------------------------------------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------------------------------------------
def load_embedder(device, log=print):
    """DINOv2-small (Apache 2.0) from the Hugging Face Hub; returns (callable, name) or (None, None) offline."""
    try:
        from transformers import AutoModel

        model = (
            AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval().half()
        )

        @torch.no_grad()
        def embed(x):  # x: float (N, 3, 224, 224) in [0, 1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            out = model(pixel_values=((x - mean) / std).half())
            feat = (
                out.pooler_output
                if getattr(out, "pooler_output", None) is not None
                else out.last_hidden_state[:, 0]
            )
            return F.normalize(feat.float(), dim=1)

        return embed, "dinov2-small"
    except Exception as exc:
        log(f"DINOv2 unavailable ({exc}); embeddings skipped, hashes only")
        return None, None


@torch.no_grad()
def embed_thumbs(thumbs, embedders, devices, batch=256):
    """Embed uint8 thumbnails (N, 256, 256, 3), splitting batches across devices (threads, one per GPU)."""
    from concurrent.futures import ThreadPoolExecutor

    chunks = [(i, thumbs[i : i + batch]) for i in range(0, len(thumbs), batch)]
    out = [None] * len(chunks)

    def work(rank):
        for k in range(rank, len(chunks), len(devices)):
            _, t = chunks[k]
            x = (
                torch.from_numpy(np.ascontiguousarray(t))
                .to(devices[rank])
                .permute(0, 3, 1, 2)
                .float()
                / 255.0
            )
            x = F.interpolate(
                x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False
            )
            out[k] = embedders[rank](x).cpu()

    with ThreadPoolExecutor(len(devices)) as pool:
        list(pool.map(work, range(len(devices))))
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------------------------
# Exhaustive neighbour search on GPU (cosine for embeddings, Hamming for hashes via +-1 dot products)
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def pairs_above(feats, threshold, device, chunk=4096):
    """All pairs (i < j) with dot(feats_i, feats_j) >= threshold. Returns int64 (M, 2) and float32 scores."""
    x = torch.from_numpy(np.ascontiguousarray(feats)).to(device).half()
    n = x.shape[0]
    ii, jj, ss = [], [], []
    for a in range(0, n, chunk):
        sim = (x[a : a + chunk] @ x.T).float()
        rows = torch.arange(a, min(a + chunk, n), device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        mask = (sim >= threshold) & (cols > rows)
        r, c = mask.nonzero(as_tuple=True)
        ii.append((r + a).cpu())
        jj.append(c.cpu())
        ss.append(sim[r, c].cpu())
    if not ii:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float32)
    return torch.stack([torch.cat(ii), torch.cat(jj)], 1).numpy(), torch.cat(ss).numpy()


def bits_to_pm1(bits):
    return np.where(bits, 1.0, -1.0).astype(np.float32)


def hamming_threshold_to_dot(h, nbits=64):
    """Hamming distance <= h  <=>  dot of +-1 vectors >= nbits - 2h."""
    return float(nbits - 2 * h)


# ---------------------------------------------------------------------------------------------------------------
# Calibration with synthetic positives and random negatives
# ---------------------------------------------------------------------------------------------------------------
def augment_copy(img, rng):
    """A near-duplicate of img as it might appear in another dataset: re-crop, rescale, recompress, recolour,
    mirror, small rotation, blur or a text/watermark overlay. img: PIL RGB."""
    w, h = img.size
    a = rng.uniform(0.6, 0.95)
    cw, ch = int(w * np.sqrt(a)), int(h * np.sqrt(a))
    x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    out = img.crop((x, y, x + cw, y + ch)).resize((256, 256), Image.BILINEAR)
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.3:
        out = out.rotate(float(rng.uniform(-4, 4)), resample=Image.BILINEAR)
    if rng.random() < 0.5:
        arr = np.asarray(out, np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    if rng.random() < 0.3:
        ImageDraw.Draw(out).text(
            (int(rng.integers(5, 120)), int(rng.integers(5, 220))),
            "SAMPLE 1080p",
            fill=(255, 255, 255),
        )
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(rng.integers(35, 92)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def calibration_pairs(thumbs, n_pos, seed):
    """Positive pairs (thumb, augmented copy) and negative pairs (two different thumbs). Returns arrays of
    thumbnails A, B (uint8, N x 256 x 256 x 3) and labels."""
    rng = np.random.default_rng(seed)
    n = len(thumbs)
    idx = rng.choice(n, size=min(n_pos, n), replace=False)
    a_list, b_list, labels = [], [], []
    for i in idx:
        a_list.append(thumbs[i])
        b_list.append(
            np.asarray(augment_copy(Image.fromarray(thumbs[i]), rng).resize((256, 256)))
        )
        labels.append(1)
    for _ in range(len(idx) * 4):
        i, j = rng.choice(n, size=2, replace=False)
        a_list.append(thumbs[i])
        b_list.append(thumbs[j])
        labels.append(0)
    return np.stack(a_list), np.stack(b_list), np.array(labels)


def choose_threshold(scores, labels, max_fpr=0.001):
    """Lowest similarity threshold whose false-positive rate on negatives stays <= max_fpr; returns the
    threshold with precision, recall and F1 at that point, plus a ROC table."""
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    tp, fp = np.cumsum(y == 1), np.cumsum(y == 0)
    P, N = max((labels == 1).sum(), 1), max((labels == 0).sum(), 1)
    fpr, tpr = fp / N, tp / P
    ok = np.nonzero(fpr <= max_fpr)[0]
    k = ok[-1] if len(ok) else 0
    thr = float(s[k])
    prec = tp[k] / max(tp[k] + fp[k], 1)
    rec = tpr[k]
    roc = {
        "fpr": fpr[:: max(1, len(fpr) // 400)].tolist(),
        "tpr": tpr[:: max(1, len(tpr) // 400)].tolist(),
    }
    return {
        "threshold": thr,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
        "roc": roc,
    }


# ---------------------------------------------------------------------------------------------------------------
# Clustering and leakage-safe split
# ---------------------------------------------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_splits(items, dup_edges, val_fraction, seed):
    """items: dicts with 'group' and 'role' ('train' or 'test'); dup_edges: (i, j) item index pairs.
    Components = union of shared leakage groups and near-duplicate edges. A component containing any test item
    becomes test; its train-role members are dropped (they would leak). Other components go to val with
    probability val_fraction (deterministic hash), else train. Returns (split list, component id list, report).
    """
    n = len(items)
    uf = UnionFind(n)
    first_of_group = {}
    for i, it in enumerate(items):
        g = it["group"]
        if g in first_of_group:
            uf.union(i, first_of_group[g])
        else:
            first_of_group[g] = i
    for i, j in dup_edges:
        uf.union(int(i), int(j))
    comp = [uf.find(i) for i in range(n)]
    has_test = {}
    for i, it in enumerate(items):
        if it["role"] == "test":
            has_test[comp[i]] = True
    split, dropped = [], 0
    for i, it in enumerate(items):
        c = comp[i]
        if has_test.get(c):
            if it["role"] == "test":
                split.append("test")
            else:
                split.append("drop")
                dropped += 1
        else:
            digest = hashlib.md5(f"{seed}/{items[c]['item_id']}".encode()).hexdigest()
            h = (int(digest, 16) % 10_000) / 10_000.0
            split.append("val" if h < val_fraction else "train")
    sizes = np.bincount(np.unique(comp, return_inverse=True)[1])
    report = {
        "items": n,
        "components": int(len(sizes)),
        "largest_component": int(sizes.max()) if n else 0,
        "dup_edges": int(len(dup_edges)),
        "train_items_dropped_for_test_leakage": dropped,
        "split_counts": {s: split.count(s) for s in ("train", "val", "test", "drop")},
    }
    return split, comp, report
