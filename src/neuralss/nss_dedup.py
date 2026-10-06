"""Perceptual near-duplicate (copy) detection across the combined dataset and leakage-safe splitting.

Two-stage copy detection, because global similarity alone cannot tell "the same picture" from "a similar picture"
(two photos of cherry trees, two skylines or two frames of one game all have high DINOv2 cosine similarity):

1. Candidates: for every keyframe, its nearest neighbours in other leakage groups by DINOv2 cosine similarity
   (exhaustive GPU search) above a high-recall threshold, plus pHash pairs within a loose Hamming radius.
2. Verification: SIFT keypoints matched with Lowe's ratio test and a RANSAC affine fit, also against the mirrored
   image. The score is the number of 8 x 8 grid cells covered by geometrically consistent matches, so a shared logo,
   HUD or caption cannot verify two different frames on its own.

A pair is a duplicate edge if it is verified, or if its cosine similarity or pHash distance lies beyond anything
seen among hard negatives. Thresholds come from synthetic copies (positives) and from negatives that include hard
ones: nearest neighbours between items known to be distinct (curated stills, different rendered environments).

Splitting: leakage groups (same sequence, video, scene family, environment, town) are the unit of assignment. Items
that share a group with a test item, or reach a test item through duplicate edges, are dropped; the rest form
components (groups joined by duplicate edges) that go to validation or training by a deterministic hash.
"""

import hashlib
import io
from collections import defaultdict, deque
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------------------------------------------
# Hashes
# ---------------------------------------------------------------------------------------------------------------
def _dct_matrix(n=32):
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


_DCT32 = _dct_matrix(32)


def phash_bits(thumbs):
    """thumbs: uint8 (N, h, w, 3) -> bool (N, 64). DCT of a 32x32 grey image, 8x8 low frequencies (DC excluded
    from the median), each bit = coefficient above the median."""
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


def hamming(a, b):
    """Row-wise Hamming distance of two bool (N, 64) arrays."""
    return (a != b).sum(1)


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


@torch.no_grad()
def topk_neighbours(query, ref, k, device, query_groups, ref_groups, chunk=2048):
    """For each query row, the k ref rows with the largest dot product, never from the query's own group (which
    also excludes the row itself when query is ref). Returns idx int64 (Q, k) with -1 for empty slots, and scores.
    """
    q = torch.from_numpy(np.ascontiguousarray(query)).to(device).half()
    r = torch.from_numpy(np.ascontiguousarray(ref)).to(device).half()
    qg = torch.from_numpy(np.asarray(query_groups, np.int64)).to(device)
    rg = torch.from_numpy(np.asarray(ref_groups, np.int64)).to(device)
    k = min(k, r.shape[0])
    idx_out, sim_out = [], []
    for a in range(0, q.shape[0], chunk):
        sim = (q[a : a + chunk] @ r.T).float()
        sim.masked_fill_(qg[a : a + chunk, None] == rg[None, :], float("-inf"))
        s, i = sim.topk(k, dim=1)
        i[~torch.isfinite(s)] = -1
        idx_out.append(i.cpu())
        sim_out.append(s.cpu())
    if not idx_out:
        return np.zeros((0, k), np.int64), np.zeros((0, k), np.float32)
    return torch.cat(idx_out).numpy(), torch.cat(sim_out).numpy()


def bits_to_pm1(bits):
    return np.where(bits, 1.0, -1.0).astype(np.float32)


def hamming_threshold_to_dot(h, nbits=64):
    """Hamming distance <= h  <=>  dot of +-1 vectors >= nbits - 2h."""
    return float(nbits - 2 * h)


# ---------------------------------------------------------------------------------------------------------------
# Geometric verification with local features (CPU process pool; tables are shared with forked workers)
# ---------------------------------------------------------------------------------------------------------------
_IMAGES = None  # image array read by forked feature workers
_FEATS = None  # feature table read by forked verification workers
GRID = (
    8  # verification score = occupied cells of a GRID x GRID grid over the first image
)


def _sift(img, n_features):
    import cv2

    grey = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    kp, desc = cv2.SIFT_create(nfeatures=n_features).detectAndCompute(grey, None)
    if desc is None or len(kp) < 3:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
    pts = np.float32([p.pt for p in kp])
    return pts, np.clip(desc, 0, 255).astype(np.uint8)


def _features_task(args):
    import cv2

    cv2.setNumThreads(1)
    i, n_features = args
    img = _IMAGES[i]
    return _sift(img, n_features), _sift(img[:, ::-1], n_features)


def local_features(images, workers, n_features=400, desc="SIFT features"):
    """SIFT keypoints and uint8 descriptors of every image and of its mirror: list of ((pts, desc), (pts, desc))."""
    global _IMAGES
    from tqdm.auto import tqdm

    _IMAGES = images
    try:
        with Pool(workers) as pool:
            return list(
                tqdm(
                    pool.imap(
                        _features_task,
                        [(i, n_features) for i in range(len(images))],
                        chunksize=32,
                    ),
                    total=len(images),
                    desc=desc,
                )
            )
    finally:
        _IMAGES = None


def match_cells(fa, fb, size=256, ratio=0.8, reproj=4.0):
    """Grid cells of image A covered by matches consistent with one orientation-preserving affine map A -> B.
    Degenerate maps (extreme or collapsing scale) score 0."""
    import cv2

    (pa, da), (pb, db) = fa, fb
    if len(da) < 3 or len(db) < 3:
        return 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        da.astype(np.float32), db.astype(np.float32), k=2
    )
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if len(good) < 6:
        return 0
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    M, mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=reproj, maxIters=1000
    )
    if M is None or mask is None:
        return 0
    sv = np.linalg.svd(M[:, :2], compute_uv=False)
    if np.linalg.det(M[:, :2]) <= 0 or sv.min() < 0.2 or sv.max() > 5.0:
        return 0
    inl = src[mask.ravel().astype(bool)]
    if len(inl) < 6:
        return 0
    cells = np.clip((inl * GRID / size).astype(int), 0, GRID - 1)
    return int(len({(int(x), int(y)) for x, y in cells}))


def _verify_task(pair):
    import cv2

    cv2.setNumThreads(1)
    i, j = pair
    a, (b, b_flip) = _FEATS[i][0], _FEATS[j]
    return max(match_cells(a, b), match_cells(a, b_flip))


def verify_pairs(feats, pairs, workers, desc="Geometric verification"):
    """Verification score (covered grid cells, 0..64) for each (i, j) pair of rows of the feature table."""
    global _FEATS
    from tqdm.auto import tqdm

    pairs = [(int(i), int(j)) for i, j in pairs]
    if not pairs:
        return np.zeros(0, np.int32)
    _FEATS = feats
    try:
        with Pool(workers) as pool:
            return np.array(
                list(
                    tqdm(
                        pool.imap(_verify_task, pairs, chunksize=64),
                        total=len(pairs),
                        desc=desc,
                    )
                ),
                np.int32,
            )
    finally:
        _FEATS = None


# ---------------------------------------------------------------------------------------------------------------
# Calibration: synthetic copies (positives); negatives are chosen by the pipeline (random and hard pairs)
# ---------------------------------------------------------------------------------------------------------------
def augment_copy(img, rng, strength="heavy"):
    """A near-duplicate of img as it might appear in another dataset. 'light': rescale, recompress, recolour and
    blur only (what a perceptual hash is meant to survive). 'heavy': additionally re-crop (60 to 95 percent of the
    area), mirror, rotate slightly and overlay text. img: PIL RGB."""
    w, h = img.size
    out = img
    if strength == "heavy":
        a = rng.uniform(0.6, 0.95)
        cw, ch = int(w * np.sqrt(a)), int(h * np.sqrt(a))
        x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
        out = out.crop((x, y, x + cw, y + ch)).resize((256, 256), Image.BILINEAR)
        if rng.random() < 0.5:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.3:
            out = out.rotate(float(rng.uniform(-4, 4)), resample=Image.BILINEAR)
    else:
        s = rng.uniform(0.4, 1.0)
        out = out.resize((max(16, int(w * s)), max(16, int(h * s))), Image.BILINEAR)
        out = out.resize((256, 256), Image.BILINEAR)
    if rng.random() < 0.5:
        arr = np.asarray(out, np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        out = out.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.5))))
    if strength == "heavy" and rng.random() < 0.3:
        ImageDraw.Draw(out).text(
            (int(rng.integers(5, 120)), int(rng.integers(5, 220))),
            "SAMPLE 1080p",
            fill=(255, 255, 255),
        )
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=int(rng.integers(35, 92)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def synthetic_copies(thumbs, candidates, n_pos, seed):
    """Copies of n_pos random thumbnails (drawn from the candidates index array), half light and half heavy.
    Returns (source indices, uint8 copies (n, 256, 256, 3), strength labels)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(candidates, size=min(n_pos, len(candidates)), replace=False)
    kinds = np.array(["light" if k % 2 else "heavy" for k in range(len(idx))])
    copies = np.stack(
        [
            np.asarray(
                augment_copy(Image.fromarray(thumbs[i]), rng, kind).resize((256, 256))
            )
            for i, kind in zip(idx, kinds)
        ]
    )
    return idx, copies, kinds


def roc_curve(scores, labels, points=400):
    """False-positive and true-positive rate as the threshold sweeps down the scores (higher = more similar)."""
    order = np.argsort(-scores, kind="stable")
    y = labels[order]
    tpr = np.cumsum(y == 1) / max((labels == 1).sum(), 1)
    fpr = np.cumsum(y == 0) / max((labels == 0).sum(), 1)
    step = max(1, len(fpr) // points)
    return {"fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist()}


def edge_rule(cos, ham, dham, cells, textured, thr):
    """Duplicate decision for arrays of pair scores. Returns an object array: 'verified' (geometric verification),
    'strong_embedding' (cosine beyond every hard negative), 'exact_hash' (pHash and dHash both nearly identical on
    textured frames) or '' (not a duplicate). cos may be NaN when embeddings are unavailable.
    """
    rule = np.full(len(cells), "", dtype=object)
    h = thr["phash"]["h_exact"]
    exact = textured & (ham <= h) & (dham <= 2 * h + 4)
    rule[exact] = "exact_hash"
    if "embedding" in thr:
        rule[np.nan_to_num(cos, nan=-1.0) >= thr["embedding"]["t_strong"]] = (
            "strong_embedding"
        )
    rule[cells >= thr["verify"]["t_cells"]] = "verified"
    return rule


# ---------------------------------------------------------------------------------------------------------------
# Leakage-safe split
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


def _components(n, members, edges):
    """Union-find over the given member indices joined by shared group and by edges; returns root per index."""
    uf = UnionFind(n)
    first = {}
    for i, g in members:
        if g in first:
            uf.union(i, first[g])
        else:
            first[g] = i
    for i, j in edges:
        uf.union(i, j)
    return uf


def assign_splits(items, dup_edges, val_fraction, seed, active=None):
    """items: dicts with 'group', 'role' ('train' or 'test') and 'item_id'; dup_edges: (i, j) item index pairs;
    active: keep flags (inactive items get split 'filtered' and are ignored).

    1. Provenance: a leakage group that holds a test item is test-only; its other items are dropped.
    2. Content: non-test items connected to a test item through duplicate edges (transitively, but never through
       shared groups, so one copied clip does not take its whole folder with it) are dropped.
    3. The remaining items form components (shared groups joined by duplicate edges); each component goes to
       validation with probability val_fraction (deterministic hash of its root item), else to training.
    Returns (split list, component id list, drop reason list, report)."""
    n = len(items)
    active = [True] * n if active is None else list(active)
    edges = [
        (int(i), int(j))
        for i, j in dup_edges
        if active[int(i)] and active[int(j)] and int(i) != int(j)
    ]
    is_test = [active[i] and it["role"] == "test" for i, it in enumerate(items)]
    test_groups = {it["group"] for i, it in enumerate(items) if is_test[i]}

    adj = defaultdict(list)
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    reached = [False] * n
    queue = deque(i for i in range(n) if is_test[i])
    for i in queue:
        reached[i] = True
    while queue:
        a = queue.popleft()
        for b in adj[a]:
            if not reached[b]:
                reached[b] = True
                queue.append(b)

    split, reason = [None] * n, [None] * n
    for i, it in enumerate(items):
        if not active[i]:
            split[i] = "filtered"
        elif is_test[i]:
            split[i] = "test"
        elif it["group"] in test_groups:
            split[i], reason[i] = "drop", "test_group"
        elif reached[i]:
            split[i], reason[i] = "drop", "test_duplicate"

    rest = [i for i in range(n) if split[i] is None]
    rest_set = set(rest)
    uf = _components(
        n,
        [(i, items[i]["group"]) for i in rest],
        [(i, j) for i, j in edges if i in rest_set and j in rest_set],
    )
    for i in rest:
        root = uf.find(i)
        digest = hashlib.md5(f"{seed}/{items[root]['item_id']}".encode()).hexdigest()
        split[i] = (
            "val" if (int(digest, 16) % 10_000) / 10_000.0 < val_fraction else "train"
        )

    # component ids over every active item (groups and all duplicate edges), for the manifest and the report
    act = [i for i in range(n) if active[i]]
    uf_all = _components(n, [(i, items[i]["group"]) for i in act], edges)
    comp = [uf_all.find(i) if active[i] else -1 for i in range(n)]
    uf_dup = _components(n, [], edges)
    dup_sizes = np.bincount([uf_dup.find(i) for i in act], minlength=n) if act else [0]
    rest_sizes = np.bincount([uf.find(i) for i in rest], minlength=n) if rest else [0]
    report = {
        "items": n,
        "active": len(act),
        "dup_edges": len(edges),
        "largest_duplicate_cluster": int(np.max(dup_sizes)),
        "trainval_components": int(np.count_nonzero(rest_sizes)),
        "largest_trainval_component": int(np.max(rest_sizes)),
        "dropped_test_group": reason.count("test_group"),
        "dropped_test_duplicate": reason.count("test_duplicate"),
        "split_counts": {
            s: split.count(s) for s in ("train", "val", "test", "drop", "filtered")
        },
    }
    return split, comp, reason, report
