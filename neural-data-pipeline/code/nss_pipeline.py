"""Unified data pipeline steps (used cell by cell in the data-pipeline notebook, and end to end by
build_manifest() when the supersampling or frame-generation notebook runs without the pipeline output attached).

discover -> materialise Hugging Face subsets -> keyframe thumbnails and quality statistics -> filters ->
perceptual hashes and embeddings -> threshold calibration -> near-duplicate search -> leakage-safe split.
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nss_data as D
import nss_dedup as DD

FILTER_RULES = {
    "min_side": 256,
    "min_std": 8.0,
    "max_letterbox": 0.35,
    "min_sharpness": {"still": 20.0, "seq": 5.0},
}


def discover_kaggle(input_root, cap, seed, log=print):
    items, rows = [], []
    kaggle = [s for s in D.SOURCES if s["host"] == "kaggle"]
    with ThreadPoolExecutor(len(kaggle)) as pool:
        results = list(
            pool.map(
                lambda s: (s, *D.catalogue_kaggle(s, input_root, cap, seed)), kaggle
            )
        )
    for src, its, info in results:
        items.extend(its)
        rows.append(
            {
                "source": src["name"],
                "ref": src["ref"],
                **info,
                "items": len(its),
                "test_items": sum(i["role"] == "test" for i in its),
            }
        )
        if not info.get("found"):
            log(f"{src['name']}: not attached ({src['ref']})")
    return items, pd.DataFrame(rows)


def materialise_hf(data_root, caps, seed, log=print):
    """caps: {"game_stills": (n_train, n_test), "tartanair": frames_per_env, "tartanair_seq": len,
    "gameir": (train_clips, test_clips), "vimeo1080p": (n_train, n_val)}; a cap of 0 skips the source.
    """
    items = []
    steps = [
        (
            "game_stills",
            lambda s: D.materialise_game_stills(
                s, data_root, *caps["game_stills"], seed, log=log
            ),
        ),
        (
            "tartanair",
            lambda s: D.materialise_tartanair(
                s,
                data_root,
                caps["tartanair"],
                s["train_envs"],
                s["test_envs"],
                caps["tartanair_seq"],
                log=log,
            ),
        ),
        (
            "gameir",
            lambda s: D.materialise_gameir(s, data_root, *caps["gameir"], log=log),
        ),
        (
            "vimeo1080p",
            lambda s: D.materialise_vimeo1080p(
                s, data_root, *caps["vimeo1080p"], log=log
            ),
        ),
    ]
    for name, fn in steps:
        cap = caps.get(name)
        if not cap or (isinstance(cap, tuple) and not any(cap)):
            continue
        try:
            items.extend(fn(D.source(name)))
        except Exception as exc:
            log(f"{name}: materialisation failed ({exc}); continuing without it")
    return items


def keyframe_stats(items, data_root, workers):
    """Thumbnails (K, 256, 256, 3) and statistics for every keyframe; returns thumbs, table (one row per keyframe)."""
    tasks, owner = [], []
    for i, it in enumerate(items):
        for uri in D.keyframes(it):
            tasks.append((str(data_root), uri))
            owner.append(i)
    thumbs = np.zeros((len(tasks), 256, 256, 3), np.uint8)
    rows = [None] * len(tasks)
    from tqdm.auto import tqdm

    with Pool(workers) as pool:
        for k, (uri, small, stats) in enumerate(
            tqdm(
                pool.imap(D.thumb_and_stats, tasks, chunksize=4),
                total=len(tasks),
                desc="Keyframe thumbnails and statistics",
            )
        ):
            if small is not None:
                thumbs[k] = small
            rows[k] = {"item": owner[k], "uri": uri, "ok": small is not None, **stats}
    return thumbs, pd.DataFrame(rows)


def apply_filters(items, kf, rules=FILTER_RULES):
    """Item-level keep flags and rejection reasons (an item is rejected if its keyframes fail)."""
    reasons = {}
    for i, g in kf.groupby("item"):
        kind = items[i]["kind"]
        r = []
        if not g["ok"].all():
            r.append("decode_error")
        else:
            if g[["width", "height"]].min(axis=1).min() < rules["min_side"]:
                r.append("too_small")
            if g["std"].min() < rules["min_std"]:
                r.append("flat")
            if g["letterbox"].max() > rules["max_letterbox"]:
                r.append("letterbox")
            if (
                g["sharpness"].median()
                < rules["min_sharpness"]["still" if kind == "still" else "seq"]
            ):
                r.append("blurry")
        reasons[i] = r
    keep = [not reasons.get(i, ["no_keyframe"]) for i in range(len(items))]
    return keep, reasons


def hashes(thumbs):
    return DD.phash_bits(thumbs), DD.dhash_bits(thumbs)


def calibrate(thumbs, embedders, devices, n_pos, seed, max_fpr=0.001, margin=0.02):
    """Thresholds for pHash Hamming distance and embedding cosine similarity from synthetic positive pairs.

    An all-pairs search over K keyframes tests about K^2 / 2 pairs, so a threshold tuned only to a 0.1 percent
    false-positive rate on random pairs would create far too many false edges. Each threshold is therefore the
    stricter of (a) the max_fpr point and (b) the most similar random negative plus a margin; find_duplicates
    then tightens it further if the edge count is still implausible."""
    a, b, y = DD.calibration_pairs(thumbs, n_pos, seed)
    pa, pb = DD.phash_bits(a), DD.phash_bits(b)
    ham = (pa != pb).sum(1)
    res = {"phash": DD.choose_threshold(-ham.astype(np.float32), y, max_fpr)}
    neg_min_ham = int(ham[y == 0].min()) if (y == 0).any() else 64
    res["phash"]["hamming"] = int(min(-res["phash"]["threshold"], neg_min_ham - 2))
    res["phash"]["recall_at_threshold"] = float(
        (ham[y == 1] <= res["phash"]["hamming"]).mean()
    )
    res["phash_scores"], res["labels"] = ham.tolist(), y.tolist()
    if embedders:
        ea, eb = DD.embed_thumbs(a, embedders, devices), DD.embed_thumbs(
            b, embedders, devices
        )
        cos = (ea * eb).sum(1)
        res["embedding"] = DD.choose_threshold(cos, y, max_fpr)
        res["embedding"]["threshold"] = float(
            max(res["embedding"]["threshold"], cos[y == 0].max() + margin)
        )
        res["embedding"]["recall_at_threshold"] = float(
            (cos[y == 1] >= res["embedding"]["threshold"]).mean()
        )
        res["embedding_scores"] = cos.tolist()
    return res


def find_duplicates(
    kf, phash, emb, thresholds, device, dhash=None, min_std=15.0, max_degree=2.0
):
    """Keyframe pairs judged duplicates by pHash (Hamming <= h, confirmed by dHash, only for keyframes with enough
    structure: low-contrast frames share hashes by accident) or by embeddings (cosine >= c), mapped to items
    (pairs inside one item are ignored). If the number of item edges exceeds max_degree per keyframe, the
    thresholds are tightened step by step (an implausible duplicate rate means false positives, which would merge
    unrelated data into giant components). Returns the pair table; final thresholds are written into thresholds.
    """
    n = len(kf)
    textured = kf["std"].fillna(0).values >= min_std
    h = thresholds["phash"]["hamming"]
    while True:
        rows = []
        pairs, scores = DD.pairs_above(
            DD.bits_to_pm1(phash), DD.hamming_threshold_to_dot(h), device
        )
        for (i, j), sc in zip(pairs, scores):
            if not (textured[i] and textured[j]):
                continue
            if dhash is not None and (dhash[i] != dhash[j]).sum() > 2 * h + 4:
                continue
            rows.append((i, j, "phash", (64 - sc) / 2))
        if len(rows) <= max_degree * n or h <= 0:
            break
        h -= 1
    thresholds["phash"]["hamming_final"] = int(h)
    if emb is not None and "embedding" in thresholds:
        c = thresholds["embedding"]["threshold"]
        while True:
            pairs, scores = DD.pairs_above(emb, c, device)
            if len(pairs) <= max_degree * n or c >= 0.995:
                break
            c = min(0.995, c + 0.01)
        thresholds["embedding"]["threshold_final"] = float(c)
        rows += [(i, j, "embedding", sc) for (i, j), sc in zip(pairs, scores)]
    df = pd.DataFrame(rows, columns=["kf_a", "kf_b", "method", "score"])
    if len(df):
        df["item_a"] = kf["item"].values[df["kf_a"]]
        df["item_b"] = kf["item"].values[df["kf_b"]]
        df = df[df["item_a"] != df["item_b"]].reset_index(drop=True)
    return df


def split_items(items, dup_pairs, val_fraction, seed):
    edges = (
        list(zip(dup_pairs["item_a"], dup_pairs["item_b"])) if len(dup_pairs) else []
    )
    split, comp, report = DD.assign_splits(items, edges, val_fraction, seed)
    for it, s, c in zip(items, split, comp):
        it["split"], it["component"] = s, int(c)
    return report


def cross_split_audit(kf, items, emb, phash, device, k=1):
    """Nearest neighbour of every test / val keyframe among train keyframes (cosine and Hamming)."""
    split = np.array([items[i]["split"] for i in kf["item"]])
    tr = np.nonzero(split == "train")[0]
    out = {}
    for s in ("val", "test"):
        q = np.nonzero(split == s)[0]
        if not len(q) or not len(tr):
            continue
        res = {}
        if emb is not None:
            x = torch.from_numpy(emb[tr]).to(device).half()
            best = []
            for a in range(0, len(q), 4096):
                sim = torch.from_numpy(emb[q[a : a + 4096]]).to(device).half() @ x.T
                best.append(sim.float().max(1).values.cpu())
            res["max_cosine_to_train"] = torch.cat(best).numpy()
        x = torch.from_numpy(DD.bits_to_pm1(phash[tr])).to(device).half()
        best = []
        for a in range(0, len(q), 4096):
            dot = (
                torch.from_numpy(DD.bits_to_pm1(phash[q[a : a + 4096]]))
                .to(device)
                .half()
                @ x.T
            )
            best.append(((64 - dot.float().max(1).values) / 2).cpu())
        res["min_hamming_to_train"] = torch.cat(best).numpy()
        out[s] = res
    return out


def build_manifest(data_root, input_root, cfg, devices, log=print):
    """End-to-end fallback used by the training notebooks when no pipeline output is attached."""
    Path(data_root).mkdir(parents=True, exist_ok=True)
    items, _ = discover_kaggle(input_root, cfg["list_cap_files"], cfg["seed"], log=log)
    items += materialise_hf(data_root, cfg["hf_caps"], cfg["seed"], log=log)
    thumbs, kf = keyframe_stats(items, data_root, cfg["workers"])
    keep, _ = apply_filters(items, kf)
    for it, k in zip(items, keep):
        it["filtered_out"] = not k
    ph, dh = hashes(thumbs)
    embedders = []
    for dvc in devices:
        e, _ = DD.load_embedder(dvc, log=log)
        if e is None:
            embedders = []
            break
        embedders.append(e)
    emb = DD.embed_thumbs(thumbs, embedders, devices) if embedders else None
    thr = calibrate(thumbs, embedders, devices, cfg["calibration_pairs"], cfg["seed"])
    pairs = find_duplicates(kf, ph, emb, thr, devices[0], dhash=dh)
    report = split_items(items, pairs, cfg["val_fraction"], cfg["seed"])
    for it in items:
        if it.get("filtered_out"):
            it["split"] = "filtered"
    D.save_manifest(items, Path(data_root) / "nss_manifest.parquet")
    log(f"manifest: {report['split_counts']}")
    shutil.rmtree(Path(data_root) / "_downloads", ignore_errors=True)
    return items, report
