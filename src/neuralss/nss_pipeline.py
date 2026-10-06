"""Unified data pipeline steps (used cell by cell in the data-pipeline notebook, and end to end by
build_manifest() when the supersampling or frame-generation notebook runs without the pipeline output attached).

discover -> materialise Hugging Face subsets -> keyframe thumbnails and quality statistics -> filters ->
perceptual hashes, embeddings and local features -> threshold calibration -> candidate search and geometric
verification -> leakage-safe split with a cross-split audit (repaired until clean).
"""

import shutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import nss_data as D
import nss_dedup as DD

FILTER_RULES = {
    "min_side": 256,
    "min_std": 8.0,
    "max_letterbox": 0.35,
    "min_sharpness": {"still": 20.0, "seq": 5.0},
}
DEDUP = {
    "k": 16,  # embedding neighbours per keyframe considered as candidates
    "retrieval_recall": 0.98,  # share of synthetic copies the candidate threshold must retrieve
    "sift_features": 400,
    "min_std_hash": 15.0,  # hash matches only count between keyframes with this much structure
    "max_hash_candidates_per_kf": 20,
    "audit_k": 5,  # nearest reference neighbours verified per val / test keyframe
    "audit_rounds": 3,
}
EDGE_COLUMNS = [
    "kf_a",
    "kf_b",
    "item_a",
    "item_b",
    "cosine",
    "hamming",
    "cells",
    "rule",
]


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
    """Item-level keep flags and rejection reasons. Stills are judged on their single keyframe; sequences on the
    median of their keyframes, so a fade-in or one dark shot does not reject a whole video.
    """
    reasons = {}
    for i, g in kf.groupby("item"):
        kind = items[i]["kind"]
        r = []
        if not g["ok"].all():
            r.append("decode_error")
        else:
            if g[["width", "height"]].min(axis=1).min() < rules["min_side"]:
                r.append("too_small")
            if g["std"].median() < rules["min_std"]:
                r.append("flat")
            if g["letterbox"].median() > rules["max_letterbox"]:
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


def keyframe_groups(kf, items):
    """Integer leakage-group id of every keyframe (pairs inside one group never matter for the split)."""
    return pd.factorize(pd.Series([items[i]["group"] for i in kf["item"]]))[0]


def _pair_scores(a, b, emb_a, emb_b, ph_a, ph_b, dh_a, dh_b, std_a, std_b, min_std):
    cos = (
        (emb_a * emb_b).sum(1)
        if emb_a is not None
        else np.full(len(a), np.nan, np.float32)
    )
    return {
        "cosine": cos,
        "hamming": DD.hamming(ph_a, ph_b),
        "dhash": DD.hamming(dh_a, dh_b),
        "textured": (std_a >= min_std) & (std_b >= min_std),
    }


def calibrate(
    thumbs,
    kf,
    items,
    emb,
    ph,
    dh,
    feats,
    embedders,
    devices,
    n_pos,
    seed,
    workers,
    cfg=DEDUP,
    log=print,
):
    """Thresholds from synthetic copies (positives, half light and half heavy) against random negatives and hard
    negatives: the nearest keyframe in another group among sources whose groups are known to be distinct.

    * embedding: t_ret retrieves cfg["retrieval_recall"] of the copies (candidate stage, precision comes from
      verification); t_strong lies above the most similar negative (accepted without verification).
    * verify: t_cells lies above the most similar negative's verification score (at least 10 of 64 cells).
    * phash: h_exact lies 4 bits below the closest negative (at most 6); h_cand covers 95 percent of light copies.
    "Most similar negative" is the extreme hard negative or the 99.9th percentile of random negatives.
    Returns (thresholds, per-pair score table)."""
    rng = np.random.default_rng(seed + 1)
    active = kf["active"].values & kf["ok"].values
    std = kf["std"].fillna(0).values
    gid = keyframe_groups(kf, items)
    act = np.nonzero(active)[0]

    src, copies, kinds = DD.synthetic_copies(thumbs, act, n_pos, seed)
    copy_feats = DD.local_features(
        copies, workers, cfg["sift_features"], desc="SIFT features (synthetic copies)"
    )
    emb_c = DD.embed_thumbs(copies, embedders, devices) if embedders else None
    ph_c, dh_c = hashes(copies)
    pos = _pair_scores(
        src,
        src,
        None if emb is None else emb[src],
        emb_c,
        ph[src],
        ph_c,
        dh[src],
        dh_c,
        std[src],
        std[src],
        cfg["min_std_hash"],
    )
    pos["cells"] = DD.verify_pairs(
        feats + copy_feats,
        [(i, len(feats) + m) for m, i in enumerate(src)],
        workers,
        desc="Verification (synthetic copies)",
    )

    i = rng.choice(act, size=2 * len(src))
    j = rng.choice(act, size=2 * len(src))
    keep = gid[i] != gid[j]
    neg_pairs = [np.stack([i[keep], j[keep]], 1)]
    neg_kind = ["random"] * int(keep.sum())
    distinct = {s["name"] for s in D.SOURCES if s.get("distinct_groups")}
    pool = np.array(
        [k for k in act if items[kf["item"].values[k]]["source"] in distinct]
    )
    if len(pool) > 1:
        feats_pool = emb[pool] if emb is not None else DD.bits_to_pm1(ph[pool])
        idx, _ = DD.topk_neighbours(
            feats_pool, feats_pool, 1, devices[0], gid[pool], gid[pool]
        )
        ok = idx[:, 0] >= 0
        hard = np.sort(np.stack([pool[ok], pool[idx[ok, 0]]], 1), axis=1)
        hard = np.unique(hard, axis=0)
        neg_pairs.append(hard)
        neg_kind += ["hard"] * len(hard)
    neg_pairs = np.concatenate(neg_pairs)
    a, b = neg_pairs[:, 0], neg_pairs[:, 1]
    neg = _pair_scores(
        a,
        b,
        None if emb is None else emb[a],
        None if emb is None else emb[b],
        ph[a],
        ph[b],
        dh[a],
        dh[b],
        std[a],
        std[b],
        cfg["min_std_hash"],
    )
    neg["cells"] = DD.verify_pairs(
        feats, neg_pairs, workers, desc="Verification (negatives)"
    )
    log(
        f"calibration: {len(src)} synthetic copies, {neg_kind.count('random')} random and "
        f"{neg_kind.count('hard')} hard negatives"
    )

    hard_mask = np.array(neg_kind) == "hard"

    def worst(values, high=True):
        """Most duplicate-like negative score: the extreme hard negative (trusted labels) or the 99.9th
        percentile of random negatives (a rare random pair may be a genuine duplicate).
        """
        q = 0.999 if high else 0.001
        rand = float(np.quantile(values[~hard_mask], q)) if (~hard_mask).any() else None
        hard = (
            (values[hard_mask].max() if high else values[hard_mask].min())
            if hard_mask.any()
            else None
        )
        vals = [v for v in (rand, hard) if v is not None]
        return float(max(vals) if high else min(vals))

    thr = {}
    if emb is not None:
        cos_p, cos_n = pos["cosine"], neg["cosine"]
        t_ret = float(
            np.clip(np.quantile(cos_p, 1 - cfg["retrieval_recall"]), 0.5, 0.9)
        )
        t_strong = float(np.clip(worst(cos_n) + 0.03, 0.92, 0.995))
        thr["embedding"] = {
            "t_ret": t_ret,
            "t_strong": t_strong,
            "recall_at_t_ret": float((cos_p >= t_ret).mean()),
            "recall_at_t_strong": float((cos_p >= t_strong).mean()),
            "worst_negative": worst(cos_n),
        }
    t_cells = int(np.clip(worst(neg["cells"]) + 2, 10, 40))
    thr["verify"] = {
        "t_cells": t_cells,
        "recall": float((pos["cells"] >= t_cells).mean()),
        "worst_negative": worst(neg["cells"]),
    }
    h_exact = int(np.clip(worst(neg["hamming"], high=False) - 4, 0, 6))
    light = kinds == "light"
    h_cand = int(np.clip(np.quantile(pos["hamming"][light], 0.95), h_exact, 12))
    thr["phash"] = {
        "h_exact": h_exact,
        "h_cand": h_cand,
        "recall_light_at_h_exact": float((pos["hamming"][light] <= h_exact).mean()),
        "recall_light_at_h_cand": float((pos["hamming"][light] <= h_cand).mean()),
        "worst_negative": worst(neg["hamming"], high=False),
    }

    def table(scores, label, kind):
        return pd.DataFrame(
            {
                "label": label,
                "kind": kind,
                "cosine": scores["cosine"],
                "hamming": scores["hamming"],
                "dhash": scores["dhash"],
                "textured": scores["textured"],
                "cells": scores["cells"],
            }
        )

    scores = pd.concat(
        [table(pos, 1, kinds), table(neg, 0, np.array(neg_kind))], ignore_index=True
    )
    scores["rule"] = DD.edge_rule(
        scores["cosine"].values,
        scores["hamming"].values,
        scores["dhash"].values,
        scores["cells"].values,
        scores["textured"].values,
        thr,
    )
    hit = scores["rule"] != ""
    tp, fp = int((hit & (scores["label"] == 1)).sum()), int(
        (hit & (scores["label"] == 0)).sum()
    )
    thr["rule"] = {
        "recall": tp / max(int((scores["label"] == 1).sum()), 1),
        "recall_light": float(
            hit[(scores["label"] == 1) & (scores["kind"] == "light")].mean()
        ),
        "recall_heavy": float(
            hit[(scores["label"] == 1) & (scores["kind"] == "heavy")].mean()
        ),
        "false_positives": fp,
        "precision": tp / max(tp + fp, 1),
    }
    if emb is not None:
        thr["embedding"]["roc"] = DD.roc_curve(
            scores["cosine"].values, scores["label"].values
        )
    return thr, scores


def find_duplicates(
    kf, items, ph, dh, emb, feats, thr, device, workers, cfg=DEDUP, log=print
):
    """Candidate pairs (embedding top-k above t_ret, pHash within h_cand on textured keyframes, never inside one
    group or with a filtered item), scored and verified; returns (edge table, statistics). If hash candidates are
    implausibly many, h_cand is tightened (dark or repetitive frames would otherwise flood verification).
    """
    n = len(kf)
    active = kf["active"].values & kf["ok"].values
    std = kf["std"].fillna(0).values
    gid = keyframe_groups(kf, items)
    act = np.nonzero(active)[0]
    stats = {"keyframes_searched": int(len(act))}
    cand = [np.zeros((0, 2), np.int64)]
    if emb is not None and len(act):
        idx, sim = DD.topk_neighbours(
            emb[act], emb[act], cfg["k"], device, gid[act], gid[act]
        )
        rows = np.repeat(np.arange(len(act)), idx.shape[1])
        ok = (idx.ravel() >= 0) & (sim.ravel() >= thr["embedding"]["t_ret"])
        cand.append(np.stack([act[rows[ok]], act[idx.ravel()[ok]]], 1))
        stats["candidates_embedding"] = int(ok.sum())
    textured = active & (std >= cfg["min_std_hash"])
    tex = np.nonzero(textured)[0]
    h, pairs = thr["phash"]["h_cand"], np.zeros((0, 2), np.int64)
    while len(tex) > 1:
        found, _ = DD.pairs_above(
            DD.bits_to_pm1(ph[tex]), DD.hamming_threshold_to_dot(h), device
        )
        pairs = tex[found]
        pairs = pairs[gid[pairs[:, 0]] != gid[pairs[:, 1]]]
        too_many = len(pairs) > cfg["max_hash_candidates_per_kf"] * n
        if not too_many or h <= thr["phash"]["h_exact"]:
            break
        h -= 1
    thr["phash"]["h_cand_final"] = int(h)
    cand.append(pairs)
    stats["candidates_phash"] = int(len(pairs))
    cand = np.unique(np.sort(np.concatenate(cand), axis=1), axis=0)
    stats["candidates"] = int(len(cand))
    log(f"duplicate candidates: {stats}")
    a, b = cand[:, 0], cand[:, 1]
    sc = _pair_scores(
        a,
        b,
        None if emb is None else emb[a],
        None if emb is None else emb[b],
        ph[a],
        ph[b],
        dh[a],
        dh[b],
        std[a],
        std[b],
        cfg["min_std_hash"],
    )
    cells = DD.verify_pairs(feats, cand, workers, desc="Verification (candidates)")
    rule = DD.edge_rule(
        sc["cosine"], sc["hamming"], sc["dhash"], cells, sc["textured"], thr
    )
    item = kf["item"].values
    edges = pd.DataFrame(
        {
            "kf_a": a,
            "kf_b": b,
            "item_a": item[a],
            "item_b": item[b],
            "cosine": sc["cosine"],
            "hamming": sc["hamming"],
            "cells": cells,
            "rule": rule,
        }
    )
    edges = edges[edges["rule"] != ""].reset_index(drop=True)
    stats.update(
        {
            f"edges_{r}": int((edges["rule"] == r).sum())
            for r in ("verified", "strong_embedding", "exact_hash")
        }
    )
    stats["edges"] = int(len(edges))
    stats["verification_score_hist"] = np.bincount(cells, minlength=65).tolist()
    return edges, stats


def split_items(items, edges, val_fraction, seed, active):
    pairs = list(zip(edges["item_a"], edges["item_b"])) if len(edges) else []
    split, comp, reason, report = DD.assign_splits(
        items, pairs, val_fraction, seed, active
    )
    for it, s, c, r in zip(items, split, comp, reason):
        it["split"], it["component"], it["drop_reason"] = s, int(c), r
    return report


def cross_split_audit(kf, items, ph, dh, emb, feats, thr, device, workers, cfg=DEDUP):
    """Every validation keyframe against training, every test keyframe against training and validation: its
    audit_k nearest reference keyframes by cosine and its nearest by pHash are scored and verified with the same
    rule as the search. Returns (per-split arrays for plots, violation edge table)."""
    split = np.array([items[i]["split"] for i in kf["item"]])
    std = kf["std"].fillna(0).values
    gid = keyframe_groups(kf, items)
    item = kf["item"].values
    pm1 = DD.bits_to_pm1(ph)
    out, violations = {}, []
    for s, refs in (("val", ("train",)), ("test", ("train", "val"))):
        q, r = np.nonzero(split == s)[0], np.nonzero(np.isin(split, refs))[0]
        if not len(q) or not len(r):
            continue
        pairs, res = [], {"reference": "+".join(refs)}
        if emb is not None:
            idx, sim = DD.topk_neighbours(
                emb[q], emb[r], cfg["audit_k"], device, gid[q], gid[r]
            )
            res["max_cosine"] = sim[:, 0]
            rows = np.repeat(np.arange(len(q)), idx.shape[1])
            ok = idx.ravel() >= 0
            pairs.append(np.stack([q[rows[ok]], r[idx.ravel()[ok]]], 1))
        hidx, hsim = DD.topk_neighbours(pm1[q], pm1[r], 1, device, gid[q], gid[r])
        res["min_hamming"] = (64 - hsim[:, 0]) / 2
        ok = hidx[:, 0] >= 0
        pairs.append(np.stack([q[ok], r[hidx[ok, 0]]], 1))
        pairs = np.unique(np.concatenate(pairs), axis=0)
        a, b = pairs[:, 0], pairs[:, 1]
        sc = _pair_scores(
            a,
            b,
            None if emb is None else emb[a],
            None if emb is None else emb[b],
            ph[a],
            ph[b],
            dh[a],
            dh[b],
            std[a],
            std[b],
            cfg["min_std_hash"],
        )
        cells = DD.verify_pairs(feats, pairs, workers, desc=f"Audit {s}")
        best = pd.Series(cells).groupby(a).max()
        res["max_cells"] = best.reindex(q).fillna(0).values
        rule = DD.edge_rule(
            sc["cosine"], sc["hamming"], sc["dhash"], cells, sc["textured"], thr
        )
        bad = rule != ""
        violations.append(
            pd.DataFrame(
                {
                    "kf_a": a[bad],
                    "kf_b": b[bad],
                    "item_a": item[a[bad]],
                    "item_b": item[b[bad]],
                    "cosine": sc["cosine"][bad],
                    "hamming": sc["hamming"][bad],
                    "cells": cells[bad],
                    "rule": rule[bad],
                    "split": s,
                }
            )
        )
        out[s] = res
    viol = (
        pd.concat(violations, ignore_index=True)
        if violations
        else pd.DataFrame(columns=EDGE_COLUMNS + ["split"])
    )
    return out, viol


def split_and_audit(
    kf,
    items,
    edges,
    active,
    ph,
    dh,
    emb,
    feats,
    thr,
    device,
    workers,
    val_fraction,
    seed,
    cfg=DEDUP,
    log=print,
):
    """Split, audit, and while the audit finds cross-split duplicates (pairs the top-k candidate search missed),
    add them as edges and split again. Returns (report, audit, remaining violations, final edges).
    """
    for rnd in range(cfg["audit_rounds"]):
        report = split_items(items, edges, val_fraction, seed, active)
        audit, viol = cross_split_audit(
            kf, items, ph, dh, emb, feats, thr, device, workers, cfg
        )
        report["audit_rounds"] = rnd + 1
        report["audit_violations"] = int(len(viol))
        if not len(viol):
            break
        log(
            f"audit round {rnd + 1}: {len(viol)} cross-split duplicates; adding them as edges and re-splitting"
        )
        edges = pd.concat([edges, viol[EDGE_COLUMNS]], ignore_index=True)
        edges = edges.drop_duplicates(["kf_a", "kf_b"]).reset_index(drop=True)
    return report, audit, viol, edges


def load_embedders(devices, log=print):
    """One DINOv2 copy per GPU, or [] if the weights are unavailable."""
    embedders = []
    for dvc in devices:
        e, _ = DD.load_embedder(dvc, log=log)
        if e is None:
            return []
        embedders.append(e)
    return embedders


def build_manifest(data_root, input_root, cfg, devices, log=print):
    """End-to-end fallback used by the training notebooks when no pipeline output is attached."""
    dd = {**DEDUP, **cfg.get("dedup", {})}
    Path(data_root).mkdir(parents=True, exist_ok=True)
    items, _ = discover_kaggle(input_root, cfg["list_cap_files"], cfg["seed"], log=log)
    items += materialise_hf(data_root, cfg["hf_caps"], cfg["seed"], log=log)
    thumbs, kf = keyframe_stats(items, data_root, cfg["workers"])
    keep, _ = apply_filters(items, kf)
    kf["active"] = [keep[i] for i in kf["item"]]
    ph, dh = hashes(thumbs)
    embedders = load_embedders(devices, log=log)
    emb = DD.embed_thumbs(thumbs, embedders, devices) if embedders else None
    feats = DD.local_features(thumbs, cfg["workers"], dd["sift_features"])
    thr, _ = calibrate(
        thumbs,
        kf,
        items,
        emb,
        ph,
        dh,
        feats,
        embedders,
        devices,
        cfg["calibration_pairs"],
        cfg["seed"],
        cfg["workers"],
        dd,
        log=log,
    )
    edges, _ = find_duplicates(
        kf, items, ph, dh, emb, feats, thr, devices[0], cfg["workers"], dd, log=log
    )
    report, _, viol, _ = split_and_audit(
        kf,
        items,
        edges,
        keep,
        ph,
        dh,
        emb,
        feats,
        thr,
        devices[0],
        cfg["workers"],
        cfg["val_fraction"],
        cfg["seed"],
        dd,
        log=log,
    )
    if len(viol):
        log(
            f"warning: {len(viol)} cross-split duplicates remain after {report['audit_rounds']} audit rounds"
        )
    D.save_manifest(items, Path(data_root) / "nss_manifest.parquet")
    log(f"manifest: {report['split_counts']}")
    shutil.rmtree(Path(data_root) / "_downloads", ignore_errors=True)
    return items, report
