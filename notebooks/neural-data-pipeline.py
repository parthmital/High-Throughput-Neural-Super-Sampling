# %% [markdown]
# # NeuralSS data pipeline: discovery, deduplication and leakage-safe splits (Kaggle GPU T4 x2)
#
# **Goal.** Build the single dataset manifest used by both training notebooks of the NeuralSS stack
# (temporal super-resolution and low-latency frame generation). Only Kaggle and Hugging Face datasets are used.
#
# **Steps.**
#
# 1. Discover attached Kaggle datasets (capped, randomised directory walk) and drop degraded copies (LR, bicubic, blurred, `x2`/`x3`/`x4`) and non-colour passes.
# 2. Materialise selected Hugging Face subsets into `/kaggle/working/nss_data` with range requests and streaming (no full archive downloads): game frames, TartanAir sequences with depth, exact optical flow and occlusion masks, GameIR native 720p / 1440p game renders with depth, and 1080p Vimeo videos.
# 3. Group frames into sequences; every item gets a leakage group (sequence, video, scene family, environment or town).
# 4. Decode keyframes into thumbnails and compute quality statistics; filter too-small, flat, letterboxed and blurry items.
# 5. Perceptual near-duplicate detection across the whole combined dataset: 64-bit DCT pHash and dHash plus DINOv2-small embeddings, exhaustive GPU nearest-neighbour search on both GPUs, thresholds calibrated on synthetic positive pairs (crops, rescales, JPEG, colour shifts, mirroring, overlays) at a 0.1 percent false-positive rate.
# 6. Union-find clustering of duplicate edges and leakage groups; one split per component. Components that touch a test item become test and their training members are dropped.
# 7. Audit: nearest training neighbour of every validation and test keyframe; save the manifest, reports, plots and a dataset card with licences.
#
# **Outputs** (in `/kaggle/working`, saved as this notebook's output; attach it to the training notebooks with **Add Input > Notebook output**):
#
# ```text
# nss_data/nss_manifest.parquet   one row per still or sequence: source, frames (URIs), motion / depth / mask URIs, group, split, component
# nss_data/<hf subsets>/          materialised Hugging Face data referenced by the manifest
# metrics/                        discovery, statistics, calibration, duplicate pairs, split and audit reports, dataset card
# plots/                          every figure as PNG
# logs/notebook.log
# outputs.zip                     metrics, plots and logs (the data itself is already in the Output tab)
# ```
#
# **Required Kaggle settings.** Accelerator **GPU T4 x2**; **Internet on** (Hugging Face data and DINOv2 weights); optional secret `HF_TOKEN` (higher Hub rate limits). Attach these Kaggle datasets:
#
# | Kaggle dataset | Use |
# | :--- | :--- |
# | `soumikrakshit/div2k-high-resolution-images` | stills; `DIV2K_valid_HR` pinned to test |
# | `daehoyang/flickr2k` | stills |
# | `amithkesavmrajagiri/reds-dataset` | 720p video sequences |
# | `wangsally/vimeo-90k-7` | 448x256 septuplets |
# | `chenshu123/vimeo-triplet` | frame-generation test (official test list when present) |
# | `artemmmtry/mpi-sintel-dataset` | rendered sequences with exact flow and occlusions (scene families `market`, `cave` pinned to test) |
# | `uom200647r/vid4-dataset` | video test only |
#
# **Run time.** About 1.5 to 2.5 hours (most of it Hugging Face transfer and decoding). `QUICK_RUN = True` shrinks every cap for a 20-minute smoke test.

# %% [markdown]
# ## 1. Library files
#
# The pipeline code is shared with the two training notebooks. The first cell creates `/kaggle/working/code`, then the following cells write the library into it so the notebook is self-contained: `nss_common` (run folders, logging, stage timer, zip), `nss_data` (source registry, discovery, Hugging Face range-request access, decoding, windows), `nss_dedup` (hashes, embeddings, calibration, GPU neighbour search, union-find split) and `nss_pipeline` (the steps run below).

# %%
import os

os.makedirs(
    "/kaggle/working/code", exist_ok=True
)  # %%writefile does not create folders

# %% [markdown]
# **`nss_common.py`**: run folders, logger, stage timer, figure and metric saving, hardware report, pip helper and the training-process launcher.

# %%
# WRITEFILE nss_common.py

# %% [markdown]
# **`nss_data.py`**: source registry (Kaggle and Hugging Face), capped discovery, HTTP range-request access to remote archives, Hugging Face materialisation, decoding of colour / depth / flow / masks, window and crop workers, manifest I/O.

# %%
# WRITEFILE nss_data.py

# %% [markdown]
# **`nss_dedup.py`**: pHash / dHash, DINOv2 embeddings on both GPUs, exhaustive GPU neighbour search, threshold calibration with synthetic near-duplicates, union-find leakage-safe split.

# %%
# WRITEFILE nss_dedup.py

# %% [markdown]
# **`nss_pipeline.py`**: the data-pipeline steps (discover, materialise, statistics, filters, hashes, calibration, duplicate search, split, audit) and the end-to-end fallback.

# %%
# WRITEFILE nss_pipeline.py

# %% [markdown]
# ## 2. Setup
#
# Imports the library, creates the output folders and logger, and reads the optional `HF_TOKEN` Kaggle secret into the environment (never printed or saved). `RUN` keeps per-stage timings for the summary.

# %%
import os
import sys
import time

sys.path.insert(0, "/kaggle/working/code")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import nss_common as C
import nss_data as D
import nss_dedup as DD
import nss_pipeline as PL

RUN = C.Run("nss_data")
log = RUN.log.info
try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    log("HF_TOKEN secret loaded")
except Exception:
    log("no HF_TOKEN secret: anonymous Hugging Face access")

# %% [markdown]
# ## 3. Hardware check
#
# Detects GPUs, CPU cores, RAM and free disk at run time and saves `metrics/hardware.json`. Both GPUs are used for embeddings and neighbour search; all CPU cores decode images. Expected output: two Tesla T4 with about 15 GB each.

# %%
HW = C.hardware_report()
RUN.save_json(HW, "hardware.json")
print(pd.Series({k: v for k, v in HW.items() if k != "gpu_list"}).to_string())
print(pd.DataFrame(HW["gpu_list"]).to_string(index=False))
assert HW["gpus"] >= 2, "Select the GPU T4 x2 accelerator (Settings > Accelerator)."
DEVICES = [torch.device("cuda", i) for i in range(HW["gpus"])]

# %% [markdown]
# ## 4. Configuration
#
# All caps and thresholds in one place. Hugging Face caps are sized to keep the materialised data under about 12 GB of the 20 GB `/kaggle/working` quota; the cell lowers them further when free disk is short.
#
# * `hf_caps`: game frames (train, test), TartanAir frames per environment and sequence length, GameIR clips (train, test), Vimeo-1080p videos (train, test).
# * `max_fpr`: calibration target for duplicate thresholds (0.1 percent of random pairs may be flagged).
# * `val_fraction`: share of non-test components assigned to validation.

# %%
QUICK_RUN = False
CFG = {
    "seed": 7,
    "list_cap_files": 4000 if QUICK_RUN else 40000,
    "hf_caps": {
        "game_stills": (150, 40) if QUICK_RUN else (1200, 250),
        "tartanair": 40 if QUICK_RUN else 120,
        "tartanair_seq": 24,
        "gameir": (4, 4) if QUICK_RUN else (40, 30),
        "vimeo1080p": (8, 4) if QUICK_RUN else (160, 40),
    },
    "calibration_pairs": 300 if QUICK_RUN else 1500,
    "max_fpr": 0.001,
    "val_fraction": 0.05,
    "workers": os.cpu_count(),
}
DATA_ROOT = C.WORK / "nss_data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
free_gb = HW["disk_free_gb_working"]
if (
    free_gb < 16
):  # keep room for the training notebooks' own outputs when this output is reused
    f = max(free_gb - 4, 2) / 12
    gs, ta, gi, vm = (
        CFG["hf_caps"]["game_stills"],
        CFG["hf_caps"]["tartanair"],
        CFG["hf_caps"]["gameir"],
        CFG["hf_caps"]["vimeo1080p"],
    )
    CFG["hf_caps"].update(
        {
            "game_stills": (int(gs[0] * f), int(gs[1] * f)),
            "tartanair": int(ta * f),
            "gameir": (int(gi[0] * f), int(gi[1] * f)),
            "vimeo1080p": (int(vm[0] * f), int(vm[1] * f)),
        }
    )
RUN.save_json({"quick_run": QUICK_RUN, "cfg": CFG, "sources": D.SOURCES}, "config.json")
print(pd.Series(CFG["hf_caps"]).to_string())

# %% [markdown]
# ## 5. Kaggle source discovery
#
# Lists each attached Kaggle dataset in parallel threads with a capped randomised walk (each directory listed completely, so a sequence is never cut in half), drops degraded copies by path token, and builds items: stills, sequences grouped by folder, and Sintel scenes with their flow and occlusion files. Test roles: `DIV2K_valid_HR`, Vid4, the Vimeo triplet test list and the Sintel `market` and `cave` scene families. Expected output: one row per source with files listed, items and test items.

# %%
with RUN.stage("kaggle_discovery"):
    kaggle_items, discovery = PL.discover_kaggle(
        C.INPUT_ROOT, CFG["list_cap_files"], CFG["seed"], log=log
    )
RUN.save_csv(discovery, "discovery_kaggle.csv", index=False)
print(discovery.to_string(index=False))

# %% [markdown]
# ## 6. Hugging Face materialisation
#
# Downloads only what is needed into `nss_data/`:
#
# * **game_stills** (`ericphann/video-game-super-resolution`): 1080p frames from game scenes; `test-hr` pinned to test.
# * **tartanair** (`theairlabcmu/tartanair`, BSD-3): contiguous runs from 12 training and 2 held-out environments, read member by member from the remote zips with HTTP range requests: RGB, depth, forward optical flow and the flow mask (occlusion / out of view).
# * **gameir** (`LLLebin/GameIR`, MIT): CARLA / Unreal Engine 4 clips streamed from the mini tars, keeping native 1440p and 720p renders and depth; town 05 is the official test town.
# * **vimeo1080p** (`danjacobellis/vimeo1080p`): MP4 videos extracted from two training parquet shards and one validation shard (validation pinned to test).
#
# Each source is optional: a failure is logged and the pipeline continues. Expected output: per-source counts and the disk used.

# %%
with RUN.stage("hf_materialisation"):
    hf_items = PL.materialise_hf(DATA_ROOT, CFG["hf_caps"], CFG["seed"], log=log)
used_gb = sum(p.stat().st_size for p in DATA_ROOT.rglob("*") if p.is_file()) / 1e9
log(f"Hugging Face items: {len(hf_items)}, data root size {used_gb:.2f} GB")
ITEMS = kaggle_items + hf_items
cat = pd.DataFrame(
    [
        {
            "source": i["source"],
            "kind": i["kind"],
            "role": i["role"],
            "frames": i["n_frames"],
            "motion": i.get("mv", "none"),
            "depth": bool(i.get("depth")),
        }
        for i in ITEMS
    ]
)
summary = (
    cat.groupby(["source", "kind", "role"])
    .agg(
        items=("frames", "size"),
        frames=("frames", "sum"),
        exact_motion=("motion", lambda m: int((m == "exact_forward").sum())),
        with_depth=("depth", "sum"),
    )
    .reset_index()
)
RUN.save_csv(summary, "catalogue_summary.csv", index=False)
print(summary.to_string(index=False))

# %% [markdown]
# ## 7. Catalogue overview
#
# Item and frame counts per source and role, and the share of data carrying game-rendering metadata (exact motion vectors, depth). Expected output: two bar charts.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
summary.pivot_table(
    index="source", columns="role", values="items", aggfunc="sum"
).fillna(0).plot.barh(ax=axes[0], stacked=True)
axes[0].set_xlabel("items (stills or sequences)")
axes[0].set_title("Items by source and role")
meta = summary.groupby("source")[["frames", "exact_motion", "with_depth"]].sum()
meta.plot.barh(ax=axes[1], logx=True)
axes[1].set_title("Frames, sequences with exact motion, sequences with depth")
RUN.show(fig, "catalogue")

# %% [markdown]
# ## 8. Keyframe thumbnails and quality statistics
#
# Decodes one keyframe per still or short clip and three (first, middle, last) per long sequence in a process pool over all CPU cores, keeping a 256x256 thumbnail and statistics: size, grey mean and contrast, Laplacian-variance sharpness, letterbox fraction and colourfulness. Expected output: a progress bar and a statistics table.

# %%
with RUN.stage("keyframe_stats"):
    THUMBS, KF = PL.keyframe_stats(ITEMS, DATA_ROOT, CFG["workers"])
KF["source"] = [ITEMS[i]["source"] for i in KF["item"]]
RUN.save_csv(KF, "keyframe_stats.csv", index=False)
print(
    KF.groupby("source")[
        ["width", "height", "std", "sharpness", "letterbox", "colourfulness"]
    ]
    .median()
    .round(2)
    .to_string()
)
print(f"keyframes: {len(KF)}, decode failures: {int((~KF['ok']).sum())}")

# %% [markdown]
# ## 9. Filtering unsuitable data
#
# Rejects items whose keyframes fail to decode or are smaller than 256 px, flat (grey standard deviation below 8), heavily letterboxed (more than 35 percent black rows) or blurry (median Laplacian variance below 20 for stills, 5 for video). Rejected items stay in the manifest with split `filtered`, so the decision is auditable. Expected output: rejection counts by reason and source, distributions with thresholds, and examples of rejected keyframes.

# %%
KEEP, REASONS = PL.apply_filters(ITEMS, KF)
rej = pd.DataFrame(
    [
        {"source": ITEMS[i]["source"], "reason": r}
        for i, rs in REASONS.items()
        for r in rs
    ]
)
if len(rej):
    print(rej.value_counts().unstack(fill_value=0).to_string())
print(f"kept {sum(KEEP)} of {len(ITEMS)} items")
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, (col, thr, logx) in zip(
    axes,
    [
        ("std", PL.FILTER_RULES["min_std"], False),
        ("sharpness", 20.0, True),
        ("letterbox", 0.35, False),
    ],
):
    for src, g in KF.groupby("source"):
        vals = g[col].dropna()
        ax.hist(np.log10(vals + 1e-3) if logx else vals, bins=40, alpha=0.4, label=src)
    ax.axvline(np.log10(thr) if logx else thr, color="k", ls="--")
    ax.set_title(f"{'log10 ' if logx else ''}{col}")
axes[0].legend(fontsize=6)
RUN.show(fig, "quality_distributions")
bad = [k for k, i in enumerate(KF["item"]) if not KEEP[i]][:16]
if bad:
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, k in zip(axes.flat, bad):
        ax.imshow(THUMBS[k])
        ax.set_title(
            f"{KF['source'][k]}: {','.join(REASONS[KF['item'][k]])}", fontsize=7
        )
    RUN.show(fig, "rejected_examples")

# %% [markdown]
# ## 10. Perceptual hashes
#
# 64-bit pHash (DCT of a 32x32 grey image, 8x8 low frequencies against their median) and dHash (horizontal gradient signs on 9x8) for every keyframe. pHash is robust to rescaling, mild compression and colour shifts but not to large crops or mirroring; embeddings cover those. Expected output: hash bit-balance check.

# %%
with RUN.stage("hashes"):
    PH, DH = PL.hashes(THUMBS)
print(
    f"pHash mean bit = {PH.mean():.3f}, dHash mean bit = {DH.mean():.3f} (about 0.5 means balanced bits)"
)

# %% [markdown]
# ## 11. Embeddings on both GPUs
#
# DINOv2-small (Apache 2.0) global descriptors (384-d, L2-normalised) for every thumbnail, batches split across the two GPUs with one thread per GPU. Self-supervised descriptors are the strongest general-purpose features for near-duplicate and copy detection after dedicated copy-detection models. Falls back to hashes only if the weights cannot be downloaded. Expected output: embedding matrix shape and throughput.

# %%
with RUN.stage("embeddings"):
    EMBEDDERS = []
    for dvc in DEVICES:
        e, name = DD.load_embedder(dvc, log=log)
        if e is None:
            EMBEDDERS = []
            break
        EMBEDDERS.append(e)
    t0 = time.time()
    EMB = DD.embed_thumbs(THUMBS, EMBEDDERS, DEVICES) if EMBEDDERS else None
log(f"embeddings: {None if EMB is None else EMB.shape} in {time.time() - t0:.0f} s")

# %% [markdown]
# ## 12. Threshold calibration
#
# Positive pairs are synthetic near-duplicates of random keyframes (60 to 95 percent crops, rescale, optional mirror, small rotation, colour and brightness shift, blur, text overlay, JPEG quality 35 to 92); negatives are random pairs of different keyframes. An all-pairs search tests hundreds of millions of pairs, so a threshold tuned only to a 0.1 percent false-positive rate would flag far too many unrelated pairs. Each threshold is therefore the stricter of the `max_fpr` point and the most similar negative plus a margin; recall at the final threshold shows what that strictness costs. Expected output: thresholds with precision, recall and F1 at the `max_fpr` point and recall at the final threshold, ROC curves and the Hamming histogram.

# %%
with RUN.stage("calibration"):
    THR = PL.calibrate(
        THUMBS,
        EMBEDDERS,
        DEVICES,
        CFG["calibration_pairs"],
        CFG["seed"],
        CFG["max_fpr"],
    )
cal = {
    k: {
        m: v[m]
        for m in ("threshold", "precision", "recall", "f1", "recall_at_threshold")
    }
    for k, v in THR.items()
    if isinstance(v, dict)
}
cal["phash"]["threshold"] = THR["phash"][
    "hamming"
]  # Hamming distance (bits) rather than the negated score
RUN.save_json(cal, "dedup_calibration.json")
print(pd.DataFrame(cal).T.round(4).to_string())
labels = np.array(THR["labels"])
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
for key, ax in (("phash", axes[0]), ("embedding", axes[1])):
    if key in THR:
        ax.plot(THR[key]["roc"]["fpr"], THR[key]["roc"]["tpr"])
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_title(f"ROC {key}")
        ax.set_xlabel("false-positive rate")
        ax.set_ylabel("recall")
ham = np.array(THR["phash_scores"])
axes[2].hist(
    [ham[labels == 1], ham[labels == 0]], bins=33, label=["near-duplicate", "different"]
)
axes[2].axvline(THR["phash"]["hamming"] + 0.5, color="k", ls="--")
axes[2].set_title("pHash Hamming distance")
axes[2].legend()
RUN.show(fig, "dedup_calibration")

# %% [markdown]
# ## 13. Near-duplicate search across the combined dataset
#
# Exhaustive search over all keyframe pairs on the GPU: pHash Hamming distances become dot products of +-1 vectors, embeddings use cosine similarity, both computed in chunked half-precision matrix products. pHash matches count only between keyframes with enough structure (grey standard deviation at least 15) and must be confirmed by dHash, because dark or flat frames share hashes by accident. If a method still yields more than two edges per keyframe, its threshold is tightened step by step (an implausible duplicate rate means false positives, which would merge unrelated data into giant components). Pairs inside one item are ignored. Expected output: final thresholds, pairs per method, a cross-source duplicate matrix (which datasets overlap) and examples of detected pairs for visual verification.

# %%
with RUN.stage("duplicate_search"):
    PAIRS = PL.find_duplicates(KF, PH, EMB, THR, DEVICES[0], dhash=DH)
RUN.save_csv(PAIRS, "duplicate_pairs.csv", index=False)
print(
    f"final thresholds: pHash Hamming <= {THR['phash']['hamming_final']}, "
    f"cosine >= {THR.get('embedding', {}).get('threshold_final')}"
)
print(
    PAIRS["method"].value_counts().to_string() if len(PAIRS) else "no duplicate pairs"
)
if len(PAIRS):
    src_a = [ITEMS[i]["source"] for i in PAIRS["item_a"]]
    src_b = [ITEMS[i]["source"] for i in PAIRS["item_b"]]
    mat = pd.crosstab(pd.Series(src_a, name="a"), pd.Series(src_b, name="b"))
    print(mat.to_string())
    show_pairs = (
        PAIRS.sort_values("score", ascending=False)
        .drop_duplicates(["item_a", "item_b"])
        .sample(n=min(8, len(PAIRS)), random_state=0)
    )
    fig, axes = plt.subplots(
        2, len(show_pairs), figsize=(2.2 * len(show_pairs), 4.6), squeeze=False
    )
    for c, (_, r) in enumerate(show_pairs.iterrows()):
        for row, kf_col in ((0, "kf_a"), (1, "kf_b")):
            axes[row, c].imshow(THUMBS[r[kf_col]])
            axes[row, c].axis("off")
            axes[row, c].set_title(
                f"{KF['source'][r[kf_col]]} {r['method'][:5]} {r['score']:.2f}",
                fontsize=7,
            )
    RUN.show(fig, "duplicate_examples")

# %% [markdown]
# ## 14. Clustering and leakage-safe split
#
# Union-find over (a) structural edges: items sharing a leakage group (same sequence, video, Sintel scene family, TartanAir environment, GameIR town) and (b) near-duplicate edges. Each connected component receives one split: components containing a test item become test and any training-role members are dropped; the rest go to validation with probability `val_fraction` (deterministic hash) or to training. Expected output: the split report and per-source split counts.

# %%
REPORT = PL.split_items(ITEMS, PAIRS, CFG["val_fraction"], CFG["seed"])
for it, k in zip(ITEMS, KEEP):
    if not k:
        it["split"] = "filtered"
RUN.save_json(REPORT, "split_report.json")
print(pd.Series(REPORT).to_string())
split_table = pd.crosstab(
    pd.Series([i["source"] for i in ITEMS], name="source"),
    pd.Series([i["split"] for i in ITEMS], name="split"),
)
RUN.save_csv(split_table, "split_by_source.csv")
print(split_table.to_string())

# %% [markdown]
# ## 15. Leakage audit
#
# For every validation and test keyframe, the most similar training keyframe (cosine) and the closest pHash (Hamming). After the split no cross-split pair may pass the final embedding threshold (asserted); pHash matches are reported, since low-structure frames are excluded from pHash edges by design. Expected output: audit statistics, histograms and violation counts.

# %%
AUDIT = PL.cross_split_audit(KF, ITEMS, EMB, PH, DEVICES[0])
rows = []
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
for s, res in AUDIT.items():
    if "max_cosine_to_train" in res:
        axes[0].hist(res["max_cosine_to_train"], bins=60, alpha=0.5, label=s)
        rows.append(
            {
                "split": s,
                "metric": "max cosine to train",
                "max": float(res["max_cosine_to_train"].max()),
                "p99": float(np.percentile(res["max_cosine_to_train"], 99)),
            }
        )
    axes[1].hist(res["min_hamming_to_train"], bins=33, alpha=0.5, label=s)
    rows.append(
        {
            "split": s,
            "metric": "min pHash Hamming to train",
            "max": float(res["min_hamming_to_train"].min()),
            "p99": float(np.percentile(res["min_hamming_to_train"], 1)),
        }
    )
if "embedding" in THR:
    axes[0].axvline(THR["embedding"]["threshold_final"], color="k", ls="--")
axes[1].axvline(THR["phash"]["hamming_final"], color="k", ls="--")
axes[0].set_title("nearest training neighbour: cosine")
axes[1].set_title("nearest training neighbour: pHash Hamming")
for ax in axes:
    ax.legend()
RUN.show(fig, "leakage_audit")
audit = pd.DataFrame(rows)
RUN.save_csv(audit, "leakage_audit.csv", index=False)
print(audit.to_string(index=False))
violations = {}
for s, res in AUDIT.items():
    violations[f"{s}_phash_matches"] = int(
        (res["min_hamming_to_train"] <= THR["phash"]["hamming_final"]).sum()
    )
    if "max_cosine_to_train" in res and "embedding" in THR:
        violations[f"{s}_embedding_matches"] = int(
            (res["max_cosine_to_train"] >= THR["embedding"]["threshold_final"]).sum()
        )
RUN.save_json(violations, "leakage_violations.json")
print(pd.Series(violations).to_string())
assert all(
    v == 0 for k, v in violations.items() if "embedding" in k
), "cross-split embedding duplicates remain"

# %% [markdown]
# ## 16. Manifest and dataset card
#
# Saves the manifest (`nss_data/nss_manifest.parquet`) and a dataset card listing every source with its host, licence statement, content type, item counts per split and whether it carries exact motion vectors or depth. Licences are as stated by the hosting page; several research datasets restrict commercial use (see `RESEARCH.md`). Expected output: the card table.

# %%
D.save_manifest(ITEMS, DATA_ROOT / "nss_manifest.parquet")
card = []
for s in D.SOURCES:
    its = [i for i in ITEMS if i["source"] == s["name"]]
    card.append(
        {
            "source": s["name"],
            "host": s["host"],
            "id": s.get("ref") or s.get("repo"),
            "licence": s["licence"],
            "content": s.get("content"),
            "items": len(its),
            **{
                sp: sum(i["split"] == sp for i in its)
                for sp in ("train", "val", "test", "drop", "filtered")
            },
            "exact_motion": any(i.get("mv") == "exact_forward" for i in its),
            "depth": any(bool(i.get("depth")) for i in its),
        }
    )
card = pd.DataFrame(card)
RUN.save_csv(card, "dataset_card.csv", index=False)
print(card.to_string(index=False))

# %% [markdown]
# ## 17. Summary
#
# Stage times against the budget and the headline numbers of the pipeline. Expected output: a summary table and a stage-time bar chart.

# %%
summary = {
    "items": len(ITEMS),
    "keyframes": len(KF),
    "duplicate_pairs": int(len(PAIRS)),
    **REPORT["split_counts"],
    "train_dropped_for_leakage": REPORT["train_items_dropped_for_test_leakage"],
    "phash_hamming_threshold": THR["phash"]["hamming_final"],
    "embedding_cosine_threshold": THR.get("embedding", {}).get("threshold_final"),
    "data_root_gb": round(used_gb, 2),
    "run_hours": round(RUN.hours(), 2),
}
RUN.save_json(summary, "summary.json")
print(pd.Series(summary).to_string())
st = pd.Series(RUN.stage_times) / 60
RUN.save_csv(st.rename("minutes").to_frame(), "stage_times.csv")
fig, ax = plt.subplots(figsize=(8, 3.5))
st.plot.barh(ax=ax)
ax.set_xlabel("minutes")
RUN.show(fig, "stage_times")

# %% [markdown]
# ## 18. Package reports
#
# Zips metrics, plots, logs and code into `outputs.zip` (the materialised data in `nss_data/` is excluded because it is already part of this notebook's Output and is consumed from there). After **Save Version > Save & Run All**, the zip and all files also appear in the notebook's **Output** tab.

# %%
import shutil
import zipfile

from IPython.display import FileLink, display
from tqdm.auto import tqdm

WORK = C.WORK
ZIP_PATH = WORK / "outputs.zip"
EXCLUDE_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".cache",
    "nss_data",
    "tmp",
    "wandb",
}
STORED_SUFFIXES = {
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".gz",
    ".npz",
    ".onnx",
    ".mp4",
    ".parquet",
}
files = [
    p
    for p in WORK.rglob("*")
    if p.is_file()
    and p != ZIP_PATH
    and not EXCLUDE_PARTS.intersection(p.relative_to(WORK).parts)
]
total_bytes = sum(p.stat().st_size for p in files)
free_bytes = shutil.disk_usage(WORK).free
assert (
    free_bytes > total_bytes * 1.05
), f"Not enough disk space to zip {total_bytes / 1e9:.2f} GB with {free_bytes / 1e9:.2f} GB free."
with zipfile.ZipFile(ZIP_PATH, "w", allowZip64=True) as archive:
    for p in tqdm(files, desc="Zipping outputs", unit="file"):
        method = (
            zipfile.ZIP_STORED
            if p.suffix.lower() in STORED_SUFFIXES
            else zipfile.ZIP_DEFLATED
        )
        archive.write(p, p.relative_to(WORK), compress_type=method)
print(
    f"Zipped {len(files)} files ({total_bytes / 1e9:.2f} GB) into {ZIP_PATH}",
    flush=True,
)
display(FileLink(os.path.relpath(ZIP_PATH)))
