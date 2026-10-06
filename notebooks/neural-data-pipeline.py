# %% [markdown]
# # NeuralSS data pipeline: discovery, copy detection and leakage-safe splits (Kaggle GPU T4 x2)
#
# **Goal.** Build the single dataset manifest used by both training notebooks of the NeuralSS stack
# (temporal super-resolution and low-latency frame generation). Only Kaggle and Hugging Face datasets are used.
#
# **Steps.**
#
# 1. Discover attached Kaggle datasets (capped, randomised directory walk; official test lists read directly) and drop degraded copies (LR, bicubic, blurred, `x2`/`x3`/`x4`) and non-colour passes.
# 2. Materialise selected Hugging Face subsets into `/kaggle/working/nss_data` with keep-alive range requests and single-stream reads (no full archive downloads): game frames, TartanAir sequences with depth, exact optical flow and occlusion masks, GameIR native 720p / 1440p game renders with depth, and 1080p Vimeo videos.
# 3. Group frames into sequences; every item gets a leakage group (sequence, video, scene family, environment or town).
# 4. Decode keyframes into thumbnails and compute quality statistics; filter too-small, flat, letterboxed and blurry items.
# 5. Copy detection across the whole combined dataset in two stages. Candidates: DINOv2-small nearest neighbours (exhaustive GPU search) and pHash pairs. Verification: SIFT matches that agree on one RANSAC affine map, scored by the image area they cover. Thresholds are calibrated on synthetic copies against random and hard negatives (nearest neighbours between items known to be distinct).
# 6. Split by leakage group: test items keep their role, anything sharing a group with a test item or connected to one by duplicate edges is dropped, and the remaining components go to validation or training.
# 7. Audit every validation and test keyframe against its nearest training (and validation) keyframes with the same rule; any duplicate found is added as an edge and the split is repeated until the audit is clean.
#
# **Why two stages.** The first run used global similarity only (pHash Hamming 14, DINOv2 cosine 0.75). Those thresholds flag *similar* pictures (two cherry-tree photos, two skylines, frames of one game), not copies. The false edges chained 8,354 items into one component that touched a test item, and 8,150 training items (72 percent) were dropped. Geometric verification separates copies from look-alikes. Splitting by provenance instead of transitive components keeps one copied clip from taking its whole folder with it.
#
# **Outputs** (in `/kaggle/working`, saved as this notebook's output; attach it to the training notebooks with **Add Input > Notebook output**):
#
# ```text
# nss_data/nss_manifest.parquet   one row per still or sequence: source, frames (URIs), motion / depth / mask URIs, group, split, component, drop_reason
# nss_data/<hf subsets>/          materialised Hugging Face data referenced by the manifest
# metrics/                        discovery, statistics, calibration, duplicate edges, split and audit reports, dataset card
# plots/                          every figure as PNG
# logs/notebook.log
# outputs.zip                     metrics, plots, logs, code and the manifest (the data itself is already in the Output tab)
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
# | `chenshu123/vimeo-triplet` | frame-generation test (clips from the official test list) |
# | `artemmmtry/mpi-sintel-dataset` | rendered sequences with exact flow and occlusions (scene families `market`, `cave` pinned to test) |
# | `uom200647r/vid4-dataset` | video test only |
#
# **Run time.** The first full run took 2.7 hours, 2.4 of them Hugging Face transfer. Connection reuse and streamed tars should shorten that, and copy detection adds roughly 10 to 20 minutes of CPU verification (not yet measured). `QUICK_RUN = True` shrinks every cap for a short smoke test.

# %% [markdown]
# ## 1. Library files
#
# The pipeline code is shared with the two training notebooks. The first cell creates `/kaggle/working/code`. The following cells write the library into it so the notebook is self-contained: `nss_common` (run folders, logging, stage timer), `nss_data` (source registry, discovery, Hugging Face access, decoding, windows), `nss_dedup` (hashes, embeddings, GPU neighbour search, SIFT verification, calibration copies, split) and `nss_pipeline` (the steps run below).

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
# **`nss_data.py`**: source registry (Kaggle and Hugging Face), capped discovery, keep-alive HTTP with range requests and streamed tars, Hugging Face materialisation, decoding of colour / depth / flow / masks, window and crop workers, manifest I/O.

# %%
# WRITEFILE nss_data.py

# %% [markdown]
# **`nss_dedup.py`**: pHash / dHash, DINOv2 embeddings on both GPUs, exhaustive GPU neighbour search, SIFT + RANSAC geometric verification in a process pool, synthetic copies for calibration, the duplicate rule and the leakage-safe split.

# %%
# WRITEFILE nss_dedup.py

# %% [markdown]
# **`nss_pipeline.py`**: the data-pipeline steps (discover, materialise, statistics, filters, hashes, calibration, candidate search and verification, split with audit) and the end-to-end fallback used by the training notebooks.

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
# Detects GPUs, CPU cores, RAM and free disk at run time and saves `metrics/hardware.json`. Both GPUs compute embeddings; all CPU cores decode images and run SIFT verification. Expected output: two Tesla T4 with about 15 GB each.

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
# All caps and thresholds in one place. Hugging Face caps are sized to keep the materialised data under about 14 GB of the 20 GB `/kaggle/working` quota (the first run used 13.4 GB); the cell lowers them when free disk is short.
#
# * `hf_caps`: game frames (train, test), TartanAir frames per environment and sequence length, GameIR clips (train, test), Vimeo-1080p videos (train, test).
# * `calibration_pairs`: synthetic copies used to calibrate the duplicate rule (half light, half heavy edits).
# * `dedup`: neighbours per keyframe (`k`), the share of synthetic copies the candidate stage must retrieve, SIFT keypoints per image, minimum grey standard deviation for hash matches, and audit depth and rounds.
# * `val_fraction`: share of train / validation components assigned to validation.

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
    "dedup": dict(PL.DEDUP),
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
print(pd.Series(CFG["dedup"]).to_string())

# %% [markdown]
# ## 5. Kaggle source discovery
#
# Lists each attached Kaggle dataset in parallel threads. A capped randomised walk lists each directory completely, so a sequence is never cut in half. Degraded copies are dropped by path token. Items are built from the listing: stills, sequences grouped by folder, and Sintel scenes with their flow and occlusion files. When a source ships an official test list (Vimeo triplets), clips are read straight from it instead of hoping a random walk finds them. Test roles: `DIV2K_valid_HR`, Vid4, the Vimeo triplet test list, and the Sintel `market` and `cave` scene families. Expected output: one row per source with files listed, items and test items.

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
# Downloads only what is needed into `nss_data/`. Each thread keeps one HTTP connection alive, range reads are length-checked and retried, and an expired CDN link is re-resolved:
#
# * **game_stills** (`ericphann/video-game-super-resolution`): 1080p frames from game scenes; `test-hr` pinned to test.
# * **tartanair** (`theairlabcmu/tartanair`, BSD-3): contiguous runs from 12 training and 2 held-out environments, read member by member from the remote zips. Each run keeps RGB, depth, forward optical flow and the flow mask (occlusion / out of view). A failed environment is retried twice, resuming from the frames already written.
# * **gameir** (`LLLebin/GameIR`, MIT): CARLA / Unreal Engine 4 clips read from the mini tars in one sequential stream, keeping native 1440p and 720p renders and depth; town 05 is the official test town.
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
# Decodes one keyframe per still or short clip and three per long sequence, in a process pool over all CPU cores. Long sequences use frames at 10, 50 and 90 percent, because first and last frames are often fades or black. Each keyframe keeps a 256x256 thumbnail and statistics: size, grey mean and contrast, Laplacian-variance sharpness, letterbox fraction and colourfulness. Expected output: a progress bar and a statistics table.

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
# Rejects items whose keyframes fail to decode or are smaller than 256 px, flat (grey standard deviation below 8), heavily letterboxed (more than 35 percent black rows) or blurry (Laplacian variance below 20 for stills, 5 for video). Sequences are judged on the median of their keyframes, so one dark shot or a fade does not reject a whole video. Rejected items stay in the manifest with split `filtered` and are left out of duplicate search, so the decision is auditable. Expected output: rejection counts by reason and source, distributions with thresholds, and examples of rejected keyframes from every source and reason.

# %%
with RUN.stage("filters"):
    KEEP, REASONS = PL.apply_filters(ITEMS, KF)
    KF["active"] = [KEEP[i] for i in KF["item"]]
rej = pd.DataFrame(
    [
        {"source": ITEMS[i]["source"], "reason": r}
        for i, rs in REASONS.items()
        for r in rs
    ],
    columns=["source", "reason"],
)
if len(rej):
    print(rej.value_counts().unstack(fill_value=0).to_string())
print(f"kept {sum(KEEP)} of {len(ITEMS)} items")
rules = PL.FILTER_RULES
fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
for ax, (col, thrs, logx) in zip(
    axes,
    [
        ("std", [rules["min_std"]], False),
        ("sharpness", list(rules["min_sharpness"].values()), True),
        ("letterbox", [rules["max_letterbox"]], False),
    ],
):
    for src, g in KF.groupby("source"):
        vals = g[col].dropna()
        ax.hist(np.log10(vals + 1e-3) if logx else vals, bins=40, alpha=0.4, label=src)
    for t in thrs:
        ax.axvline(np.log10(t) if logx else t, color="k", ls="--")
    ax.set_title(f"{'log10 ' if logx else ''}{col}")
axes[0].legend(fontsize=6)
RUN.show(fig, "quality_distributions")
rej_kf = pd.DataFrame(
    [
        {"k": k, "source": KF["source"][k], "reason": ",".join(REASONS[i])}
        for k, i in enumerate(KF["item"])
        if not KEEP[i]
    ],
    columns=["k", "source", "reason"],
)
if len(rej_kf):
    show = rej_kf.groupby(["source", "reason"]).head(2).head(16)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, show.iterrows()):
        ax.imshow(THUMBS[r["k"]])
        ax.set_title(f"{r['source']}: {r['reason']}", fontsize=7)
    RUN.show(fig, "rejected_examples")

# %% [markdown]
# ## 10. Perceptual hashes
#
# 64-bit pHash (DCT of a 32x32 grey image, 8x8 low frequencies against their median) and dHash (horizontal gradient signs on 9x8) for every keyframe. Hashes catch near-exact copies (rescaled, recompressed, recoloured) but not crops or mirrored copies; embeddings and verification cover those. Expected output: hash bit-balance check.

# %%
with RUN.stage("hashes"):
    PH, DH = PL.hashes(THUMBS)
print(
    f"pHash mean bit = {PH.mean():.3f}, dHash mean bit = {DH.mean():.3f} (about 0.5 means balanced bits)"
)

# %% [markdown]
# ## 11. Embeddings on both GPUs
#
# DINOv2-small (Apache 2.0) global descriptors (384-d, L2-normalised) for every thumbnail, with batches split across the two GPUs, one thread per GPU. Here they only *retrieve* candidates: global similarity is high for look-alikes as well as copies, so it cannot decide alone. If the weights cannot be downloaded, the pipeline falls back to hash candidates. Expected output: embedding matrix shape and time.

# %%
with RUN.stage("embeddings"):
    EMBEDDERS = PL.load_embedders(DEVICES, log=log)
    t0 = time.time()
    EMB = DD.embed_thumbs(THUMBS, EMBEDDERS, DEVICES) if EMBEDDERS else None
log(f"embeddings: {None if EMB is None else EMB.shape} in {time.time() - t0:.0f} s")

# %% [markdown]
# ## 12. Local features for geometric verification
#
# SIFT keypoints (up to `sift_features` per image) and descriptors for every thumbnail and for its mirror image, computed in a process pool over all CPU cores and kept in RAM (uint8 descriptors). Verification later matches two keyframes with Lowe's ratio test and fits one affine map with RANSAC. The score is the number of cells of an 8x8 grid covered by consistent matches, so a shared logo, HUD or caption alone cannot verify two different frames. Expected output: a progress bar and keypoint statistics.

# %%
with RUN.stage("local_features"):
    FEATS = DD.local_features(THUMBS, CFG["workers"], CFG["dedup"]["sift_features"])
n_kp = np.array([len(f[0][0]) for f in FEATS])
print(
    f"keypoints per keyframe: median {np.median(n_kp):.0f}, "
    f"under 20: {int((n_kp < 20).sum())} of {len(n_kp)}"
)

# %% [markdown]
# ## 13. Threshold calibration
#
# Positives are synthetic copies of random keyframes. Half are *light* edits: rescale, JPEG quality 35 to 92, colour and brightness shift, blur. Half are *heavy*: these also crop to 60 to 95 percent of the area, mirror, rotate slightly and overlay text. Negatives are random pairs from different groups plus *hard* negatives: each keyframe's nearest neighbour in another group, among sources whose groups are known to be distinct (DIV2K images, TartanAir environments, Sintel scene families).
#
# Thresholds, from the scores of the most similar negative (the extreme hard negative, or the 99.9th percentile of random negatives):
#
# * candidate cosine `t_ret` retrieves 98 percent of the copies; precision comes from verification;
# * strong cosine `t_strong` lies above the most similar negative, so such pairs count without verification;
# * verification `t_cells` lies above the most similar negative's score (at least 10 of 64 cells);
# * exact hash `h_exact` lies 4 bits below the closest negative's pHash distance (at most 6, also confirmed by dHash on textured frames); `h_cand` covers 95 percent of light copies and only proposes candidates.
#
# Expected output: the thresholds, recall of the combined rule on light and heavy copies, false positives on the negatives, and four diagnostic plots.

# %%
with RUN.stage("calibration"):
    THR, CAL = PL.calibrate(
        THUMBS,
        KF,
        ITEMS,
        EMB,
        PH,
        DH,
        FEATS,
        EMBEDDERS,
        DEVICES,
        CFG["calibration_pairs"],
        CFG["seed"],
        CFG["workers"],
        CFG["dedup"],
        log=log,
    )
RUN.save_json(
    {k: {m: x for m, x in v.items() if m != "roc"} for k, v in THR.items()},
    "dedup_calibration.json",
)
RUN.save_csv(CAL, "calibration_scores.csv", index=False)
for key, val in THR.items():
    print(
        key,
        {
            m: (round(x, 4) if isinstance(x, float) else x)
            for m, x in val.items()
            if m != "roc"
        },
    )
groups = {
    "copy (light)": (CAL["label"] == 1) & (CAL["kind"] == "light"),
    "copy (heavy)": (CAL["label"] == 1) & (CAL["kind"] == "heavy"),
    "hard negative": CAL["kind"] == "hard",
    "random negative": CAL["kind"] == "random",
}
fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
if EMB is not None:
    for name, m in groups.items():
        axes[0].hist(CAL.loc[m, "cosine"], bins=50, range=(0, 1), alpha=0.5, label=name)
    for t in (THR["embedding"]["t_ret"], THR["embedding"]["t_strong"]):
        axes[0].axvline(t, color="k", ls="--")
    axes[0].set_title("DINOv2 cosine (t_ret, t_strong)")
    axes[0].legend(fontsize=7)
    roc = THR["embedding"]["roc"]
    axes[3].plot(roc["fpr"], roc["tpr"])
    axes[3].set_xscale("symlog", linthresh=1e-3)
    axes[3].set_xlabel("false-positive rate")
    axes[3].set_ylabel("recall")
    axes[3].set_title("ROC of cosine alone")
for name, m in groups.items():
    axes[1].hist(CAL.loc[m, "cells"], bins=33, range=(0, 65), alpha=0.5, label=name)
axes[1].axvline(THR["verify"]["t_cells"] - 0.5, color="k", ls="--")
axes[1].set_yscale("log")
axes[1].set_title("verification: covered grid cells (t_cells)")
for name, m in groups.items():
    axes[2].hist(CAL.loc[m, "hamming"], bins=33, range=(0, 64), alpha=0.5, label=name)
for t in (THR["phash"]["h_exact"], THR["phash"]["h_cand"]):
    axes[2].axvline(t + 0.5, color="k", ls="--")
axes[2].set_title("pHash Hamming (h_exact, h_cand)")
RUN.show(fig, "dedup_calibration")

# %% [markdown]
# ## 14. Copy detection across the combined dataset
#
# Candidates come from two sources. The first is each kept keyframe's `k` nearest DINOv2 neighbours outside its own leakage group with cosine at least `t_ret`, from an exhaustive half-precision search on the GPU. The second is pHash pairs within `h_cand` between textured keyframes. If the hash candidates are implausibly many, `h_cand` is tightened. Every candidate is verified with SIFT + RANSAC in a process pool. A pair becomes a duplicate edge if it is `verified`, `strong_embedding` or `exact_hash` (section 13). Expected output: candidate and edge counts per rule, a cross-source edge matrix (which datasets overlap) and examples of edges for visual checking, cross-source pairs first.

# %%
with RUN.stage("duplicate_search"):
    EDGES, DUP_STATS = PL.find_duplicates(
        KF,
        ITEMS,
        PH,
        DH,
        EMB,
        FEATS,
        THR,
        DEVICES[0],
        CFG["workers"],
        CFG["dedup"],
        log=log,
    )
RUN.save_csv(EDGES, "duplicate_pairs.csv", index=False)
RUN.save_json(DUP_STATS, "duplicate_search.json")
print(
    pd.Series(
        {k: v for k, v in DUP_STATS.items() if k != "verification_score_hist"}
    ).to_string()
)
if len(EDGES):
    EDGES["source_a"] = [ITEMS[i]["source"] for i in EDGES["item_a"]]
    EDGES["source_b"] = [ITEMS[i]["source"] for i in EDGES["item_b"]]
    print(pd.crosstab(EDGES["source_a"], EDGES["source_b"]).to_string())
    cross = EDGES["source_a"] != EDGES["source_b"]
    show_pairs = (
        pd.concat(
            [
                EDGES[cross].sample(frac=1, random_state=0),
                EDGES[~cross].sample(frac=1, random_state=0),
            ]
        )
        .groupby("rule", group_keys=False)
        .head(4)
        .head(10)
    )
    fig, axes = plt.subplots(
        2, len(show_pairs), figsize=(2.2 * len(show_pairs), 4.8), squeeze=False
    )
    for c, (_, r) in enumerate(show_pairs.iterrows()):
        for row, kf_col in ((0, "kf_a"), (1, "kf_b")):
            axes[row, c].imshow(THUMBS[r[kf_col]])
            axes[row, c].axis("off")
        axes[1, c].set_title(KF["source"][r["kf_b"]], fontsize=7)
        axes[0, c].set_title(
            f"{KF['source'][r['kf_a']]}\n{r['rule']} cos {r['cosine']:.2f} cells {r['cells']}",
            fontsize=7,
        )
    RUN.show(fig, "duplicate_examples")

# %% [markdown]
# ## 15. Leakage-safe split with audit
#
# 1. **Provenance.** A leakage group (sequence, video, Sintel scene family, TartanAir environment, GameIR town) that holds a test item is test-only, and its other items are dropped.
# 2. **Content.** Non-test items connected to a test item through duplicate edges are dropped. The walk runs through duplicate edges only, never through shared groups, so one copied Vimeo-90K clip does not take its 1,000-clip folder with it.
# 3. **Train and validation.** The remaining items form components (groups joined by duplicate edges). Each component goes to validation with probability `val_fraction` (deterministic hash) or to training.
# 4. **Audit.** Every validation keyframe is compared with its `audit_k` nearest training keyframes, and every test keyframe with its nearest training and validation keyframes, by cosine and by pHash, using the same rule. Any duplicate the top-`k` search missed becomes an edge and the split is repeated, up to `audit_rounds` times. The cell fails if duplicates remain.
#
# Expected output: the split report, drop reasons, items per source and split (table and chart), and zero audit violations.

# %%
with RUN.stage("split_and_audit"):
    REPORT, AUDIT, VIOL, EDGES = PL.split_and_audit(
        KF,
        ITEMS,
        EDGES,
        KEEP,
        PH,
        DH,
        EMB,
        FEATS,
        THR,
        DEVICES[0],
        CFG["workers"],
        CFG["val_fraction"],
        CFG["seed"],
        CFG["dedup"],
        log=log,
    )
RUN.save_json(REPORT, "split_report.json")
RUN.save_csv(
    EDGES, "duplicate_pairs.csv", index=False
)  # final edges, including audit repairs
RUN.save_csv(VIOL, "leakage_violations.csv", index=False)
print(pd.Series({k: v for k, v in REPORT.items() if k != "split_counts"}).to_string())
print(pd.Series(REPORT["split_counts"]).to_string())
split_table = pd.crosstab(
    pd.Series([i["source"] for i in ITEMS], name="source"),
    pd.Series([i["split"] for i in ITEMS], name="split"),
).reindex(columns=["train", "val", "test", "drop", "filtered"], fill_value=0)
RUN.save_csv(split_table, "split_by_source.csv")
print(split_table.to_string())
drops = pd.crosstab(
    pd.Series([i["source"] for i in ITEMS if i["drop_reason"]], name="source"),
    pd.Series(
        [i["drop_reason"] for i in ITEMS if i["drop_reason"]], name="drop reason"
    ),
)
print(drops.to_string() if len(drops) else "no items dropped")
fig, ax = plt.subplots(figsize=(10, 4.5))
split_table.plot.barh(ax=ax, stacked=True)
ax.set_xlabel("items")
ax.set_title("Items per source and split")
RUN.show(fig, "split_by_source")
assert not len(
    VIOL
), f"{len(VIOL)} cross-split duplicates remain; see metrics/leakage_violations.csv"

# %% [markdown]
# ## 16. Audit statistics
#
# How close each validation and test keyframe comes to the data it is compared with: the highest cosine similarity, the highest verification score and the smallest pHash distance among its nearest reference keyframes, each against the duplicate thresholds. Values beyond a threshold are allowed only when the other criteria reject the pair (a high cosine without geometric support is a look-alike, not a copy). Expected output: three histograms and the audit table.

# %%
rows = []
fig, axes = plt.subplots(1, 3, figsize=(18, 3.6))
for s, res in AUDIT.items():
    label = f"{s} vs {res['reference']}"
    if "max_cosine" in res:
        axes[0].hist(res["max_cosine"], bins=60, range=(0, 1), alpha=0.5, label=label)
    axes[1].hist(res["max_cells"], bins=33, range=(0, 65), alpha=0.5, label=label)
    axes[2].hist(res["min_hamming"], bins=33, range=(0, 64), alpha=0.5, label=label)
    for metric, vals, worst in (
        ("max cosine", res.get("max_cosine"), np.max),
        ("max verification cells", res["max_cells"], np.max),
        ("min pHash Hamming", res["min_hamming"], np.min),
    ):
        if vals is not None:
            rows.append(
                {
                    "split": s,
                    "reference": res["reference"],
                    "metric": metric,
                    "worst": float(worst(vals)),
                    "median": float(np.median(vals)),
                }
            )
if "embedding" in THR:
    axes[0].axvline(THR["embedding"]["t_strong"], color="k", ls="--")
axes[1].axvline(THR["verify"]["t_cells"] - 0.5, color="k", ls="--")
axes[1].set_yscale("log")
axes[2].axvline(THR["phash"]["h_exact"] + 0.5, color="k", ls="--")
axes[0].set_title("nearest reference keyframe: cosine (t_strong)")
axes[1].set_title("best verification score (t_cells)")
axes[2].set_title("nearest pHash Hamming (h_exact)")
for ax in axes:
    ax.legend(fontsize=7)
RUN.show(fig, "leakage_audit")
audit = pd.DataFrame(rows)
RUN.save_csv(audit, "leakage_audit.csv", index=False)
print(audit.to_string(index=False))

# %% [markdown]
# ## 17. Manifest and dataset card
#
# Saves the manifest (`nss_data/nss_manifest.parquet`) and a dataset card. The card lists every source with its host, licence statement, content type, item counts per split, and whether it carries exact motion vectors or depth. Licences are as stated by the hosting page; several research datasets restrict commercial use (see `RESEARCH.md`). Expected output: the card table.

# %%
with RUN.stage("manifest"):
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
# ## 18. Summary
#
# Stage times and the headline numbers of the pipeline. Expected output: a summary table and a stage-time bar chart.

# %%
summary = {
    "items": len(ITEMS),
    "keyframes": len(KF),
    "candidates": DUP_STATS["candidates"],
    "duplicate_edges": int(len(EDGES)),
    **{f"edges_{k}": int(v) for k, v in EDGES["rule"].value_counts().items()},
    **REPORT["split_counts"],
    "dropped_test_group": REPORT["dropped_test_group"],
    "dropped_test_duplicate": REPORT["dropped_test_duplicate"],
    "largest_duplicate_cluster": REPORT["largest_duplicate_cluster"],
    "audit_rounds": REPORT["audit_rounds"],
    "audit_violations": REPORT["audit_violations"],
    "t_ret": THR.get("embedding", {}).get("t_ret"),
    "t_strong": THR.get("embedding", {}).get("t_strong"),
    "t_cells": THR["verify"]["t_cells"],
    "h_exact": THR["phash"]["h_exact"],
    "rule_recall_light": round(THR["rule"]["recall_light"], 4),
    "rule_recall_heavy": round(THR["rule"]["recall_heavy"], 4),
    "rule_false_positives": THR["rule"]["false_positives"],
    "data_root_gb": round(used_gb, 2),
    "run_hours": round(RUN.hours(), 2),
}
RUN.save_json(summary, "summary.json")
print(pd.Series(summary).to_string())
st = pd.Series(RUN.stage_times) / 60
RUN.save_csv(st.rename("minutes").to_frame(), "stage_times.csv")
fig, ax = plt.subplots(figsize=(8, 3.5))
st.plot.barh(ax=ax, logx=True)
ax.set_xlabel("minutes (log scale)")
RUN.show(fig, "stage_times")

# %% [markdown]
# ## 19. Package reports
#
# Zips metrics, plots, logs, code and the manifest into `outputs.zip`. The materialised data in `nss_data/` is left out because it is already part of this notebook's Output and is consumed from there. After **Save Version > Save & Run All**, the zip and all files also appear in the notebook's **Output** tab.

# %%
import shutil
import zipfile

from IPython.display import FileLink, display
from tqdm.auto import tqdm

WORK = C.WORK
ZIP_PATH = WORK / "outputs.zip"
MANIFEST_PATH = DATA_ROOT / "nss_manifest.parquet"
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
    and (
        p == MANIFEST_PATH or not EXCLUDE_PARTS.intersection(p.relative_to(WORK).parts)
    )
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
