# %% [markdown]
# # NeuralSS temporal super-resolution for games (Kaggle GPU T4 x2)
#
# **Goal.** Train and evaluate the NeuralSS temporal upscaler: a recurrent, motion-compensated network that reconstructs native-quality frames from jittered low-resolution game renders (primary 3x, for example 640x360 to 1920x1080; also 2x), stable under heavy motion and converging to near-native detail in static scenes, small enough for integrated GPUs.
#
# **Approach.**
#
# 1. **Data.** Uses the leakage-safe manifest from the `neural-data-pipeline` notebook when attached (otherwise builds a smaller one inline with the same code). Two training streams:
#    * *Synthetic game-like sequences* rendered on the GPU from cached high-resolution stills (game frames, DIV2K, Flickr2K, GameIR, video frames): layered 2.5D scenes with independent camera and object motion, thin wires and fences, particles that write no motion vectors, world-space text, exposure changes. Low-resolution inputs are rendered like a game renders at reduced resolution (one jittered sample per pixel, Halton 2,3, no mip filtering, so thin geometry aliases); targets are 4-sample anti-aliased. Motion vectors, disocclusions and depth are exact.
#    * *Real sequences*: TartanAir and Sintel with exact optical flow (played backwards so forward flow becomes the engine-style motion vector) and depth; REDS, Vimeo-90K, Vimeo-1080p and GameIR video with RAFT teacher motion (marked as estimated).
# 2. **Model.** Per frame, at low resolution: jittered colour, depth and derived normals, motion vectors, a depth disocclusion cue, reactive mask, exposure, jitter offset, availability flags, the previous output warped by the upsampled motion vectors and folded to low resolution (space-to-depth), and a warped recurrent hidden state. A RepConv trunk (with an optional half-resolution U-Net level) predicts a per-pixel history weight and a residual, unfolded by pixel shuffle: `out = a * warped_history + (1 - a) * jitter-aware upsample + residual`. RepConv branches fold into plain 3x3 convolutions for deployment. Tiers: `igpu`, `low`, `mid`, `high`, and a `teacher` used for distillation.
# 3. **Training.** DDP on both GPUs, FP16, EMA, wall-clock cosine schedule. Loss per frame: Charbonnier + gradient + Fourier magnitude + LPIPS (perceptual) + temporal-change loss along exact motion; students also match the teacher's outputs (knowledge distillation).
# 4. **Evaluation.** Spatial quality (PSNR / SSIM on Y, LPIPS, FLIP), temporal stability (warping error and temporal-change error along exact motion, tOF with RAFT), region metrics (disocclusions, fine detail, particles), static-scene convergence, motion-speed classes, a native-render generalisation test (GameIR 720p to 1440p), against bilinear, bicubic and an analytic TAA upscaler. Ablations: no history, no G-buffers, bicubic degradation, no temporal loss, no distillation. Latency of fused tiers at 1080p and 4K outputs, ONNX export.
#
# **Required Kaggle settings.** Accelerator **GPU T4 x2**; **Internet on** (pip packages, Hugging Face data in the fallback path, RAFT / AlexNet / DINOv2 weights). Attach the output of the `neural-data-pipeline` notebook (**Add Input > Notebook output**) and the same Kaggle datasets it lists (`soumikrakshit/div2k-high-resolution-images`, `daehoyang/flickr2k`, `amithkesavmrajagiri/reds-dataset`, `wangsally/vimeo-90k-7`, `chenshu123/vimeo-triplet`, `artemmmtry/mpi-sintel-dataset`, `uom200647r/vid4-dataset`). Optional secret `HF_TOKEN`.
#
# **Budget** (enforced; numbers in the configuration cell): caches about 25 min, teacher 100 min, main student 80 min, parallel `igpu` 3x and `low` 2x students 50 min, three parallel ablation pairs 20 min each, evaluation and export about 45 min: about 6.5 hours of a 9.5-hour ceiling. `QUICK_RUN = True` gives a 40-minute smoke test of every cell.
#
# **Outputs** in `/kaggle/working/`:
#
# ```text
# code/          library and training scripts written by this notebook, run configs
# weights/       *_best.pt / *_last.pt per run, fused students (*.pt), ONNX models
# plots/         every figure as PNG
# metrics/       JSON / CSV: hardware, caches, histories, per-frame and aggregated results, ablations, latency, summary
# logs/          notebook log and one log per training run
# predictions/   sample output frames and temporal profiles
# outputs.zip    everything above (the training caches are deleted before zipping)
# ```

# %% [markdown]
# ## 1. Library files
#
# The first cell creates `/kaggle/working/code`; the shared NeuralSS library and the DDP training script are then written into it (the repository's `src/neuralss` is their single source). Training runs as separate processes that import the same files.

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
# **`nss_dedup.py`**: pHash / dHash, DINOv2 embeddings on both GPUs, exhaustive GPU neighbour search, SIFT + RANSAC geometric verification, synthetic copies for calibration, the duplicate rule and the leakage-safe split.

# %%
# WRITEFILE nss_dedup.py

# %% [markdown]
# **`nss_pipeline.py`**: the data-pipeline steps (discover, materialise, statistics, filters, hashes, calibration, duplicate search, split, audit) and the end-to-end fallback.

# %%
# WRITEFILE nss_pipeline.py

# %% [markdown]
# **`nss_cache.py`**: memory-mapped training caches (still crops, sequence windows, frame-generation quads) and RAFT teacher motion.

# %%
# WRITEFILE nss_cache.py

# %% [markdown]
# **`nss_synth.py`**: GPU renderer of layered game-like scenes with exact motion, disocclusion, thin geometry, particles, HUD / menus / cuts, jittered low-resolution rendering; batch builders for real sequences.

# %%
# WRITEFILE nss_synth.py

# %% [markdown]
# **`nss_models.py`**: RepConv blocks with online re-parameterisation, the temporal super-resolution network, the frame generator, tiers, fusion and ONNX wrappers.

# %%
# WRITEFILE nss_models.py

# %% [markdown]
# **`nss_metrics.py`**: losses, PSNR / SSIM / LPIPS / FLIP, RAFT teacher, temporal metrics, analytic baselines and the frame-pacing latency model.

# %%
# WRITEFILE nss_metrics.py

# %% [markdown]
# **`nss_train.py`**: DDP harness: batch streaming from memory maps, wall-clock schedule, EMA, checkpoints and the PROGRESS / ROUND / DONE protocol.

# %%
# WRITEFILE nss_train.py

# %% [markdown]
# **`nss_eval.py`**: test clips, input construction, method runners and per-frame metric rows.

# %%
# WRITEFILE nss_eval.py

# %% [markdown]
# **`train_tsr.py`**: DDP training entry point of the temporal super-resolution network.

# %%
# WRITEFILE train_tsr.py

# %% [markdown]
# ## 2. Setup and dependencies
#
# Installs the two packages Kaggle does not ship (`lpips` for the perceptual loss and metric, `flip-evaluator` for NVIDIA FLIP) plus `onnxruntime` for the export check; each is optional and its absence is logged. Imports the library and creates the run folders and logger. Expected output: install status per package.

# %%
import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/kaggle/working/code")
import nss_common as C

RUN = C.Run("nss_sr")
log = RUN.log.info
PKGS = C.pip_install(
    {
        "lpips": "lpips",
        "flip-evaluator": "flip_evaluator",
        "onnxruntime": "onnxruntime",
    },
    log=RUN.log,
)
print(PKGS)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display

import nss_cache as K
import nss_data as D
import nss_eval as E
import nss_metrics as M
import nss_models as NM
import nss_pipeline as PL
import nss_synth as S

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

# %% [markdown]
# ## 3. Hardware check
#
# GPUs, CPU cores, RAM and disk detected at run time; stops early unless two GPUs are visible, because training uses one process per GPU. Expected output: two Tesla T4 rows.

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
# Every tunable value. Key choices:
#
# * `scale` 3 is the primary ratio (360p to 1080p, the extreme case); `scale_alt` 2 is the less aggressive ratio (540p to 1080p) trained for the `low` tier.
# * `patch` 192 output pixels (64 low-resolution pixels at 3x) and `seq_len` 6 frames per training sequence (truncated back-propagation through the recurrence); `still_size` 512 texture crops leave room for fast camera paths.
# * `p_real` is the share of each batch drawn from real sequences; the rest is synthetic.
# * Loss weights follow the research plan: temporal 0.5, LPIPS 0.05, distillation 0.5.
# * `minutes` are wall-clock training budgets per stage; they are scaled down automatically if the run budget is short.

# %%
QUICK_RUN = False
RUN_BUDGET_H = 9.5
CFG = {
    "seed": 11,
    "scale": 3,
    "scale_alt": 2,
    "patch": 192,
    "seq_len": 6,
    "still_size": 512,
    "student_tier": "low",
    "small_tier": "igpu",
    "teacher_tier": "teacher",
    "cache": {
        "still_train": 600 if QUICK_RUN else 6500,
        "still_val": 64 if QUICK_RUN else 256,
        "win_train": 300 if QUICK_RUN else 3600,
        "win_val": 48 if QUICK_RUN else 192,
        "build_min": 6 if QUICK_RUN else 30,
    },
    "p_real": 0.4,
    "lr": 2e-3,
    "lr_teacher": 1e-3,
    "min_lr": 1e-6,
    "warmup_frac": 0.03,
    "weight_decay": 1e-4,
    "ema_decay": 0.999,
    "grad_clip": 1.0,
    "w_grad": 0.1,
    "w_fft": 0.05,
    "w_temporal": 0.5,
    "w_lpips": 0.05,
    "w_distill": 0.5,
    "val_bs": 8,
    "val_batches": 2 if QUICK_RUN else 4,
    "val_interval_min": 8,
    "minutes": {
        "teacher": 4 if QUICK_RUN else 100,
        "student": 4 if QUICK_RUN else 80,
        "pair": 3 if QUICK_RUN else 50,
        "ablation": 3 if QUICK_RUN else 20,
    },
    "eval": {
        "synthetic_per_class": 2 if QUICK_RUN else 6,
        "synthetic_T": 8 if QUICK_RUN else 16,
        "synthetic_P": 384,
        "real_clips": 2 if QUICK_RUN else 8,
        "real_T": 8 if QUICK_RUN else 16,
        "eval_reserve_h": 1.0,
    },
    "pipeline": {
        "list_cap_files": 3000 if QUICK_RUN else 20000,
        "seed": 7,
        "workers": os.cpu_count(),
        "hf_caps": {
            "game_stills": (120, 30) if QUICK_RUN else (600, 120),
            "tartanair": 30 if QUICK_RUN else 80,
            "tartanair_seq": 24,
            "gameir": (3, 3) if QUICK_RUN else (20, 12),
            "vimeo1080p": (6, 3) if QUICK_RUN else (60, 16),
        },
        "calibration_pairs": 300,
        "val_fraction": 0.05,
    },
}
ABLATIONS = {
    "reference": {},
    "no_history": {"use_history": False, "find_unused": True},
    "no_gbuffers": {"ablate_gbuffers": True},
    "bicubic_degradation": {"degradation": "bicubic"},
    "no_temporal_loss": {"w_temporal": 0.0},
    "no_distillation": {"teacher_ckpt": None},
}
RUN.save_json(
    {
        "quick_run": QUICK_RUN,
        "budget_h": RUN_BUDGET_H,
        "cfg": CFG,
        "ablations": ABLATIONS,
    },
    "config.json",
)
print(json.dumps(CFG["minutes"]), json.dumps(CFG["cache"]))

# %% [markdown]
# ## 5. Dataset manifest
#
# Looks for `nss_manifest.parquet` in an attached `neural-data-pipeline` output. If it is missing, the same pipeline runs inline with smaller caps (Kaggle discovery, Hugging Face subsets in `/tmp/nss_data`, quality filters, copy detection with DINOv2 candidates and SIFT verification, leakage-safe split with audit), which adds roughly 30 to 60 minutes. Expected output: where the manifest came from and the item counts per split.

# %%
with RUN.stage("manifest"):
    DATA_ROOT = D.locate_data_root(C.INPUT_ROOT)
    if DATA_ROOT is None:
        log("no attached pipeline output: building the manifest inline (smaller caps)")
        DATA_ROOT = Path("/tmp/nss_data")
        PL.build_manifest(DATA_ROOT, C.INPUT_ROOT, CFG["pipeline"], DEVICES, log=log)
    MANIFEST = D.load_manifest(DATA_ROOT / "nss_manifest.parquet")
log(f"data root: {DATA_ROOT}")
print(pd.crosstab(MANIFEST["source"], MANIFEST["split"]).to_string())

# %% [markdown]
# ## 6. Training and validation item selection
#
# Textures for the synthetic renderer come from stills and from frames of high-resolution sequences (at least 512 px on the shorter side); real training windows come from every sequence source. Test items are never touched until the evaluation section. Expected output: item counts per role.

# %%
TEX_SOURCES = ["div2k", "flickr2k", "game_stills", "gameir", "vimeo1080p", "reds"]
SEQ_SOURCES = ["tartanair", "sintel", "reds", "vimeo_septuplet", "vimeo1080p", "gameir"]
T, P = CFG["seq_len"], CFG["patch"]
ITEMS = {}
for split in ("train", "val"):
    ITEMS[f"tex_{split}"] = K.items_for(MANIFEST, split, TEX_SOURCES)
    ITEMS[f"seq_{split}"] = K.items_for(
        MANIFEST, split, SEQ_SOURCES, kinds=["seq"], min_frames=T
    )
print(pd.Series({k: len(v) for k, v in ITEMS.items()}).to_string())
assert ITEMS[
    "tex_train"
], "No texture sources found: attach the pipeline output or the Kaggle datasets."

# %% [markdown]
# ## 7. Training caches
#
# Decodes once, with all CPU cores, into memory-mapped arrays in `/kaggle/working/cache` (shared by both training processes through the page cache):
#
# * still crops of 512x512 for the synthetic renderer (random down-scale per source for varied detail density, flat crops rejected);
# * windows of 6 frames at 192x192 with motion vectors, validity and depth at half resolution.
#
# Windows from video sources get RAFT motion (frame to previous frame) with a forward-backward consistency mask, computed on both GPUs. Expected output: progress bars, counts per source and the cache size.

# %%
CACHE = C.WORK / "cache"
CACHE.mkdir(exist_ok=True)
CACHES = {}
with RUN.stage("caches"):
    deadline = CFG["cache"]["build_min"] * 60
    for split in ("train", "val"):
        n = CFG["cache"][f"still_{split}"]
        tasks = K.still_tasks(
            ITEMS[f"tex_{split}"],
            DATA_ROOT,
            n,
            CFG["still_size"],
            2 if split == "train" else 1,
            CFG["seed"],
        )
        cnt = K.build_still_cache(
            tasks,
            CACHE / f"still_{split}.npy",
            n,
            CFG["still_size"],
            os.cpu_count(),
            deadline,
            log=log,
        )
        CACHES[f"still_{split}"] = {
            "arrays": {"rgb": str(CACHE / f"still_{split}.npy")},
            "count": cnt,
        }
    RAFTS = [M.load_raft(d, log=log) for d in DEVICES]
    RAFTS = RAFTS if all(r is not None for r in RAFTS) else []
    WIN_COUNTS = {}
    for split in ("train", "val"):
        prefix = CACHE / f"win_{split}"
        cnt, counts, need = K.build_window_cache(
            ITEMS[f"seq_{split}"],
            DATA_ROOT,
            prefix,
            CFG["cache"][f"win_{split}"],
            T,
            P,
            os.cpu_count(),
            deadline,
            CFG["seed"] + (split == "val"),
            log=log,
        )
        K.fill_estimated_motion(prefix, need, RAFTS, DEVICES[: len(RAFTS)], log=log)
        CACHES[f"seq_{split}"] = K.cache_spec(
            prefix, cnt, ["rgb", "mv", "valid", "depth", "meta"]
        )
        WIN_COUNTS[split] = counts
cache_gb = sum(p.stat().st_size for p in CACHE.glob("*.npy")) / 1e9
RUN.save_json(
    {"caches": CACHES, "window_sources": WIN_COUNTS, "cache_gb": round(cache_gb, 2)},
    "caches.json",
)
print(
    pd.DataFrame(WIN_COUNTS).fillna(0).astype(int).to_string(),
    f"\ncache size {cache_gb:.2f} GB",
)

# %% [markdown]
# ## 8. Cache inspection
#
# Random still crops, and for real windows: first and last frame, the motion-vector field of the last frame (colour wheel: hue = direction, brightness = magnitude) and its validity (black = disoccluded or out of view). Expected output: two image grids and per-window statistics (share with exact motion, depth, mean motion magnitude).


# %%
def flow_to_rgb(f):
    mag = np.linalg.norm(f, axis=-1)
    ang = np.arctan2(f[..., 1], f[..., 0])
    hsv = np.stack(
        [
            (ang + np.pi) / (2 * np.pi),
            np.ones_like(mag),
            np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1),
        ],
        -1,
    )
    import matplotlib.colors as mcolors

    return mcolors.hsv_to_rgb(hsv)


st = np.load(CACHES["still_train"]["arrays"]["rgb"], mmap_mode="r")
fig, axes = plt.subplots(2, 8, figsize=(16, 4.2))
for ax, i in zip(
    axes.flat,
    np.random.default_rng(0).choice(CACHES["still_train"]["count"], 16, replace=False),
):
    ax.imshow(st[i])
    ax.axis("off")
fig.suptitle("Still crops (textures for the synthetic renderer)")
RUN.show(fig, "cache_stills")
if CACHES["seq_train"]["count"]:
    arr = {
        k: np.load(p, mmap_mode="r") for k, p in CACHES["seq_train"]["arrays"].items()
    }
    pick = np.random.default_rng(1).choice(
        CACHES["seq_train"]["count"],
        min(4, CACHES["seq_train"]["count"]),
        replace=False,
    )
    fig, axes = plt.subplots(len(pick), 4, figsize=(12, 3 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        for c, (img, title) in enumerate(
            [
                (arr["rgb"][i, 0], "frame 0"),
                (arr["rgb"][i, -1], f"frame {T - 1}"),
                (flow_to_rgb(arr["mv"][i, -1].astype(np.float32)), "motion vectors"),
                (arr["valid"][i, -1], "validity"),
            ]
        ):
            axes[r, c].imshow(img, cmap="gray" if c == 3 else None)
            axes[r, c].set_title(
                f"{title} (mv quality {arr['meta'][i, 1]:.1f})", fontsize=8
            )
            axes[r, c].axis("off")
    RUN.show(fig, "cache_windows")
    meta = arr["meta"][: CACHES["seq_train"]["count"]]
    print(
        pd.Series(
            {
                "exact_motion_share": float((meta[:, 1] >= 1).mean()),
                "estimated_motion_share": float((meta[:, 1] == 0.5).mean()),
                "depth_share": float(meta[:, 0].mean()),
                "mean_mv_px": float(
                    np.abs(arr["mv"][: min(256, len(meta))].astype(np.float32)).mean()
                ),
            }
        )
        .round(3)
        .to_string()
    )

# %% [markdown]
# ## 9. Synthetic renderer preview
#
# One synthetic training sequence: jittered low-resolution input (nearest-neighbour enlarged, showing aliasing), the anti-aliased target, exact motion vectors, disocclusion mask, depth and particle (reactive) mask. Expected output: a 6-column grid for three frames.

# %%
g = torch.Generator(device=DEVICES[0])
g.manual_seed(3)
TEXT_BANK = torch.from_numpy(
    S.make_text_bank(64, CFG["seed"], height=24, width=128)
).to(DEVICES[0])
stills_t = torch.from_numpy(np.ascontiguousarray(st[:4])).to(DEVICES[0])
demo = S.sr_batch(stills_t, CFG["scale"], T, P, g, text_bank=TEXT_BANK)
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for r, t in enumerate([1, 3, 5]):
    panels = [
        (
            F.interpolate(
                demo["lr"][0, t : t + 1], scale_factor=CFG["scale"], mode="nearest"
            )[0],
            "LR input (jittered)",
        ),
        (demo["hr"][0, t], "target (4x SSAA)"),
        (
            torch.from_numpy(
                flow_to_rgb(demo["mv_hr"][0, t].permute(1, 2, 0).cpu().numpy())
            ).permute(2, 0, 1),
            "motion vectors",
        ),
        (demo["valid_hr"][0, t].expand(3, -1, -1), "valid history"),
        (demo["depth_lr"][0, t].expand(3, -1, -1), "depth (1 = near)"),
        (demo["reactive_lr"][0, t].expand(3, -1, -1), "reactive (particles)"),
    ]
    for c, (img, title) in enumerate(panels):
        axes[r, c].imshow(img.permute(1, 2, 0).float().cpu().clamp(0, 1).numpy())
        axes[r, c].set_title(f"t={t}: {title}", fontsize=8)
        axes[r, c].axis("off")
RUN.show(fig, "synthetic_preview")

# %% [markdown]
# ## 10. Model tiers, cost and fusion parity
#
# For each tier: trainable parameters, deployed (fused) parameters, multiply-accumulates per frame at 640x360 to 1920x1080 (3x) and 960x540 to 1920x1080 (2x), and the maximum difference between the multi-branch and fused networks on random input (should be about 1e-5 or less). Expected output: a tier table and a bar chart.

# %%
tier_rows = []
for tier in NM.TSR_TIERS:
    for s in (CFG["scale"], CFG["scale_alt"]):
        net = NM.build_tsr(tier, s).eval()
        f = {
            "color": torch.rand(1, 3, 32, 32),
            "depth": torch.rand(1, 1, 32, 32),
            "mv": torch.randn(1, 2, 32, 32),
            "reactive": torch.zeros(1, 1, 32, 32),
            "exposure": torch.zeros(1),
            "jitter": torch.rand(1, 2) - 0.5,
            "has_depth": torch.ones(1),
            "mv_quality": torch.ones(1),
        }
        with torch.no_grad():
            for p in net.out.parameters():
                p.normal_(
                    0, 0.01
                )  # non-zero head so parity is tested on a non-trivial output
            fused = NM.fuse_model(net)
            o1, st1 = net(f, None)
            o2, _ = fused(f, None)
            o1b, _ = net(f, st1)
            o2b, _ = fused(f, st1)
        lr_px = (1920 // s) * (1080 // s)
        macs = NM.tsr_macs_per_lr_pixel(net)
        tier_rows.append(
            {
                "tier": tier,
                "scale": s,
                "train_params": NM.count_params(net),
                "fused_params": NM.count_params(fused),
                "gmac_per_frame_1080p": round(macs * lr_px / 1e9, 2),
                "parity_max_abs": float(
                    max((o1 - o2).abs().max(), (o1b - o2b).abs().max())
                ),
            }
        )
TIERS = pd.DataFrame(tier_rows)
RUN.save_csv(TIERS, "model_tiers.csv", index=False)
print(TIERS.to_string(index=False))
assert TIERS["parity_max_abs"].max() < 1e-3, "fusion parity failed"
fig, ax = plt.subplots(figsize=(9, 3.5))
TIERS.pivot(index="tier", columns="scale", values="gmac_per_frame_1080p").plot.bar(
    ax=ax, rot=0
)
ax.set_ylabel("GMAC per 1080p output frame")
RUN.show(fig, "model_tiers")

# %% [markdown]
# ## 11. Batch-size probe
#
# Runs full training steps (synthetic batch generation, recurrent unroll, losses, backward) on GPU 0 for increasing batch sizes, for the teacher alone and for the student with the frozen teacher alongside (distillation), and keeps the largest batch below 85 percent of VRAM. Expected output: a table of batch size, peak VRAM share and sequences per second.


# %%
def probe(student_tier, teacher_tier, scale, sizes):
    rows = []
    dev = DEVICES[0]
    total = torch.cuda.get_device_properties(dev).total_memory
    stills = torch.from_numpy(np.ascontiguousarray(st[: max(sizes)])).to(dev)
    for bs in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            net = NM.build_tsr(student_tier, scale).to(dev)
            teacher = (
                NM.build_tsr(teacher_tier, scale).to(dev).eval().requires_grad_(False)
                if teacher_tier
                else None
            )
            opt = torch.optim.AdamW(net.parameters(), 1e-4)
            scaler = torch.amp.GradScaler("cuda")
            times = []
            for _ in range(3):
                torch.cuda.synchronize(dev)
                t0 = time.time()
                b = S.sr_batch(stills[:bs], scale, T, P, g)
                with torch.autocast("cuda", dtype=torch.float16):
                    outs, state = [], None
                    for t in range(T):
                        o, state = net(E.frame_inputs(b, t), state)
                        outs.append(o)
                    if teacher is not None:
                        with torch.no_grad():
                            st_ = None
                            for t in range(T):
                                _, st_ = teacher(E.frame_inputs(b, t), st_)
                    loss = sum(
                        M.charbonnier(o.float(), b["hr"][:, t])
                        for t, o in enumerate(outs)
                    )
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                torch.cuda.synchronize(dev)
                times.append(time.time() - t0)
            rows.append(
                {
                    "batch": bs,
                    "fits": True,
                    "peak_vram_pct": round(
                        100 * torch.cuda.max_memory_allocated(dev) / total, 1
                    ),
                    "seq_per_s": round(bs / np.mean(times[1:]), 1),
                }
            )
            del net, teacher, opt, outs, state, loss, b
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": bs, "fits": False})
            break
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    ok = df[df["fits"] & (df["peak_vram_pct"] <= 85)]
    return df, int(ok["batch"].max()) if len(ok) else sizes[0]


with RUN.stage("batch_probe"):
    sizes = [4, 8] if QUICK_RUN else [4, 8, 12, 16, 24, 32, 48]
    PROBE_T, BS_TEACHER = probe(CFG["teacher_tier"], None, CFG["scale"], sizes)
    PROBE_S, BS_STUDENT = probe(
        CFG["student_tier"], CFG["teacher_tier"], CFG["scale"], sizes
    )
print(
    "teacher\n",
    PROBE_T.to_string(index=False),
    "\nstudent + teacher\n",
    PROBE_S.to_string(index=False),
)
print(f"batch per GPU: teacher {BS_TEACHER}, student {BS_STUDENT}")
RUN.save_csv(
    pd.concat([PROBE_T.assign(run="teacher"), PROBE_S.assign(run="student")]),
    "batch_probe.csv",
    index=False,
)

# %% [markdown]
# ## 12. Training launcher and budget
#
# Builds one JSON config per run and launches `code/train_tsr.py`: DDP on both GPUs for the main stages, or two single-GPU runs in parallel threads for the paired stages and ablations. Each run streams `PROGRESS` lines into a progress bar (percent of its time budget, loss, sequences per second, learning rate, per-GPU memory) and `ROUND` lines (validation PSNR-Y, temporal-change error) into a live chart. Stage budgets are scaled down if the run budget is short. Expected output: the planned minutes per stage.

# %%
remaining_h = RUN_BUDGET_H - RUN.hours() - CFG["eval"]["eval_reserve_h"]
plan_min = (
    CFG["minutes"]["teacher"]
    + CFG["minutes"]["student"]
    + CFG["minutes"]["pair"]
    + 3 * CFG["minutes"]["ablation"]
)
SCALE_T = min(1.0, max(0.1, remaining_h * 60 / plan_min))
MINUTES = {k: v * SCALE_T for k, v in CFG["minutes"].items()}
print(
    pd.Series(MINUTES).round(1).to_string(),
    f"\n(scale {SCALE_T:.2f}, {remaining_h:.2f} h available for training)",
)
HISTORY = {}
PORT = [29500]


def run_cfg(name, tier, scale, minutes, batch, teacher_ckpt=None, **over):
    cfg = {
        "name": name,
        "tier": tier,
        "scale": scale,
        "seq_len": T,
        "patch": P,
        "seed": CFG["seed"],
        "batch_per_gpu": batch,
        "p_real": CFG["p_real"],
        "lr": CFG["lr_teacher"] if tier == "teacher" else CFG["lr"],
        "min_lr": CFG["min_lr"],
        "warmup_frac": CFG["warmup_frac"],
        "weight_decay": CFG["weight_decay"],
        "ema_decay": CFG["ema_decay"],
        "grad_clip": CFG["grad_clip"],
        "w_grad": CFG["w_grad"],
        "w_fft": CFG["w_fft"],
        "w_temporal": CFG["w_temporal"],
        "w_lpips": CFG["w_lpips"] if PKGS.get("lpips") else 0.0,
        "w_distill": CFG["w_distill"],
        "teacher_ckpt": teacher_ckpt,
        "teacher_tier": CFG["teacher_tier"],
        "train_seconds": minutes * 60,
        "val_interval_s": min(CFG["val_interval_min"] * 60, max(minutes * 60 / 6, 60)),
        "progress_interval_s": 15,
        "sync_every": 10,
        "select_metric": "psnr_y",
        "caches": {"still": CACHES["still_train"], "seq": CACHES["seq_train"]},
        "val_caches": {"still": CACHES["still_val"], "seq": CACHES["seq_val"]},
        "val_bs": CFG["val_bs"],
        "val_batches": CFG["val_batches"],
        "workers_per_rank": max(1, os.cpu_count() // 2 - 0),
        "weights_dir": str(RUN.dirs["weights"]),
        "metrics_dir": str(RUN.dirs["metrics"]),
    }
    cfg.update(over)
    return cfg


def live_plot(title):
    handle = display(plt.figure(), display_id=True)

    def update(rounds):
        h = pd.DataFrame(rounds)
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.2))
        axes[0].plot(h["elapsed_min"], h["train_loss"])
        axes[0].set_title("train loss")
        axes[1].plot(h["elapsed_min"], h["val_psnr_y"])
        axes[1].set_title("val PSNR-Y (dB)")
        axes[2].plot(h["elapsed_min"], h["val_tce"])
        axes[2].set_title("val temporal-change error")
        for ax in axes:
            ax.set_xlabel("minutes")
        fig.suptitle(title)
        handle.update(fig)
        plt.close(fig)

    return update


def launch(cfg, gpus, live=True):
    PORT[0] += 1
    cfg_path = RUN.dirs["code"] / f"cfg_{cfg['name']}.json"
    rounds, done = C.launch_training(
        RUN.work / "code" / "train_tsr.py",
        cfg,
        cfg_path,
        RUN.dirs["logs"] / f"{cfg['name']}.log",
        gpus,
        cfg["name"],
        port=PORT[0],
        on_round=live_plot(cfg["name"]) if live else None,
    )
    HISTORY[cfg["name"]] = rounds
    return done


def launch_pair(cfg_a, cfg_b):
    with ThreadPoolExecutor(2) as pool:
        fa = pool.submit(launch, cfg_a, [0], False)
        fb = pool.submit(launch, cfg_b, [1], False)
        return fa.result(), fb.result()


# %% [markdown]
# ## 13. Stage A: teacher (both GPUs)
#
# The `teacher` tier (64 channels, 8 blocks, 6 U-Net blocks) at 3x on both GPUs. It is too large for deployment; it only provides distillation targets for the students. Expected output: a live training chart and the final validation scores.

# %%
with RUN.stage("train_teacher"):
    S3 = CFG["scale"]
    done_teacher = launch(
        run_cfg(
            "tsr_teacher_s3", CFG["teacher_tier"], S3, MINUTES["teacher"], BS_TEACHER
        ),
        [0, 1],
    )
TEACHER_CKPT = str(RUN.dirs["weights"] / "tsr_teacher_s3_best.pt")
print(done_teacher)

# %% [markdown]
# ## 14. Stage B: main student with distillation (both GPUs)
#
# The `low` tier at 3x, trained with the full loss plus distillation toward the frozen teacher. Expected output: live chart and final scores.

# %%
with RUN.stage("train_student"):
    done_student = launch(
        run_cfg(
            f"tsr_{CFG['student_tier']}_s3",
            CFG["student_tier"],
            S3,
            MINUTES["student"],
            BS_STUDENT,
            teacher_ckpt=TEACHER_CKPT,
        ),
        [0, 1],
    )
print(done_student)

# %% [markdown]
# ## 15. Stage C: integrated-GPU student (3x) and 2x student, in parallel
#
# GPU 0 trains the smallest `igpu` tier at 3x with distillation; GPU 1 trains the `low` tier at 2x (the teacher is 3x only, so no distillation). Expected output: both runs' final scores.

# %%
with RUN.stage("train_pair"):
    S2 = CFG["scale_alt"]
    done_pair = launch_pair(
        run_cfg(
            f"tsr_{CFG['small_tier']}_s3",
            CFG["small_tier"],
            S3,
            MINUTES["pair"],
            BS_STUDENT,
            teacher_ckpt=TEACHER_CKPT,
        ),
        run_cfg(
            f"tsr_{CFG['student_tier']}_s2",
            CFG["student_tier"],
            S2,
            MINUTES["pair"],
            BS_STUDENT,
        ),
    )
print(done_pair)

# %% [markdown]
# ## 16. Ablations (three parallel pairs)
#
# Short runs of the `low` 3x student with one change each, against a `reference` run with the full recipe and the same short budget: no history (single-frame upscaling), no G-buffers (colour only: no depth or motion vectors), bicubic degradation (classic image super-resolution inputs instead of render-like jittered sampling), no temporal loss, and no distillation. Expected output: final scores of the six runs.

# %%
with RUN.stage("ablations"):
    names = list(ABLATIONS)
    for a, b in zip(names[0::2], names[1::2]):
        cfgs = []
        for name in (a, b):
            over = dict(ABLATIONS[name])
            ckpt = over.pop("teacher_ckpt", TEACHER_CKPT)
            cfgs.append(
                run_cfg(
                    f"abl_{name}",
                    CFG["student_tier"],
                    S3,
                    MINUTES["ablation"],
                    BS_STUDENT,
                    teacher_ckpt=ckpt,
                    **over,
                )
            )
        print(launch_pair(*cfgs))

# %% [markdown]
# ## 17. Training curves
#
# Validation PSNR-Y and temporal-change error against time for every run, and the train loss of the main stages. Expected output: three charts and a table of the best validation round per run.

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
best_rows = []
for name, rounds in HISTORY.items():
    if not rounds:
        continue
    h = pd.DataFrame(rounds)
    RUN.save_csv(h, f"history_{name}.csv", index=False)
    style = "-" if not name.startswith("abl_") else "--"
    axes[0].plot(h["elapsed_min"], h["val_psnr_y"], style, label=name)
    axes[1].plot(h["elapsed_min"], h["val_tce"], style, label=name)
    if not name.startswith("abl_"):
        axes[2].plot(h["elapsed_min"], h["train_loss"], label=name)
    b = h.loc[h["val_psnr_y"].idxmax()]
    best_rows.append(
        {
            "run": name,
            "best_round": int(b["round"]),
            "val_psnr_y": b["val_psnr_y"],
            "val_psnr_y_last": b["val_psnr_y_last"],
            "val_tce": b["val_tce"],
            "samples_per_s": h["samples_per_s"].mean(),
            "peak_mem_gb": max(max(m) for m in h["mem_gb"]),
        }
    )
for ax, t in zip(axes, ["val PSNR-Y (dB)", "val temporal-change error", "train loss"]):
    ax.set_title(t)
    ax.set_xlabel("minutes")
axes[0].legend(fontsize=7)
RUN.show(fig, "training_curves")
BEST = pd.DataFrame(best_rows)
RUN.save_csv(BEST, "best_validation.csv", index=False)
print(BEST.round(4).to_string(index=False))

# %% [markdown]
# ## 18. Load, fuse and place models
#
# Loads the best EMA weights of every run, folds RepConv branches into plain convolutions, checks fused against unfused output on a real validation batch, and keeps FP16 fused replicas on GPU 0 for evaluation. Expected output: a table of runs with fused parameter count and parity.


# %%
def load_tsr(name, tier, scale):
    path = RUN.dirs["weights"] / f"{name}_best.pt"
    if not path.exists():
        return None
    net = NM.build_tsr(tier, scale)
    sd = torch.load(path, map_location="cpu")["ema"]
    net.load_state_dict(
        {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
    )
    return net.eval()


RUNS = {
    "teacher_s3": ("tsr_teacher_s3", CFG["teacher_tier"], S3),
    "student_s3": (f"tsr_{CFG['student_tier']}_s3", CFG["student_tier"], S3),
    "igpu_s3": (f"tsr_{CFG['small_tier']}_s3", CFG["small_tier"], S3),
    "student_s2": (f"tsr_{CFG['student_tier']}_s2", CFG["student_tier"], S2),
}
RUNS.update({f"abl_{a}": (f"abl_{a}", CFG["student_tier"], S3) for a in ABLATIONS})
MODELS, rows = {}, []
vb = {k: v.to(DEVICES[0]) for k, v in S.sr_batch(stills_t[:2], S3, 4, P, g).items()}
for key, (name, tier, scale) in RUNS.items():
    net = load_tsr(name, tier, scale)
    if net is None:
        continue
    fused = NM.fuse_model(net).to(DEVICES[0]).half()
    torch.save(
        {"state_dict": fused.state_dict(), "tier": tier, "scale": scale},
        RUN.dirs["weights"] / f"{name}_fused.pt",
    )
    if scale == S3:  # FP32 parity on a synthetic validation batch
        a = E.run_tsr(net.to(DEVICES[0]).float(), vb, amp=False)
        b2 = E.run_tsr(NM.fuse_model(net).to(DEVICES[0]).float(), vb, amp=False)
        parity = float((a - b2).abs().max())
    else:
        parity = None
    MODELS[key] = (fused, scale)
    net.cpu()
    rows.append(
        {
            "model": key,
            "tier": tier,
            "scale": scale,
            "fused_params": NM.count_params(fused),
            "parity_max_abs": parity,
        }
    )
print(pd.DataFrame(rows).to_string(index=False))
LPIPS = M.load_lpips(DEVICES[0], log=log)
RAFT = RAFTS[0] if RAFTS else M.load_raft(DEVICES[0], log=log)

# %% [markdown]
# ## 19. Test suites
#
# Built only from the test split:
#
# * **synthetic-game**: layered scenes from held-out game frames and DIV2K validation images at 384x384 output, one set per motion class (static, slow 0.3 to 3 px/frame, medium 3 to 10, fast 10 to 28), with exact motion, disocclusions, thin geometry and particles;
# * **tartanair**: held-out environments (Japanese alley, seaside town) with exact motion and depth, 576x480;
# * **sintel**: held-out `market` and `cave` scene families with exact motion, 1008x432;
# * **video**: Vid4 and held-out Vimeo-1080p clips with RAFT motion;
# * **gameir-native**: held-out town 05, native 720p renders to 1440p ground truth (2x, a real renderer's low-resolution image instead of a simulated one).
#
# Expected output: the number of sequences per suite.

# %%
ev = CFG["eval"]
TEST_ITEMS = D.items_from_frame(MANIFEST[MANIFEST["split"] == "test"])
test_still_items = [i for i in TEST_ITEMS if i["source"] in ("game_stills", "div2k")]
SUITES = {}
with RUN.stage("test_suites"):
    S_EV = 1024  # texture crop size for 384-px test views
    tasks = K.still_tasks(
        test_still_items, DATA_ROOT, 4 * ev["synthetic_per_class"], S_EV, 1, 99
    )
    crops = []
    for task in tasks:
        _, c = D.still_crop(task)
        if c is not None:
            crops.append(c[0])
        if len(crops) >= 4 * ev["synthetic_per_class"]:
            break
    TEST_STILLS = torch.from_numpy(np.stack(crops)).to(DEVICES[0]) if crops else None

    def clips(source, crop, n, Tn, max_side=None):
        out = []
        for it in [
            i for i in TEST_ITEMS if i["source"] == source and i["n_frames"] >= Tn
        ][: n * 3]:
            c = E.load_clip(DATA_ROOT, it, 0, Tn, crop, max_side=max_side)
            if c is not None:
                out.append((it["item_id"], c))
            if len(out) >= n:
                break
        return out

    REAL = {
        "tartanair": clips("tartanair", (480, 576), ev["real_clips"], ev["real_T"]),
        "sintel": clips("sintel", (432, 1008), ev["real_clips"], ev["real_T"]),
        "vid4": clips("vid4", (480, 696), ev["real_clips"], ev["real_T"]),
        "vimeo1080p": clips(
            "vimeo1080p", (540, 960), ev["real_clips"], ev["real_T"], max_side=540
        ),
        "gameir_native": clips(
            "gameir", (1440, 1920), max(2, ev["real_clips"] // 2), 6
        ),
    }
print(
    pd.Series(
        {
            "synthetic stills": 0 if TEST_STILLS is None else len(TEST_STILLS),
            **{k: len(v) for k, v in REAL.items()},
        }
    ).to_string()
)

# %% [markdown]
# ## 20. Evaluation
#
# Runs every method on every suite at 3x (and the 2x student on 2x inputs) and records per-frame metrics, skipping the first frame (no history yet): PSNR-Y, SSIM-Y, PSNR inside disoccluded regions, fine-detail regions and particles, LPIPS and tOF every fourth frame, warping error (exact motion only) and temporal-change error. Methods: bicubic, jitter-aware bilinear, analytic TAA upscaler, the teacher, the students and the ablations. Expected output: progress logs and the number of metric rows.

# %%
ROWS = []


def evaluate(suite, b, scale, methods, extra=None):
    ref = (
        MODELS["student_s3"][0] if scale == S3 else MODELS.get("student_s2", (None,))[0]
    )
    for mname in methods:
        if mname == "bicubic":
            out = E.run_bicubic(b, scale)
        elif (
            mname == "bilinear"
        ):  # the reference network only provides its parameter-free upsampler
            out = E.run_bilinear(ref, b)
        elif mname == "analytic_taau":
            out = E.run_taau(ref, b, scale)
        else:
            net, s_ = MODELS[mname]
            if s_ != scale:
                continue
            kw = {"use_history": False} if mname == "abl_no_history" else {}
            kw.update({"ablate_gbuffers": True} if mname == "abl_no_gbuffers" else {})
            out = E.run_tsr(net, b, **kw)
        for r in E.sr_frame_metrics(out, b, scale, LPIPS, RAFT):
            ROWS.append(
                {"suite": suite, "method": mname, "scale": scale, **(extra or {}), **r}
            )


METHODS_S3 = ["bicubic", "bilinear", "analytic_taau"] + [
    k for k, v in MODELS.items() if v[1] == S3
]
with RUN.stage("evaluation"):
    if TEST_STILLS is not None:
        for cls, name in enumerate(S.MOTION_PROFILE["names"]):
            sub = TEST_STILLS[
                cls * ev["synthetic_per_class"] : (cls + 1) * ev["synthetic_per_class"]
            ]
            if len(sub):
                b = E.synthetic_suite(
                    sub,
                    S3,
                    ev["synthetic_T"],
                    ev["synthetic_P"],
                    cls,
                    1000 + cls,
                    DEVICES[0],
                    text_bank=TEXT_BANK,
                )
                evaluate("synthetic-game", b, S3, METHODS_S3, {"motion": name})
                if "student_s2" in MODELS:
                    b2 = E.synthetic_suite(
                        sub,
                        S2,
                        ev["synthetic_T"],
                        ev["synthetic_P"],
                        cls,
                        1000 + cls,
                        DEVICES[0],
                        text_bank=TEXT_BANK,
                    )
                    evaluate(
                        "synthetic-game",
                        b2,
                        S2,
                        ["bicubic", "bilinear", "analytic_taau", "student_s2"],
                        {"motion": name},
                    )
            log(f"synthetic {name} done")
    for suite in ("tartanair", "sintel", "vid4", "vimeo1080p"):
        for cid, clip in REAL[suite]:
            b = E.sr_inputs(clip, S3, DEVICES[0], raft=RAFT)
            evaluate(suite, b, S3, METHODS_S3, {"clip": cid, "motion": "real"})
            if "student_s2" in MODELS:
                b2 = E.sr_inputs(clip, S2, DEVICES[0], raft=RAFT)
                evaluate(
                    suite,
                    b2,
                    S2,
                    ["bicubic", "bilinear", "analytic_taau", "student_s2"],
                    {"clip": cid, "motion": "real"},
                )
        log(f"{suite} done")
    for cid, clip in REAL["gameir_native"]:
        b2 = E.sr_inputs(clip, S2, DEVICES[0], raft=RAFT, native=True)
        evaluate(
            "gameir-native",
            b2,
            S2,
            ["bicubic", "bilinear", "analytic_taau", "student_s2"],
            {"clip": cid, "motion": "real"},
        )
RES = pd.DataFrame(ROWS)
METRIC_COLS = [
    "psnr_y",
    "ssim_y",
    "psnr_disocc",
    "psnr_detail",
    "psnr_particles",
    "lpips",
    "warp_err",
    "temporal_err",
    "tof",
    "mv_px",
]
for (
    col
) in (
    METRIC_COLS
):  # metrics that were not computable are None: make every metric column numeric (NaN)
    RES[col] = pd.to_numeric(RES[col], errors="coerce")
RUN.save_csv(RES, "test_per_frame.csv", index=False)
print(len(RES), "metric rows")

# %% [markdown]
# ## 21. Results by suite and method
#
# Mean of every metric per suite, scale and method (lower is better for LPIPS, warping, temporal-change error and tOF). Expected output: the results table and a chart of PSNR-Y against temporal-change error (top-left is better: sharp and stable).

# %%
AGG = (
    RES.groupby(["suite", "scale", "method"])
    .mean(numeric_only=True)
    .drop(columns=["sample", "frame", "speed_class"], errors="ignore")
)
RUN.save_csv(AGG.round(4), "test_results.csv")
print(AGG.round(3).to_string())
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, scale in zip(axes, (S3, S2)):
    sub = (
        AGG.xs(scale, level="scale")
        if scale in AGG.index.get_level_values("scale")
        else None
    )
    if sub is None:
        continue
    for (suite, method), r in sub.iterrows():
        ax.scatter(r["temporal_err"], r["psnr_y"], s=30)
        ax.annotate(f"{method}\n{suite}", (r["temporal_err"], r["psnr_y"]), fontsize=6)
    ax.set_xlabel("temporal-change error (lower is more stable)")
    ax.set_ylabel("PSNR-Y (dB)")
    ax.set_title(f"{scale}x")
RUN.show(fig, "quality_vs_stability")

# %% [markdown]
# ## 22. Motion classes, regions and perceptual metrics (synthetic-game, 3x)
#
# How quality changes from static to fast motion, and quality inside the hardest regions: disocclusions (no valid history), fine detail (top 15 percent gradients: thin geometry, foliage, text) and particles (transparent effects without motion vectors). Expected output: three grouped bar charts.

# %%
syn = RES[(RES["suite"] == "synthetic-game") & (RES["scale"] == S3)]
if len(syn):
    print(
        "measured motion per class (output px / frame):",
        syn.groupby("motion")["mv_px"].mean().round(2).to_dict(),
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    syn.groupby(["motion", "method"])["psnr_y"].mean().unstack().reindex(
        S.MOTION_PROFILE["names"]
    ).plot.bar(ax=axes[0], rot=0)
    axes[0].set_title("PSNR-Y by motion class")
    syn.groupby("method")[
        ["psnr_disocc", "psnr_detail", "psnr_particles"]
    ].mean().plot.bar(ax=axes[1], rot=30)
    axes[1].set_title("PSNR-Y in hard regions")
    syn.groupby("method")[["lpips", "tof"]].mean().plot.bar(ax=axes[2], rot=30)
    axes[2].set_title("LPIPS and tOF (lower is better)")
    for ax in axes:
        ax.legend(fontsize=7)
    RUN.show(fig, "motion_regions")

# %% [markdown]
# ## 23. Static-scene convergence
#
# A static camera with only sub-pixel jitter: a temporal upscaler should accumulate detail frame after frame and approach the native image, while single-frame methods stay flat. Expected output: PSNR-Y against frame index per method, and FLIP for the last frame.

# %%
if TEST_STILLS is not None:
    sub = TEST_STILLS[: ev["synthetic_per_class"]]
    b = E.synthetic_suite(
        sub, S3, max(ev["synthetic_T"], 16), ev["synthetic_P"], 0, 4242, DEVICES[0]
    )
    conv, flip_rows = {}, []
    for mname in [
        "bilinear",
        "analytic_taau",
        "student_s3",
        "igpu_s3",
        "teacher_s3",
        "abl_no_history",
    ]:
        if mname in ("bilinear", "analytic_taau"):
            ref_net = MODELS["student_s3"][
                0
            ]  # only its parameter-free jitter-aware upsampler is used
            out = (
                E.run_bilinear(ref_net, b)
                if mname == "bilinear"
                else E.run_taau(ref_net, b, S3)
            )
        elif mname in MODELS:
            out = E.run_tsr(
                MODELS[mname][0], b, use_history=(mname != "abl_no_history")
            )
        else:
            continue
        conv[mname] = [
            M.psnr(M.rgb_to_y(out[:, t]), M.rgb_to_y(b["hr"][:, t]), border=S3)
            .mean()
            .item()
            for t in range(out.shape[1])
        ]
        fe = M.flip_error(b["hr"][0, -1], out[0, -1])
        flip_rows.append(
            {"method": mname, "flip_last_frame": fe, "psnr_y_last": conv[mname][-1]}
        )
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, v in conv.items():
        ax.plot(v, label=k)
    ax.set_xlabel("frame")
    ax.set_ylabel("PSNR-Y (dB)")
    ax.set_title("Static camera with jitter: convergence toward native")
    ax.legend()
    RUN.show(fig, "static_convergence")
    FLIP_STATIC = pd.DataFrame(flip_rows)
    RUN.save_csv(FLIP_STATIC, "static_convergence.csv", index=False)
    print(FLIP_STATIC.round(4).to_string(index=False))

# %% [markdown]
# ## 24. Visual comparisons, history weights and temporal profiles
#
# For one fast-motion synthetic sequence and one TartanAir clip: crops of the last frame for each method, the student's learned history weight `a` (white = history kept, black = history rejected, which should light up at disocclusions and particles), and an x-t slice (one image row over time) where flicker and ghosting appear as streaks. Expected output: two figures; full frames are saved to `predictions/`.


# %%
def visual(b, title, fname, crop=160):
    outs = {
        "bilinear": E.run_bilinear(MODELS["student_s3"][0], b),
        "analytic_taau": E.run_taau(MODELS["student_s3"][0], b, S3),
    }
    for k in ("igpu_s3", "student_s3", "teacher_s3"):
        if k in MODELS:
            outs[k] = E.run_tsr(MODELS[k][0], b)
    _, alpha = E.run_tsr(MODELS["student_s3"][0], b, keep_alpha=True)
    outs["ground truth"] = b["hr"]
    H, W = b["hr"].shape[-2:]
    y0, x0 = (H - crop) // 2, (W - crop) // 2
    fig, axes = plt.subplots(2, len(outs) + 1, figsize=(3 * (len(outs) + 1), 6.4))
    row = H // 2
    for c, (k, o) in enumerate(outs.items()):
        axes[0, c].imshow(
            o[0, -1, :, y0 : y0 + crop, x0 : x0 + crop].permute(1, 2, 0).cpu().numpy()
        )
        axes[0, c].set_title(k, fontsize=8)
        axes[1, c].imshow(
            o[0, :, :, row, :].permute(0, 2, 1).cpu().numpy(), aspect="auto"
        )  # time down, x across
        axes[1, c].set_title("x-t slice", fontsize=8)
        import PIL.Image as PImage

        PImage.fromarray(
            (o[0, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ).save(RUN.dirs["predictions"] / f"{fname}_{k.replace(' ', '_')}.png")
    axes[0, -1].imshow(
        alpha[0, -1, 0, y0 : y0 + crop, x0 : x0 + crop].cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, -1].set_title("history weight a", fontsize=8)
    axes[1, -1].imshow(
        1 - b["valid_hr"][0, -1, 0, y0 : y0 + crop, x0 : x0 + crop].cpu().numpy(),
        cmap="gray",
    )
    axes[1, -1].set_title("true disocclusion", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(title)
    RUN.show(fig, fname)


if TEST_STILLS is not None:
    visual(
        E.synthetic_suite(
            TEST_STILLS[-2:],
            S3,
            12,
            ev["synthetic_P"],
            3,
            77,
            DEVICES[0],
            text_bank=TEXT_BANK,
        ),
        "synthetic, fast motion",
        "visual_synthetic_fast",
    )
if REAL["tartanair"]:
    visual(
        E.sr_inputs(REAL["tartanair"][0][1], S3, DEVICES[0]),
        "TartanAir held-out environment",
        "visual_tartanair",
    )

# %% [markdown]
# ## 25. Ablations
#
# Same short budget, one change each, on the synthetic-game and TartanAir suites. The difference to `abl_reference` isolates the value of history, G-buffers, render-like training degradation, the temporal loss and distillation. Expected output: a table of deltas and a chart.

# %%
abl = RES[
    RES["method"].str.startswith("abl_")
    & RES["suite"].isin(["synthetic-game", "tartanair"])
    & (RES["scale"] == S3)
]
if len(abl):
    ABL = abl.groupby(["suite", "method"])[
        ["psnr_y", "ssim_y", "lpips", "temporal_err", "psnr_disocc"]
    ].mean()
    ref = ABL.xs("abl_reference", level="method")
    delta = ABL.sub(ref, level="suite")
    RUN.save_csv(ABL.round(4), "ablations.csv")
    print(
        ABL.round(3).to_string(), "\n\ndelta to reference\n", delta.round(3).to_string()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    delta["psnr_y"].unstack(0).plot.bar(ax=axes[0], rot=30)
    axes[0].set_title("PSNR-Y change vs reference (dB)")
    delta["temporal_err"].unstack(0).plot.bar(ax=axes[1], rot=30)
    axes[1].set_title("temporal-change error change (negative is better)")
    RUN.show(fig, "ablations")

# %% [markdown]
# ## 26. Latency of fused models
#
# Times one fused FP16 step (including warps, jitter-aware upsampling and the final blend) on one T4 with CUDA events after warm-up, batch 1, for 640x360 to 1920x1080 (3x), 960x540 to 1920x1080 (2x) and 1280x720 to 3840x2160 (3x). PyTorch eager on a T4 ranks variants; in-engine latency on target GPUs needs the ONNX / DirectML path. The arithmetic estimate column converts multiply-accumulates to milliseconds for an assumed sustained FP16 throughput (a parameter, not a measurement). Expected output: latency table and chart.


# %%
@torch.no_grad()
def time_step(net, scale, out_hw, iters=100):
    dev = DEVICES[0]
    h, w = out_hw[0] // scale, out_hw[1] // scale
    f = {
        "color": torch.rand(1, 3, h, w, device=dev).half(),
        "depth": torch.rand(1, 1, h, w, device=dev).half(),
        "mv": torch.zeros(1, 2, h, w, device=dev).half(),
        "reactive": torch.zeros(1, 1, h, w, device=dev).half(),
        "exposure": torch.zeros(1, device=dev),
        "jitter": torch.zeros(1, 2, device=dev),
        "has_depth": torch.ones(1, device=dev),
        "mv_quality": torch.ones(1, device=dev),
    }
    _, state = net(f, None)
    for _ in range(10):
        _, state = net(f, state)
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(iters):
        a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        _, state = net(f, state)
        b_.record()
        torch.cuda.synchronize(dev)
        times.append(a.elapsed_time(b_))
    return np.array(times)


lat_rows = []
SUSTAINED_TFLOPS = {
    "assumed iGPU (4 TFLOPS FP16 sustained)": 4.0,
    "assumed low-end dGPU (8 TFLOPS)": 8.0,
}
for key in ("igpu_s3", "student_s3", "student_s2", "teacher_s3"):
    if key not in MODELS:
        continue
    net, scale = MODELS[key]
    for out_hw in ((1080, 1920), (2160, 3840)):
        if scale == S2 and out_hw[0] > 1080:
            continue
        t_ms = time_step(net, scale, out_hw, iters=30 if QUICK_RUN else 100)
        tier = RUNS[key][1]
        gmac = (
            TIERS[(TIERS["tier"] == tier) & (TIERS["scale"] == scale)][
                "gmac_per_frame_1080p"
            ].iloc[0]
            * (out_hw[0] * out_hw[1])
            / (1920 * 1080)
        )
        lat_rows.append(
            {
                "model": key,
                "scale": scale,
                "output": f"{out_hw[1]}x{out_hw[0]}",
                "t4_mean_ms": t_ms.mean(),
                "t4_p95_ms": np.percentile(t_ms, 95),
                "gmac": gmac,
                **{
                    f"est_ms {k}": M.estimate_ms(gmac, v)
                    for k, v in SUSTAINED_TFLOPS.items()
                },
            }
        )
LAT = pd.DataFrame(lat_rows)
RUN.save_csv(LAT, "latency.csv", index=False)
print(LAT.round(2).to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 3.8))
LAT.pivot_table(index="model", columns="output", values="t4_mean_ms").plot.bar(
    ax=ax, rot=0
)
ax.set_ylabel("ms per frame (T4, PyTorch FP16)")
RUN.show(fig, "latency")

# %% [markdown]
# ## 27. ONNX export
#
# Exports one fused recurrent step of the main 3x student and the integrated-GPU student (explicit state tensors: previous output, hidden state, previous depth), opset 17, with static shapes (a deployment builds one optimised graph per render resolution; re-export with the target size), and compares ONNX Runtime (CPU) with PyTorch FP32. Expected output: export status and maximum difference.

# %%
onnx_rows = []
for key in ("student_s3", "igpu_s3"):
    if key not in MODELS:
        continue
    net = copy.deepcopy(MODELS[key][0]).float().cpu().eval()
    wrap = NM.TSRExport(net)
    h, w = 36, 64
    s = MODELS[key][1]
    args = (
        torch.rand(1, 3, h, w),
        torch.rand(1, 1, h, w),
        torch.zeros(1, 2, h, w),
        torch.zeros(1, 1, h, w),
        torch.zeros(1),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.rand(1, 3, h * s, w * s),
        torch.zeros(1, net.hidden, h, w),
        torch.rand(1, 1, h, w),
    )
    names = [
        "color",
        "depth",
        "mv",
        "reactive",
        "exposure",
        "jitter",
        "flags",
        "prev_out",
        "prev_hidden",
        "prev_depth",
    ]
    path = RUN.dirs["weights"] / f"tsr_{key}_fused.onnx"
    row = {"model": key, "exported": False, "ort_max_abs": None}
    try:  # static shapes: engines build one optimised graph per render resolution anyway
        torch.onnx.export(
            wrap,
            args,
            str(path),
            input_names=names,
            output_names=["out", "hidden", "depth_out"],
            opset_version=17,
            dynamo=False,
        )
        row["exported"] = True
        row["onnx_mb"] = round(path.stat().st_size / 1e6, 3)
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})[0]
        with torch.no_grad():
            ref_out = wrap(*args)[0].numpy()
        row["ort_max_abs"] = float(np.abs(ort_out - ref_out).max())
    except Exception as exc:
        row["error"] = repr(exc)[:300]
    onnx_rows.append(row)
print(pd.DataFrame(onnx_rows).to_string(index=False))
RUN.save_json(onnx_rows, "onnx_export.json")

# %% [markdown]
# ## 28. Summary
#
# Headline results of the main 3x student against the analytic TAA upscaler on the game-like suites, the integrated-GPU tier, the 2x student on native GameIR renders, latency, and stage times against the budget. Expected output: the summary table and a stage-time chart.


# %%
def pick(suite, method, col, scale=S3):
    try:
        return round(float(AGG.loc[(suite, scale, method), col]), 4)
    except KeyError:
        return None


summary = {"run_hours": round(RUN.hours(), 2)}
for suite in ("synthetic-game", "tartanair", "sintel", "vid4", "vimeo1080p"):
    for m in ("analytic_taau", "student_s3", "igpu_s3"):
        summary[f"{suite}/{m}/psnr_y"] = pick(suite, m, "psnr_y")
        summary[f"{suite}/{m}/temporal_err"] = pick(suite, m, "temporal_err")
        summary[f"{suite}/{m}/lpips"] = pick(suite, m, "lpips")
for m in ("analytic_taau", "student_s2"):
    summary[f"gameir-native/{m}/psnr_y"] = pick("gameir-native", m, "psnr_y", S2)
if len(LAT):
    for _, r in LAT.iterrows():
        summary[f"latency_ms/{r['model']}/{r['output']}"] = round(r["t4_mean_ms"], 2)
RUN.save_json(summary, "summary.json")
print(pd.Series(summary).to_string())
st_t = pd.Series(RUN.stage_times) / 60
RUN.save_csv(st_t.rename("minutes").to_frame(), "stage_times.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
st_t.plot.barh(ax=ax)
ax.set_xlabel("minutes")
RUN.show(fig, "stage_times")

# %% [markdown]
# ## 29. Remove training caches
#
# The memory-mapped caches are intermediates; deleting them keeps the saved output small and frees disk for the zip. Expected output: the space freed.

# %%
import shutil

freed = sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file())
shutil.rmtree(CACHE, ignore_errors=True)
print(f"removed caches: {freed / 1e9:.2f} GB")

# %% [markdown]
# ## 30. Package outputs
#
# Zips everything in `/kaggle/working` (weights, ONNX models, plots, metrics, logs, predictions, code and configs) into `outputs.zip`, storing already-compressed files without recompression, after a disk-space check. After **Save Version > Save & Run All**, the zip and all files also appear in the notebook's **Output** tab.

# %%
import shutil
import zipfile

from IPython.display import FileLink
from tqdm.auto import tqdm

WORK = C.WORK
ZIP_PATH = WORK / "outputs.zip"
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".cache", "cache", "tmp", "wandb"}
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
