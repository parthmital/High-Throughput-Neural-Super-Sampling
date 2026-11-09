# %% [markdown]
# # NeuralSS low-latency frame generation (Kaggle GPU T4 x2)
#
# **Goal.** Train and evaluate the NeuralSS frame generator: one network that produces in-between frames at any time (**interpolation**, 0 < tau < 1, so 2x to 6x multi-frame generation) or a future frame (**extrapolation**, tau = 1.5) from two real frames, **works on ordinary video with no side information** and uses engine motion vectors, depth and the latest camera input when they exist, never hallucinates or interpolates UI (HUD, subtitles, menus, pause screens), falls back safely on scene cuts, and has tiers down to integrated GPUs. Generated-frame latency (when a generated image reaches the screen) is reported separately from game input latency (when a real frame reflecting an input reaches the screen).
#
# **Approach.**
#
# 1. **Data.** The leakage-safe manifest from `neural-data-pipeline` (or an inline build). Synthetic layered scenes rendered on the GPU (exact intermediate flows at random in-between times, disocclusions, thin geometry, particles; HUD tiles and subtitles that change between frames, full-screen menus, scene cuts, rapid camera reversals) and real quads of consecutive video frames (Vimeo-90K, REDS, Vimeo-1080p, TartanAir, Sintel) with RAFT teacher flows, including 3x multi-frame samples (frames k and k+3, targets at 1/3 and 2/3).
# 2. **Model.** An IFNet coarse-to-fine intermediate-flow estimator with the design of RIFE 4.26 (Practical-RIFE, MIT), which estimates motion from the two colour frames alone. The `high` tier is RIFE 4.26's network and starts from its pre-trained weights (downloaded from the author's Hugging Face repository and hash-checked); smaller tiers inherit leading channel slices. Engine data enter as extra input channels with zero initial weight (linear-motion priors from the motion vector of the newest frame, the camera-only flow of the latest input for late-latched extrapolation, depth, availability flags), so a fresh model outputs RIFE's blend mixed with I1 at weight sigmoid(-4), about 0.018, and fine-tuning learns how much to trust each signal. A half-resolution head adds a residual and the UI mask: output = UI-mask * newest real frame + (1 - UI-mask) * (blend of the two warped frames + residual). Tiers: `igpu`, `low`, `mid`, `high` (= `teacher`, which distils into the others). Each tier estimates flow at no more than a set number of image lines (576 for `igpu` and `low`, 1152 for `mid` and `high`), halving taller frames, the per-resolution scale choice RIFE recommends; earlier runs (2 to 3 November 2026) halved `igpu` and `low` always, so they trained with flow at 128 pixels and did no better than frame averaging on the Middlebury interpolation set; this rule replaced that fixed scale (saved Middlebury results are in the README).
# 3. **Training.** DDP on both GPUs, FP16, EMA, wall-clock schedule; fine-tuning of the RIFE-initialised `high` tier first, then the smaller tiers with distillation from it. Loss: Charbonnier + census + LPIPS + flow supervision + UI-mask cross-entropy + distillation. Engine inputs are randomly withheld, on synthetic and on real video, so one model serves engine integrations and plain video.
# 4. **Evaluation.** Every suite with engine data (**engine**) and without (**plain**: ordinary video). PSNR, SSIM, LPIPS, interpolation error, quality in occluded regions, UI-region PSNR and UI ghosting rate (with and without the engine's separate UI layer), scene-cut detection, menus, rapid camera reversals (with and without the camera prior), 3x multi-frame generation on real clips, temporal consistency of the displayed sequence (tOF), motion classes; against frame repeat, frame averaging, motion-vector reprojection and **pre-trained RIFE 4.26 unchanged** (the open state of the art this model starts from). Ablations: no UI training, no motion priors, extrapolation-only, no distillation, no weight inheritance. Latency of fused tiers at 1080p and 1440p, an input-to-photon pacing model for native, interpolation, extrapolation and late-latched extrapolation, ONNX export.
#
# **Required Kaggle settings.** Accelerator **GPU T4 x2**; **Internet on** (pip packages, RIFE 4.26 / RAFT / AlexNet weights, Hugging Face data in the fallback). Attach the `neural-data-pipeline` output and the Kaggle datasets it lists. Optional secret `HF_TOKEN`. If Kaggle reports `ERRORED_MOUNTING_DATASET` for one dataset, the session never starts and shows no logs: remove that dataset from the inputs (or re-add it) and save again. Section 5 leaves out sources that are not attached.
#
# **Budget.** The ceiling is Kaggle's 12-hour session limit minus a 72-minute margin (`RUN_BUDGET_H` 10.8). Caches at most 30 min, `high` tier (teacher) 150 min, main student 200 min, parallel `igpu` / `mid` students 120 min, three ablation pairs of 30 min, evaluation in both input modes plus the RIFE reference, multi-frame and native-1080p suites and export about 30 min: about 10.3 hours. Stage budgets shrink automatically when caches take longer, and the measured stage times of each saved run are in its `outputs/logs/notebook.log`. A full run uses about 10.3 hours of Kaggle's weekly GPU quota. `QUICK_RUN = True` gives a 40-minute smoke test.
#
# **Outputs** in `/kaggle/working/`: `code/`, `weights/` (checkpoints, fused models, ONNX), `plots/`, `metrics/`, `logs/`, `predictions/`, `outputs.zip` (training caches are deleted first).

# %% [markdown]
# ## 1. Library files
#
# The first cell creates `/kaggle/working/code`; the shared NeuralSS library and the frame-generation training script are then written into it from the repository's single source.

# %%
import os

os.makedirs("/kaggle/working/code", exist_ok=True)  # %%writefile does not create it

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
# **`train_fg.py`**: DDP training entry point of the frame generator.

# %%
# WRITEFILE train_fg.py

# %% [markdown]
# ## 2. Setup and dependencies
#
# Installs `lpips`, `flip-evaluator` and `onnxruntime` when missing (optional; absence is logged), imports the library and creates run folders and the logger. Expected output: install status.

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

RUN = C.Run("nss_fg")
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
from IPython.display import display

import nss_cache as K
import nss_data as D
import nss_eval as E
import nss_metrics as M
import nss_models as NM
import nss_pipeline as PL
import nss_synth as S
import nss_train as TR

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

# %% [markdown]
# ## 3. Hardware check
#
# Run-time detection of GPUs, CPU, RAM and disk; two GPUs are required for DDP. The library supports PyTorch 2.0 (the Kaggle image of the data-pipeline run) and later. Expected output: two Tesla T4 rows.

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
# * `patch` 256: training crops (RIFE's own crop size; its coarsest level works at 1/16 of the crop); textures are 512x512 so synthetic cameras can move up to about 35 px per frame.
# * `p_extrap` share of extrapolation samples (the low-latency mode); `p_tau` share of synthetic interpolation samples at a random time in [0.15, 0.85] and `p_mfg` share of real interpolation samples taken as 3x multi-frame generation (both for 2x to 6x generation); `p_ui`, `p_menu`, `p_cut`, `p_flip` control UI overlays, menus, scene cuts and rapid camera reversals in synthetic batches.
# * `p_drop_mv` / `p_drop_mv_real`: shares of synthetic and real samples whose motion vectors are withheld, so the model learns to estimate motion itself on video (the run of 24 October 2026 lost 5.8 to 7.8 dB without motion priors because it never had to).
# * Learning rates are fine-tuning rates: 1e-4 for the RIFE-initialised `high` tier, 4e-4 for the inherited smaller tiers.
# * Loss weights: census 0.5, LPIPS 0.1, flow 0.01 (per pixel of flow error; RAFT labels on video at a quarter of that), UI 0.2, distillation 0.5.
# * `minutes`: wall-clock budgets per stage, sized so the run fills `RUN_BUDGET_H` and scaled down to the remaining run budget when caches take longer.
# * `cache.build_min`: the total decoding budget of all training caches.

# %%
QUICK_RUN = False
RUN_BUDGET_H = 10.8  # Kaggle's 12-hour session limit minus a 72-minute margin
CFG = {
    "seed": 21,
    "patch": 256,
    "still_size": 512,
    "student_tier": "low",
    "small_tier": "igpu",
    "mid_tier": "mid",
    "cache": {
        "still_train": 600 if QUICK_RUN else 6000,
        "still_val": 64 if QUICK_RUN else 256,
        "quad_train": 400 if QUICK_RUN else 6000,
        "quad_val": 64 if QUICK_RUN else 256,
        "build_min": 6 if QUICK_RUN else 30,
    },
    "p_real": 0.4,
    "p_extrap": 0.4,
    "p_tau": 0.5,
    "p_mfg": 0.25,
    "p_drop_mv": 0.3,
    "p_drop_mv_real": 0.5,
    "p_ui": 0.5,
    "p_menu": 0.03,
    "p_cut": 0.03,
    "p_flip": 0.2,
    "lr": 4e-4,
    "lr_teacher": 1e-4,
    "min_lr": 1e-6,
    "warmup_frac": 0.03,
    "weight_decay": 1e-4,
    "ema_decay": 0.999,
    "grad_clip": 1.0,
    "w_census": 0.5,
    "w_lpips": 0.1,
    "w_flow": 0.01,
    "w_ui": 0.2,
    "w_distill": 0.5,
    "val_bs": 16,
    "val_batches": 2 if QUICK_RUN else 4,
    "val_interval_min": 8,
    "minutes": {
        "teacher": 4 if QUICK_RUN else 150,
        "student": 4 if QUICK_RUN else 200,
        "pair": 3 if QUICK_RUN else 120,
        "ablation": 3 if QUICK_RUN else 30,
    },
    "eval": {
        "synthetic_per_suite": 8 if QUICK_RUN else 48,
        "real_clips": 3 if QUICK_RUN else 12,
        "native_clips": 2 if QUICK_RUN else 6,
        "eval_reserve_h": 1.0,
    },
    "pipeline": {
        "list_cap_files": 3000 if QUICK_RUN else 40000,
        "seed": 7,
        "workers": os.cpu_count(),
        "hf_caps": {
            "game_stills": (120, 30) if QUICK_RUN else (1200, 250),
            "tartanair": 30 if QUICK_RUN else 120,
            "tartanair_seq": 24,
            "gameir": (0, 0),
            "vimeo1080p": (6, 3) if QUICK_RUN else (160, 40),
        },
        "calibration_pairs": 1500,
        "val_fraction": 0.05,
    },
}
ABLATIONS = {
    "reference": {},
    "no_ui_training": {"ui_training": False, "w_ui": 0.0},
    "no_motion_prior": {"ablate_prior": True},
    "extrap_only": {"mode": "extrap"},
    "no_distillation": {"teacher_ckpt": None},
    "no_inheritance": {"rife_weights": None},
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
# Uses the attached `neural-data-pipeline` output, or builds a smaller manifest inline with the same code (adds about 30 to 60 minutes). Then checks that every source in the manifest is readable and leaves out sources whose Kaggle dataset is not attached (or failed to mount), so a run can go ahead with a subset of the datasets. Expected output: the data root, any sources left out, and item counts per split.

# %%
with RUN.stage("manifest"):
    DATA_ROOT = D.locate_data_root(C.INPUT_ROOT)
    if DATA_ROOT is None:
        log("no attached pipeline output: building the manifest inline (smaller caps)")
        DATA_ROOT = Path("/tmp/nss_data")
        PL.build_manifest(DATA_ROOT, C.INPUT_ROOT, CFG["pipeline"], DEVICES, log=log)
    MANIFEST = D.load_manifest(DATA_ROOT / "nss_manifest.parquet")
    REACHABLE = D.reachable_sources(MANIFEST, DATA_ROOT, C.INPUT_ROOT)
    RUN.save_json(REACHABLE, "reachable_sources.json")
    missing = sorted(s for s, ok in REACHABLE.items() if not ok)
    if missing:
        log(f"sources left out (dataset not attached or not mounted): {missing}")
        MANIFEST = MANIFEST[~MANIFEST["source"].isin(missing)]
log(f"data root: {DATA_ROOT}")
print(pd.crosstab(MANIFEST["source"], MANIFEST["split"]).to_string())

# %% [markdown]
# ## 6. Item selection
#
# Textures for synthetic scenes from stills and high-resolution sequences; real quads (four consecutive frames) from every video and rendered-sequence source. GameIR is not used for frame generation because its frames are ten rendered frames apart. Expected output: item counts.

# %%
TEX_SOURCES = ["div2k", "flickr2k", "game_stills", "vimeo1080p", "reds"]
QUAD_SOURCES = ["vimeo_septuplet", "reds", "vimeo1080p", "tartanair", "sintel"]
P = CFG["patch"]
ITEMS = {}
for split in ("train", "val"):
    ITEMS[f"tex_{split}"] = K.items_for(MANIFEST, split, TEX_SOURCES)
    ITEMS[f"quad_{split}"] = K.items_for(
        MANIFEST, split, QUAD_SOURCES, kinds=["seq"], min_frames=4
    )
print(pd.Series({k: len(v) for k, v in ITEMS.items()}).to_string())
assert ITEMS[
    "tex_train"
], "No texture sources found: attach the pipeline output or the Kaggle datasets."

# %% [markdown]
# ## 7. Training caches
#
# Memory-mapped 512x512 still crops (synthetic textures) and quads of four 256x256 frames in `/tmp/nss_cache` (local disk outside `/kaggle/working`, so about 12 GB of caches never count toward the 20 GB saved output), decoded with all CPU cores; RAFT teacher flows for each quad on both GPUs: 2 -> 0 (the engine-style motion vector of the newest real frame), 1 -> 0 and 1 -> 2 (interpolation targets) and 3 -> 2 (extrapolation target). Decoding shares one time budget, `build_min`: a quarter for stills, three quarters for quads; the small validation caches are built first and the training caches get the rest of each share, so a slow dataset mount shrinks the caches instead of the training time. Expected output: progress bars, counts per source, cache size.

# %%
CACHE = Path("/tmp/nss_cache")
CACHE.mkdir(parents=True, exist_ok=True)
CACHES = {}
with RUN.stage("caches"):
    budget_s = CFG["cache"]["build_min"] * 60
    t0 = time.time()
    for split in ("val", "train"):
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
            max(60, 0.25 * budget_s - (time.time() - t0)),
            log=log,
        )
        CACHES[f"still_{split}"] = {
            "arrays": {"rgb": str(CACHE / f"still_{split}.npy")},
            "count": cnt,
        }
    RAFTS = [M.load_raft(d, log=log) for d in DEVICES]
    RAFTS = RAFTS if all(r is not None for r in RAFTS) else []
    QUAD_COUNTS = {}
    t0 = time.time()
    for split in ("val", "train"):
        prefix = CACHE / f"quad_{split}"
        cnt, counts, need = K.build_window_cache(
            ITEMS[f"quad_{split}"],
            DATA_ROOT,
            prefix,
            CFG["cache"][f"quad_{split}"],
            4,
            P,
            os.cpu_count(),
            max(60, 0.75 * budget_s - (time.time() - t0)),
            CFG["seed"] + (split == "val"),
            flows=True,
            log=log,
        )
        K.fill_estimated_motion(
            prefix, need, RAFTS, DEVICES[: len(RAFTS)], quads=True, log=log
        )
        CACHES[f"quad_{split}"] = K.cache_spec(prefix, cnt, ["rgb", "flows", "meta"])
        QUAD_COUNTS[split] = counts
cache_gb = sum(p.stat().st_size for p in CACHE.glob("*.npy")) / 1e9
RUN.save_json(
    {"caches": CACHES, "quad_sources": QUAD_COUNTS, "cache_gb": round(cache_gb, 2)},
    "caches.json",
)
print(
    pd.DataFrame(QUAD_COUNTS).fillna(0).astype(int).to_string(),
    f"\ncache size {cache_gb:.2f} GB",
)

# %% [markdown]
# ## 8. Synthetic frame-generation preview
#
# Synthetic samples for every special case: interpolation, extrapolation with a camera reversal, HUD with a subtitle change, a menu overlay and a scene cut. Each row shows I0, I1, the target, the UI mask and the exact target-to-I1 flow. Expected output: a 5 x 5 grid.


# %%
def flow_to_rgb(f):
    import matplotlib.colors as mcolors

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
    return mcolors.hsv_to_rgb(hsv)


st = np.load(CACHES["still_train"]["arrays"]["rgb"], mmap_mode="r")
TILES = (
    torch.from_numpy(S.make_text_bank(256, CFG["seed"], height=28, width=160))
    .to(DEVICES[0])
    .float()
    / 255.0
)
stills_t = torch.from_numpy(np.array(st[:8])).to(DEVICES[0])
cases = [
    ("interpolation", dict()),
    ("extrapolation + reversal", dict(extrap=True, p_flip=1.0)),
    ("HUD + subtitle change", dict(p_ui=1.0)),
    ("menu", dict(p_ui=1.0, p_menu=1.0)),
    ("scene cut", dict(p_cut=1.0)),
]
fig, axes = plt.subplots(len(cases), 5, figsize=(16, 3.3 * len(cases)))
for r, (name, kw) in enumerate(cases):
    b = E.fg_suite(stills_t[:2], P, 5 + r, DEVICES[0], tiles=TILES, **kw)
    for c, (img, title) in enumerate(
        [
            (b["i0"][0], "I0"),
            (b["i1"][0], "I1"),
            (b["target"][0], f"target tau={b['tau'][0]:.1f}"),
            (b["ui_mask"][0].expand(3, -1, -1), "UI mask"),
            (
                torch.from_numpy(
                    flow_to_rgb(b["flow_t1"][0].permute(1, 2, 0).cpu().numpy())
                ).permute(2, 0, 1),
                "flow target->I1",
            ),
        ]
    ):
        axes[r, c].imshow(img.permute(1, 2, 0).float().cpu().clamp(0, 1).numpy())
        axes[r, c].set_title(f"{name}: {title}", fontsize=8)
        axes[r, c].axis("off")
RUN.show(fig, "synthetic_preview")

# %% [markdown]
# ## 9. Model tiers, cost and fusion parity
#
# Downloads the pre-trained RIFE 4.26 weights (MIT) from the author's Hugging Face repository, checks both SHA-256 hashes and loads the checkpoint with `weights_only=True`. `RIFE_REF` is a `high`-tier generator holding only these weights: with no engine data its blend is RIFE 4.26's own output (checked to 3e-6 against RIFE's code on CPU before this run), the unchanged reference for every test suite. If the download fails, every tier starts from random weights and the reference is left out (logged). Then, per tier: parameters, the flow scale at 1080p (each tier estimates flow at no more than its `flow_lines` image lines, halving the frame until it fits), multiply-accumulates per generated 1080p frame (every convolution, including the per-frame feature encoder, which a running pipeline computes once per real frame and reuses; warps and blends are not counted) and fused-versus-unfused parity (the frame generator has no RepConv blocks, so fusion is a copy). Expected output: the inheritance counts and a tier table.


# %%
RIFE_STATE = NM.fetch_rife_weights("/tmp/nss_rife", log=log)
RIFE_PT, RIFE_REF = None, None
if RIFE_STATE is not None:
    RIFE_PT = "/tmp/nss_rife/rife426.pt"  # plain state dict for the training processes
    torch.save(RIFE_STATE, RIFE_PT)
    RIFE_REF = NM.build_fg("high")
    log(
        f"RIFE 4.26 into the high tier: {NM.load_ifnet_weights(RIFE_REF.ifnet, RIFE_STATE)}"
    )
    RIFE_REF = RIFE_REF.to(DEVICES[0]).eval()
tier_rows = []
for tier in NM.FG_TIERS:
    net = NM.build_fg(tier).eval()
    with torch.no_grad():
        net.refine.up.weight.normal_(0, 0.01)  # non-trivial output for the parity check
        fused = NM.fuse_model(net)
        f = NM.fg_inputs(1, 128, 128)
        parity = float((net(f)[0] - fused(f)[0]).abs().max())
    # counted at 272x480 and scaled: convolution cost is linear in pixel count; flow_lines is scaled by the
    # same factor so the flow runs at the scale it uses for a 1080p frame
    s1080 = fused.flow_scale(1080)
    fused.flow_lines = fused.flow_lines * 272 / 1080
    f = NM.fg_inputs(1, 272, 480)
    px = (1080 * 1920) / (272 * 480)
    macs = NM.count_macs(fused, lambda m: m(f))
    # the encoder runs on frames scaled by the flow scale (convolution cost scales with the pixel count)
    enc = 2 * NM.count_macs(fused.ifnet.encode, lambda m: m(f["i0"])) * s1080**2
    tier_rows.append(
        {
            "tier": tier,
            "params": NM.count_params(net),
            "flow_scale_1080p": s1080,
            "gmac_per_1080p_frame": round(macs * px / 1e9, 2),
            "gmac_per_frame_encoder": round(enc * px / 1e9, 2),
            "parity_max_abs": parity,
        }
    )
TIERS = pd.DataFrame(tier_rows)
RUN.save_csv(TIERS, "model_tiers.csv", index=False)
print(TIERS.to_string(index=False))
assert TIERS["parity_max_abs"].max() < 1e-3, "fusion parity failed"

# %% [markdown]
# ## 10. Batch-size probe
#
# Full training steps (synthetic batch with UI at 256x256, forward, pixel and census losses, backward) on GPU 0 for the `high` tier (teacher, RIFE 4.26's width) and for a student with the teacher alongside; keeps the largest batch below 75 percent of VRAM. The margin covers what the probe leaves out: the LPIPS, flow, UI and distillation losses, real-quad batches and the notebook's own CUDA context. RAFT's cached GPU memory is released first, so both GPUs start equal. Expected output: probe tables and chosen batch sizes.


# %%
def probe(tier, with_teacher, sizes):
    rows, dev = [], DEVICES[0]
    total = torch.cuda.get_device_properties(dev).total_memory
    stills = torch.from_numpy(np.array(st[: max(sizes)])).to(dev)
    for bs in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            net = NM.build_fg(tier).to(dev)
            teacher = (
                NM.build_fg("teacher").to(dev).eval().requires_grad_(False)
                if with_teacher
                else None
            )
            opt = torch.optim.AdamW(net.parameters(), 1e-4)
            scaler = TR.grad_scaler()
            times = []
            for _ in range(3):
                torch.cuda.synchronize(dev)
                t0 = time.time()
                b = E.fg_suite(stills[:bs], P, 1, dev, tiles=TILES, p_ui=0.5)
                with torch.autocast("cuda", dtype=torch.float16):
                    out, aux = net(b)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher(b)
                    loss = M.charbonnier(
                        out.float(), b["target"]
                    ) + 0.5 * M.census_loss(out.float(), b["target"])
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
                    "samples_per_s": round(bs / np.mean(times[1:]), 1),
                }
            )
            del net, teacher, opt, out, aux, loss, b
        except RuntimeError as exc:  # CUDA OOM, or cuDNN / cuBLAS failing to allocate
            msg = str(exc)
            if "out of memory" not in msg.lower() and "ALLOC_FAILED" not in msg:
                raise
            rows.append({"batch": bs, "fits": False})
            break
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    ok = df[df["fits"] & (df["peak_vram_pct"] <= 75)]
    return df, int(ok["batch"].max()) if len(ok) else sizes[0]


with RUN.stage("batch_probe"):
    TR.free_gpu_memory()
    sizes = [4, 8] if QUICK_RUN else [4, 8, 12, 16, 24, 32, 48, 64]
    PROBE_T, BS_TEACHER = probe("teacher", False, sizes)
    PROBE_S, BS_STUDENT = probe(CFG["student_tier"], True, sizes)
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
# ## 11. Training launcher and budget
#
# One JSON config per run; `code/train_fg.py` runs with DDP on both GPUs for main stages, or as two single-GPU processes in parallel for paired stages and ablations. Progress bars follow `PROGRESS` lines; validation rounds (PSNR overall, interpolation, extrapolation and UI regions) update a live chart.
#
# A failed run never stops the notebook. A run that runs out of GPU memory is retried once with half the batch for the rest of its stage budget (the failed attempt's log is kept as `logs/<run>.oom.log`). A run still alive 25 percent plus 15 minutes past its budget (a hung data-loader worker or collective) is killed with all its processes by a watchdog. Any other failure is logged and recorded in `FAILED`, and later stages use whatever checkpoints exist: students train without distillation if the teacher failed, and evaluation skips missing models. Expected output: planned minutes per stage.

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
print(pd.Series(MINUTES).round(1).to_string(), f"\n(scale {SCALE_T:.2f})")
HISTORY, FAILED, PORT = {}, {}, [29600]


def run_cfg(name, tier, minutes, batch, teacher_ckpt=None, **over):
    cfg = {
        "name": name,
        "tier": tier,
        "patch": P,
        "seed": CFG["seed"],
        "batch_per_gpu": batch,
        "p_real": CFG["p_real"],
        "p_extrap": CFG["p_extrap"],
        "p_tau": CFG["p_tau"],
        "p_mfg": CFG["p_mfg"],
        "p_drop_mv": CFG["p_drop_mv"],
        "p_drop_mv_real": CFG["p_drop_mv_real"],
        "rife_weights": RIFE_PT,
        "p_ui": CFG["p_ui"],
        "p_menu": CFG["p_menu"],
        "p_cut": CFG["p_cut"],
        "p_flip": CFG["p_flip"],
        "lr": CFG["lr_teacher"] if tier == "teacher" else CFG["lr"],
        "min_lr": CFG["min_lr"],
        "warmup_frac": CFG["warmup_frac"],
        "weight_decay": CFG["weight_decay"],
        "ema_decay": CFG["ema_decay"],
        "grad_clip": CFG["grad_clip"],
        "w_census": CFG["w_census"],
        "w_lpips": CFG["w_lpips"] if PKGS.get("lpips") else 0.0,
        "w_flow": CFG["w_flow"],
        "w_ui": CFG["w_ui"],
        "w_distill": CFG["w_distill"],
        "teacher_ckpt": teacher_ckpt,
        "ui_training": True,
        "train_seconds": minutes * 60,
        "val_interval_s": min(CFG["val_interval_min"] * 60, max(minutes * 60 / 6, 60)),
        "progress_interval_s": 15,
        "sync_every": 10,
        "select_metric": "psnr",
        "caches": {"still": CACHES["still_train"], "quad": CACHES["quad_train"]},
        "val_caches": {"still": CACHES["still_val"], "quad": CACHES["quad_val"]},
        "val_bs": CFG["val_bs"],
        "val_batches": CFG["val_batches"],
        "workers_per_rank": max(1, os.cpu_count() // 2),
        "weights_dir": str(RUN.dirs["weights"]),
        "metrics_dir": str(RUN.dirs["metrics"]),
    }
    cfg.update(over)
    return cfg


def live_plot(title):
    placeholder = plt.figure()
    handle = display(placeholder, display_id=True)
    # otherwise the inline backend shows it again, empty, when the cell ends
    plt.close(placeholder)

    def update(rounds):
        h = pd.DataFrame(rounds)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
        axes[0].plot(h["elapsed_min"], h["train_loss"])
        axes[0].set_title("train loss")
        for col in (
            "val_psnr_engine",
            "val_psnr_plain",
            "val_psnr_interp",
            "val_psnr_extrap",
            "val_psnr_ui",
        ):
            if col in h:
                axes[1].plot(h["elapsed_min"], h[col], label=col[4:])
        axes[1].legend(fontsize=7)
        axes[1].set_title("validation PSNR (dB)")
        fig.suptitle(title)
        handle.update(fig)
        plt.close(fig)

    return update


def launch(cfg, gpus, live=True):
    """Run one training process; returns its DONE line, or None if it failed (recorded in FAILED)."""
    PORT[0] += 2
    rounds, done, error = TR.launch_with_retry(
        RUN.work / "code" / "train_fg.py",
        cfg,
        RUN.dirs["code"],
        RUN.dirs["logs"],
        gpus,
        PORT[0],
        on_round=live_plot(cfg["name"]) if live else None,
        log=log,
    )
    HISTORY[cfg["name"]] = rounds
    if error:
        FAILED[cfg["name"]] = error
        return None
    return done


def checkpoint(name):
    """Path of a run's best checkpoint, or None if the run produced none."""
    path = RUN.dirs["weights"] / f"{name}_best.pt"
    return str(path) if path.exists() else None


def launch_pair(cfg_a, cfg_b):
    with ThreadPoolExecutor(2) as pool:
        fa, fb = pool.submit(launch, cfg_a, [0], False), pool.submit(
            launch, cfg_b, [1], False
        )
        return fa.result(), fb.result()


# %% [markdown]
# ## 12. Stage A: high tier fine-tuned from RIFE 4.26 (both GPUs)
#
# The `high` tier (`teacher`), RIFE 4.26's network with the engine-hint channels and the residual / UI head, starts from the pre-trained RIFE weights and is fine-tuned on game-like scenes, UI, cuts, extrapolation and engine data. It is deployable on high-end GPUs (about 77 GMAC per generated 1080p frame) and teaches the smaller tiers. Unlike the privileged teacher of earlier runs it does not see the target frame, so its scores are achievable at run time. Expected output: live chart and final scores (with engine data, without, and their mean).

# %%
with RUN.stage("train_teacher"):
    print(
        launch(run_cfg("fg_teacher", "teacher", MINUTES["teacher"], BS_TEACHER), [0, 1])
    )
TEACHER_CKPT = checkpoint("fg_teacher")
if TEACHER_CKPT is None:
    log("no teacher checkpoint: students train without distillation")

# %% [markdown]
# ## 13. Stage B: main student with distillation (both GPUs)
#
# The `low` tier (inherited RIFE slices; flow at full resolution on the 256-pixel training crops, at half resolution on 1080p frames) with the full loss and distillation of flows and output toward the fine-tuned `high` tier. Expected output: live chart and final scores.

# %%
with RUN.stage("train_student"):
    print(
        launch(
            run_cfg(
                f"fg_{CFG['student_tier']}",
                CFG["student_tier"],
                MINUTES["student"],
                BS_STUDENT,
                teacher_ckpt=TEACHER_CKPT,
            ),
            [0, 1],
        )
    )

# %% [markdown]
# ## 14. Stage C: integrated-GPU and mid tiers in parallel
#
# GPU 0 trains the `igpu` tier, GPU 1 the `mid` tier (flow at full resolution up to 1152 lines), both distilled. Expected output: both runs' final scores.

# %%
with RUN.stage("train_pair"):
    print(
        launch_pair(
            run_cfg(
                f"fg_{CFG['small_tier']}",
                CFG["small_tier"],
                MINUTES["pair"],
                BS_STUDENT,
                teacher_ckpt=TEACHER_CKPT,
            ),
            run_cfg(
                f"fg_{CFG['mid_tier']}",
                CFG["mid_tier"],
                MINUTES["pair"],
                max(8, BS_STUDENT // 2),
                teacher_ckpt=TEACHER_CKPT,
            ),
        )
    )

# %% [markdown]
# ## 15. Ablations (three parallel pairs)
#
# Short `low`-tier runs against a `reference` with the same budget: no UI training (no HUD / subtitle / menu overlays, no UI loss), no motion priors (never any engine motion vectors or camera prior), extrapolation only, no distillation, and no weight inheritance (random initialisation instead of RIFE slices). The budgets are equal in wall-clock time, so the `no_distillation` run, which skips the teacher's forward pass, takes more steps. Without a teacher checkpoint the `no_distillation` run is the same recipe as `reference` (logged). Each run validates on batches built with its own recipe (for example no UI for `no_ui_training`), so validation scores are not comparable across ablations; section 25 compares them on the test suites. Expected output: final scores of the six runs.

# %%
with RUN.stage("ablations"):
    if TEACHER_CKPT is None:
        log("no teacher checkpoint: abl_no_distillation repeats abl_reference")
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
                    MINUTES["ablation"],
                    BS_STUDENT,
                    teacher_ckpt=ckpt,
                    **over,
                )
            )
        print(launch_pair(*cfgs))

# %% [markdown]
# ## 16. Training curves
#
# Validation PSNR over time for every run (the mean of the same validation samples with their engine data and without any, the checkpoint selection metric) and train loss for the main stages; table of best rounds. Expected output: two charts, a table and the list of failed runs (`metrics/failed_runs.json`, normally empty).

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
best_rows = []
for name, rounds in HISTORY.items():
    if not rounds:
        continue
    h = pd.DataFrame(rounds)
    RUN.save_csv(h, f"history_{name}.csv", index=False)
    if "val_psnr" not in h or h["val_psnr"].isna().all():
        continue
    axes[0].plot(
        h["elapsed_min"],
        h["val_psnr"],
        "--" if name.startswith("abl_") else "-",
        label=name,
    )
    if not name.startswith("abl_"):
        axes[1].plot(h["elapsed_min"], h["train_loss"], label=name)
    b = h.loc[h["val_psnr"].idxmax()]
    best_rows.append(
        {
            "run": name,
            **{
                k: b.get(k)
                for k in (
                    "round",
                    "val_psnr",
                    "val_psnr_engine",
                    "val_psnr_plain",
                    "val_psnr_interp",
                    "val_psnr_extrap",
                    "val_psnr_ui",
                )
            },
            "samples_per_s": h["samples_per_s"].mean(),
            "peak_mem_gb": max(max(m) for m in h["mem_gb"]),
        }
    )
for ax, t in zip(axes, ["validation PSNR (dB)", "train loss"]):
    ax.set_title(t)
    ax.set_xlabel("minutes")
    ax.legend(fontsize=7)
RUN.show(fig, "training_curves")
BEST = pd.DataFrame(best_rows)
RUN.save_csv(BEST, "best_validation.csv", index=False)
print(BEST.round(3).to_string(index=False))
RUN.save_json(FAILED, "failed_runs.json")
print("failed runs:", sorted(FAILED) or "none")

# %% [markdown]
# ## 17. Load and fuse models
#
# Best EMA weights of each run, kept as FP16 replicas on GPU 0 (`high` is the fine-tuned teacher run, a deployable tier). Expected output: a table of models with parameter counts.

# %%
RUNS = {
    "high": ("fg_teacher", "teacher"),
    "student": (f"fg_{CFG['student_tier']}", CFG["student_tier"]),
    "igpu": (f"fg_{CFG['small_tier']}", CFG["small_tier"]),
    "mid": (f"fg_{CFG['mid_tier']}", CFG["mid_tier"]),
}
RUNS.update({f"abl_{a}": (f"abl_{a}", CFG["student_tier"]) for a in ABLATIONS})
MODELS, rows = {}, []
for key, (name, tier) in RUNS.items():
    path = RUN.dirs["weights"] / f"{name}_best.pt"
    if not path.exists():
        continue
    net = NM.build_fg(tier)
    net.load_state_dict(torch.load(path, map_location="cpu")["ema"])
    fused = NM.fuse_model(net).to(DEVICES[0]).half().eval()
    torch.save(
        {"state_dict": fused.state_dict(), "tier": tier},
        RUN.dirs["weights"] / f"{name}_fused.pt",
    )
    MODELS[key] = fused
    rows.append({"model": key, "tier": tier, "params": NM.count_params(fused)})
print(pd.DataFrame(rows).to_string(index=False))
LPIPS = M.load_lpips(DEVICES[0], log=log)
RAFT = RAFTS[0] if RAFTS else M.load_raft(DEVICES[0], log=log)

# %% [markdown]
# ## 18. Scene-cut detector calibration
#
# A cheap analytic detector decides when not to generate (output the newest real frame instead). With engine motion vectors its score is the luma-histogram distance plus the motion-compensated difference between the two real frames; without them (ordinary video) it is the histogram distance alone, because a raw frame difference reads ordinary fast motion as a cut (in the run of 2 to 3 November 2026 that one shared threshold gated 35 to 100 percent of the real test samples without engine data, so the plain-video scores there mostly measured frame repeat). Each score gets its own threshold on synthetic validation pairs (cut versus continuous motion, including fast motion) for at most 1 percent false alarms. Expected output: both thresholds, detection rates and the score distributions.

# %%
sv = torch.from_numpy(
    np.load(CACHES["still_val"]["arrays"]["rgb"], mmap_mode="r")[
        : min(64, CACHES["still_val"]["count"])
    ].copy()
).to(DEVICES[0])
cal = [E.fg_suite(sv, P, 900 + k, DEVICES[0], p_cut=0.5) for k in range(4)]
labels = torch.cat([b["cut"] for b in cal]).cpu().numpy()


def calibrate(scores):
    """Threshold for at most 1 percent false alarms on continuous pairs, and the detection rate of cuts."""
    neg = np.sort(scores[labels < 0.5])
    thr = float(neg[int(0.99 * (len(neg) - 1))]) if len(neg) else 1.0
    rate = float((scores[labels > 0.5] > thr).mean()) if (labels > 0.5).any() else None
    return thr, rate


scores = torch.cat([E.cut_score(b["i0"], b["i1"], b["mv1"]) for b in cal]).cpu().numpy()
scores_plain = torch.cat([E.cut_score(b["i0"], b["i1"]) for b in cal]).cpu().numpy()
CUT_THR, cut_rate = calibrate(scores)
CUT_THR_PLAIN, cut_rate_plain = calibrate(scores_plain)
RUN.save_json(
    {
        "threshold": CUT_THR,
        "detection_rate": cut_rate,
        "threshold_plain": CUT_THR_PLAIN,
        "detection_rate_plain": cut_rate_plain,
        "false_alarm_target": 0.01,
    },
    "cut_detector.json",
)
print(f"with motion vectors: threshold {CUT_THR:.4f}, detection rate {cut_rate}")
print(
    f"without (histogram only): threshold {CUT_THR_PLAIN:.4f}, detection rate {cut_rate_plain}"
)
fig, axes = plt.subplots(1, 2, figsize=(14, 3.2))
for ax, s, thr, name in (
    (axes[0], scores, CUT_THR, "with motion vectors"),
    (axes[1], scores_plain, CUT_THR_PLAIN, "without (histogram distance)"),
):
    ax.hist([s[labels < 0.5], s[labels > 0.5]], bins=40, label=["continuous", "cut"])
    ax.axvline(thr, color="k", ls="--")
    ax.legend()
    ax.set_title(f"scene-cut score, {name}")
RUN.show(fig, "cut_detector")

# %% [markdown]
# ## 19. Test suites
#
# Test split only. Synthetic suites from held-out game frames and DIV2K validation images (256x256 crops of 512-px textures): interpolation and extrapolation per motion class, HUD / subtitles, menus, scene cuts and rapid camera reversals (extrapolation with and without the camera prior). Real suites at full clip resolution: the Vimeo-90K triplet test set (interpolation, the standard benchmark), held-out TartanAir and Sintel (exact flow available for the motion-vector input), held-out Vimeo-1080p (scaled to 544 lines) and Vid4 (both modes, RAFT motion vectors), and the same Vimeo-1080p videos at full 1080p (`vimeo1080p_native`, 1072x1904 crops), where the `igpu` and `low` tiers estimate flow at half resolution as they would in a 1080p game; every other suite has at most 544 lines, so there all tiers estimate flow at full resolution. RAFT motion of the native clips is estimated at most 1024 px on the longer side, and its flow is scaled back. Clips that cannot be decoded are logged and skipped, and so are real clips that cannot test generation (`E.test_clips`): blank frames, frozen (duplicated) frames, where every method reaches the 100 dB PSNR cap, and scene cuts or flashes between the frames, which the synthetic cut suite covers. In the run of 20 to 21 October 2026, two blank Vimeo-1080p clips and one with a cut decided that suite's ranking. Expected output: suite sizes.

# %%
ev = CFG["eval"]
TEST_ITEMS = D.items_from_frame(MANIFEST[MANIFEST["split"] == "test"])
test_still_items = [i for i in TEST_ITEMS if i["source"] in ("game_stills", "div2k")]
with RUN.stage("test_suites"):
    tasks = K.still_tasks(
        test_still_items, DATA_ROOT, ev["synthetic_per_suite"], CFG["still_size"], 1, 99
    )
    crops = []
    for task in tasks:
        _, c = D.still_crop(task)
        if c is not None:
            crops.append(c[0])
        if len(crops) >= ev["synthetic_per_suite"]:
            break
    TEST_STILLS = torch.from_numpy(np.stack(crops)).to(DEVICES[0]) if crops else None

    def real_quads(source, crop, n, max_side=None):
        return E.test_clips(
            TEST_ITEMS,
            DATA_ROOT,
            source,
            crop,
            n,
            3 if source == "vimeo_triplet" else 4,
            max_side=max_side,
            reject=("blank", "frozen", "cut"),
            log=log,
        )

    REAL = {
        "vimeo_triplet": real_quads("vimeo_triplet", (256, 448), ev["real_clips"] * 4),
        "tartanair": real_quads("tartanair", (480, 640), ev["real_clips"]),
        "sintel": real_quads("sintel", (432, 1024), ev["real_clips"]),
        "vimeo1080p": real_quads(
            "vimeo1080p", (544, 960), ev["real_clips"], max_side=544
        ),
        "vid4": real_quads("vid4", (480, 704), ev["real_clips"]),
        # full 1080p: the igpu and low tiers estimate flow at half resolution here, as deployed
        "vimeo1080p_native": real_quads("vimeo1080p", (1072, 1904), ev["native_clips"]),
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
# For each suite and mode (interpolation, extrapolation, 3x multi-frame generation), with two input modes: **engine** (motion vectors, depth and camera where the suite has them) and **plain** (no engine data at all: ordinary video). Methods: every network, **RIFE 4.26 unchanged** (pre-trained, colour only, the open state of the art this model starts from), the analytic baselines (frame repeat, average and, with engine data, motion-vector reprojection) and, on UI suites, the engine path (generate on HUD-less frames, composite the newest UI layer). The cut detector gates every network: frames it flags are replaced by the newest real frame; the share of gated samples per suite, mode and input mode goes to `cut_gate_rates.csv` (on real suites every gated sample is a false alarm, since the test clips contain no cuts). On real suites the motion-vector input of I1 is the exact flow on TartanAir and Sintel (frame 2 -> 1 composed with frame 1 -> 0, as an engine would provide) and RAFT on the video suites; the target's occlusion mask always comes from RAFT and is used by the metrics only. The 3x multi-frame suite generates frames k+1 and k+2 (tau = 1/3 and 2/3) from real frames k and k+3 of the 4-frame real clips, without engine data. Models whose training failed are left out, and a model that fails on one suite is logged and skipped there. Expected output: number of metric rows.

# %%
ROWS, GATES = [], []


def quad_batch(clip, extrap, dev):
    fr = torch.from_numpy(clip["rgb"]).to(dev).permute(0, 3, 1, 2).float()[None] / 255.0
    # triplets: (I0, middle, I1); quads add the extrapolation target as frame 3
    i0, i1 = fr[:, 0], fr[:, 2]
    target = fr[:, 3] if extrap else fr[:, 1]
    n, _, H, W = i0.shape
    z1 = torch.zeros(n, device=dev)
    zeros2 = torch.zeros(n, 2, H, W, device=dev)
    flow_t1, valid_t1 = zeros2, torch.ones(n, 1, H, W, device=dev)
    mv1 = zeros2
    # engine-style motion I1 -> I0 on exact-flow clips: compose frame 2 -> 1 with frame 1 -> 0
    if clip.get("exact"):
        mv = torch.from_numpy(clip["mv"][1:3]).to(dev).permute(0, 3, 1, 2).float()
        m21, m10 = mv[1:2], mv[0:1]
        mv1 = m21 + NM.backward_warp(m10, m21)
    elif RAFT is not None:
        mv1, _ = M.raft_mv_and_valid(RAFT, i1, i0)
    if RAFT is not None:  # occlusion mask of the target, used by the metrics only
        flow_t1, valid_t1 = M.raft_mv_and_valid(RAFT, target, i1)
    return {
        "i0": i0,
        "i1": i1,
        "target": target,
        "tau": torch.full((n,), 1.5 if extrap else 0.5, device=dev),
        "extrap": torch.full((n,), float(extrap), device=dev),
        "mv1": mv1,
        "mv_avail": torch.ones(n, device=dev)
        * float(bool(clip.get("exact")) or RAFT is not None),
        "depth1": torch.zeros(n, 1, H, W, device=dev),
        "depth_avail": z1,
        "cam_prior": torch.zeros_like(mv1),
        "cam_avail": z1,
        "flow_t1": flow_t1,
        "valid_t1": valid_t1,
        "ui_mask": torch.zeros(n, 1, H, W, device=dev),
        "speed_class": torch.full((n,), -1, device=dev),
        "cut": z1,
        "menu": z1,
        "flip": z1,
    }


def mfg_batch(clip, second, dev):
    """3x multi-frame generation from real frames 0 and 3: target frame 1 (tau 1/3) or 2 (tau 2/3), no
    engine data."""
    fr = torch.from_numpy(clip["rgb"]).to(dev).permute(0, 3, 1, 2).float()[None] / 255.0
    b = quad_batch(clip, False, dev)
    tgt = fr[:, 2] if second else fr[:, 1]
    b.update(
        i1=fr[:, 3],
        target=tgt,
        tau=torch.full_like(b["tau"], 2 / 3 if second else 1 / 3),
    )
    b.update(mv1=torch.zeros_like(b["mv1"]), mv_avail=torch.zeros_like(b["mv_avail"]))
    if RAFT is not None:  # occlusion mask of the target, used by the metrics only
        b["flow_t1"], b["valid_t1"] = M.raft_mv_and_valid(RAFT, tgt, fr[:, 3])
    return b


def evaluate(
    suite, b, extra=None, models=None, ui_engine=False, inputs=("engine", "plain")
):
    for mode in inputs:
        bm = b if mode == "engine" else E.without_engine(b)
        evaluate_inputs(
            suite,
            bm,
            {"inputs": mode, **(extra or {})},
            models,
            ui_engine and mode == "engine",
        )


def training_flags(key):
    """Training options of an ablation model (its ABLATIONS entry); {} for the main runs."""
    return ABLATIONS.get(key[4:], {}) if key.startswith("abl_") else {}


def evaluate_inputs(suite, b, extra, models, ui_engine):
    gate = E.cut_gate(b, CUT_THR, CUT_THR_PLAIN)
    GATES.append(
        {
            "suite": suite,
            "mode": extra["mode"],
            "inputs": extra["inputs"],
            "samples": len(gate),
            "gated": int(gate.sum()),
            "true_cuts": int((b["cut"] > 0.5).sum()),
        }
    )
    gate = gate.view(-1, 1, 1, 1)
    outs = dict(E.fg_baselines(b))
    if b["mv_avail"].max() <= 0:  # reprojection needs motion vectors
        outs.pop("mv_reprojection")
    if RIFE_REF is not None:
        try:
            outs["rife_4.26"] = torch.where(gate, b["i1"], E.run_rife(RIFE_REF, b))
        except Exception as exc:
            log(f"evaluation {suite} / rife_4.26 skipped: {exc}")
            torch.cuda.empty_cache()
    for key in models or MODELS:
        if key not in MODELS:
            continue
        net = MODELS[key]
        f = dict(b)
        if training_flags(key).get(
            "ablate_prior"
        ):  # as in training: no motion vectors or camera prior, depth kept
            f.update(
                mv_avail=torch.zeros_like(b["mv_avail"]),
                cam_avail=torch.zeros_like(b["cam_avail"]),
            )
        try:  # one failing model (for example out of memory on a large clip) must not end the evaluation
            out, _ = E.run_fg(net, f)
            outs[key] = torch.where(gate, b["i1"], out)
            if ui_engine and key in ("student", "igpu"):
                out_e, _ = E.run_fg(net, f, hudless=True)
                outs[f"{key}+ui_layer"] = torch.where(gate, b["i1"], out_e)
        except Exception as exc:
            log(f"evaluation {suite} / {key} skipped: {exc}")
            torch.cuda.empty_cache()
    for mname, out in outs.items():
        for r in E.fg_metrics(out, b, LPIPS):
            ROWS.append({"suite": suite, "method": mname, **extra, **r})
    return outs


with RUN.stage("evaluation"):
    if TEST_STILLS is not None:
        n = len(TEST_STILLS)
        for extrap in (False, True):
            mode = "extrap" if extrap else "interp"
            for cls, cname in enumerate(S.MOTION_PROFILE["names"]):
                evaluate(
                    "synthetic",
                    E.fg_suite(
                        TEST_STILLS,
                        P,
                        2000 + cls,
                        DEVICES[0],
                        extrap=extrap,
                        motion_class=cls,
                    ),
                    {"mode": mode, "motion": cname},
                )
            evaluate(
                "synthetic-ui",
                E.fg_suite(
                    TEST_STILLS,
                    P,
                    3000 + extrap,
                    DEVICES[0],
                    tiles=TILES,
                    extrap=extrap,
                    p_ui=1.0,
                ),
                {"mode": mode, "motion": "mixed"},
                ui_engine=True,
            )
            evaluate(
                "synthetic-menu",
                E.fg_suite(
                    TEST_STILLS,
                    P,
                    3100 + extrap,
                    DEVICES[0],
                    tiles=TILES,
                    extrap=extrap,
                    p_ui=1.0,
                    p_menu=1.0,
                ),
                {"mode": mode, "motion": "mixed"},
                ui_engine=True,
            )
            evaluate(
                "synthetic-cut",
                E.fg_suite(
                    TEST_STILLS, P, 3200 + extrap, DEVICES[0], extrap=extrap, p_cut=1.0
                ),
                {"mode": mode, "motion": "cut"},
            )
        rev = E.fg_suite(TEST_STILLS, P, 3300, DEVICES[0], extrap=True, p_flip=1.0)
        evaluate(
            "synthetic-reversal",
            rev,
            {"mode": "extrap", "motion": "reversal", "camera_prior": True},
            inputs=("engine",),
        )
        rev_nc = dict(rev)
        rev_nc["cam_avail"] = torch.zeros(n, device=DEVICES[0])
        evaluate(
            "synthetic-reversal",
            rev_nc,
            {"mode": "extrap", "motion": "reversal", "camera_prior": False},
            inputs=("engine",),
        )
    for suite, quads in REAL.items():
        for extrap in ((False,) if suite == "vimeo_triplet" else (False, True)):
            for cid, clip in quads:
                if clip["rgb"].shape[0] < (4 if extrap else 3):
                    continue
                evaluate(
                    suite,
                    quad_batch(clip, extrap, DEVICES[0]),
                    {
                        "mode": "extrap" if extrap else "interp",
                        "clip": cid,
                        "motion": "real",
                    },
                )
        if suite != "vimeo_triplet":
            for cid, clip in quads:
                for second in (False, True):
                    evaluate(
                        suite,
                        mfg_batch(clip, second, DEVICES[0]),
                        {"mode": "mfg3x", "clip": cid, "motion": "real"},
                        inputs=("plain",),
                    )
        log(f"{suite} done")
RES = pd.DataFrame(ROWS)
for col in [
    "psnr",
    "ssim",
    "ie",
    "lpips",
    "psnr_ui",
    "ui_ghost_rate",
    "psnr_occluded",
    "mv_px",
]:  # None -> NaN
    RES[col] = pd.to_numeric(RES[col], errors="coerce")
if "camera_prior" in RES:
    RES["camera_prior"] = RES["camera_prior"].fillna("n/a").astype(str)
RUN.save_csv(RES, "test_per_sample.csv", index=False)
print(len(RES), "metric rows")
# real test clips contain no cuts (E.test_clips rejects them), so every gated real sample is a false alarm
GATE = pd.DataFrame(GATES).groupby(["suite", "mode", "inputs"]).sum()
GATE["gated_share"] = GATE["gated"] / GATE["samples"]
RUN.save_csv(GATE.round(4), "cut_gate_rates.csv")
print(GATE.round(3).to_string())

# %% [markdown]
# ## 21. Results by suite, mode and method
#
# Mean metrics (PSNR, SSIM, LPIPS, interpolation error, PSNR in occluded regions) per suite, mode, input mode and method. Expected output: the results table and a chart of PSNR by method on real suites for interpolation, extrapolation and 3x multi-frame generation, with and without engine data.

# %%
AGG = (
    RES.groupby(["suite", "mode", "inputs", "method"])
    .mean(numeric_only=True)
    .drop(columns=["speed_class"], errors="ignore")
)
RUN.save_csv(AGG.round(4), "test_results.csv")
print(AGG[["psnr", "ssim", "lpips", "ie", "psnr_occluded"]].round(3).to_string())
real = RES[RES["suite"].isin(list(REAL)) & ~RES["method"].str.startswith("abl_")]
if len(real):
    panels = [
        (m, i) for m in ("interp", "extrap", "mfg3x") for i in ("plain", "engine")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(24, 8.4))
    for ax, (mode, inp) in zip(axes.T.flat, panels):
        sub = real[(real["mode"] == mode) & (real["inputs"] == inp)]
        if not len(sub):
            # 3x multi-frame generation has no engine-data variant: hide the empty panel
            ax.set_visible(False)
            continue
        sub.groupby(["suite", "method"])["psnr"].mean().unstack().plot.bar(
            ax=ax, rot=20
        )
        ax.legend(fontsize=6)
        ax.set_title(f"real suites, {mode}, {inp}: PSNR (dB)")
    RUN.show(fig, "real_results")

# %% [markdown]
# ## 22. Motion, UI, menus, cuts and input reversals
#
# * PSNR by motion class for both modes (synthetic).
# * UI: PSNR inside UI pixels and the UI ghosting rate (share of UI pixels off by more than 10 percent), for the network alone, with the engine's separate UI layer, and for the baselines.
# * Menus and cuts: PSNR against the correct output (the newest real frame).
# * Rapid camera reversal in extrapolation: with the late-latched camera prior versus without.
#
# Expected output: four charts and the UI table.

# %%
syn = RES[(RES["suite"] == "synthetic") & (RES["inputs"] == "engine")]
if len(syn):
    print(
        "measured motion between real frames per class (px):",
        syn.groupby("motion")["mv_px"].mean().round(2).to_dict(),
    )
fig, axes = plt.subplots(1, 4, figsize=(22, 4.2))
if len(syn):
    for mode, ls in (("interp", "-"), ("extrap", "--")):
        t = (
            syn[syn["mode"] == mode]
            .groupby(["motion", "method"])["psnr"]
            .mean()
            .unstack()
            .reindex(S.MOTION_PROFILE["names"])
        )
        for col in [
            c
            for c in t.columns
            if c in ("student", "high", "rife_4.26", "mv_reprojection", "repeat")
        ]:
            axes[0].plot(t.index, t[col], ls, marker="o", label=f"{col} ({mode})")
    axes[0].set_title("PSNR by motion class")
    axes[0].legend(fontsize=6)
ui = RES[RES["suite"] == "synthetic-ui"]
if len(ui):
    UI = ui.groupby(["mode", "inputs", "method"])[
        ["psnr", "psnr_ui", "ui_ghost_rate"]
    ].mean()
    RUN.save_csv(UI.round(4), "ui_results.csv")
    print(UI.round(3).to_string())
    UI.xs("engine", level="inputs")["ui_ghost_rate"].unstack(0).plot.bar(
        ax=axes[1], rot=45
    )
    axes[1].set_title("UI ghosting rate (lower is better)")
for suite in ("synthetic-menu", "synthetic-cut"):
    d = RES[(RES["suite"] == suite) & (RES["inputs"] == "engine")]
    if len(d):
        d.groupby("method")["psnr"].mean().plot(ax=axes[2], marker="o", label=suite)
axes[2].legend(fontsize=7)
axes[2].set_title("menus and cuts: PSNR vs correct output")
axes[2].tick_params(axis="x", rotation=60)
rv = RES[RES["suite"] == "synthetic-reversal"]
if len(rv):
    rv.groupby(["camera_prior", "method"])["psnr"].mean().unstack(0).plot.bar(
        ax=axes[3], rot=60
    )
    axes[3].set_title("camera reversal (extrapolation)")
RUN.show(fig, "special_cases")

# %% [markdown]
# ## 23. Temporal consistency of the displayed sequence
#
# Generated frames are shown between real ones, so motion must stay smooth across real -> generated -> real. tOF compares RAFT flow along the displayed sequence (I0 -> generated, generated -> I1) with the same flow along the true sequence (I0 -> true middle, true middle -> I1). Interpolation only (extrapolated frames are followed by the next real frame, which the test clips do not always contain). The networks run without engine data (the ordinary-video case); motion-vector reprojection uses the RAFT motion of I1. Expected output: tOF per suite and method; per-clip values go to `temporal_consistency_per_clip.csv`, because a suite mean can be set by a few clips with large motion.

# %%
tof_rows = []
if RAFT is not None:
    for suite in ("vimeo_triplet", "tartanair", "vimeo1080p"):
        for cid, clip in REAL.get(suite, [])[: ev["real_clips"]]:
            b = quad_batch(clip, False, DEVICES[0])
            base = E.fg_baselines(b)
            outs = {
                "mv_reprojection": base["mv_reprojection"],
                "average": base["average"],
            }
            if RIFE_REF is not None:
                outs["rife_4.26"] = E.run_rife(RIFE_REF, b)
            for key in ("student", "igpu", "mid", "high"):
                if key in MODELS:  # without engine data: the ordinary-video case
                    outs[key] = E.run_fg(MODELS[key], E.without_engine(b))[0]
            for mname, out in outs.items():
                t1 = (
                    (
                        M.raft_flow(RAFT, b["i0"], out)
                        - M.raft_flow(RAFT, b["i0"], b["target"])
                    )
                    .norm(dim=1)
                    .mean()
                    .item()
                )
                t2 = (
                    (
                        M.raft_flow(RAFT, out, b["i1"])
                        - M.raft_flow(RAFT, b["target"], b["i1"])
                    )
                    .norm(dim=1)
                    .mean()
                    .item()
                )
                tof_rows.append(
                    {
                        "suite": suite,
                        "clip": cid,
                        "method": mname,
                        "tof": 0.5 * (t1 + t2),
                    }
                )
TOF = pd.DataFrame(tof_rows)
if len(TOF):
    RUN.save_csv(TOF.round(4), "temporal_consistency_per_clip.csv", index=False)
    TOFA = TOF.groupby(["suite", "method"])["tof"].mean().unstack()
    RUN.save_csv(TOFA.round(4), "temporal_consistency.csv")
    print(TOFA.round(3).to_string())

# %% [markdown]
# ## 24. Visual comparisons
#
# For a UI sample and a fast-motion extrapolation sample: target, baselines, RIFE 4.26 unchanged, the student with and without engine data, the predicted UI mask, blend weight and flow. Every image panel is also saved as a full-size PNG in `predictions/`. Expected output: two figures.


# %%
import PIL.Image as PImage


def visual(b, title, fname):
    outs = dict(E.fg_baselines(b))
    if RIFE_REF is not None:
        outs["rife_4.26"] = E.run_rife(RIFE_REF, b)
    out, aux = E.run_fg(MODELS["student"], b)
    outs["student"] = out
    outs["student, no engine data"] = E.run_fg(MODELS["student"], E.without_engine(b))[
        0
    ]
    if "hudless_i0" in b:
        outs["student+ui_layer"] = E.run_fg(MODELS["student"], b, hudless=True)[0]
    panels = [("target", b["target"])] + list(outs.items())
    fig, axes = plt.subplots(1, len(panels) + 3, figsize=(3 * (len(panels) + 3), 3.4))
    for ax, (k, img) in zip(axes, panels):
        rgb = img[0].permute(1, 2, 0).float().cpu().clamp(0, 1).numpy()
        ax.imshow(rgb)
        ax.set_title(k, fontsize=8)
        PImage.fromarray((rgb * 255).round().astype(np.uint8)).save(
            RUN.dirs["predictions"]
            / f"{fname}_{k.replace('+', '_').replace(', ', '_').replace(' ', '_')}.png"
        )
    axes[len(panels)].imshow(
        aux["ui"][0, 0].float().cpu().numpy(), cmap="gray", vmin=0, vmax=1
    )
    axes[len(panels)].set_title("predicted UI mask", fontsize=8)
    axes[len(panels) + 1].imshow(
        aux["blend"][0, 0].float().cpu().numpy(), cmap="coolwarm", vmin=0, vmax=1
    )
    axes[len(panels) + 1].set_title("blend weight (I0)", fontsize=8)
    axes[len(panels) + 2].imshow(
        flow_to_rgb(aux["flow1"][0].permute(1, 2, 0).float().cpu().numpy())
    )
    axes[len(panels) + 2].set_title("flow target->I1", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    RUN.show(fig, fname)


if TEST_STILLS is not None and "student" in MODELS:
    visual(
        E.fg_suite(TEST_STILLS[:2], P, 4000, DEVICES[0], tiles=TILES, p_ui=1.0),
        "HUD and subtitles (interpolation)",
        "visual_ui",
    )
    visual(
        E.fg_suite(TEST_STILLS[:2], P, 4001, DEVICES[0], extrap=True, motion_class=3),
        "fast motion (extrapolation)",
        "visual_extrap",
    )

# %% [markdown]
# ## 25. Ablations
#
# Same short budget, one change each, on synthetic and real suites; deltas to `abl_reference`. Expected output: table and chart.

# %%
abl = RES[RES["method"].str.startswith("abl_")]
if len(abl):
    ABL = abl.groupby(["suite", "mode", "inputs", "method"])[
        ["psnr", "lpips", "psnr_ui", "ui_ghost_rate", "psnr_occluded"]
    ].mean()
    RUN.save_csv(ABL.round(4), "ablations.csv")
    print(ABL.round(3).to_string())
    piv = abl.groupby(["method", "mode", "inputs"])["psnr"].mean().unstack([1, 2])
    if "abl_reference" in piv.index:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        (piv - piv.loc["abl_reference"]).drop(
            index="abl_reference", errors="ignore"
        ).plot.bar(ax=ax, rot=30)
        ax.set_title("PSNR change vs reference (all suites)")
        RUN.show(fig, "ablations")
    else:
        print("abl_reference did not train: deltas cannot be computed")

# %% [markdown]
# ## 26. Latency and the input-to-photon pacing model
#
# Model latency: one fused FP16 generation on a T4 (CUDA events, batch 1, cuDNN autotuning on, every engine input present) at 1920x1080 and 2560x1440, plus the arithmetic estimate for an assumed integrated-GPU throughput.
#
# Pacing model (`nss_metrics.simulate_latency`): a GPU-bound game loop with just-in-time CPU scheduling (Reflex / Anti-Lag style) on a variable-refresh display, for base render times of 33, 25 and 17 ms. It separates **game input latency** (input -> first real frame reflecting it), **camera latency** (input -> first displayed frame whose camera reflects it; late-latched extrapolation re-samples the camera just before generating) and **content age** of generated frames. Interpolation holds each real frame for half a frame plus the generation time; extrapolation holds nothing. Measured T4 model times feed the model (2.0 ms is assumed, and logged, if the student did not train); on other GPUs scale them by the measured / estimated ratio. Expected output: latency tables and a chart of displayed frame rate against latency.


# %%
# cuDNN picks the fastest kernel per shape, as a deployment runtime would; the run of 2 to 3 November 2026 timed with
# PyTorch's default heuristics
torch.backends.cudnn.benchmark = True


@torch.no_grad()
def time_fg(net, hw, iters=100):
    dev = DEVICES[0]
    f = {
        k: (v.half() if v.dim() == 4 else v)
        for k, v in NM.fg_inputs(1, hw[0], hw[1], dev).items()
    }
    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(10):
            net(f)
        torch.cuda.synchronize(dev)
        times = []
        for _ in range(iters):
            a, b_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            net(f)
            b_.record()
            torch.cuda.synchronize(dev)
            times.append(a.elapsed_time(b_))
    return np.array(times)


lat_rows = []
for key in ("igpu", "student", "mid", "high"):
    if key not in MODELS:
        continue
    for hw in ((1080, 1920), (1440, 2560)):
        t = time_fg(MODELS[key], hw, 30 if QUICK_RUN else 100)
        gmac = (
            TIERS[TIERS["tier"] == RUNS[key][1]]["gmac_per_1080p_frame"].iloc[0]
            * hw[0]
            * hw[1]
            / (1920 * 1080)
        )
        lat_rows.append(
            {
                "model": key,
                "output": f"{hw[1]}x{hw[0]}",
                "t4_mean_ms": t.mean(),
                "t4_p95_ms": np.percentile(t, 95),
                "gmac": gmac,
                "est_ms_igpu_4tflops": M.estimate_ms(gmac, 4.0),
            }
        )
LAT = pd.DataFrame(lat_rows)
RUN.save_csv(LAT, "latency.csv", index=False)
print(LAT.round(2).to_string(index=False))
sel = (
    LAT[(LAT["model"] == "student") & (LAT["output"] == "1920x1080")]
    if len(LAT)
    else LAT
)
fg_ms = float(sel["t4_mean_ms"].iloc[0]) if len(sel) else 2.0
if not len(sel):
    log(
        "no measured student latency: the pacing model assumes 2.0 ms per generated frame"
    )
sim_rows = []
for render_ms in (33.3, 25.0, 16.7):
    for mode in ("native", "interp", "extrap", "extrap_late"):
        sim_rows.append(M.simulate_latency(mode, render_ms, sr_ms=0.0, fg_ms=fg_ms))
SIM = pd.DataFrame(sim_rows)
RUN.save_csv(SIM, "pacing_model.csv", index=False)
print(SIM.round(2).to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
for mode, g_ in SIM.groupby("mode"):
    axes[0].plot(g_["displayed_fps"], g_["game_latency_ms"], marker="o", label=mode)
    axes[1].plot(g_["displayed_fps"], g_["camera_latency_ms"], marker="o", label=mode)
for ax, t in zip(
    axes, ["game input latency (real frames)", "camera latency (any displayed frame)"]
):
    ax.set_xlabel("displayed frames per second")
    ax.set_ylabel("ms")
    ax.set_title(t)
    ax.legend()
RUN.show(fig, "pacing_model")

# %% [markdown]
# ## 27. ONNX export
#
# Exports the fused `low` and `igpu` generators (opset 17 with the TorchScript exporter on any PyTorch 2.x via `NM.export_onnx`, one static graph for the 128x128 example input, so another input size needs a new export) and compares ONNX Runtime CPU output with PyTorch FP32. Expected output: export status and maximum difference.

# %%
onnx_rows = []
for key in ("student", "igpu"):
    if key not in MODELS:
        continue
    net = copy.deepcopy(MODELS[key]).float().cpu().eval()
    wrap = NM.FGExport(net)
    f = NM.fg_inputs(1, 128, 128)
    args = (
        f["i0"],
        f["i1"],
        f["mv1"],
        f["depth1"],
        f["cam_prior"],
        f["tau"],
        torch.ones(1, 3),
    )
    names = ["i0", "i1", "mv1", "depth1", "cam_prior", "tau", "avail"]
    path = RUN.dirs["weights"] / f"fg_{key}_fused.onnx"
    row = {"model": key, "exported": False, "ort_max_abs": None}
    try:
        NM.export_onnx(wrap, args, path, names, ["frame"])
        row.update(exported=True, onnx_mb=round(path.stat().st_size / 1e6, 3))
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {n: a.numpy() for n, a in zip(names, args)})[0]
        with torch.no_grad():
            row["ort_max_abs"] = float(np.abs(ort_out - wrap(*args).numpy()).max())
    except Exception as exc:
        row["error"] = repr(exc)[:300]
    onnx_rows.append(row)
print(pd.DataFrame(onnx_rows).to_string(index=False))
RUN.save_json(onnx_rows, "onnx_export.json")

# %% [markdown]
# ## 28. Summary
#
# Headline numbers for the students and the `high` tier against RIFE 4.26 unchanged (without engine data) and motion-vector reprojection (with engine data) on every suite and mode, UI handling, cut and menu correctness, latency, the pacing model at a 30 fps base and failed runs (normally none). Expected output: summary table and stage times.


# %%
def agg(suite, mode, method, col, inputs="plain"):
    try:
        return round(float(AGG.loc[(suite, mode, inputs, method), col]), 4)
    except KeyError:
        return None


summary = {
    "run_hours": round(RUN.hours(), 2),
    "failed_runs": sorted(FAILED),
    "cut_threshold": CUT_THR,
    "cut_detection_rate": cut_rate,
    "cut_threshold_plain": CUT_THR_PLAIN,
    "cut_detection_rate_plain": cut_rate_plain,
}
for suite in (
    "vimeo_triplet",
    "tartanair",
    "sintel",
    "vid4",
    "vimeo1080p",
    "vimeo1080p_native",
    "synthetic",
):
    # triplets have no fourth frame, so no extrapolation or multi-frame target
    modes = ("interp",) if suite == "vimeo_triplet" else ("interp", "extrap", "mfg3x")
    for mode in modes if suite != "synthetic" else ("interp", "extrap"):
        for inp in ("plain",) if mode == "mfg3x" else ("plain", "engine"):
            for m in ("rife_4.26", "student", "igpu", "high") + (
                ("mv_reprojection",) if inp == "engine" else ()
            ):
                summary[f"{suite}/{mode}/{inp}/{m}/psnr"] = agg(
                    suite, mode, m, "psnr", inp
                )
for m in ("mv_reprojection", "rife_4.26", "student", "student+ui_layer"):
    summary[f"ui/interp/{m}/ghost_rate"] = agg(
        "synthetic-ui", "interp", m, "ui_ghost_rate", "engine"
    )
for _, r in LAT.iterrows():
    summary[f"latency_ms/{r['model']}/{r['output']}"] = round(r["t4_mean_ms"], 2)
for _, r in SIM[SIM["render_ms"] == 33.3].iterrows():
    summary[f"pacing30/{r['mode']}"] = (
        f"{r['displayed_fps']:.0f} fps, game {r['game_latency_ms']:.1f} ms, camera {r['camera_latency_ms']:.1f} ms"
    )
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
# Deletes the memory-mapped caches in `/tmp/nss_cache` (intermediates) to free local disk. Expected output: the space freed.

# %%
import shutil

freed = sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file())
shutil.rmtree(CACHE, ignore_errors=True)
print(f"removed caches: {freed / 1e9:.2f} GB")

# %% [markdown]
# ## 30. Package outputs
#
# Zips everything in `/kaggle/working` into `outputs.zip` after a disk check. After **Save Version > Save & Run All**, the zip and all files also appear in the notebook's **Output** tab.

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
