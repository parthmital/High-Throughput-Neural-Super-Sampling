# %% [markdown]
# # NeuralSS low-latency frame generation (Kaggle GPU T4 x2)
#
# **Goal.** Train and evaluate the NeuralSS frame generator: one small network that produces an in-between frame (**interpolation**, tau = 0.5) or a future frame (**extrapolation**, tau = 1.5) from two real frames, uses engine motion vectors, depth and the latest camera input when available, never hallucinates or interpolates UI (HUD, subtitles, menus, pause screens), falls back safely on scene cuts, and is cheap enough for integrated GPUs. Generated-frame latency (when a generated image reaches the screen) is reported separately from game input latency (when a real frame reflecting an input reaches the screen).
#
# **Approach.**
#
# 1. **Data.** The leakage-safe manifest from `neural-data-pipeline` (or an inline build). Synthetic layered scenes rendered on the GPU (exact intermediate flows, disocclusions, thin geometry, particles; HUD tiles and subtitles that change between frames, full-screen menus, scene cuts, rapid camera reversals) and real quads of consecutive video frames (Vimeo-90K, REDS, Vimeo-1080p, TartanAir, Sintel) with RAFT teacher flows.
# 2. **Model.** Coarse-to-fine intermediate-flow estimator at 1/4 and 1/2 resolution initialised from motion priors: linear-motion flows from the engine motion vector of the newest frame and, for extrapolation, the camera-only flow of the latest input (late latching, as in asynchronous reprojection). Output = UI-mask * newest real frame + (1 - UI-mask) * (blend of the two warped frames + residual). A privileged teacher that also sees the true target frame provides flow and output distillation. Tiers: `igpu`, `low`, `mid`, `high`, `teacher`.
# 3. **Training.** DDP on both GPUs, FP16, EMA, wall-clock schedule. Loss: Charbonnier + census + LPIPS + flow supervision + UI-mask cross-entropy + distillation. Engine inputs are randomly dropped so one model serves engine integrations and post-process integrations.
# 4. **Evaluation.** PSNR, SSIM, LPIPS, interpolation error, quality in occluded regions, UI-region PSNR and UI ghosting rate (with and without the engine's separate UI layer), scene-cut detection, menus, rapid camera reversals (with and without the camera prior), temporal consistency of the displayed sequence (tOF), motion classes; against frame repeat, frame averaging and motion-vector reprojection. Ablations: no UI training, no motion priors, interpolation-only, extrapolation-only, no distillation. Latency of fused tiers at 1080p and 1440p, an input-to-photon pacing model for native, interpolation, extrapolation and late-latched extrapolation, ONNX export.
#
# **Required Kaggle settings.** Accelerator **GPU T4 x2**; **Internet on** (pip packages, RAFT / AlexNet weights, Hugging Face data in the fallback). Attach the `neural-data-pipeline` output and the Kaggle datasets it lists. Optional secret `HF_TOKEN`.
#
# **Budget.** Caches about 25 min, teacher 70 min, main student 80 min, parallel `igpu` / `mid` students 50 min, three ablation pairs of 20 min, evaluation about 40 min: about 6 hours of a 9.5-hour ceiling. `QUICK_RUN = True` gives a 40-minute smoke test.
#
# **Outputs** in `/kaggle/working/`: `code/`, `weights/` (checkpoints, fused models, ONNX), `plots/`, `metrics/`, `logs/`, `predictions/`, `outputs.zip` (training caches are deleted first).

# %% [markdown]
# ## 1. Library files
#
# The first cell creates `/kaggle/working/code`; the shared NeuralSS library and the frame-generation training script are then written into it from the repository's single source.

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

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    pass

# %% [markdown]
# ## 3. Hardware check
#
# Run-time detection of GPUs, CPU, RAM and disk; two GPUs are required for DDP. Expected output: two Tesla T4 rows.

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
# * `patch` 192: training crops; textures are 512x512 so synthetic cameras can move up to about 35 px per frame.
# * `p_extrap` share of extrapolation samples (the low-latency mode); `p_ui`, `p_menu`, `p_cut`, `p_flip` control UI overlays, menus, scene cuts and rapid camera reversals in synthetic batches.
# * Loss weights: census 0.5, LPIPS 0.1, flow 0.01 (per pixel of flow error), UI 0.2, distillation 0.5.
# * `minutes`: wall-clock budgets per stage, scaled to the remaining run budget.

# %%
QUICK_RUN = False
RUN_BUDGET_H = 9.5
CFG = {
    "seed": 21,
    "patch": 192,
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
    "p_extrap": 0.5,
    "p_ui": 0.5,
    "p_menu": 0.03,
    "p_cut": 0.03,
    "p_flip": 0.1,
    "lr": 2e-3,
    "lr_teacher": 1e-3,
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
        "teacher": 4 if QUICK_RUN else 70,
        "student": 4 if QUICK_RUN else 80,
        "pair": 3 if QUICK_RUN else 50,
        "ablation": 3 if QUICK_RUN else 20,
    },
    "eval": {
        "synthetic_per_suite": 8 if QUICK_RUN else 48,
        "real_clips": 3 if QUICK_RUN else 12,
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
            "gameir": (0, 0),
            "vimeo1080p": (6, 3) if QUICK_RUN else (60, 16),
        },
        "calibration_pairs": 300,
        "val_fraction": 0.05,
    },
}
ABLATIONS = {
    "reference": {},
    "no_ui_training": {"ui_training": False, "w_ui": 0.0},
    "no_motion_prior": {"ablate_prior": True},
    "interp_only": {"mode": "interp"},
    "extrap_only": {"mode": "extrap"},
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
# Uses the attached `neural-data-pipeline` output, or builds a smaller manifest inline with the same code (adds about 30 to 60 minutes). Expected output: the data root and item counts per split.

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
# Memory-mapped 512x512 still crops (synthetic textures) and quads of four 192x192 frames, decoded with all CPU cores; RAFT teacher flows for each quad on both GPUs: 2 -> 0 (the engine-style motion vector of the newest real frame), 1 -> 0 and 1 -> 2 (interpolation targets) and 3 -> 2 (extrapolation target). Expected output: progress bars, counts per source, cache size.

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
    QUAD_COUNTS = {}
    for split in ("train", "val"):
        prefix = CACHE / f"quad_{split}"
        cnt, counts, need = K.build_window_cache(
            ITEMS[f"quad_{split}"],
            DATA_ROOT,
            prefix,
            CFG["cache"][f"quad_{split}"],
            4,
            P,
            os.cpu_count(),
            deadline,
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
stills_t = torch.from_numpy(np.ascontiguousarray(st[:8])).to(DEVICES[0])
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
# Parameters, multiply-accumulates per generated 1080p frame (the networks run at 1/4 and 1/2 resolution; warping and blending at full resolution are not counted) and fused-versus-unfused parity. Expected output: a tier table.


# %%
def fg_inputs(n, h, w, dev="cpu"):
    return {
        "i0": torch.rand(n, 3, h, w, device=dev),
        "i1": torch.rand(n, 3, h, w, device=dev),
        "mv1": torch.randn(n, 2, h, w, device=dev),
        "depth1": torch.rand(n, 1, h, w, device=dev),
        "cam_prior": torch.randn(n, 2, h, w, device=dev),
        "tau": torch.full((n,), 1.5, device=dev),
        "extrap": torch.ones(n, device=dev),
        "mv_avail": torch.ones(n, device=dev),
        "cam_avail": torch.ones(n, device=dev),
        "depth_avail": torch.ones(n, device=dev),
        "target": torch.rand(n, 3, h, w, device=dev),
    }


tier_rows = []
for tier in NM.FG_TIERS:
    net = NM.build_fg(tier).eval()
    with torch.no_grad():
        for mod in net.nets:
            mod["head"].weight.normal_(0, 0.01)
        fused = NM.fuse_model(net)
        f = fg_inputs(1, 64, 64)
        parity = float((net(f)[0] - fused(f)[0]).abs().max())
    macs = [0]
    hooks = [
        m.register_forward_hook(
            lambda mod, i, o: macs.__setitem__(
                0,
                macs[0]
                + o.numel() * mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1],
            )
        )
        for m in fused.modules()
        if isinstance(m, torch.nn.Conv2d)
    ]
    with torch.no_grad():  # counted at 272x480 and scaled: convolution cost is linear in pixel count
        fused(fg_inputs(1, 272, 480))
    for hk in hooks:
        hk.remove()
    tier_rows.append(
        {
            "tier": tier,
            "train_params": NM.count_params(net),
            "fused_params": NM.count_params(fused),
            "gmac_per_1080p_frame": round(
                macs[0] * (1080 * 1920) / (272 * 480) / 1e9, 2
            ),
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
# Full training steps (synthetic batch with UI, forward, losses, backward) on GPU 0 for the teacher and for a student with the teacher alongside; keeps the largest batch below 85 percent of VRAM. Expected output: probe tables and chosen batch sizes.


# %%
def probe(tier, with_teacher, sizes):
    rows, dev = [], DEVICES[0]
    total = torch.cuda.get_device_properties(dev).total_memory
    stills = torch.from_numpy(np.ascontiguousarray(st[: max(sizes)])).to(dev)
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
            scaler = torch.amp.GradScaler("cuda")
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
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch": bs, "fits": False})
            break
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    ok = df[df["fits"] & (df["peak_vram_pct"] <= 85)]
    return df, int(ok["batch"].max()) if len(ok) else sizes[0]


with RUN.stage("batch_probe"):
    sizes = [8, 16] if QUICK_RUN else [8, 16, 24, 32, 48, 64, 96]
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
# One JSON config per run; `code/train_fg.py` runs with DDP on both GPUs for main stages, or as two single-GPU processes in parallel for paired stages and ablations. Progress bars follow `PROGRESS` lines; validation rounds (PSNR overall, interpolation, extrapolation and UI regions) update a live chart. Expected output: planned minutes per stage.

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
HISTORY, PORT = {}, [29600]


def run_cfg(name, tier, minutes, batch, teacher_ckpt=None, **over):
    cfg = {
        "name": name,
        "tier": tier,
        "patch": P,
        "seed": CFG["seed"],
        "batch_per_gpu": batch,
        "p_real": CFG["p_real"],
        "p_extrap": CFG["p_extrap"],
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
    handle = display(plt.figure(), display_id=True)

    def update(rounds):
        h = pd.DataFrame(rounds)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
        axes[0].plot(h["elapsed_min"], h["train_loss"])
        axes[0].set_title("train loss")
        for col in ("val_psnr", "val_psnr_interp", "val_psnr_extrap", "val_psnr_ui"):
            if col in h:
                axes[1].plot(h["elapsed_min"], h[col], label=col[4:])
        axes[1].legend(fontsize=7)
        axes[1].set_title("validation PSNR (dB)")
        fig.suptitle(title)
        handle.update(fig)
        plt.close(fig)

    return update


def launch(cfg, gpus, live=True):
    PORT[0] += 1
    rounds, done = C.launch_training(
        RUN.work / "code" / "train_fg.py",
        cfg,
        RUN.dirs["code"] / f"cfg_{cfg['name']}.json",
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
        fa, fb = pool.submit(launch, cfg_a, [0], False), pool.submit(
            launch, cfg_b, [1], False
        )
        return fa.result(), fb.result()


# %% [markdown]
# ## 12. Stage A: privileged teacher (both GPUs)
#
# The `teacher` tier sees the true target frame as an extra input (privileged information, as in RIFE's distillation), so its flows and outputs are better than any deployable model's; it is never deployed. Expected output: live chart and final scores.

# %%
with RUN.stage("train_teacher"):
    print(
        launch(run_cfg("fg_teacher", "teacher", MINUTES["teacher"], BS_TEACHER), [0, 1])
    )
TEACHER_CKPT = str(RUN.dirs["weights"] / "fg_teacher_best.pt")

# %% [markdown]
# ## 13. Stage B: main student with distillation (both GPUs)
#
# The `low` tier with the full loss and distillation of flows and output toward the privileged teacher. Expected output: live chart and final scores.

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
# GPU 0 trains the `igpu` tier, GPU 1 the `mid` tier (two-level), both distilled. Expected output: both runs' final scores.

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
# Short `low`-tier runs against a `reference` with the same budget: no UI training (no HUD / subtitle / menu overlays, no UI loss), no motion priors (no engine motion vectors or camera prior), interpolation only, extrapolation only, and no distillation. Expected output: final scores of the six runs.

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
# Validation PSNR over time for every run and train loss for the main stages; table of best rounds. Expected output: two charts and a table.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 4))
best_rows = []
for name, rounds in HISTORY.items():
    if not rounds:
        continue
    h = pd.DataFrame(rounds)
    RUN.save_csv(h, f"history_{name}.csv", index=False)
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

# %% [markdown]
# ## 17. Load and fuse models
#
# Best EMA weights of each run, folded into plain convolutions, kept as FP16 replicas on GPU 0. Expected output: a table of models with fused parameters.

# %%
RUNS = {
    "teacher": ("fg_teacher", "teacher"),
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
    rows.append({"model": key, "tier": tier, "fused_params": NM.count_params(fused)})
print(pd.DataFrame(rows).to_string(index=False))
LPIPS = M.load_lpips(DEVICES[0], log=log)
RAFT = RAFTS[0] if RAFTS else M.load_raft(DEVICES[0], log=log)

# %% [markdown]
# ## 18. Scene-cut detector calibration
#
# A cheap analytic detector decides when not to generate (output the newest real frame instead): luma-histogram distance plus motion-compensated difference between the two real frames. Its threshold is chosen on synthetic validation pairs (cut versus continuous motion, including fast motion) for at most 1 percent false alarms. Expected output: threshold, detection rate and the score distributions.

# %%
sv = torch.from_numpy(
    np.load(CACHES["still_val"]["arrays"]["rgb"], mmap_mode="r")[
        : min(64, CACHES["still_val"]["count"])
    ].copy()
).to(DEVICES[0])
cal = [E.fg_suite(sv, P, 900 + k, DEVICES[0], p_cut=0.5) for k in range(4)]
scores = torch.cat([E.cut_score(b["i0"], b["i1"], b["mv1"]) for b in cal]).cpu().numpy()
labels = torch.cat([b["cut"] for b in cal]).cpu().numpy()
neg = np.sort(scores[labels < 0.5])
CUT_THR = float(neg[int(0.99 * (len(neg) - 1))]) if len(neg) else 1.0
cut_rate = (
    float((scores[labels > 0.5] > CUT_THR).mean()) if (labels > 0.5).any() else None
)
RUN.save_json(
    {"threshold": CUT_THR, "detection_rate": cut_rate, "false_alarm_target": 0.01},
    "cut_detector.json",
)
print(f"cut threshold {CUT_THR:.4f}, detection rate {cut_rate}")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(
    [scores[labels < 0.5], scores[labels > 0.5]], bins=40, label=["continuous", "cut"]
)
ax.axvline(CUT_THR, color="k", ls="--")
ax.legend()
ax.set_title("scene-cut score")
RUN.show(fig, "cut_detector")

# %% [markdown]
# ## 19. Test suites
#
# Test split only. Synthetic suites from held-out game frames and DIV2K validation images (192x192 crops of 512-px textures): interpolation and extrapolation per motion class, HUD / subtitles, menus, scene cuts and rapid camera reversals (extrapolation with and without the camera prior). Real suites at full clip resolution: the Vimeo-90K triplet test set (interpolation, the standard benchmark), held-out TartanAir and Sintel (exact flow available for the motion-vector input), held-out Vimeo-1080p and Vid4 (both modes, RAFT motion vectors). Expected output: suite sizes.

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
        out = []
        for it in [
            i
            for i in TEST_ITEMS
            if i["source"] == source
            and i["n_frames"] >= (3 if source == "vimeo_triplet" else 4)
        ][: n * 3]:
            Tn = 3 if source == "vimeo_triplet" else 4
            c = E.load_clip(DATA_ROOT, it, 0, Tn, crop, max_side=max_side)
            if c is not None:
                out.append((it["item_id"], c))
            if len(out) >= n:
                break
        return out

    REAL = {
        "vimeo_triplet": real_quads("vimeo_triplet", (256, 448), ev["real_clips"] * 4),
        "tartanair": real_quads("tartanair", (480, 640), ev["real_clips"]),
        "sintel": real_quads("sintel", (432, 1024), ev["real_clips"]),
        "vimeo1080p": real_quads(
            "vimeo1080p", (544, 960), ev["real_clips"], max_side=544
        ),
        "vid4": real_quads("vid4", (480, 704), ev["real_clips"]),
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
# For each suite and mode: every network (the privileged teacher is listed as an upper bound only, since it sees the target), the analytic baselines (frame repeat, average, motion-vector reprojection) and, on UI suites, the engine path (generate on HUD-less frames, composite the newest UI layer). The cut detector gates every network: frames it flags are replaced by the newest real frame. Expected output: number of metric rows.

# %%
ROWS = []


def quad_batch(clip, extrap, dev):
    fr = torch.from_numpy(clip["rgb"]).to(dev).permute(0, 3, 1, 2).float()[None] / 255.0
    i0, i1 = (
        fr[:, 0],
        fr[:, 2],
    )  # triplets: (I0, middle, I1); quads add the extrapolation target as frame 3
    target = fr[:, 3] if extrap else fr[:, 1]
    n, _, H, W = i0.shape
    z1 = torch.zeros(n, device=dev)
    if RAFT is not None:
        mv1, _ = M.raft_mv_and_valid(RAFT, i1, i0)
        flow_t1, valid_t1 = M.raft_mv_and_valid(RAFT, target, i1)
    else:
        mv1, flow_t1, valid_t1 = (
            torch.zeros(n, 2, H, W, device=dev),
            torch.zeros(n, 2, H, W, device=dev),
            torch.ones(n, 1, H, W, device=dev),
        )
    return {
        "i0": i0,
        "i1": i1,
        "target": target,
        "tau": torch.full((n,), 1.5 if extrap else 0.5, device=dev),
        "extrap": torch.full((n,), float(extrap), device=dev),
        "mv1": mv1,
        "mv_avail": torch.ones(n, device=dev) * (RAFT is not None),
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


def evaluate(suite, b, extra=None, models=None, ui_engine=False):
    gate = (
        E.cut_score(b["i0"], b["i1"], b["mv1"] * b["mv_avail"].view(-1, 1, 1, 1))
        > CUT_THR
    ).view(-1, 1, 1, 1)
    outs = dict(E.fg_baselines(b))
    for key in models or MODELS:
        if key not in MODELS:
            continue
        net = MODELS[key]
        f = dict(b)
        if key == "abl_no_motion_prior":
            f["mv_avail"], f["cam_avail"] = torch.zeros_like(
                f["mv_avail"]
            ), torch.zeros_like(f["cam_avail"])
        out, _ = E.run_fg(net, f)
        outs[key] = torch.where(gate, b["i1"], out)
        if ui_engine and key in ("student", "igpu"):
            out_e, _ = E.run_fg(net, f, hudless=True)
            outs[f"{key}+ui_layer"] = torch.where(gate, b["i1"], out_e)
    for mname, out in outs.items():
        for r in E.fg_metrics(out, b, LPIPS):
            ROWS.append({"suite": suite, "method": mname, **(extra or {}), **r})
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
        )
        rev_nc = dict(rev)
        rev_nc["cam_avail"] = torch.zeros(n, device=DEVICES[0])
        evaluate(
            "synthetic-reversal",
            rev_nc,
            {"mode": "extrap", "motion": "reversal", "camera_prior": False},
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

# %% [markdown]
# ## 21. Results by suite, mode and method
#
# Mean metrics (PSNR, SSIM, LPIPS, interpolation error, PSNR in occluded regions) per suite, mode and method. Expected output: the results table and a chart of PSNR by method for interpolation and extrapolation on real suites.

# %%
AGG = (
    RES.groupby(["suite", "mode", "method"])
    .mean(numeric_only=True)
    .drop(columns=["speed_class"], errors="ignore")
)
RUN.save_csv(AGG.round(4), "test_results.csv")
print(AGG[["psnr", "ssim", "lpips", "ie", "psnr_occluded"]].round(3).to_string())
real = RES[RES["suite"].isin(list(REAL))]
if len(real):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
    for ax, mode in zip(axes, ("interp", "extrap")):
        sub = real[real["mode"] == mode]
        if len(sub):
            sub.groupby(["suite", "method"])["psnr"].mean().unstack().plot.bar(
                ax=ax, rot=0
            )
            ax.set_title(f"real suites, {mode}: PSNR (dB)")
            ax.legend(fontsize=6)
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
syn = RES[RES["suite"] == "synthetic"]
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
            if c in ("student", "igpu", "mv_reprojection", "repeat")
        ]:
            axes[0].plot(t.index, t[col], ls, marker="o", label=f"{col} ({mode})")
    axes[0].set_title("PSNR by motion class")
    axes[0].legend(fontsize=6)
ui = RES[RES["suite"] == "synthetic-ui"]
if len(ui):
    UI = ui.groupby(["mode", "method"])[["psnr", "psnr_ui", "ui_ghost_rate"]].mean()
    RUN.save_csv(UI.round(4), "ui_results.csv")
    print(UI.round(3).to_string())
    UI["ui_ghost_rate"].unstack(0).plot.bar(ax=axes[1], rot=45)
    axes[1].set_title("UI ghosting rate (lower is better)")
for suite in ("synthetic-menu", "synthetic-cut"):
    d = RES[RES["suite"] == suite]
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
# Generated frames are shown between real ones, so motion must stay smooth across real -> generated -> real. tOF compares RAFT flow along the displayed sequence (I0 -> generated, generated -> I1) with the same flow along the true sequence (I0 -> true middle, true middle -> I1). Interpolation only (extrapolated frames are followed by the next real frame, which the test clips do not always contain). Expected output: tOF per suite and method.

# %%
tof_rows = []
if RAFT is not None:
    for suite in ("vimeo_triplet", "tartanair", "vimeo1080p"):
        for cid, clip in REAL.get(suite, [])[: ev["real_clips"]]:
            b = quad_batch(clip, False, DEVICES[0])
            outs = {
                "mv_reprojection": E.fg_baselines(b)["mv_reprojection"],
                "average": E.fg_baselines(b)["average"],
            }
            for key in ("student", "igpu", "mid"):
                if key in MODELS:
                    outs[key] = E.run_fg(MODELS[key], b)[0]
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
                    {"suite": suite, "method": mname, "tof": 0.5 * (t1 + t2)}
                )
TOF = pd.DataFrame(tof_rows)
if len(TOF):
    TOFA = TOF.groupby(["suite", "method"])["tof"].mean().unstack()
    RUN.save_csv(TOFA.round(4), "temporal_consistency.csv")
    print(TOFA.round(3).to_string())

# %% [markdown]
# ## 24. Visual comparisons
#
# For a UI sample and a fast-motion extrapolation sample: target, baselines and student output, the predicted UI mask, blend weight and flow. Saved to `predictions/`. Expected output: two figures.


# %%
def visual(b, title, fname):
    outs = dict(E.fg_baselines(b))
    out, aux = E.run_fg(MODELS["student"], b)
    outs["student"] = out
    if "hudless_i0" in b:
        outs["student+ui_layer"] = E.run_fg(MODELS["student"], b, hudless=True)[0]
    panels = [("target", b["target"])] + list(outs.items())
    fig, axes = plt.subplots(1, len(panels) + 3, figsize=(3 * (len(panels) + 3), 3.4))
    for ax, (k, img) in zip(axes, panels):
        ax.imshow(img[0].permute(1, 2, 0).float().cpu().clamp(0, 1).numpy())
        ax.set_title(k, fontsize=8)
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
    ABL = abl.groupby(["suite", "mode", "method"])[
        ["psnr", "lpips", "psnr_ui", "ui_ghost_rate", "psnr_occluded"]
    ].mean()
    RUN.save_csv(ABL.round(4), "ablations.csv")
    print(ABL.round(3).to_string())
    piv = abl.groupby(["method", "mode"])["psnr"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    (piv - piv.loc["abl_reference"]).drop(
        index="abl_reference", errors="ignore"
    ).plot.bar(ax=ax, rot=30)
    ax.set_title("PSNR change vs reference (all suites)")
    RUN.show(fig, "ablations")

# %% [markdown]
# ## 26. Latency and the input-to-photon pacing model
#
# Model latency: one fused FP16 generation on a T4 (CUDA events, batch 1) at 1920x1080 and 2560x1440, plus the arithmetic estimate for an assumed integrated-GPU throughput.
#
# Pacing model (`nss_metrics.simulate_latency`): a GPU-bound game loop with just-in-time CPU scheduling (Reflex / Anti-Lag style) on a variable-refresh display, for base render times of 33, 25 and 17 ms. It separates **game input latency** (input -> first real frame reflecting it), **camera latency** (input -> first displayed frame whose camera reflects it; late-latched extrapolation re-samples the camera just before generating) and **content age** of generated frames. Interpolation holds each real frame for half a frame plus the generation time; extrapolation holds nothing. Measured T4 model times feed the model; on other GPUs scale them by the measured / estimated ratio. Expected output: latency tables and a chart of displayed frame rate against latency.


# %%
@torch.no_grad()
def time_fg(net, hw, iters=100):
    dev = DEVICES[0]
    f = {
        k: (v.half() if v.dim() == 4 else v)
        for k, v in fg_inputs(1, hw[0], hw[1], dev).items()
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
for key in ("igpu", "student", "mid"):
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
fg_ms = (
    float(
        LAT[(LAT["model"] == "student") & (LAT["output"] == "1920x1080")][
            "t4_mean_ms"
        ].iloc[0]
    )
    if len(LAT)
    else 2.0
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
# Exports the fused `low` and `igpu` generators (opset 17, static shapes: one graph per output resolution) and compares ONNX Runtime CPU output with PyTorch FP32. Expected output: export status and maximum difference.

# %%
onnx_rows = []
for key in ("student", "igpu"):
    if key not in MODELS:
        continue
    net = copy.deepcopy(MODELS[key]).float().cpu().eval()
    wrap = NM.FGExport(net)
    f = fg_inputs(1, 64, 96)
    args = (
        f["i0"],
        f["i1"],
        f["mv1"],
        f["depth1"],
        f["cam_prior"],
        f["tau"],
        f["extrap"],
        torch.ones(1, 3),
    )
    names = ["i0", "i1", "mv1", "depth1", "cam_prior", "tau", "extrap", "avail"]
    path = RUN.dirs["weights"] / f"fg_{key}_fused.onnx"
    row = {"model": key, "exported": False, "ort_max_abs": None}
    try:
        torch.onnx.export(
            wrap,
            args,
            str(path),
            input_names=names,
            output_names=["frame"],
            opset_version=17,
            dynamo=False,
        )
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
# Headline numbers for the main student against motion-vector reprojection on real suites, UI handling, cut and menu correctness, latency and the pacing model at a 30 fps base. Expected output: summary table and stage times.


# %%
def agg(suite, mode, method, col):
    try:
        return round(float(AGG.loc[(suite, mode, method), col]), 4)
    except KeyError:
        return None


summary = {
    "run_hours": round(RUN.hours(), 2),
    "cut_threshold": CUT_THR,
    "cut_detection_rate": cut_rate,
}
for suite in ("vimeo_triplet", "tartanair", "vimeo1080p", "synthetic"):
    for mode in ("interp", "extrap"):
        for m in ("mv_reprojection", "student", "igpu"):
            summary[f"{suite}/{mode}/{m}/psnr"] = agg(suite, mode, m, "psnr")
for m in ("mv_reprojection", "student", "student+ui_layer"):
    summary[f"ui/interp/{m}/ghost_rate"] = agg(
        "synthetic-ui", "interp", m, "ui_ghost_rate"
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
# Deletes the memory-mapped caches (intermediates) before zipping. Expected output: the space freed.

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
