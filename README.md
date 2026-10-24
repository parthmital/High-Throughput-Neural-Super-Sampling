# NeuralSS: neural super-sampling and frame generation for games

NeuralSS is a research project for open, hardware-agnostic neural reconstruction in games. It has two networks:

- **Temporal super-resolution (TSR):** a small recurrent network that turns jittered low-resolution game frames (3x and 2x, for example 640x360 to 1920x1080) into full-resolution frames, using motion vectors, depth and its own history.
- **Frame generation (FG):** a small network that creates an in-between frame (interpolation) or a future frame (extrapolation) from two real frames, copies HUD and subtitles from the newest real frame instead of blending them, and skips generation on scene cuts.

Both networks fold into plain 3x3 convolution chains and export to ONNX. Everything is trained and evaluated in three Kaggle notebooks (GPU T4 x2) built from one shared Python library. All three notebooks have run on Kaggle, and their downloaded outputs (metrics, plots, logs, weights) are in this repository.

The full research plan, design rationale, competitive analysis and dataset licences are in [RESEARCH.md](RESEARCH.md). This README covers how the repository works, how to run it, and what each notebook produced.

## Table of contents

- [Quick start](#quick-start)
- [Project status](#project-status)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Local development](#local-development)
- [Notebook 1: data pipeline](#notebook-1-data-pipeline)
- [Notebook 2: temporal super-resolution](#notebook-2-temporal-super-resolution)
- [Notebook 3: frame generation](#notebook-3-frame-generation)
- [Weights and exported models](#weights-and-exported-models)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Licence](#licence)

## Quick start

The notebooks run on Kaggle, not locally. Training and evaluation need two GPUs and the attached datasets.

1. **Create the data-pipeline notebook on Kaggle.** Upload [neural-data-pipeline/neural-data-pipeline.ipynb](neural-data-pipeline/neural-data-pipeline.ipynb). In the notebook settings choose **Accelerator: GPU T4 x2** and turn **Internet on**. Attach these Kaggle datasets (**Add Input**): `soumikrakshit/div2k-high-resolution-images`, `daehoyang/flickr2k`, `amithkesavmrajagiri/reds-dataset`, `wangsally/vimeo-90k-7`, `chenshu123/vimeo-triplet`, `artemmmtry/mpi-sintel-dataset`, `uom200647r/vid4-dataset`. An optional Kaggle secret `HF_TOKEN` raises Hugging Face download limits.
2. **Run it** with **Save Version > Save & Run All**. Expected result: about 0.6 hours; the output contains `nss_data/nss_manifest.parquet` and `outputs.zip`.
3. **Create the two training notebooks** from [neural-supersampling/neural-supersampling.ipynb](neural-supersampling/neural-supersampling.ipynb) and [neural-frame-generation/neural-frame-generation.ipynb](neural-frame-generation/neural-frame-generation.ipynb) with the same accelerator, Internet and datasets, and also attach the pipeline notebook's output (**Add Input > Notebook output**).
4. **Run each one** with **Save & Run All**. Expected result: about 7.1 hours (super-resolution) and 6.5 hours (frame generation) with the current training budgets (7.13 and 6.47 hours in the runs of 24 October 2026), within the notebooks' 9.5-hour ceiling and Kaggle's 12-hour session limit; `outputs.zip` and every file appear in the notebook's **Output** tab. Together the two runs use about 14 hours of Kaggle's weekly GPU quota.
5. **Optional smoke test:** set `QUICK_RUN = True` in the configuration cell for a short run of every cell (about 40 minutes).

Common errors:

| Problem                                                            | Fix                                                                                                                                                       |
| :----------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `AssertionError: Select the GPU T4 x2 accelerator`                 | Set **Accelerator** to **GPU T4 x2**; the notebooks run one training process per GPU                                                                      |
| The version shows no logs and ends with `ERRORED_MOUNTING_DATASET` | One attached dataset failed to mount, so no code ran. Remove or re-add that dataset and save again; the notebooks leave out sources that are not attached |
| `No texture sources found`                                         | Neither the pipeline output nor the Kaggle datasets are attached                                                                                          |
| A pip install fails                                                | Internet is off. `lpips`, `flip-evaluator` and `onnxruntime` are optional; their absence is logged and the matching metric or check is skipped            |

## Project status

| Notebook                  | Kaggle run      | Duration | Result                                                                                                                                                                                            |
| :------------------------ | :-------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `neural-data-pipeline`    | 18 October 2026 | 0.63 h   | 12,300 items split into 8,874 train, 1,087 val and 1,432 test with zero cross-split duplicates                                                                                                    |
| `neural-supersampling`    | 24 October 2026 | 7.13 h   | 3x student beats every baseline on PSNR-Y on all five suites: 33.94 dB on synthetic game scenes against 29.35 dB for an analytic TAA upscaler, and 2.0 to 2.4 dB above the best baseline on video |
| `neural-frame-generation` | 24 October 2026 | 6.47 h   | Interpolation beats motion-vector reprojection on 5 of 6 suites and ties on Sintel; UI ghosting rate 0.046 against 0.342                                                                          |

Both training runs started at about 10:25 on 24 October 2026 (Kaggle clock; the date is from the output files, the times from the notebook logs). No training run failed in either notebook (`metrics/failed_runs.json` is empty). Kaggle had updated its image since the previous runs: these ran on Python 3.13.15 and PyTorch 2.11.0 with CUDA 12.8, with unchanged notebook code.

**What these runs measured.** They are the first runs of the fixes made after the runs of 20 to 21 October 2026, and the notebooks that ran are identical to the ones in this repository apart from the two later code changes listed after this one and updated run notes in the markdown cells:

1. **Jitter-aware bicubic TSR baseline** (`nss_models.jitter_upsample`). Bicubic now beats bilinear on TartanAir, Sintel, Vid4 and Vimeo-1080p, as expected of a sharper kernel; before the fix it lost on every suite with jitter.
2. **RAFT motion from full-resolution frames on video suites.** The 3x students went from losing to bilinear on Vid4 and Vimeo-1080p to beating the best baseline by 2.4 dB (Vid4) and 2.0 dB (Vimeo-1080p).
3. **Test-clip filter** (`nss_eval.test_clips`). The TSR run skipped 2 blank Vimeo-1080p clips; the FG run skipped 4 (blank, frozen or with a cut). No suite mean is now set by a clip at the 100 dB cap.
4. **Longer budgets** (TSR 140 / 120 / 75 / 20 minutes, FG 100 / 120 / 75 / 20). Validation curves are now flat at the end (at most 0.06 dB over each run's last three rounds), which suggests that more time alone would add little.

**Changes made after these runs, not yet measured on Kaggle.** Neither affects training or the numbers below:

1. The FG notebook saves tOF per clip to `metrics/temporal_consistency_per_clip.csv`. On Vimeo-1080p, frame averaging has the lowest tOF (6.37 px against 9.11 to 11.76 px), the opposite of every image metric; the suite mean cannot be traced to clips from this run's outputs.
2. `nss_models.export_onnx` silences the tracer and constant-folding warnings that static-shape export printed in both notebooks; the export call itself is unchanged.

The executed notebooks of these runs, with every output, are kept at [neural-supersampling/outputs/neural-supersampling.executed.ipynb](neural-supersampling/outputs/neural-supersampling.executed.ipynb) and [neural-frame-generation/outputs/neural-frame-generation.executed.ipynb](neural-frame-generation/outputs/neural-frame-generation.executed.ipynb); the notebooks next to them are the current versions without outputs. The exact library code each run used is in its `outputs/code/` folder.

## How it works

```mermaid
flowchart LR
    K[Kaggle datasets] --> P[neural-data-pipeline]
    H[Hugging Face subsets] --> P
    P -->|nss_manifest.parquet| T[neural-supersampling]
    P -->|nss_manifest.parquet| G[neural-frame-generation]
    T --> TO[TSR weights, ONNX, metrics, plots]
    G --> GO[FG weights, ONNX, metrics, plots]
```

1. **Data pipeline.** Discovers images and videos on Kaggle and Hugging Face, filters unusable items, detects copies across all sources with hashes, DINOv2 embeddings and SIFT verification, and writes a leakage-safe train / validation / test manifest.
2. **Training data.** Both training notebooks decode the manifest into memory-mapped caches. Each training batch mixes _synthetic_ game-like scenes, rendered on the GPU from still images with exact motion vectors, disocclusions, thin geometry, particles and HUD overlays, with _real_ sequences (TartanAir and Sintel with exact flow; video with RAFT-estimated motion).
3. **Training.** DDP on both GPUs with FP16, EMA weights and a wall-clock cosine schedule: a teacher first, then students that learn from the teacher and the ground truth, then smaller tiers and ablations, two runs in parallel (one per GPU).
4. **Evaluation.** Test-split suites only, against analytic baselines, with spatial, temporal, region, UI and latency metrics, then fusion to plain convolutions and ONNX export.

**Library and build.** The library lives once in [src/neuralss/](src/neuralss/). [tools/build_notebooks.py](tools/build_notebooks.py) turns the percent-format sources in [notebooks/](notebooks/) into `.ipynb` files and embeds every library module as a `%%writefile` cell, so each notebook is self-contained on Kaggle.

```mermaid
flowchart LR
    S[src/neuralss/*.py] --> B[tools/build_notebooks.py]
    N[notebooks/*.py] --> B
    B --> I[name/name.ipynb]
    I -->|on Kaggle: %%writefile| W[/kaggle/working/code/]
    W --> TR[train_tsr.py / train_fg.py subprocesses]
```

## Tech stack

| Component                              | Version (where known)                                                   | Purpose and where used                                                                                          |
| :------------------------------------- | :---------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| Python                                 | 3.10.12 (data pipeline), 3.13.15 (TSR and FG runs)                      | Notebook and library language                                                                                   |
| PyTorch with CUDA                      | 2.0.0 / CUDA 11.8 (data pipeline), 2.11.0 / CUDA 12.8 (TSR and FG runs) | Models, GPU renderer, DDP training (`torch.distributed.run`), FP16 autocast; the library supports 2.0 and later |
| torchvision                            | Kaggle image                                                            | RAFT-small optical flow (motion for video, tOF metric), AlexNet weights for the LPIPS fallback                  |
| Hugging Face Transformers              | Kaggle image                                                            | DINOv2-small embeddings for copy detection (`nss_dedup.py`)                                                     |
| OpenCV, Pillow, NumPy, pandas, PyArrow | Kaggle image                                                            | Decoding, SIFT features, HUD text tiles, caches, manifest (`nss_manifest.parquet`)                              |
| matplotlib, tqdm                       | Kaggle image                                                            | Plots and progress bars                                                                                         |
| `lpips`                                | installed by the notebooks                                              | Perceptual loss and metric (optional)                                                                           |
| `flip-evaluator`                       | installed by the notebooks                                              | NVIDIA FLIP metric for static convergence (optional)                                                            |
| `onnxruntime`                          | installed by the notebooks                                              | ONNX export parity check (optional)                                                                             |
| nbformat                               | 5.11.1 (local `.venv`)                                                  | Local notebook build and check (`tools/build_notebooks.py`)                                                     |
| black, pyflakes                        | 26.10.1, 4.0.3 (local `.venv`)                                          | Local formatting and lint checks                                                                                |

## Repository structure

```text
.
|-- README.md                       this file
|-- RESEARCH.md                     research plan, design, datasets, measured results, limitations, licences
|-- src/neuralss/                   single source of the shared library (embedded into every notebook)
|   |-- nss_common.py               run folders, logger, stage timer, hardware report, pip helper, training launcher with watchdog
|   |-- nss_data.py                 source registry, discovery, Hugging Face range access, decoding, manifest I/O
|   |-- nss_dedup.py                pHash / dHash, DINOv2 embeddings, GPU neighbour search, SIFT + RANSAC, leakage-safe split
|   |-- nss_pipeline.py             data-pipeline steps and the inline fallback used by the training notebooks
|   |-- nss_cache.py                memory-mapped training caches and RAFT teacher motion
|   |-- nss_synth.py                GPU renderer of synthetic game-like scenes and batch builders
|   |-- nss_models.py               RepConv, jitter-aware upsampling, TSR and FG networks, tiers, fusion, ONNX wrappers
|   |-- nss_metrics.py              losses, PSNR / SSIM / LPIPS / FLIP, RAFT, temporal metrics, baselines, latency model
|   |-- nss_train.py                DDP training loop, batch streaming, EMA, checkpoints, retry launcher
|   |-- nss_eval.py                 test clips with a defect filter, input construction, method runners, metric rows
|   |-- train_tsr.py                TSR training entry point (one process per GPU)
|   `-- train_fg.py                 FG training entry point
|-- notebooks/                      percent-format notebook sources (markdown and code cells)
|-- tools/build_notebooks.py        builds the .ipynb files from notebooks/ and src/neuralss
|-- neural-data-pipeline/
|   |-- neural-data-pipeline.ipynb  executed notebook with saved outputs (run of 18 October 2026)
|   `-- outputs/                    code/, logs/, metrics/, plots/, nss_data/nss_manifest.parquet
|-- neural-supersampling/
|   |-- neural-supersampling.ipynb  current notebook, no outputs
|   `-- outputs/                    run of 24 October 2026: code/, logs/, metrics/, plots/, predictions/, weights/,
|                                   neural-supersampling.executed.ipynb (the run's notebook with all outputs)
|-- neural-frame-generation/
|   |-- neural-frame-generation.ipynb  current notebook, no outputs
|   `-- outputs/                    run of 24 October 2026: code/, logs/, metrics/, plots/, predictions/, weights/,
|                                   neural-frame-generation.executed.ipynb
|-- .gitignore                      ignores .venv/, caches, *.zip archives and Kaggle __notebook__.ipynb copies
|-- .gitattributes                  line-ending normalisation
`-- .prettierrc.json                Markdown / JSON formatting settings
```

Inside every `outputs/` folder: `code/` holds the library and per-run JSON configs exactly as written on Kaggle, `logs/` the notebook log and one log per training run, `metrics/` every CSV and JSON result, `plots/` every figure.

## Local development

There is no local training or test suite: the notebooks are verified on Kaggle. Local work is editing the library or notebook sources and rebuilding the notebooks.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install nbformat black pyflakes --cache-dir .cache\pip
```

| Command (repository root)                                            | What it does                                                                                                                                  |
| :------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------- |
| `.venv\Scripts\python tools\build_notebooks.py`                      | Rebuilds every notebook whose sources changed (written without outputs); unchanged notebooks, including executed copies, are left as they are |
| `.venv\Scripts\python tools\build_notebooks.py --check`              | Exits with code 1 and lists stale notebooks if any `.ipynb` cell id, type or source differs from a fresh build; outputs are ignored           |
| `.venv\Scripts\python -m pyflakes src\neuralss tools notebooks`      | Lint (no output means clean)                                                                                                                  |
| `.venv\Scripts\python -m black --check src\neuralss tools notebooks` | Formatting check                                                                                                                              |

Workflow: edit `src/neuralss/*.py` or `notebooks/*.py`, run the build, then `--check`, pyflakes and black. Never edit the `.ipynb` files by hand; they are generated. Note that the build rewrites a notebook without outputs whenever its sources change, so copy an executed notebook into its `outputs/` folder first if its outputs must be kept.

## Notebook 1: data pipeline

[neural-data-pipeline/neural-data-pipeline.ipynb](neural-data-pipeline/neural-data-pipeline.ipynb) keeps its saved outputs from the run of 18 October 2026 (Kaggle GPU T4 x2, PyTorch 2.0.0, 4 CPU cores, 31.3 GB RAM). All numbers below are printed in the notebook or saved in [neural-data-pipeline/outputs/metrics/](neural-data-pipeline/outputs/metrics/).

**Sections 1 to 4: library, setup, hardware, configuration.** Writes the four library modules used by the pipeline (`nss_common`, `nss_data`, `nss_dedup`, `nss_pipeline`), loads the `HF_TOKEN` secret, and reports 2 Tesla T4 GPUs (14.56 GB each). Hugging Face caps: game stills (1,200 train, 250 test), TartanAir 120 frames per environment in sequences of 24, GameIR (40, 30) clips, Vimeo-1080p (160, 40) videos. Copy-detection settings: 16 neighbours, retrieval recall 0.98, 400 SIFT features, audit with 5 neighbours and at most 3 rounds.

**Section 5: Kaggle discovery (1.7 min).** Saved to `discovery_kaggle.csv`.

| Source          | Files listed | Items | Test items |
| :-------------- | -----------: | ----: | ---------: |
| div2k           |          900 |   900 |        100 |
| flickr2k        |        2,650 | 2,650 |          0 |
| reds            |       22,100 |   221 |          0 |
| vimeo_septuplet |       40,005 | 5,715 |          0 |
| vimeo_triplet   |        3,000 | 1,000 |      1,000 |
| sintel          |        3,256 |    46 |         10 |
| vid4            |          171 |     4 |          4 |

**Section 6: Hugging Face materialisation (14.2 min).** 1,764 items, 13.64 GB: 1,200 + 250 game stills, 12 TartanAir training environments and 2 test environments (`japanesealley`, `seasidetown`), 40 + 30 GameIR clips, 134 + 40 Vimeo-1080p videos. Catalogue (`catalogue_summary.csv`): 12,300 items; exact motion for Sintel and TartanAir, depth for TartanAir and GameIR.

**Section 7: catalogue overview.**

![Items and frames per source and role](neural-data-pipeline/outputs/plots/catalogue.png)

_Items and frames per source, split by training and test role._

**Section 8: keyframe statistics (12.0 min).** 13,322 keyframes, 0 decode failures (`keyframe_stats.csv`). Median size per source ranges from 448x256 (Vimeo-90K) to 2040x1356 (DIV2K, Flickr2K); median sharpness from 216 (Vimeo-90K septuplet) to 1,704 (Vid4).

![Quality statistics per source](neural-data-pipeline/outputs/plots/quality_distributions.png)

_Distributions of size, contrast, sharpness, letterboxing and colourfulness per source._

**Section 9: filtering (0.3 min).** Kept 12,149 of 12,300 items. Rejections by reason (an item can have several): Vimeo-90K septuplet 43 blurry, 17 flat, 71 letterboxed; game stills 2, 3, 26; Vimeo-1080p 1, 2, 2; Vimeo-90K triplet 1, 0, 2.

![Rejected examples](neural-data-pipeline/outputs/plots/rejected_examples.png)

_Examples of rejected keyframes with their reasons._

**Sections 10 to 12: hashes, embeddings, local features.** pHash mean bit 0.500 and dHash 0.480 (balanced). DINOv2 embeddings (13,322 x 384) on both GPUs in 16 s. SIFT: median 368 keypoints per keyframe; 273 keyframes have fewer than 20.

**Section 13: threshold calibration (0.9 min).** 1,500 synthetic copies against 2,930 random and 998 hard negatives (`dedup_calibration.json`, `calibration_scores.csv`).

| Signal            | Threshold               | Recall                            | Most similar negative              |
| :---------------- | :---------------------- | :-------------------------------- | :--------------------------------- |
| DINOv2 retrieval  | `t_ret` 0.6075          | 0.98                              | cosine 0.9348                      |
| DINOv2 strong     | `t_strong` 0.9648       | 0.038                             | cosine 0.9348                      |
| SIFT verification | `t_cells` 10            | 0.9493                            | 8 cells                            |
| pHash             | `h_exact` 6, `h_cand` 6 | 0.9987 (light copies)             | Hamming 18                         |
| Combined rule     |                         | 0.9673 (light 0.9947, heavy 0.94) | 1 false positive, precision 0.9993 |

![Calibration score distributions](neural-data-pipeline/outputs/plots/dedup_calibration.png)

_Score distributions of synthetic copies against negatives, with the chosen thresholds._

**Section 14: copy detection (1.4 min).** 13,165 keyframes searched; 45,213 embedding and 849 pHash candidates, 27,044 unique pairs; 3,346 edges (3,304 verified, 37 strong embedding, 5 exact hash). Most edges join game frames to each other (2,873); cross-source edges link Vimeo-90K septuplets to Vimeo-1080p (78) and to Vimeo-90K triplet test clips (104). Pairs are in `duplicate_pairs.csv`.

![Duplicate examples](neural-data-pipeline/outputs/plots/duplicate_examples.png)

_Examples of detected duplicate pairs with the rule that matched them._

**Section 15: leakage-safe split with audit (1.3 min).** Audit round 1 found 24 cross-split duplicates (added as edges and re-split); round 2 found none. Final: 3,370 edges, largest duplicate cluster 50 items, 3,956 train and validation components (largest 2,936).

| Source          |     Train |       Val |      Test |    Drop | Filtered |
| :-------------- | --------: | --------: | --------: | ------: | -------: |
| div2k           |       755 |        43 |       100 |       2 |        0 |
| flickr2k        |     2,497 |       153 |         0 |       0 |        0 |
| game_stills     |       502 |        41 |       243 |     636 |       28 |
| gameir          |        40 |         0 |        30 |       0 |        0 |
| reds            |       209 |        12 |         0 |       0 |        0 |
| sintel          |        36 |         0 |        10 |       0 |        0 |
| tartanair       |        60 |         0 |        10 |       0 |        0 |
| vid4            |         0 |         0 |         4 |       0 |        0 |
| vimeo1080p      |       120 |         3 |        38 |      10 |        3 |
| vimeo_septuplet |     4,655 |       835 |         0 |     108 |      117 |
| vimeo_triplet   |         0 |         0 |       997 |       0 |        3 |
| **Total**       | **8,874** | **1,087** | **1,432** | **756** |  **151** |

Every dropped item is a duplicate of a test item (`split_report.json`, `split_by_source.csv`).

![Split by source](neural-data-pipeline/outputs/plots/split_by_source.png)

_Items per split and source after the audit._

**Section 16: audit statistics.** `leakage_audit.csv`; `leakage_violations.csv` has a header and no rows (no cross-split duplicate after round 2).

| Split | Reference | Max cosine (median) | Max verified cells (median) | Min pHash Hamming (median) |
| :---- | :-------- | :------------------ | :-------------------------- | :------------------------- |
| val   | train     | 0.9614 (0.5757)     | 9 (0)                       | 8 (16)                     |
| test  | train+val | 0.9570 (0.5745)     | 9 (0)                       | 0 (16)                     |

The pHash distance of 0 is not a duplicate under the rule (untextured keyframe or rejected by dHash).

![Leakage audit](neural-data-pipeline/outputs/plots/leakage_audit.png)

_Nearest-neighbour similarity of validation and test keyframes to the training split, against the thresholds._

**Sections 17 to 19: manifest, summary, packaging.** Writes `nss_data/nss_manifest.parquet` and `dataset_card.csv` (source, host, licence, counts per split), `summary.json`, and a 31-file `outputs.zip`. Stage times (`stage_times.csv`, minutes): discovery 1.66, Hugging Face 14.24, keyframe stats 11.95, filters 0.26, hashes 0.13, embeddings 0.36, SIFT 5.09, calibration 0.94, duplicate search 1.38, split and audit 1.34.

![Pipeline stage times](neural-data-pipeline/outputs/plots/stage_times.png)

_Minutes per pipeline stage._

## Notebook 2: temporal super-resolution

Run of 24 October 2026 (PyTorch 2.11.0, CUDA 12.8, Python 3.13.15, 2 x Tesla T4, 4 CPU cores, 31.3 GB RAM; 7.13 hours; no failed runs). Outputs: [neural-supersampling/outputs/](neural-supersampling/outputs/); executed notebook: [neural-supersampling.executed.ipynb](neural-supersampling/outputs/neural-supersampling.executed.ipynb). The current notebook differs from the one that ran only by the ONNX warning filter listed under [Project status](#project-status); the numbers below are from the run.

**Sections 1 to 4: library, setup, hardware, configuration.** Writes the ten library modules and `train_tsr.py`; `lpips`, `flip-evaluator` and `onnxruntime` all installed. Main settings (`metrics/config.json`): 3x primary and 2x secondary scale, 192-px output patches, 6-frame sequences, 40 percent real windows per batch, learning rate 2e-3 (teacher 1e-3), loss weights gradient 0.1, Fourier 0.05, temporal 0.5, LPIPS 0.05, distillation 0.5. Budgets: teacher 140 min, student 120, pair 75, ablations 20 per pair; caches 30 min.

**Section 5: manifest.** Loaded from the attached pipeline output; every source was reachable (`reachable_sources.json`).

**Section 6: item selection.** 4,123 texture and 5,120 sequence items for training, 252 and 850 for validation.

**Section 7: caches (20.7 min, 9.30 GB).** 256 validation and 6,500 training still crops (4 decode failures); 192 validation and 3,600 training windows, with RAFT motion for the 192 validation and 3,533 training windows without exact motion. Training windows by source: Vimeo-90K 3,267, REDS 149, Vimeo-1080p 91, TartanAir 39, Sintel 28, GameIR 26; validation windows: Vimeo-90K 188, REDS 4 (`caches.json`).

**Section 8: cache inspection.** 1.9 percent of windows have exact motion and 98.1 percent RAFT motion; 1.8 percent have depth; mean motion 1.50 px per frame.

![Still crops](neural-supersampling/outputs/plots/cache_stills.png)

_Random 512x512 still crops used as textures by the synthetic renderer._

![Cached windows](neural-supersampling/outputs/plots/cache_windows.png)

_Real windows: first and last frame, motion vectors (hue = direction, brightness = magnitude) and validity._

**Section 9: synthetic renderer preview.**

![Synthetic training sequence](neural-supersampling/outputs/plots/synthetic_preview.png)

_One synthetic sequence at three time steps: aliased jittered input, anti-aliased target, exact motion vectors, valid history, depth and particle (reactive) mask._

**Section 10: model tiers.** Measured with forward hooks (`model_tiers.csv`); fused-versus-unfused parity 0.0 for every tier.

| Tier    | Scale | Train params | Fused params | GMAC per 1080p frame |
| :------ | ----: | -----------: | -----------: | -------------------: |
| igpu    |     3 |       27,852 |       21,196 |                 4.84 |
| igpu    |     2 |       21,832 |       16,136 |                 8.29 |
| low     |     3 |      132,284 |       99,020 |                12.91 |
| low     |     2 |      123,264 |       91,440 |                25.12 |
| mid     |     3 |      282,100 |      207,924 |                23.96 |
| mid     |     2 |      270,080 |      197,824 |                48.69 |
| high    |     3 |      783,316 |      567,220 |                62.57 |
| high    |     2 |      765,296 |      552,080 |               132.95 |
| teacher |     3 |    1,920,452 |    1,373,508 |               144.42 |
| teacher |     2 |    1,896,432 |    1,353,328 |               314.49 |

![GMAC per tier](neural-supersampling/outputs/plots/model_tiers.png)

_GMAC per 1080p output frame for each tier at 3x and 2x._

**Section 11: batch probe (0.5 min).** Every size up to 48 fit with a finite loss at initialisation; 48 per GPU was chosen for all runs (`batch_probe.csv`).

| Batch | Teacher VRAM % | Teacher seq/s | Student + teacher VRAM % | Student seq/s |
| ----: | -------------: | ------------: | -----------------------: | ------------: |
|     4 |            3.9 |          12.8 |                      1.9 |          13.8 |
|     8 |            6.8 |          24.7 |                      3.1 |          24.7 |
|    12 |            9.7 |          29.6 |                      4.2 |          32.1 |
|    16 |           12.5 |          32.3 |                      5.4 |          35.5 |
|    24 |           18.3 |          34.9 |                      7.6 |          39.0 |
|    32 |           24.0 |          36.0 |                      9.8 |          40.9 |
|    48 |           35.5 |          36.6 |                     14.3 |          42.3 |

**Section 12: launcher and budget.** 8.14 hours were available, so the budgets ran at full scale.

**Sections 13 to 16: training.** Logs in `logs/<run>.log`, histories in `metrics/<run>_history.json` and `history_<run>.csv`, configs in `code/cfg_<run>.json`. Every 3x student was distilled from the trained teacher.

| Stage             | Run                                           |        Steps |      Train s | Best val PSNR-Y (round) |
| :---------------- | :-------------------------------------------- | -----------: | -----------: | :---------------------- |
| A, both GPUs      | `tsr_teacher_s3`                              |        6,590 |        8,408 | 33.83 dB (17)           |
| B, both GPUs      | `tsr_low_s3`                                  |        6,650 |        7,202 | 32.51 dB (14)           |
| C, GPU 0          | `tsr_igpu_s3`                                 |        4,420 |        4,501 | 31.59 dB (9)            |
| C, GPU 1          | `tsr_low_s2`                                  |        4,610 |        4,505 | 36.99 dB (9)            |
| Ablations, pair 1 | `abl_reference`, `abl_no_history`             | 1,110, 1,170 | 1,209, 1,209 | 31.69 (5), 30.63 (5)    |
| Ablations, pair 2 | `abl_no_gbuffers`, `abl_bicubic_degradation`  | 1,110, 1,110 | 1,203, 1,203 | 30.47 (5), 32.94 (5)    |
| Ablations, pair 3 | `abl_no_temporal_loss`, `abl_no_distillation` | 1,110, 1,410 | 1,207, 1,201 | 31.87 (5), 31.80 (5)    |

**Section 17: training curves.** Validation PSNR-Y (mean over the 6 frames of each validation sequence) and temporal-change error per round (`best_validation.csv`, `history_<run>.csv`):

| Run              | Rounds 0 to last, val PSNR-Y (dB)                                                                                            | Val temporal-change error, first to last | Train loss, first to last | Peak memory (GB) | Samples/s |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------- | :------------------------ | ---------------: | --------: |
| `tsr_teacher_s3` | 31.95, 32.66, 32.87, 33.09, 33.25, 33.36, 33.45, 33.52, 33.58, 33.63, 33.68, 33.72, 33.76, 33.78, 33.80, 33.82, 33.83, 33.83 | 0.0150 to 0.0130                         | 0.0505 to 0.0325          |             6.49 |      75.3 |
| `tsr_low_s3`     | 31.41, 31.90, 32.08, 32.20, 32.28, 32.33, 32.37, 32.40, 32.43, 32.45, 32.47, 32.49, 32.50, 32.51, 32.51                      | 0.0159 to 0.0147                         | 0.0593 to 0.0414          |             3.49 |      88.7 |
| `tsr_low_s2`     | 35.80, 36.32, 36.59, 36.73, 36.83, 36.89, 36.93, 36.97, 36.98, 36.99                                                         | 0.0118 to 0.0107                         | 0.0335 to 0.0235          |             4.47 |      49.2 |
| `tsr_igpu_s3`    | 31.05, 31.29, 31.42, 31.49, 31.53, 31.56, 31.57, 31.58, 31.59, 31.59                                                         | 0.0163 to 0.0158                         | 0.0598 to 0.0467          |             3.03 |      47.2 |

The longer budgets added 0.11 dB (teacher), 0.05 (`low` 3x), 0.04 (`igpu`) and 0.11 (`low` 2x) of best validation PSNR-Y over the run of 20 to 21 October 2026, and every curve is now flat at the end (at most 0.05 dB over the last three rounds), which suggests that more time alone would add little.

Each ablation validates on batches built with its own recipe, so ablation validation scores are not comparable (the bicubic ablation validates on easier bicubic inputs and reports the highest score). Throughput is limited by the GPU batch synthesis, not the student: the `igpu` student (27,852 parameters) trains at about the same rate per GPU as the `low` student, and the run without a teacher made 27 percent more steps than the reference in the same time.

![TSR training curves](neural-supersampling/outputs/plots/training_curves.png)

_Validation PSNR-Y, validation temporal-change error and train loss against minutes; dashed lines are ablations._

**Section 18: load and fuse.** Ten models loaded (teacher, three students, six ablations); fused FP32 parity 0.0 for every one.

**Section 19: test suites (2.4 min).** 24 synthetic test stills (6 per motion class), 8 TartanAir, 8 Sintel, 4 Vid4, 8 Vimeo-1080p and 4 GameIR native clips. Two blank Vimeo-1080p clips were skipped and replaced by the next clips of that source.

**Section 20: evaluation (6.4 min).** 12,560 per-frame metric rows (`test_per_frame.csv`): synthetic-game 5,760, TartanAir 1,920, Sintel 1,920, Vimeo-1080p 1,920, Vid4 960, GameIR native 80. Video suites use RAFT motion estimated on the full-resolution frames and sampled at the jittered positions.

**Section 21: results by suite and method** (`test_results.csv`, mean over frames after the first). Higher is better for PSNR and SSIM; lower is better for LPIPS, temporal-change error (TCE) and tOF. `bicubic` and `bilinear` sample at the true jittered positions.

3x:

| Suite          | Method        | PSNR-Y | SSIM-Y | PSNR disocc. | PSNR detail | LPIPS |    TCE |   tOF |
| :------------- | :------------ | -----: | -----: | -----------: | ----------: | ----: | -----: | ----: |
| synthetic-game | bicubic       |  29.50 |  0.832 |        34.08 |       23.53 | 0.288 | 0.0321 | 0.844 |
| synthetic-game | bilinear      |  29.80 |  0.833 |        34.47 |       23.87 | 0.273 | 0.0253 | 0.782 |
| synthetic-game | analytic_taau |  29.35 |  0.817 |        31.51 |       23.61 | 0.261 | 0.0110 | 0.790 |
| synthetic-game | igpu_s3       |  32.63 |  0.905 |        34.37 |       26.54 | 0.104 | 0.0123 | 0.631 |
| synthetic-game | student_s3    |  33.94 |  0.927 |        35.38 |       27.96 | 0.066 | 0.0105 | 0.485 |
| synthetic-game | teacher_s3    |  35.56 |  0.947 |        36.44 |       29.49 | 0.041 | 0.0085 | 0.387 |
| tartanair      | bicubic       |  31.83 |  0.896 |        31.18 |       24.92 | 0.216 | 0.0149 | 0.486 |
| tartanair      | bilinear      |  31.52 |  0.890 |        30.85 |       24.64 | 0.216 | 0.0132 | 0.485 |
| tartanair      | analytic_taau |  30.64 |  0.875 |        27.92 |       23.85 | 0.243 | 0.0104 | 0.415 |
| tartanair      | igpu_s3       |  33.01 |  0.921 |        29.79 |       26.15 | 0.109 | 0.0106 | 0.445 |
| tartanair      | student_s3    |  33.13 |  0.925 |        29.93 |       26.33 | 0.085 | 0.0106 | 0.444 |
| tartanair      | teacher_s3    |  33.64 |  0.932 |        30.50 |       26.81 | 0.065 | 0.0102 | 0.327 |
| sintel         | bicubic       |  32.82 |  0.881 |        30.84 |       26.86 | 0.275 | 0.0148 | 1.564 |
| sintel         | bilinear      |  32.55 |  0.874 |        30.49 |       26.61 | 0.269 | 0.0127 | 1.703 |
| sintel         | analytic_taau |  31.07 |  0.848 |        27.58 |       25.19 | 0.271 | 0.0092 | 1.519 |
| sintel         | igpu_s3       |  33.80 |  0.911 |        29.79 |       27.89 | 0.144 | 0.0099 | 1.350 |
| sintel         | student_s3    |  34.49 |  0.920 |        30.65 |       28.53 | 0.107 | 0.0095 | 1.413 |
| sintel         | teacher_s3    |  35.40 |  0.930 |        31.21 |       29.36 | 0.081 | 0.0089 | 1.253 |
| vid4           | bicubic       |  24.58 |  0.738 |        22.33 |       19.60 | 0.391 | 0.0433 | 0.398 |
| vid4           | bilinear      |  24.55 |  0.722 |        22.32 |       19.71 | 0.362 | 0.0368 | 0.397 |
| vid4           | analytic_taau |  23.96 |  0.679 |        20.53 |       19.29 | 0.382 | 0.0236 | 0.183 |
| vid4           | igpu_s3       |  26.94 |  0.855 |        22.05 |       21.64 | 0.130 | 0.0250 | 0.155 |
| vid4           | student_s3    |  26.95 |  0.852 |        22.15 |       21.78 | 0.138 | 0.0252 | 0.148 |
| vid4           | teacher_s3    |  27.96 |  0.883 |        22.92 |       22.68 | 0.103 | 0.0238 | 0.141 |
| vimeo1080p     | bicubic       |  35.21 |  0.902 |        41.50 |       28.67 | 0.191 | 0.0136 | 4.080 |
| vimeo1080p     | bilinear      |  34.80 |  0.899 |        40.65 |       28.19 | 0.189 | 0.0116 | 4.215 |
| vimeo1080p     | analytic_taau |  33.43 |  0.888 |        36.98 |       26.79 | 0.203 | 0.0075 | 5.942 |
| vimeo1080p     | igpu_s3       |  36.59 |  0.937 |        39.99 |       30.37 | 0.076 | 0.0085 | 3.372 |
| vimeo1080p     | student_s3    |  37.23 |  0.939 |        41.76 |       30.91 | 0.064 | 0.0080 | 3.433 |
| vimeo1080p     | teacher_s3    |  38.54 |  0.951 |        43.13 |       32.24 | 0.039 | 0.0072 | 3.061 |

Particle-region PSNR (synthetic-game only): bicubic 33.01, bilinear 33.13, analytic TAAU 29.64, `igpu` 33.47, student 34.68, teacher 37.26 dB. Warping error along exact motion (synthetic-game): teacher 0.0105, student 0.0119, analytic TAAU 0.0092, bilinear 0.0268.

2x (`low` 2x student against baselines; GameIR native uses the engine's own 720p render as input):

| Suite          | Method        | PSNR-Y | SSIM-Y | LPIPS |    TCE |   tOF |
| :------------- | :------------ | -----: | -----: | ----: | -----: | ----: |
| synthetic-game | bicubic       |  33.41 |  0.919 | 0.126 | 0.0214 | 0.466 |
| synthetic-game | bilinear      |  33.35 |  0.916 | 0.108 | 0.0172 | 0.430 |
| synthetic-game | analytic_taau |  30.85 |  0.871 | 0.193 | 0.0083 | 0.631 |
| synthetic-game | student_s2    |  38.63 |  0.972 | 0.023 | 0.0076 | 0.275 |
| tartanair      | bicubic       |  34.71 |  0.939 | 0.115 | 0.0107 | 0.244 |
| tartanair      | bilinear      |  33.98 |  0.931 | 0.121 | 0.0100 | 0.273 |
| tartanair      | analytic_taau |  31.44 |  0.896 | 0.212 | 0.0099 | 0.347 |
| tartanair      | student_s2    |  35.46 |  0.950 | 0.066 | 0.0092 | 0.320 |
| sintel         | bicubic       |  36.37 |  0.941 | 0.139 | 0.0090 | 0.829 |
| sintel         | bilinear      |  35.45 |  0.929 | 0.142 | 0.0083 | 0.897 |
| sintel         | analytic_taau |  31.70 |  0.875 | 0.220 | 0.0082 | 1.277 |
| sintel         | student_s2    |  38.61 |  0.965 | 0.044 | 0.0067 | 0.886 |
| vid4           | bicubic       |  28.98 |  0.886 | 0.202 | 0.0253 | 0.150 |
| vid4           | bilinear      |  27.78 |  0.854 | 0.191 | 0.0238 | 0.161 |
| vid4           | analytic_taau |  25.00 |  0.752 | 0.347 | 0.0218 | 0.138 |
| vid4           | student_s2    |  31.23 |  0.942 | 0.043 | 0.0194 | 0.113 |
| vimeo1080p     | bicubic       |  38.79 |  0.944 | 0.103 | 0.0093 | 1.945 |
| vimeo1080p     | bilinear      |  37.78 |  0.937 | 0.106 | 0.0082 | 1.902 |
| vimeo1080p     | analytic_taau |  34.47 |  0.912 | 0.173 | 0.0066 | 5.046 |
| vimeo1080p     | student_s2    |  42.27 |  0.976 | 0.015 | 0.0058 | 1.677 |
| gameir-native  | bicubic       |  32.33 |  0.912 | 0.169 | 0.0169 | 4.133 |
| gameir-native  | bilinear      |  32.30 |  0.907 | 0.190 | 0.0163 | 4.377 |
| gameir-native  | analytic_taau |  30.83 |  0.879 | 0.227 | 0.0168 | 2.332 |
| gameir-native  | student_s2    |  32.58 |  0.913 | 0.134 | 0.0160 | 4.207 |

What the numbers say:

- The 3x student beats every baseline on PSNR-Y and LPIPS on all five suites: by 2.5 to 4.6 dB over the analytic TAA upscaler and by 1.3 to 4.1 dB over the best of bicubic and bilinear. Its temporal-change error is now below the analytic TAA upscaler's on synthetic scenes (0.0105 against 0.0110) and 2 to 7 percent above it on TartanAir, Sintel, Vid4 and Vimeo-1080p.
- On video, with RAFT motion from full-resolution frames, the 3x student leads the best baseline by 2.37 dB on Vid4 and 2.02 dB on Vimeo-1080p; in the previous run, with motion estimated on jittered inputs, it lost to bilinear on both. Per clip (from `test_per_frame.csv`) it wins 11 of the 12 video clips; the exception is one Vimeo-1080p clip on which every method scores above 47 dB (`Soft Fangs`, bicubic 52.87 against student 50.48 dB).
- The 2x student beats every baseline on PSNR-Y, SSIM-Y, LPIPS and TCE on all six suites. On the native GameIR renders (a real renderer's low-resolution image, not a simulated one) its lead is small: 0.25 dB PSNR-Y over bicubic with 21 percent lower LPIPS, while the analytic TAA upscaler has the lowest tOF there.
- The teacher has the best PSNR-Y, SSIM-Y, LPIPS and tOF on every 3x suite and leads the student by 0.5 to 1.6 dB PSNR-Y; the analytic TAA upscaler keeps a lower warping error on synthetic scenes and a slightly lower TCE on Vid4.
- The `igpu` student (21,196 fused parameters) is within 0.12 dB of the `low` student on TartanAir and Vid4 and 1.3 dB behind on synthetic scenes.

![Quality against stability](neural-supersampling/outputs/plots/quality_vs_stability.png)

_PSNR-Y against temporal-change error per suite and method, 3x (left) and 2x (right); top-left is sharp and stable._

**Section 22: motion classes and regions (synthetic-game, 3x).** Measured motion per class: static 0.0, slow 2.65, medium 10.44, fast 17.62 output px per frame. PSNR-Y (dB):

| Method        | Static |  Slow | Medium |  Fast |
| :------------ | -----: | ----: | -----: | ----: |
| bicubic       |  26.25 | 30.19 |  24.93 | 36.65 |
| bilinear      |  26.70 | 30.37 |  25.56 | 36.58 |
| analytic_taau |  27.68 | 29.40 |  25.48 | 34.85 |
| igpu_s3       |  31.00 | 32.98 |  27.60 | 38.94 |
| student_s3    |  32.70 | 34.41 |  28.44 | 40.23 |
| teacher_s3    |  34.60 | 36.53 |  29.55 | 41.55 |

The fast class scores higher than medium for every method, including bicubic, so the classes also differ in texture content; compare methods within a class, not classes with each other.

![Motion classes and regions](neural-supersampling/outputs/plots/motion_regions.png)

_PSNR-Y by motion class, PSNR-Y inside disocclusions, fine detail and particles, and LPIPS / tOF per method._

**Section 23: static-scene convergence** (`static_convergence.csv`, 16 frames, static camera with sub-pixel jitter):

| Method         | Last-frame PSNR-Y (dB) | Last-frame FLIP |
| :------------- | ---------------------: | --------------: |
| bilinear       |                  27.29 |          0.0679 |
| analytic_taau  |                  28.42 |          0.0621 |
| student_s3     |                  33.97 |          0.0524 |
| igpu_s3        |                  32.05 |          0.0398 |
| teacher_s3     |                  38.30 |          0.0194 |
| abl_no_history |                  28.08 |          0.0545 |

![Static convergence](neural-supersampling/outputs/plots/static_convergence.png)

_PSNR-Y against frame index with a static camera: the recurrent models accumulate detail, single-frame methods stay flat. The teacher keeps rising to frame 15; both students peak around frames 6 to 10 (student about 35.4 dB, `igpu` about 33.3 dB) and then lose 1.3 to 1.4 dB._

The students' late decline is new: in the run of 20 to 21 October 2026 the student held about 35.5 dB to frame 15 (35.30 dB at the last frame). Training unrolls only 6 frames, so the recurrence is never trained beyond that horizon. Since that run, the TSR training code is unchanged apart from the longer budgets, and the Kaggle image moved from PyTorch 2.0 to 2.11; the likely cause is that the longer training fitted the recurrence more tightly to the 6-frame horizon. This is a hypothesis: each configuration ran once, so run-to-run variance is unknown, and the notebook saves only last-frame values, so the per-frame curve is read from the plot. See [Known limitations](#known-limitations).

**Section 24: visual comparisons.** Full last frames per method are saved in `predictions/`.

![Synthetic fast motion](neural-supersampling/outputs/plots/visual_synthetic_fast.png)

_Fast-motion synthetic sequence: last-frame crops per method, the student's history weight, x-t slices and the true disocclusion mask. The history weight drops to zero in and around the true disocclusion. On the flat blue surface the student's output shows a fine noise pattern that the teacher's shows much less._

![TartanAir held-out environment](neural-supersampling/outputs/plots/visual_tartanair.png)

_TartanAir held-out environment: the networks recover the window frames and railings that bilinear and analytic TAAU blur. The student's history weight varies with the 3x sub-pixel phase (a dot grid)._

| Prediction frame | Synthetic fast motion                                                                     | TartanAir                                                                            |
| :--------------- | :---------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| Ground truth     | ![gt](neural-supersampling/outputs/predictions/visual_synthetic_fast_ground_truth.png)    | ![gt](neural-supersampling/outputs/predictions/visual_tartanair_ground_truth.png)    |
| Bilinear         | ![bilinear](neural-supersampling/outputs/predictions/visual_synthetic_fast_bilinear.png)  | ![bilinear](neural-supersampling/outputs/predictions/visual_tartanair_bilinear.png)  |
| Analytic TAAU    | ![taau](neural-supersampling/outputs/predictions/visual_synthetic_fast_analytic_taau.png) | ![taau](neural-supersampling/outputs/predictions/visual_tartanair_analytic_taau.png) |
| `igpu` 3x        | ![igpu](neural-supersampling/outputs/predictions/visual_synthetic_fast_igpu_s3.png)       | ![igpu](neural-supersampling/outputs/predictions/visual_tartanair_igpu_s3.png)       |
| `low` 3x student | ![student](neural-supersampling/outputs/predictions/visual_synthetic_fast_student_s3.png) | ![student](neural-supersampling/outputs/predictions/visual_tartanair_student_s3.png) |
| 3x teacher       | ![teacher](neural-supersampling/outputs/predictions/visual_synthetic_fast_teacher_s3.png) | ![teacher](neural-supersampling/outputs/predictions/visual_tartanair_teacher_s3.png) |

**Section 25: ablations** (`ablations.csv`, `low` 3x, 20 minutes each). Delta to `abl_reference`:

| Ablation            | Synthetic PSNR-Y | Synthetic TCE | TartanAir PSNR-Y | TartanAir TCE | TartanAir PSNR disocc. |
| :------------------ | ---------------: | ------------: | ---------------: | ------------: | ---------------------: |
| no history          |           -1.941 |       +0.0104 |           -0.669 |       +0.0024 |                 +1.350 |
| no G-buffers        |           -2.328 |       +0.0093 |           -1.506 |       +0.0048 |                 +0.874 |
| bicubic degradation |           -6.103 |       +0.0296 |           -3.362 |       +0.0104 |                 -0.234 |
| no temporal loss    |           +0.186 |       +0.0014 |           +0.245 |       +0.0000 |                 +0.309 |
| no distillation     |           +0.236 |       -0.0004 |           +0.018 |       -0.0001 |                 +0.052 |

Reference absolute values: synthetic-game 32.73 dB, TCE 0.0122; TartanAir 32.94 dB, TCE 0.0107. Render-like jittered training (against bicubic) matters most, then G-buffers, then history. Without history the network does better inside disocclusions (no stale history to reject) but worse overall. The temporal loss costs 0.2 dB of PSNR-Y and buys 10 percent lower TCE on synthetic scenes (0.0122 against 0.0136), with no measurable effect on TartanAir. Distillation from the trained teacher again shows no gain within 20 minutes; with equal wall-clock budgets the run without a teacher also made 27 percent more steps (1,410 against 1,110).

![TSR ablations](neural-supersampling/outputs/plots/ablations.png)

_PSNR-Y change and temporal-change error change against the reference per ablation and suite._

**Section 26: latency** (`latency.csv`, fused FP16 step, batch 1, PyTorch eager on one T4; the estimate columns assume a sustained FP16 throughput and are not measurements):

| Model      | Scale | Output    | T4 mean (ms) | T4 p95 (ms) |   GMAC | Estimate at 4 TFLOPS (ms) | Estimate at 8 TFLOPS (ms) |
| :--------- | ----: | :-------- | -----------: | ----------: | -----: | ------------------------: | ------------------------: |
| igpu_s3    |     3 | 1920x1080 |        10.26 |       12.85 |   4.84 |                      2.42 |                      1.21 |
| igpu_s3    |     3 | 3840x2160 |        40.03 |       40.61 |  19.36 |                      9.68 |                      4.84 |
| student_s3 |     3 | 1920x1080 |        14.06 |       14.22 |  12.91 |                      6.46 |                      3.23 |
| student_s3 |     3 | 3840x2160 |        55.75 |       56.48 |  51.64 |                     25.82 |                     12.91 |
| student_s2 |     2 | 1920x1080 |        24.30 |       24.92 |  25.12 |                     12.56 |                      6.28 |
| teacher_s3 |     3 | 1920x1080 |        45.58 |       46.37 | 144.42 |                     72.21 |                     36.10 |
| teacher_s3 |     3 | 3840x2160 |       183.48 |      189.98 | 577.68 |                    288.84 |                    144.42 |

Means are within 4 percent of the previous run's on the older PyTorch 2.0 image.

![TSR latency](neural-supersampling/outputs/plots/latency.png)

_Mean T4 milliseconds per frame for each model and output size._

**Section 27: ONNX export** (`onnx_export.json`, opset 17, static shapes): `student_s3` 0.449 MB, ONNX Runtime max difference 5.96e-7; `igpu_s3` 0.135 MB, 5.96e-7. The export printed a tracer warning and three constant-folding warnings; the current library silences them.

**Section 28: summary** (`summary.json`, `stage_times.csv`). Stage minutes: manifest 0.05, caches 20.71, batch probe 0.47, teacher 140.44, student 120.22, pair 75.21, ablations 60.70, test suites 2.36, evaluation 6.44.

![TSR stage times](neural-supersampling/outputs/plots/stage_times.png)

_Minutes per stage of the TSR run._

**Sections 29 and 30.** Deleted 9.30 GB of caches and zipped 126 files (0.06 GB) into `outputs.zip`.

## Notebook 3: frame generation

Run of 24 October 2026 (PyTorch 2.11.0, CUDA 12.8, Python 3.13.15, 2 x Tesla T4, 4 CPU cores, 31.3 GB RAM; 6.47 hours; no failed runs). Outputs: [neural-frame-generation/outputs/](neural-frame-generation/outputs/); executed notebook: [neural-frame-generation.executed.ipynb](neural-frame-generation/outputs/neural-frame-generation.executed.ipynb). The numbers below are from the run; the current notebook adds only the per-clip tOF file and the ONNX warning filter listed under [Project status](#project-status).

**Sections 1 to 4: library, setup, hardware, configuration.** All three optional packages installed. Settings: 192-px patches, 40 percent real quads, 50 percent extrapolation, HUD in 50 percent of synthetic samples, menus 3, cuts 3 and camera reversals 10 percent; loss weights census 0.5, LPIPS 0.1, flow 0.01, UI 0.2, distillation 0.5. Budgets: teacher 100, student 120, pair 75, ablations 20 min; caches 30 min.

**Sections 5 and 6: manifest and items.** Same manifest as the TSR run (all sources reachable). 4,083 texture and 5,080 quad items for training, 252 and 850 for validation. GameIR is not used (its frames are 10 rendered frames apart).

**Section 7: caches (26.3 min, 10.22 GB).** 256 validation and 5,690 training still crops (1 decode failure; the stills' share of the cache budget ended at 5,690 of 6,000), 256 validation and 6,000 training quads, all with RAFT flows. Training quads: Vimeo-90K 5,493, REDS 252, Vimeo-1080p 142, TartanAir 70, Sintel 43; validation quads: Vimeo-90K 251, REDS 5.

**Section 8: synthetic preview.**

![Synthetic special cases](neural-frame-generation/outputs/plots/synthetic_preview.png)

_Interpolation, extrapolation with a camera reversal, HUD with a subtitle change, a menu and a scene cut: I0, I1, target, UI mask and exact target-to-I1 flow._

**Section 9: model tiers** (`model_tiers.csv`, parity 0.0 for every tier):

| Tier    | Train params | Fused params | GMAC per 1080p frame (network only) |
| :------ | -----------: | -----------: | ----------------------------------: |
| igpu    |       16,953 |       12,089 |                                1.55 |
| low     |       40,497 |       28,545 |                                3.67 |
| mid     |      134,610 |       94,546 |                               30.42 |
| high    |      342,066 |      238,770 |                               76.98 |
| teacher |      701,202 |      487,954 |                              157.51 |

**Section 10: batch probe (0.5 min).** Every size up to 96 fit; 96 per GPU was chosen (`batch_probe.csv`).

| Batch | Teacher VRAM % | Teacher samples/s | Student + teacher VRAM % | Student samples/s |
| ----: | -------------: | ----------------: | -----------------------: | ----------------: |
|     8 |            6.6 |              46.0 |                      5.4 |              54.8 |
|    16 |           11.9 |              59.3 |                      9.6 |              61.3 |
|    24 |           17.4 |              62.0 |                     13.9 |              66.7 |
|    32 |           22.7 |              64.2 |                     18.1 |              69.1 |
|    48 |           33.6 |              65.3 |                     26.6 |              70.7 |
|    64 |           44.5 |              66.0 |                     35.2 |              70.6 |
|    96 |           66.0 |              66.6 |                     52.2 |              70.5 |

**Sections 11 to 15: training** (budgets at full scale).

| Stage             | Run                                      |        Steps |      Train s | Best val PSNR (round) |
| :---------------- | :--------------------------------------- | -----------: | -----------: | :-------------------- |
| A, both GPUs      | `fg_teacher` (privileged)                |        4,560 |        6,002 | 33.92 dB (12)         |
| B, both GPUs      | `fg_low`                                 |        6,170 |        7,209 | 27.17 dB (14)         |
| C, GPU 0 / GPU 1  | `fg_igpu`, `fg_mid`                      | 3,950, 6,830 | 4,506, 4,502 | 26.94 (9), 27.79 (9)  |
| Ablations, pair 1 | `abl_reference`, `abl_no_ui_training`    | 1,040, 1,170 | 1,205, 1,203 | 26.56 (5), 28.87 (5)  |
| Ablations, pair 2 | `abl_no_motion_prior`, `abl_interp_only` | 1,030, 1,030 | 1,200, 1,205 | 24.77 (5), 28.03 (5)  |
| Ablations, pair 3 | `abl_extrap_only`, `abl_no_distillation` | 1,040, 1,120 | 1,208, 1,203 | 25.66 (4), 26.54 (5)  |

**Section 16: training curves** (`best_validation.csv`, `history_<run>.csv`):

| Run          | Val PSNR per round (dB)                                                                                 | Interp / extrap / UI at best | Train loss first to last | Peak memory (GB) |
| :----------- | :------------------------------------------------------------------------------------------------------ | :--------------------------- | :----------------------- | ---------------: |
| `fg_teacher` | 30.41, 31.97, 32.45, 32.86, 33.24, 33.48, 33.64, 33.74, 33.82, 33.86, 33.90, 33.92, 33.92               | 34.38 / 33.47 / 33.36        | 0.1856 to 0.0496         |             9.22 |
| `fg_low`     | 26.33, 26.63, 26.94, 27.04, 27.09, 27.11, 27.13, 27.14, 27.15, 27.16, 27.16, 27.16, 27.16, 27.16, 27.17 | 27.99 / 26.37 / 27.54        | 0.2039 to 0.1021         |             7.21 |
| `fg_mid`     | 26.61, 27.24, 27.45, 27.57, 27.64, 27.69, 27.73, 27.77, 27.79, 27.79                                    | 28.83 / 26.78 / 30.92        | 0.1816 to 0.0914         |             4.05 |
| `fg_igpu`    | 26.33, 26.46, 26.71, 26.86, 26.88, 26.92, 26.93, 26.93, 26.93, 26.94                                    | 27.65 / 26.25 / 25.43        | 0.2091 to 0.1182         |             7.16 |

The longer budgets added 0.33 dB (teacher), 0.29 (`low`), 0.40 (`mid`) and 0.43 (`igpu`) of best validation PSNR over the run of 20 to 21 October 2026; the `low` student gained only 0.01 dB over its last five rounds. As for TSR, each ablation validates on its own recipe (for example `abl_no_ui_training` has no UI in its validation batches), so ablation validation scores are not comparable.

![FG training curves](neural-frame-generation/outputs/plots/training_curves.png)

_Validation PSNR and train loss against minutes for every run._

**Section 17: load and fuse.** Ten models fused to FP16 (teacher, student, `igpu`, `mid`, six ablations).

**Section 18: cut detector** (`cut_detector.json`): threshold 0.1707 for 1 percent false alarms on continuous synthetic pairs; detection rate 1.0.

![Cut detector](neural-frame-generation/outputs/plots/cut_detector.png)

_Scene-cut score for continuous and cut pairs, with the chosen threshold._

**Section 19: test suites (1.9 min).** 48 synthetic test stills; 48 Vimeo-90K triplets, 10 TartanAir, 10 Sintel, 12 Vimeo-1080p and 4 Vid4 clips. Four Vimeo-1080p clips were skipped (blank, frozen or with a cut) and replaced by the next clips of that source.

**Section 20: evaluation (1.1 min).** 11,928 metric rows (`test_per_sample.csv`). TartanAir and Sintel feed the exact motion vector of I1 (frame 2 to 1 composed with 1 to 0); the video suites use RAFT.

**Section 21: results by suite, mode and method** (`test_results.csv`). The privileged teacher sees the target frame, so it is an upper bound, not a deployable model. PSNR in dB:

| Suite         | Mode   | Repeat | Average | MV reprojection | `igpu` | `low` student | `mid` | Teacher |
| :------------ | :----- | -----: | ------: | --------------: | -----: | ------------: | ----: | ------: |
| vimeo_triplet | interp |  23.06 |   26.03 |           31.83 |  32.60 |         33.04 | 33.22 |   39.29 |
| tartanair     | interp |  18.84 |   20.81 |           27.98 |  28.98 |         29.23 | 29.25 |   34.16 |
| tartanair     | extrap |  18.88 |   18.71 |           26.91 |  26.80 |         26.68 | 26.65 |   32.90 |
| sintel        | interp |  20.93 |   22.87 |           24.46 |  24.24 |         24.46 | 24.61 |   31.00 |
| sintel        | extrap |  20.95 |   20.50 |           23.29 |  23.07 |         22.98 | 23.04 |   29.84 |
| vid4          | interp |  20.69 |   24.09 |           27.35 |  27.80 |         27.96 | 28.12 |   31.81 |
| vid4          | extrap |  21.41 |   19.73 |           25.89 |  25.81 |         25.78 | 25.88 |   31.25 |
| vimeo1080p    | interp |  27.54 |   30.77 |           32.84 |  33.64 |         34.00 | 33.50 |   37.60 |
| vimeo1080p    | extrap |  29.46 |   26.79 |           30.72 |  32.50 |         32.57 | 32.24 |   37.04 |
| synthetic     | interp |  28.10 |   33.85 |           34.87 |  35.54 |         35.72 | 35.55 |   40.15 |
| synthetic     | extrap |  28.08 |   25.66 |           30.86 |  31.86 |         31.83 | 31.78 |   37.51 |

With the defective clips skipped, the teacher now leads on Vimeo-1080p as on every other suite (in the previous run, blank clips had put it below the student).

SSIM / LPIPS / interpolation error (lower is better) for the `low` student on Vimeo-90K triplet: 0.949 / 0.042 / 6.85 (MV reprojection 0.946 / 0.041 / 7.78). Occluded-region PSNR, student against MV reprojection: Vimeo-90K triplet 25.05 vs 23.43, TartanAir interpolation 32.71 vs 28.76, synthetic interpolation 25.03 vs 20.44 dB.

What the numbers say: interpolation beats motion-vector reprojection on Vimeo-90K triplet (+1.22 dB), TartanAir (+1.25), Vid4 (+0.61), Vimeo-1080p (+1.16) and synthetic scenes (+0.86), and ties on Sintel (24.455 against 24.456 dB). With exact engine motion vectors, extrapolation does not beat reprojection on TartanAir (-0.23 dB) or Sintel (-0.32), nor on Vid4 (-0.11); it beats it on synthetic scenes (+0.98) and Vimeo-1080p (+1.85). The `mid` tier, with 8 times the GMAC of the student, gains at most 0.2 dB on real suites.

![Real suites](neural-frame-generation/outputs/plots/real_results.png)

_PSNR per method on the real suites, interpolation (left) and extrapolation (right)._

**Section 22: motion, UI, menus, cuts and reversals.** Measured motion between real frames: static 0.0, slow 1.72, medium 6.65, fast 18.62 px. Synthetic PSNR by motion class (dB):

| Mode   | Method          | Static |  Slow | Medium |  Fast |
| :----- | :-------------- | -----: | ----: | -----: | ----: |
| interp | repeat          |  45.95 | 27.82 |  21.49 | 17.14 |
| interp | MV reprojection |  53.78 | 33.76 |  28.56 | 23.36 |
| interp | `igpu`          |  53.33 | 35.20 |  30.23 | 23.40 |
| interp | student         |  53.71 | 35.32 |  30.43 | 23.44 |
| interp | `mid`           |  51.05 | 36.28 |  31.24 | 23.63 |
| interp | teacher         |  56.37 | 39.06 |  36.93 | 28.26 |
| extrap | repeat          |  45.96 | 27.79 |  21.49 | 17.08 |
| extrap | MV reprojection |  42.07 | 31.94 |  27.21 | 22.20 |
| extrap | `igpu`          |  45.47 | 32.91 |  27.78 | 21.27 |
| extrap | student         |  45.59 | 32.90 |  27.68 | 21.15 |
| extrap | `mid`           |  44.89 | 33.13 |  27.86 | 21.23 |
| extrap | teacher         |  52.85 | 37.37 |  33.96 | 25.88 |

In fast extrapolation every student tier is about 1 dB below MV reprojection.

UI suite (`ui_results.csv`; ghosting = share of UI pixels off by more than 10 percent):

| Mode   | Method           |  PSNR | UI PSNR | UI ghosting rate |
| :----- | :--------------- | ----: | ------: | ---------------: |
| interp | repeat           | 25.29 |   44.47 |            0.029 |
| interp | average          | 24.51 |   28.06 |            0.290 |
| interp | MV reprojection  | 21.44 |   14.84 |            0.342 |
| interp | `igpu`           | 27.68 |   31.41 |            0.067 |
| interp | student          | 28.04 |   32.91 |            0.046 |
| interp | `mid`            | 28.91 |   35.29 |            0.036 |
| interp | student+ui_layer | 29.54 |   48.71 |            0.025 |
| interp | `igpu`+ui_layer  | 29.46 |   48.72 |            0.025 |
| interp | teacher          | 31.66 |   36.36 |            0.032 |
| extrap | repeat           | 25.96 |   54.37 |            0.020 |
| extrap | average          | 22.38 |   38.78 |            0.254 |
| extrap | MV reprojection  | 22.75 |   15.49 |            0.396 |
| extrap | `igpu`           | 27.67 |   30.86 |            0.092 |
| extrap | student          | 28.12 |   32.97 |            0.060 |
| extrap | `mid`            | 28.55 |   35.07 |            0.040 |
| extrap | student+ui_layer | 29.21 |   56.98 |            0.013 |
| extrap | `igpu`+ui_layer  | 29.24 |   56.83 |            0.013 |
| extrap | teacher          | 31.84 |   37.00 |            0.031 |

Menus: interpolation student 34.91 dB, student with the UI layer 38.68, `mid` 35.31, repeat 33.11, MV reprojection 23.24; extrapolation student 35.29, with the UI layer 37.81, repeat 34.58. Cuts: every gated network outputs the newest real frame, so its PSNR is capped at 100 dB, against 25.61 (interpolation) and 24.50 (extrapolation) for MV reprojection and about 14.8 for averaging. Camera reversal (extrapolation): student 23.87 dB without and 23.88 with the camera prior, `igpu` 23.92 / 23.92, teacher 31.00 / 31.01, frame repeat 24.86, MV reprojection 23.55, and the no-motion-prior ablation 25.08; the camera prior still makes no measurable difference (see [Known limitations](#known-limitations)).

![FG special cases](neural-frame-generation/outputs/plots/special_cases.png)

_PSNR by motion class, UI ghosting rate per method, menu and cut correctness, and camera reversal with and without the camera prior._

**Section 23: temporal consistency of the displayed sequence** (`temporal_consistency.csv`, tOF in px, lower is better):

| Suite         | Average | MV reprojection | `igpu` | Student | `mid` |
| :------------ | ------: | --------------: | -----: | ------: | ----: |
| tartanair     |   8.282 |           0.750 |  0.779 |   0.758 | 0.739 |
| vimeo1080p    |   6.368 |          11.664 | 11.758 |  10.039 | 9.114 |
| vimeo_triplet |   1.774 |           0.306 |  0.313 |   0.312 | 0.291 |

`mid` is the most temporally consistent network on all three suites and beats MV reprojection on each; the student is within 0.01 px of reprojection on TartanAir and Vimeo-90K triplet and ahead of it on Vimeo-1080p. On Vimeo-1080p frame averaging has the lowest tOF although it has the lowest PSNR of the non-repeat methods; this run saved no per-clip values to explain it, and the current notebook saves them (`temporal_consistency_per_clip.csv`).

**Section 24: visual comparisons.**

![HUD and subtitles](neural-frame-generation/outputs/plots/visual_ui.png)

_HUD and subtitle sample (interpolation): averaging blends in a subtitle plate from one real frame and fades the HUD bar, and MV reprojection drags the HUD bar and crosshair. The student's predicted UI mask covers the HUD, so the bar is copied from the newest real frame, but a faint second crosshair remains; with the engine UI layer the output is clean._

![Fast extrapolation](neural-frame-generation/outputs/plots/visual_extrap.png)

_Fast-motion extrapolation: target, baselines, student output, UI mask, blend weight and predicted flow. Both MV reprojection and the student leave a sliver of the moving object behind at its upper edge, where the background is disoccluded._

Full-size frames of both samples, saved in `predictions/`:

| Prediction frame   | HUD and subtitles (interpolation)                                                         | Fast extrapolation                                                                   |
| :----------------- | :---------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| Target             | ![target](neural-frame-generation/outputs/predictions/visual_ui_target.png)               | ![target](neural-frame-generation/outputs/predictions/visual_extrap_target.png)      |
| Repeat             | ![repeat](neural-frame-generation/outputs/predictions/visual_ui_repeat.png)               | ![repeat](neural-frame-generation/outputs/predictions/visual_extrap_repeat.png)      |
| Average            | ![average](neural-frame-generation/outputs/predictions/visual_ui_average.png)             | ![average](neural-frame-generation/outputs/predictions/visual_extrap_average.png)    |
| MV reprojection    | ![mv](neural-frame-generation/outputs/predictions/visual_ui_mv_reprojection.png)          | ![mv](neural-frame-generation/outputs/predictions/visual_extrap_mv_reprojection.png) |
| Student            | ![student](neural-frame-generation/outputs/predictions/visual_ui_student.png)             | ![student](neural-frame-generation/outputs/predictions/visual_extrap_student.png)    |
| Student + UI layer | ![student ui](neural-frame-generation/outputs/predictions/visual_ui_student_ui_layer.png) | Not applicable (no UI in this sample)                                                |

**Section 25: ablations** (`ablations.csv`, `low` tier, 20 minutes each). PSNR in dB:

| Ablation        | Vimeo-90K triplet interp | TartanAir interp | TartanAir extrap | Synthetic interp | Synthetic extrap | UI ghosting interp |
| :-------------- | -----------------------: | ---------------: | ---------------: | ---------------: | ---------------: | -----------------: |
| reference       |                    31.84 |            28.21 |            26.83 |            33.03 |            31.64 |              0.073 |
| no UI training  |                    33.68 |            29.52 |            26.76 |            36.86 |            32.11 |              0.176 |
| no motion prior |                    26.01 |            20.44 |            19.06 |            31.88 |            27.42 |              0.046 |
| interp only     |                    32.81 |            28.84 |            26.58 |            35.92 |            30.06 |              0.073 |
| extrap only     |                    30.95 |            27.48 |            26.76 |            32.18 |            31.91 |              0.088 |
| no distillation |                    31.92 |            28.21 |            26.93 |            33.27 |            31.55 |              0.066 |

Engine motion vectors are the most valuable input (5.8 to 7.8 dB on real suites). Training with UI costs 1.8 dB on Vimeo-90K triplet and 3.8 dB on synthetic interpolation without UI, but is what keeps UI ghosting low (0.073 against 0.176); the no-motion-prior model's low ghosting comes from barely moving anything (it also has the best camera-reversal score, close to frame repeat). One model for both modes costs 1.0 dB of interpolation quality on Vimeo-90K triplet against an interpolation-only model, while the extrapolation-only model is no better than the shared one on TartanAir extrapolation. Distillation from the privileged teacher shows no gain in this run (the run without a teacher is between 0.09 dB worse and 0.24 dB better than the reference, with 8 percent more steps); the 0.2 to 0.4 dB gain of the previous run did not repeat.

![FG ablations](neural-frame-generation/outputs/plots/ablations.png)

_PSNR change against the reference per ablation, interpolation and extrapolation, all suites._

**Section 26: latency and pacing model** (`latency.csv`, `pacing_model.csv`). T4 eager PyTorch, fused FP16, batch 1:

| Model   | Output    | T4 mean (ms) | T4 p95 (ms) |  GMAC | Estimate at 4 TFLOPS (ms) |
| :------ | :-------- | -----------: | ----------: | ----: | ------------------------: |
| igpu    | 1920x1080 |        15.32 |       17.02 |  1.55 |                      0.78 |
| igpu    | 2560x1440 |        26.57 |       26.75 |  2.76 |                      1.38 |
| student | 1920x1080 |        16.09 |       16.19 |  3.67 |                      1.84 |
| student | 2560x1440 |        28.33 |       28.50 |  6.52 |                      3.26 |
| mid     | 1920x1080 |        37.56 |       38.10 | 30.42 |                     15.21 |
| mid     | 2560x1440 |        66.55 |       67.02 | 54.08 |                     27.04 |

The pacing model is a simulation (`nss_metrics.simulate_latency`), driven by the measured 16.09 ms student time:

| Base render (ms) | Mode        | Displayed fps | Game latency (ms) | Camera latency (ms) | Content age (ms) |
| ---------------: | :---------- | ------------: | ----------------: | ------------------: | ---------------: |
|             33.3 | native      |         30.01 |             38.33 |               38.33 |            38.33 |
|             33.3 | interp      |         40.48 |             79.13 |               54.42 |            66.77 |
|             33.3 | extrap      |         40.48 |             38.33 |               50.69 |            50.69 |
|             33.3 | extrap_late |         40.48 |             38.33 |               27.71 |            50.69 |
|             25.0 | native      |         39.97 |             30.02 |               30.02 |            30.02 |
|             25.0 | interp      |         48.66 |             66.67 |               46.11 |            56.39 |
|             25.0 | extrap      |         48.66 |             30.02 |               40.30 |            40.30 |
|             25.0 | extrap_late |         48.66 |             30.02 |               23.56 |            40.30 |
|             16.7 | native      |         59.84 |             21.72 |               21.72 |            21.72 |
|             16.7 | interp      |         60.98 |             54.20 |               37.80 |            46.00 |
|             16.7 | extrap      |         60.98 |             21.72 |               29.92 |            29.92 |
|             16.7 | extrap_late |         60.98 |             21.72 |               19.40 |            29.92 |

With a 16.1 ms generation time on the same GPU the displayed rate rises only to 40 fps from a 30 fps base, and interpolation multiplies game latency by 2.1 to 2.5; extrapolation keeps native game latency, and late-latched extrapolation has the lowest camera latency.

![Pacing model](neural-frame-generation/outputs/plots/pacing_model.png)

_Displayed frame rate against game input latency (left) and camera latency (right) for each mode and base render time._

**Section 27: ONNX export** (`onnx_export.json`): student 0.325 MB, ONNX Runtime max difference 3.22e-6; `igpu` 0.259 MB, 2.44e-6. The export printed a tracer warning; the current library silences it.

**Section 28: summary** (`summary.json`, `stage_times.csv`). Stage minutes: manifest 0.20, caches 26.34, batch probe 0.48, teacher 100.35, student 120.33, pair 75.23, ablations 60.71, test suites 1.93, evaluation 1.11. The summary covers every real suite in both modes (Vimeo-90K triplets in interpolation only).

![FG stage times](neural-frame-generation/outputs/plots/stage_times.png)

_Minutes per stage of the FG run._

**Sections 29 and 30.** Deleted 10.22 GB of caches and zipped 125 files (0.03 GB) into `outputs.zip`.

## Weights and exported models

Each training run saved three PyTorch checkpoints in its notebook's `outputs/weights/`: `<run>_best.pt` (EMA weights of the best validation round), `<run>_last.pt` (model and EMA weights of the last round) and `<run>_fused.pt` (RepConv folded into plain convolutions, FP16). They are research weights trained on research-use data (see [RESEARCH.md](RESEARCH.md) section 16). All are from the runs of 24 October 2026.

| File                                                             | Size (best / last / fused)      | Purpose                                                    |
| :--------------------------------------------------------------- | :------------------------------ | :--------------------------------------------------------- |
| `neural-supersampling/outputs/weights/tsr_low_s3_*.pt`           | 539 / 1,075 / 204 KB            | Main 3x TSR student                                        |
| `neural-supersampling/outputs/weights/tsr_igpu_s3_*.pt`          | 122 / 242 / 48 KB               | Integrated-GPU 3x TSR student                              |
| `neural-supersampling/outputs/weights/tsr_low_s2_*.pt`           | 503 / 1,005 / 189 KB            | 2x TSR student                                             |
| `neural-supersampling/outputs/weights/tsr_teacher_s3_*.pt`       | 7,544 / 15,087 / 2,702 KB       | 3x TSR teacher (distillation source)                       |
| `neural-supersampling/outputs/weights/abl_*`                     | about 539 / 1,076 / 205 KB each | Six TSR ablation runs                                      |
| `neural-supersampling/outputs/weights/tsr_student_s3_fused.onnx` | 439 KB                          | ONNX, one recurrent step of the 3x student, 64x36 input    |
| `neural-supersampling/outputs/weights/tsr_igpu_s3_fused.onnx`    | 132 KB                          | ONNX, one recurrent step of the `igpu` student             |
| `neural-frame-generation/outputs/weights/fg_low_*.pt`            | 173 / 343 / 62 KB               | Main FG student                                            |
| `neural-frame-generation/outputs/weights/fg_igpu_*.pt`           | 79 / 156 / 29 KB                | Integrated-GPU FG student                                  |
| `neural-frame-generation/outputs/weights/fg_mid_*.pt`            | 554 / 1,106 / 198 KB            | Two-level FG model                                         |
| `neural-frame-generation/outputs/weights/fg_teacher_*.pt`        | 2,777 / 5,552 / 970 KB          | Privileged teacher (needs the target frame; training only) |
| `neural-frame-generation/outputs/weights/abl_*`                  | about 173 / 344 / 62 KB each    | Six FG ablation runs                                       |
| `neural-frame-generation/outputs/weights/fg_student_fused.onnx`  | 317 KB                          | ONNX, FG student, 96x64 input                              |
| `neural-frame-generation/outputs/weights/fg_igpu_fused.onnx`     | 253 KB                          | ONNX, FG `igpu`                                            |

The ONNX files use static shapes (the small export size); re-export at the target resolution with `nss_models.export_onnx` for deployment. The changes made after the runs do not touch the model definitions, so these weights load into the current models.

## Known limitations

A full list with reasons is in [RESEARCH.md](RESEARCH.md) section 15. The most important ones, from the runs:

- **The TSR students lose quality on long static sequences.** With a static camera they peak around frames 6 to 10 and lose 1.3 to 1.4 dB by frame 15, while the teacher keeps improving; training unrolls only 6 frames (section 23 of the TSR notebook). Training on longer unrolls, or carrying the recurrent state across training batches, is the likely fix and needs a new Kaggle run to verify.
- **The 3x student shows a faint noise or dot pattern at the pixel-shuffle period** on flat areas (section 24 of the TSR notebook); the teacher shows much less of it.
- **TSR temporal stability trails the analytic TAA upscaler on real suites**: the 3x student's temporal-change error is 2 to 7 percent higher on TartanAir, Sintel, Vid4 and Vimeo-1080p (it is lower on synthetic scenes).
- **The FG camera prior is unused.** `FGNet.priors` replaces the linear-motion flow with the camera flow only when no motion vector is available; with motion vectors, the network learned to ignore the camera channel, so late latching does not improve camera reversals (frame repeat is better there).
- **With exact engine motion vectors, FG extrapolation does not beat plain motion-vector reprojection** on TartanAir, Sintel or Vid4, and is about 1 dB behind it on fast synthetic motion.
- **Distillation gives no measurable gain** in 20-minute ablations, for TSR (in both runs) and FG (in this run).
- **FG Vimeo-1080p tOF ranks frame averaging first**, against every image metric; the cause is not established (per-clip values are saved from the next run).
- **T4 latencies are PyTorch eager timings**, 2.5x to 20x the arithmetic estimates for FG; in-engine latency on target GPUs is `Not measured in the current repository.`
- **Each configuration ran once**, so run-to-run variance is `Not measured in the current repository.`; differences of a few tenths of a dB between runs or ablations may be noise.
- **Validation is not comparable across ablations**, and TSR validation has no exact-motion source (TartanAir and Sintel have no validation items).
- **Real training data is mostly Vimeo-90K** (about 91 percent of real training windows and quads), and only about 2 percent of TSR windows have exact motion.
- **No comparison with FSR, DLSS or XeSS** yet; it needs engine captures.

## Troubleshooting

| Problem                                                                     | Likely cause                                                          | Diagnostic                                                                       | Resolution                                                                                                |
| :-------------------------------------------------------------------------- | :-------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------- |
| `build_notebooks.py --check` lists a stale notebook                         | A file in `src/neuralss/` or `notebooks/` changed without a rebuild   | `.venv\Scripts\python tools\build_notebooks.py --check`                          | Run `tools\build_notebooks.py`; copy an executed notebook to `outputs/` first if its outputs must be kept |
| A training run is listed in `metrics/failed_runs.json`                      | Out of memory twice, a non-finite loss for 200 steps, or a crash      | Read `outputs/logs/<run>.log` (and `<run>.oom.log` after an out-of-memory retry) | Lower the batch, check the loss terms; later stages continue without that run                             |
| A run is killed past its budget                                             | Hung data-loader worker or NCCL collective                            | The run log ends with a timeout line from the watchdog                           | Re-run; reduce `workers_per_rank` in `run_cfg` if it repeats                                              |
| `evaluation <suite> / <method> skipped` in the notebook log                 | That method failed on that suite (for example out of memory)          | `outputs/logs/notebook.log`                                                      | Other methods and suites still run; the skipped rows are missing from `test_per_frame.csv`                |
| `<source> <item>: skipped (blank)` (or `frozen`, `cut`) in the notebook log | The test clip cannot measure the method (see `nss_eval.clip_defects`) | `outputs/logs/notebook.log`                                                      | Expected; the suite loads the next clip of that source (at most 4 n clips are tried)                      |
| Sources left out after the manifest cell                                    | Their Kaggle dataset is not attached or did not mount                 | `metrics/reachable_sources.json`                                                 | Attach the dataset and run again                                                                          |
| `ImportError: nbformat` locally                                             | Local tooling not installed                                           | `.venv\Scripts\python -c "import nbformat"`                                      | `.venv\Scripts\python -m pip install nbformat --cache-dir .cache\pip`                                     |

## Licence

The repository has no licence file, so no licence is granted for the code. Dataset and dependency licences, and the policy that released weights must be retrained on permissively licensed data, are in [RESEARCH.md](RESEARCH.md) section 16.
