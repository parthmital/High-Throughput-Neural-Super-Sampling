# NeuralSS: open, hardware-agnostic neural reconstruction for games

Technical research plan for an open neural rendering and reconstruction stack that combines **temporal super-resolution** and **low-latency frame generation**, aims at native-quality output in static scenes and heavy motion, and runs on dedicated and integrated GPUs through vendor-neutral APIs.

**Status (5 October 2026).**

- The repository holds three Kaggle notebooks built from one shared library (`src/neuralss`): a data pipeline, temporal super-resolution (TSR) and frame generation (FG).
- They have passed static checks only: notebook validation, a Kaggle-structure check, pyflakes, and cross-module API and signature checks.
- **None of the new notebooks has run on Kaggle yet.** Every quality, speed and latency number in this document is a target, an arithmetic estimate, or a measurement explicitly attributed to an earlier run.
- Labels: _fact_ (with a source), _estimate_ (arithmetic), _hypothesis_ (to be tested).

## 1. Objective

Build an original, open and hardware-agnostic replacement for DLSS / FSR / XeSS-class reconstruction:

1. **Temporal super-resolution.** Reconstructs frames from jittered low-resolution renders plus engine metadata (motion vectors, depth, exposure, reactive / transparency masks, normals where available).
   - Primary ratio 3x (640x360 to 1920x1080, matching FSR's Ultra Performance ratio); 2x and 1.5x also validated.
   - Must be stable under heavy motion, recover detail in static scenes, and handle disocclusions, foliage, particles, transparencies, thin geometry and text.
2. **Frame generation.** Minimises input latency rather than maximising frames per second.
   - Interpolation and extrapolation (with late-latched camera input).
   - UI-correct: never interpolates or hallucinates HUD, subtitles, menus or pause screens.
   - Safe on camera cuts.
3. **Deployment.** Both networks reduce to plain 3x3 convolution chains plus warps, exportable through ONNX to DirectML, Vulkan compute (ncnn), Metal and TensorRT, with tiers from integrated GPUs to high-end discrete GPUs.

We do not reproduce DLSS, FSR or XeSS internals. The design draws on public research and open documentation (sections 3 and 20), and on what the open FidelityFX documentation says an engine must provide.

## 2. Constraints

| Constraint                                                                                                                                 | Consequence for the design                                                                                                                                                                                                 |
| :----------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training runs on Kaggle **GPU T4 x2** (2 x 15 GB, FP16 tensor cores, no fast BF16, 4 CPU cores, about 30 GB RAM, about 20 GB saved output) | Small models; FP16 autocast; GPU-side data generation; memory-mapped caches; wall-clock schedules; each notebook fits a 9.5-hour run budget, under the 12-hour session limit Kaggle documents (re-check the current quota) |
| Data only from **Kaggle and Hugging Face** for now                                                                                         | Game G-buffer data is scarce, so we use rendered datasets with exact flow and depth (TartanAir, Sintel, GameIR) plus a synthetic layered renderer with exact ground truth (section 5.5)                                    |
| Deployment on **integrated GPUs**                                                                                                          | Network cost is reported per output frame (GMAC); the `igpu` tiers target about 5 GMAC for TSR at 1080p 3x and under 2 GMAC for FG (section 4.5)                                                                           |
| Vendor-neutral runtime                                                                                                                     | Plain convolutions, PReLU, pixel (un)shuffle, bilinear grid sampling: operators available in DirectML, Vulkan compute and ONNX opset 16+ (`GridSample`)                                                                    |
| Legal                                                                                                                                      | Original architecture; only open-licensed code and weights as dependencies (section 16); research-only datasets are kept out of any commercial weight release                                                              |

## 3. Competitive analysis

### 3.1 Commercial and open upscalers and frame generators

| System                                                        | Public facts                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Lessons for NeuralSS                                                                                                                                                                                                          |
| :------------------------------------------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **NVIDIA DLSS 4**                                             | Transformer models for Super Resolution and Ray Reconstruction with "four times the computations and twice the number of parameters" of the previous CNN at a similar budget. Multi Frame Generation produces three frames per rendered frame. Its network is split: one half runs once per input frame pair, a much smaller half once per generated frame (about 1 ms per generated 4K frame on an RTX 5090). Pacing is moved to the display engine (flip metering) ([NVIDIA research](https://research.nvidia.com/labs/adlr/DLSS4)). Frame generation needs depth, motion vectors, HUD-less colour and UI colour, plus Reflex ([integration guide](https://developer.nvidia.com/blog/how-to-integrate-nvidia-dlss-4-into-your-game-with-nvidia-streamline)).                                                                                                                                                                    | Split FG into a per-pair part and a cheap per-frame part; separate UI from the scene; pacing matters as much as the model; closed weights and RTX-only, so it is a quality reference rather than a baseline we can reproduce. |
| **NVIDIA Reflex 2 Frame Warp**                                | Re-samples the camera from the latest input and warps the rendered frame just before display. Holes are in-painted from previous frames' camera, colour and depth. Up to 75 percent lower system latency in the examples given ([NVIDIA](https://www.nvidia.com/en-us/geforce/news/reflex-2-even-lower-latency-gameplay-with-frame-warp/)).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Late-latching the camera is the biggest latency lever; NeuralSS extrapolation takes a camera-only prior from the latest input (section 10).                                                                                   |
| **AMD FSR 3.1** (open, FidelityFX SDK)                        | The upscaler takes colour, depth, motion vectors, optional reactive and transparency/composition masks and exposure. Jitter uses a Halton or uniform sequence with `ceil(scale^2 / 2)` phases, with mip bias `-log2(scale)` ([FSR 3 upscaler manual](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/)). Frame interpolation uses two back buffers, dilated depth and motion vectors, its own optical flow with scene-change detection, and an optional HUD-less colour buffer ([FSR frame interpolation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/frame-interpolation/)). The swap chain composites UI by callback, separate surface or HUD-less detection. It paces frames on dedicated threads, and recommends rendering slightly below half the target output rate ([swap chain manual](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/frame-interpolation-swap-chain/)). | This documented engine contract defines our inputs: jittered colour, depth, motion vectors, exposure and reactive masks for TSR; HUD-less frames plus a UI layer for FG, with a fallback when the UI cannot be separated.     |
| **AMD FSR "Redstone" (FSR 4 upscaling, ML frame generation)** | ML upscaler "trained on millions of high-quality captures from modern games". ML frame generation takes previous and current frames with depth and motion vectors, estimates optical flow, predicts per-pixel motion and appearance, and blends with motion-vector reprojection. It should be used with Anti-Lag 2 "to minimize the latency that frame generation inherently introduces". The ML path targets RDNA 4, with analytical fallbacks on older GPUs ([GPUOpen, Dec 2025](https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/)).                                                                                                                                                                                                                                                                                                                                                                     | Blending learned motion with engine reprojection is our FG design too; vendor ML paths need specific hardware, which is the gap a vendor-neutral stack fills.                                                                 |
| **Intel XeSS 2 / 3**                                          | XeSS-SR runs on all GPUs with SM 6.4 (DP4a); XeSS-FG and Xe Low Latency (XeLL) also run on non-Intel GPUs meeting SM 6.4 ([intel/xess](https://github.com/intel/xess)). XMX matrix engines on Arc, DP4a elsewhere ([press summary](https://www.guru3d.com/story/9ae754096c391d2ec34bec425201ec214cc93ca6/)).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | A DP4a (INT8) fallback makes neural reconstruction cross-vendor; INT8 quantisation is on our roadmap (section 13). Weights are closed.                                                                                        |

### 3.2 Research systems

| Work                                                                                                                                                          | Idea relevant to NeuralSS                                                                                                                                                                                                                   |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| NSRR, Xiao et al., SIGGRAPH 2020 ([Meta](https://research.facebook.com/blog/2020/7/introducing-neural-supersampling-for-real-time-rendering/))                | Colour, depth and dense motion vectors of the current and previous frames; 16x supersampling; trained against high-resolution anti-aliased references.                                                                                      |
| Thomas et al., HPG 2022 ([EG library](https://diglib.eg.org/handle/10.1145/3543870))                                                                          | Joint denoising and supersampling, with a shared low-precision feature extractor and higher-precision filter stages; temporally stable.                                                                                                     |
| Mercier et al., ICCV 2023, QRISP dataset ([arXiv](https://arxiv.org/abs/2308.01483))                                                                          | Efficient neural supersampling, about 4x more efficient than prior work at equal accuracy. Dataset with jittered and mip-biased renders at several resolutions, depth and motion vectors (research licence; not on Kaggle or Hugging Face). |
| FuseSR, SIGGRAPH Asia 2023 ([arXiv](https://arxiv.org/abs/2310.09726))                                                                                        | Cheap high-resolution G-buffers fused with low-resolution colour (H-Net); 4x4 and 8x8 upsampling at 4K.                                                                                                                                     |
| RDG, CVPR 2025 ([code, Apache-2.0](https://github.com/sunny2109/RDG))                                                                                         | Asymmetric U-Net with decoupled G-buffer guidance for real-time rendering; Blender-rendered dataset.                                                                                                                                        |
| ExtraNet, SIGGRAPH Asia 2021 ([paper](https://sites.cs.ucsb.edu/~lingqi/publications/paper_extranet.pdf))                                                     | Frame extrapolation with G-buffers of the extrapolated frame; hole in-painting and shading prediction; 1.5x to 2x frame rate with minimal latency.                                                                                          |
| ExtraSS, SIGGRAPH Asia 2023 ([paper](https://sites.cs.ucsb.edu/~lingqi/publications/paper_siga23extrass.pdf), [DOI](https://doi.org/10.1145/3610548.3618224)) | Joint super sampling and frame extrapolation; G-buffer guided warping plus shading refinement; no added interpolation latency.                                                                                                              |
| STSS, AAAI 2024 ([arXiv](https://arxiv.org/abs/2312.10890))                                                                                                   | Unified space-time supersampling: aliasing and warping holes are treated as one reshading problem; about 4 ms against 17 ms for two-stage pipelines.                                                                                        |
| GFFE, 2024 ([arXiv](https://arxiv.org/abs/2406.18551))                                                                                                        | G-buffer-free frame extrapolation for low-latency rendering; motion analysis, disocclusion handling and a light shading-correction network.                                                                                                 |
| Mob-FGSR, SIGGRAPH 2024 ([paper](https://sites.cs.ucsb.edu/~lingqi/publications/paper_sig24mobfgsr.pdf))                                                      | Frame generation and super-resolution for mobile GPUs; motion vectors reconstructed by splatting, without optical-flow hardware.                                                                                                            |
| RIFE, ECCV 2022 ([arXiv](https://arxiv.org/abs/2011.06294), [code, MIT](https://github.com/hzwer/Practical-RIFE))                                             | Intermediate-flow estimation; privileged distillation (the teacher sees the true middle frame).                                                                                                                                             |
| Discontinuity-aware VFI, CVPR 2023 ([arXiv](https://arxiv.org/abs/2202.07291))                                                                                | Figure-text mixing augmentation and a discontinuity map so static UI and text are not interpolated; a benchmark of game and chat videos.                                                                                                    |
| FlashVSR, 2025 ([arXiv](https://arxiv.org/abs/2510.12747), [code, Apache-2.0](https://github.com/OpenImagingLab/FlashVSR))                                    | One-step diffusion streaming VSR, about 17 FPS at 768x1408 on an A100; far too heavy for games, but a possible offline teacher.                                                                                                             |
| TecoGAN, SIGGRAPH 2020 ([arXiv](https://arxiv.org/abs/1811.09393))                                                                                            | Temporal metrics (tOF, tLP) and a ping-pong loss for long-term consistency.                                                                                                                                                                 |

### 3.3 What we take from this

- **Engine signals.** All production systems rely on jitter, motion vectors, depth and separated UI. NeuralSS consumes the same signals and degrades gracefully when they are missing.
- **Speed.** Fast systems reuse history (TSR) and split per-pair from per-frame work (FG).
- **Latency.** Interpolation adds latency by construction; vendors offset it with Reflex, Anti-Lag 2 and XeLL. Extrapolation (ExtraSS, GFFE) and late-latched warping (Frame Warp) avoid holding frames. NeuralSS supports both and reports their latency separately (section 10).
- **Hardware.** Vendor ML paths are tied to tensor, XMX or RDNA 4 hardware; DP4a and plain FP16 compute are the cross-vendor path.

## 4. Architecture

### 4.1 Per-frame pipeline (engine integration target)

```text
engine renders jittered LR colour + depth + motion vectors (+ reactive mask, exposure)
  -> TSR step (LR space): history weight + residual, pixel shuffle to HR, history buffers updated
  -> post-processing / tone mapping at HR
  -> [optional] FG: interpolation (holds the real frame) or extrapolation (holds nothing, late camera latch)
  -> UI composited on every displayed frame (real and generated)
  -> paced present on a variable-refresh display
```

The TSR state (previous output, a small hidden state and the previous depth) lives in ping-pong buffers. It is reset on scene cuts, resolution changes and camera teleports.

### 4.2 Temporal super-resolution network

Inputs are at render resolution `h x w`; the output is `s` times larger.

| Input                                                                   | Channels         | Source in training                                                 |
| :---------------------------------------------------------------------- | :--------------- | :----------------------------------------------------------------- |
| Jittered colour (one sample per pixel, Halton 2,3 offset)               | 3                | Synthetic renderer, or jittered single-sample reads of real frames |
| Depth, normalised inverse-depth style `1 / (1 + d / median)`            | 1                | TartanAir, GameIR, synthetic layers                                |
| Normals derived from depth (central differences)                        | 3                | Derived on the fly                                                 |
| Motion vectors, current to previous frame, render pixels                | 2                | Exact (synthetic, TartanAir, Sintel) or RAFT (video)               |
| Depth disocclusion cue `                                                | d - warp(d_prev) | / d`                                                               | 1   | Derived |
| Reactive mask (transparent effects without motion vectors)              | 1                | Synthetic particles                                                |
| Exposure (log2 gain) and jitter offset                                  | 1 + 2            | Augmentation and sequence                                          |
| Flags: depth available, motion quality (1 exact, 0.5 estimated, 0 none) | 2                | Per sample                                                         |
| Previous output warped by upsampled motion vectors, space-to-depth      | 3 s^2            | Recurrent                                                          |
| `abs(avgpool(warped history) - colour)`                                 | 3                | Recurrent                                                          |
| Warped recurrent hidden state                                           | 8 to 32          | Recurrent                                                          |

**Trunk.** RepConv blocks: 3x3 + 1x1 + identity + fixed Sobel / Laplacian filters with a learnable 1x1 projection, after [ECBSR](https://github.com/xindongzhang/ECBSR) (Apache-2.0). They are folded into one 3x3 kernel on every forward pass (online re-parameterisation), so training costs one convolution per block and deployment is a plain chain. An optional U-Net level at half resolution (strided convolution down, pixel shuffle up) enlarges the receptive field for fast motion and disocclusions.

**Output.** `out = a * warped_history + (1 - a) * jitter-aware bilinear upsample + residual`. The per-pixel history weight `a` and the residual are predicted at low resolution and unfolded by pixel shuffle.

- This is a learned version of temporal accumulation with rectification: `a` is driven toward 0 at disocclusions and particles and toward 1 on static content, where jittered samples accumulate toward native detail.
- The first frame after a reset uses `a = 0`.

### 4.3 Frame-generation network

- **Inputs.** Real frames I0 (t = 0) and I1 (t = 1); target time tau = 0.5 (interpolation) or 1.5 (extrapolation).
- **Motion priors.** The engine motion vector of I1 (I1 to I0) gives linear-motion flows `F(tau->0) = tau * M1` and `F(tau->1) = (tau - 1) * M1`. For extrapolation, the camera-only flow from the target to I1 is computed from the latest camera input. Depth of I1 is optional. Availability flags let one model serve engine and post-process integrations.
- **Levels.** At 1/4 and then 1/2 resolution, a RepConv trunk predicts flow corrections, a blend weight, a residual and a UI / static-overlay logit.
- **Output.** `out = u * I1 + (1 - u) * (m * warp(I0) + (1 - m) * warp(I1) + residual)`. Pixels classified as UI are copied from the newest real frame instead of being interpolated.
- **Cuts.** A cut detector (luma histogram plus motion-compensated difference, threshold calibrated for 1 percent false alarms) bypasses generation and repeats the newest real frame.

### 4.4 Teacher networks

- **TSR teacher.** The same architecture at 64 channels, 8 blocks and a 6-block U-Net level (about 1.37M parameters, estimate). It is trained first; students learn from its outputs as well as from ground truth.
- **FG privileged teacher.** It also sees the true target frame, following RIFE's privileged distillation; its flows and outputs supervise the students.

### 4.5 Tiers

Arithmetic estimates from the definitions in `src/neuralss/nss_models.py` (`.agent-local/scripts/tier_arithmetic.py`). The notebooks measure the same numbers with forward hooks.

| TSR tier  | Width / blocks / U-Net blocks / hidden | Fused params (3x) | GMAC per 1080p frame, 3x (2x) | Intended hardware   |
| :-------- | :------------------------------------- | :---------------- | :---------------------------- | :------------------ |
| `igpu`    | 16 / 3 / 0 / 8                         | 21K               | 4.8 (8.3)                     | Integrated GPUs     |
| `low`     | 24 / 4 / 2 / 8                         | 99K               | 12.9 (25.1)                   | Entry discrete GPUs |
| `mid`     | 32 / 4 / 3 / 16                        | 208K              | 24.0 (48.7)                   | Mid-range           |
| `high`    | 48 / 6 / 4 / 16                        | 567K              | 62.6 (133.0)                  | High-end            |
| `teacher` | 64 / 8 / 6 / 32                        | 1.37M             | 144.4 (314.5)                 | Training only       |

| FG tier | Width / levels / blocks | Fused params | GMAC per generated 1080p frame (network only) |
| :------ | :---------------------- | :----------- | :-------------------------------------------- |
| `igpu`  | 16 / 1 / 3              | 12K          | 1.55                                          |
| `low`   | 24 / 1 / 4              | 29K          | 3.7                                           |
| `mid`   | 32 / 2 / 4              | 95K          | 30.4                                          |
| `high`  | 48 / 2 / 5              | 239K         | 77.0                                          |

Arithmetic latency at an assumed sustained FP16 throughput T (TFLOPS) is `2 * GMAC / T` ms. For example, the TSR `igpu` tier at 3x costs about 2.4 ms at 4 TFLOPS sustained (estimate). Real latency on target GPUs must be measured through the DirectML / Vulkan path; T4 PyTorch timings only rank variants.

## 5. Dataset strategy

### 5.1 Principles

1. **High-resolution sources only as targets.** Low-resolution inputs are generated (jittered render-like sampling), except where a native low-resolution render exists (GameIR 720p) for generalisation tests.
2. **Game-rendering metadata wherever it exists.** Exact flow, occlusion and depth (TartanAir, Sintel), depth and native multi-resolution renders (GameIR), plus a synthetic layered renderer that produces every engine signal exactly.
3. **Natural video for motion diversity**, with RAFT teacher motion marked as estimated.
4. **One manifest for both tasks**, with leakage groups and perceptual deduplication across every source (section 6).

### 5.2 Inventory (checked 5 October 2026 through the public Kaggle and Hugging Face APIs)

| Source              | Host / id                                           | Content and metadata                                                                                                                                                                                   | Licence as stated by the host                                                                                                     | Role                                                                                     |
| :------------------ | :-------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| DIV2K               | Kaggle `soumikrakshit/div2k-high-resolution-images` | 900 2K images (`DIV2K_train_HR`, `DIV2K_valid_HR`)                                                                                                                                                     | "Unknown" on Kaggle; DIV2K terms are academic                                                                                     | Textures; valid set pinned to test                                                       |
| Flickr2K            | Kaggle `daehoyang/flickr2k`                         | 2,650 2K images                                                                                                                                                                                        | "Unknown"; Flickr licences vary                                                                                                   | Textures                                                                                 |
| REDS                | Kaggle `amithkesavmrajagiri/reds-dataset`           | 720p sequences (`train_sharp/hr`)                                                                                                                                                                      | REDS is CC BY 4.0 ([HF mirror tag](https://huggingface.co/datasets/snah/REDS))                                                    | Real sequences (RAFT motion)                                                             |
| Vimeo-90K septuplet | Kaggle `wangsally/vimeo-90k-7`                      | 448x256 seven-frame clips                                                                                                                                                                              | "Unknown"                                                                                                                         | Real sequences and FG quads                                                              |
| Vimeo-90K triplet   | Kaggle `chenshu123/vimeo-triplet`                   | 448x256 triplets (official test list used when present)                                                                                                                                                | "Unknown"                                                                                                                         | FG test (capped at 1,000 items)                                                          |
| MPI Sintel          | Kaggle `artemmmtry/mpi-sintel-dataset`              | Clean and final passes, `.flo` forward flow, occlusion masks (6.2 GB)                                                                                                                                  | CC BY 3.0 on Kaggle; the original site uses its own licence terms (to be confirmed before any release)                            | Exact-motion sequences; `market` and `cave` scene families test-only                     |
| Vid4                | Kaggle `uom200647r/vid4-dataset`                    | 4 test sequences                                                                                                                                                                                       | Apache 2.0 on Kaggle                                                                                                              | Test only                                                                                |
| Game frames         | HF `ericphann/video-game-super-resolution`          | 14,431 1080p game frames (Unreal-style scenes) with 480x270 copies (only the 1080p side is used); stills, not sequences                                                                                | Apache-2.0 tag (captures of third-party content: research use)                                                                    | Game textures; `test-hr` test-only                                                       |
| TartanAir           | HF `theairlabcmu/tartanair`                         | Unreal Engine environments: RGB, depth, forward optical flow, flow masks (0 = valid; non-zero = occluded or out of view), 640x480, per-environment zips                                                | BSD-3-Clause ([HF](https://huggingface.co/datasets/theairlabcmu/tartanair), [tools](https://github.com/castacks/tartanair_tools)) | Exact-motion game-like sequences with depth; `japanesealley` and `seasidetown` test-only |
| GameIR              | HF `LLLebin/GameIR`                                 | CARLA (Unreal Engine 4) clips with native 720p and 1440p renders, depth (CARLA RGB encoding), segmentation and camera data; frames 10 rendered frames apart; mini split about 11 GB train, 1.5 GB test | MIT ([HF](https://huggingface.co/datasets/LLLebin/GameIR), [paper](https://arxiv.org/abs/2408.16866))                             | Game textures and sequences with depth; town 05 test-only (native-render generalisation) |
| Vimeo-1080p         | HF `danjacobellis/vimeo1080p`                       | 3,000 train and 200 validation 1080p MP4 videos in parquet shards (about 24 GB)                                                                                                                        | Not stated                                                                                                                        | 1080p real sequences; validation pinned to test                                          |

**Considered and not used now.**

| Candidate                                                                                                                                                            | Reason                                                                                                                                                                       |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Origin Lab game recordings v4 (HF): 1080p60 RGB with and without HUD, depth, normals, inputs                                                                         | Evaluation-only sample terms, no redistribution, 30-day deletion, 5.3 TB ([HF](https://huggingface.co/datasets/originlab/game-recordings-v4))                                |
| Generative World Renderer (Cyberpunk 2077 and Black Myth: Wukong, 4M frames with G-buffers)                                                                          | Announced gated release under CC BY-NC-SA 4.0; no Hugging Face id in the paper ([HF paper page](https://huggingface.co/papers/2604.02329))                                   |
| QRISP (Qualcomm)                                                                                                                                                     | Distributed by Qualcomm under a research licence, not on Kaggle or Hugging Face ([Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset)) |
| Kaggle `uhecoms/super-resolution-for-real-time-computer-graphics` (GPU-simulator renders of game traces at 320 to 1600 px with depth, normals and edges; MIT; 54 GB) | Promising for native multi-resolution renders but large; a candidate for the native-LR ablation D3                                                                           |
| Inter4K (HF `Shilin-LU/Inter4k`)                                                                                                                                     | The HF tag says MIT, but the original release is CC BY-NC-SA ([repo](https://github.com/alexandrosstergiou/Inter4K))                                                         |
| HF `Kim2091/8k-video-game-dataset`                                                                                                                                   | CC BY-NC-SA 4.0 and 7z archives                                                                                                                                              |
| X4K1000FPS                                                                                                                                                           | Not on Kaggle or Hugging Face ([XVFI](https://github.com/JihyongOh/XVFI)); research and education only                                                                       |
| Spring (HF `supmelon/spring_processed`)                                                                                                                              | Licence not stated on the mirror                                                                                                                                             |
| OpenVid-1M and similar                                                                                                                                               | Several TB                                                                                                                                                                   |

### 5.3 Pipeline (`neural-data-pipeline` notebook, `nss_pipeline.py`)

1. **Discovery.**
   - Kaggle: a capped randomised directory walk, listing directories completely so sequences are never split. Degraded copies (LR, bicubic, blurred, `x2`/`x3`/`x4`) and non-colour passes are dropped by path token.
   - Hugging Face: HTTPS with retries. TartanAir is read member by member from remote zips with HTTP range requests; GameIR tars are streamed and stopped early; Vimeo-1080p parquet shards are unpacked into MP4 files. Both access patterns were checked against the live repositories on 5 October 2026.
2. **Normalisation.** RGB uint8 frames; depth to an inverse-depth style code; flow to engine motion-vector semantics.
3. **Leakage groups.** Every item gets one: Kaggle video folder or video, Sintel scene family, TartanAir environment, GameIR town, Vimeo-1080p video, still file.
4. **Quality filter.** Rejects keyframes that are smaller than 256 px, flat, more than 35 percent letterboxed, or blurry (Laplacian variance). Rejected items are kept in the manifest as `filtered`.
5. **Deduplication and split.** Section 6. The output is `nss_manifest.parquet` with URIs that resolve on Kaggle (`kaggle:<ref>|path`, `data:path`, `mp4:path#frame`).

The two training notebooks load this manifest from an attached notebook output, or rebuild a smaller one inline with the same code.

### 5.4 Temporal sequences and motion-vector semantics

- All windows use the engine convention: for each pixel of frame k, the displacement to frame k-1.
- TartanAir and Sintel provide forward flow (frame k to k+1). Their windows are played **backwards**, so this forward flow becomes exactly the engine vector, and the occlusion mask becomes the disocclusion mask.
- Video windows get RAFT motion with a forward-backward consistency mask.
- Frame-generation quads (four consecutive frames) store teacher flows 2 to 0 (the newest real frame's motion vector), 1 to 0 and 1 to 2 (interpolation targets), and 3 to 2 (extrapolation target).

### 5.5 Synthetic game-like renderer (`nss_synth.py`)

Rendered on the GPU from cached high-resolution stills:

- **Scene.** A background plane and a foreground layer with a procedural alpha: solid blobs, wires one to two texels wide, and fences of thin bars.
- **Motion.** Each layer has its own continuous affine camera path. Speed classes are static, slow (0.3 to 3 px/frame), medium (3 to 10) and fast (10 to 28), with rotation, zoom, acceleration and optional camera reversal.
- **Effects.** Alpha-blended particles that write no motion vectors (like real engines) and set a reactive mask; world-space text; exposure changes.
- **Overlays (FG).** Screen-space HUD tiles, subtitle changes, full-screen menus and scene cuts.
- **Rendering.** Low-resolution frames take one jittered sample per pixel from the full-resolution texture with no mip filtering, so minification produces real aliasing on thin geometry. Targets use 4 samples per pixel (rotated grid).
- **Ground truth.** Motion vectors, intermediate flows, disocclusion masks and depth are exact because every pixel's layer and trajectory are known.

**Hypothesis.** This closes most of the gap left by the lack of open game G-buffer data for motion, disocclusion and aliasing behaviour. It does not model lighting effects such as specular highlights, reflections and shadows (section 15).

## 6. Deduplication and leakage strategy

| Step                     | Method                                                                                                                                                                                                      | Notes                                                                                                                                                                                                                |
| :----------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Keyframes                | One per still or short clip; first, middle and last for long sequences                                                                                                                                      | Bounds compute; mid-sequence content changes are covered by the three keyframes                                                                                                                                      |
| Exact and near-exact     | 64-bit DCT pHash and dHash on 256x256 thumbnails                                                                                                                                                            | Robust to rescaling, mild compression and colour shifts; weak under large crops and mirroring                                                                                                                        |
| Semantic near-duplicates | [DINOv2-small](https://huggingface.co/facebook/dinov2-small) (Apache-2.0) global descriptors, cosine similarity                                                                                             | Catches crops, mirroring and re-encodes; may also link different images of the same landmark (conservative error)                                                                                                    |
| Search                   | Exhaustive all-pairs on GPU: Hamming as dot products of +-1 vectors, cosine as half-precision matrix products, chunked, both GPUs                                                                           | No approximate index needed at about 30K keyframes; [FAISS](https://github.com/facebookresearch/faiss) (MIT) is the path to millions                                                                                 |
| Calibration              | Synthetic positives (60 to 95 percent crops, rescale, mirror, rotation, colour shift, blur, text overlay, JPEG 35 to 92) against random negatives                                                           | Threshold = stricter of the 0.1 percent false-positive point and the most similar negative plus a margin. All-pairs search tests about K^2/2 pairs, so a plain 0.1 percent false-positive rate would flood the graph |
| Guards                   | pHash edges only between keyframes with enough structure (grey standard deviation at least 15), confirmed by dHash; thresholds are tightened automatically while there are more than two edges per keyframe | Prevents dark or flat frames, or semantically similar scenes, from merging unrelated data into giant components                                                                                                      |
| Clustering               | Union-find over leakage-group edges and duplicate edges                                                                                                                                                     | A component is the unit of splitting                                                                                                                                                                                 |
| Split                    | A component containing any test item becomes test, and its training members are dropped. Other components go to validation with probability 0.05 (deterministic hash), else training                        | Test sets are pinned by source role (benchmarks, held-out environments, towns, scene families, official test lists)                                                                                                  |
| Audit                    | Nearest training neighbour of every validation and test keyframe                                                                                                                                            | The notebook asserts no cross-split embedding match above the final threshold, and reports pHash matches                                                                                                             |

**Limitations and leakage risks.**

1. Duplicates are judged on keyframes. Two sequences that share only mid-sequence content can slip through.
2. Similarity thresholds are global; per-source calibration would be tighter.
3. Strong augmentations beyond the calibration set (heavy colour grading, large overlays, extreme crops under 50 percent) are missed.
4. Scenes that are related but different, such as other trajectories in the same TartanAir environment, other Sintel shots of the same set, or REDS clips from the same city walk, are kept together only through the coarse leakage groups (environment, scene family). Groups that are too fine would leak.
5. Synthetic training scenes reuse textures from training stills only. Test synthetic scenes use test stills (held-out game frames and DIV2K validation images).
6. RAFT was trained on Sintel and other flow datasets ([RAFT](https://github.com/princeton-vl/RAFT)), so motion estimated on Sintel-like content is optimistic. Sintel test sequences use exact flow, not RAFT.

## 7. Temporal super-resolution training strategy

- **Data mix per step.**
  - About 60 percent synthetic sequences (exact everything).
  - About 40 percent real windows: TartanAir and Sintel exact; REDS, Vimeo, Vimeo-1080p and GameIR with RAFT motion and `mv_quality = 0.5`.
  - Real windows get motion-consistent flips and exposure augmentation.
- **Unroll.** Six frames per sequence (truncated back-propagation through the recurrence), 192x192 output patches (64x64 input at 3x).
- **Loss per frame.** The first frame, which has no history, is down-weighted.
  - Charbonnier, plus 0.1 times a gradient loss, plus 0.05 times an L1 loss on Fourier magnitudes.
  - Plus 0.05 times LPIPS on the last frame.
  - Plus 0.5 times a temporal-change loss `|(o_t - W o_{t-1}) - (g_t - W g_{t-1})|` over pixels visible in both frames, weighted by motion quality.
  - Plus 0.5 times distillation toward the teacher (students only).
- **Schedule.** Teacher first (both GPUs), then the main 3x student with distillation, then the `igpu` 3x and `low` 2x students in parallel (one per GPU), then six ablation runs in three parallel pairs. AdamW, cosine schedule over each run's wall-clock budget, EMA weights, best checkpoint by validation PSNR-Y.
- **Ratios.** 3x is primary and 2x is trained. **1.5x is not trained yet.** The pixel-shuffle head needs an integer factor; options (section 14): an integer-factor model with a resampled output, or a fractional head with a learned resampling kernel.

## 8. Frame-generation strategy

- **Two modes, one model.**
  - Interpolation (tau = 0.5) gives the best quality at the cost of one held frame.
  - Extrapolation (tau = 1.5) holds nothing. Its camera prior is re-sampled from the latest input, so camera motion in generated frames responds without waiting for the next rendered frame.
- **Training data.** Synthetic scenes provide exact intermediate flows, disocclusions, HUD with subtitle changes, menus, cuts and rapid camera reversals. Real quads provide RAFT flows.
- **Robustness.** Engine inputs (motion vectors, depth, camera prior) are dropped at random, so the same model works for engine integrations and post-process integrations.
- **Loss.** Charbonnier, plus 0.5 times census, plus 0.1 times LPIPS, plus 0.01 times flow L1 (exact or RAFT), plus 0.2 times UI binary cross-entropy (positive weight 4), plus 0.5 times distillation from the privileged teacher.
- **Special cases.**
  - Cuts: the detector forces a frame repeat.
  - Menus and pause screens: the UI mask covers the panel and the newest real frame is shown.
  - Rapid input reversal: measured with and without the camera prior.
  - Low base frame rate: generation becomes less reliable as motion between real frames grows, so integrations should offer a minimum base frame rate (FSR recommends rendering slightly below half the output rate; section 3.1).

## 9. UI and text handling

| Situation                                            | Handling                                                                                                                                                                                                                                                                                                                            |
| :--------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TSR, engine integration                              | Upscale before UI composition; UI is drawn at native resolution after reconstruction (the FSR / DLSS convention). Training never puts screen-space UI into TSR targets; world-space text is part of the scene and is evaluated as fine detail                                                                                       |
| FG, engine provides HUD-less frames and a UI layer   | Generate from HUD-less frames, then composite the newest UI layer on every displayed frame (`student+ui_layer` in the evaluation)                                                                                                                                                                                                   |
| FG, UI not separable (post-process injection, video) | The learned UI / static-overlay mask copies those pixels from the newest real frame. Trained with HUD tiles, health bars, crosshairs, subtitle plates and changes, and menus, in the spirit of figure-text mixing (CVPR 2023). Measured with UI-region PSNR and a UI ghosting rate (share of UI pixels off by more than 10 percent) |
| Subtitle or HUD changes between real frames          | Target = the UI of the newest real frame, never a blend                                                                                                                                                                                                                                                                             |
| Menus, pause screens, full-screen overlays           | Covered by the mask; a static scene makes generation trivial; the cut detector handles screen switches                                                                                                                                                                                                                              |

## 10. Latency strategy

**Definitions** (reported separately in the FG notebook):

- _Game input latency_: input sample to first photons of a **real** frame that reflects it (game logic only advances on real frames).
- _Camera latency_: input sample to the first displayed frame of any kind whose camera reflects it.
- _Generated-frame content age_: time since the newest real input in a displayed generated frame.

**Model** (`simulate_latency`): a GPU-bound loop with just-in-time CPU scheduling (Reflex / Anti-Lag / XeLL style, no queued frames) on a variable-refresh display, with log-normal render-time jitter.

- **Interpolation** holds real frame k until the frame between k-1 and k has been generated and shown. Game latency rises by about half a frame plus the generation time.
- **Extrapolation** holds nothing, so game latency equals native.
- **Late-latched extrapolation** re-samples the camera just before generating. Camera latency of generated frames then drops to about the generation time plus present.

The model is driven by T4-measured generation times; on target hardware these must be replaced with measured times.

**Engineering rules.**

1. Never queue frames: CPU work starts just in time.
2. Run FG on an asynchronous compute queue, with presentation paced on a dedicated thread (as FSR does).
3. Split FG into a per-pair part and a per-frame part when generating several frames (DLSS 4 MFG).
4. Disable FG below a base-rate threshold.
5. Click-to-photon measurement on hardware (PresentMon or an LDAT-like sensor) is planned for the integration phase; Kaggle cannot measure display latency.

## 11. Teacher and distillation strategy

| Teacher                                                                                                                                                                                                                                                                                                                  | Status          | Use                                                                                                                                                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| In-notebook TSR teacher (1.37M params)                                                                                                                                                                                                                                                                                   | Implemented     | Output distillation for the `low` and `igpu` students                                                                                                                                |
| In-notebook privileged FG teacher                                                                                                                                                                                                                                                                                        | Implemented     | Flow and output distillation                                                                                                                                                         |
| RAFT small ([torchvision](https://pytorch.org/vision/stable/models/raft.html), BSD-3)                                                                                                                                                                                                                                    | Implemented     | Motion for video, flow supervision, tOF metric                                                                                                                                       |
| DINOv2-small                                                                                                                                                                                                                                                                                                             | Implemented     | Deduplication embeddings                                                                                                                                                             |
| Large open VSR models as offline teachers or pseudo-ground-truth generators: [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) (Apache-2.0), [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) (Apache-2.0), [SeedVR](https://github.com/ByteDance-Seed/SeedVR) (Apache-2.0)                          | Not implemented | Only relevant where no ground truth exists, such as sharpening native-resolution game captures. Every Kaggle source here has ground truth, so the extra compute is not justified yet |
| Open interpolation models as FG references: [RIFE](https://github.com/hzwer/Practical-RIFE) (MIT, weights on Google Drive), [IFRNet](https://github.com/ltkong218/IFRNet) (MIT), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) (Apache-2.0), [FILM](https://github.com/google-research/frame-interpolation) (Apache-2.0) | Not implemented | Quality ceilings for offline comparison; weights are not on Kaggle or Hugging Face in official form, so they are not in the notebooks                                                |

## 12. Losses and metrics

| Metric                                                                                                  | Task                     | Implementation                                                                                  |
| :------------------------------------------------------------------------------------------------------ | :----------------------- | :---------------------------------------------------------------------------------------------- |
| PSNR and SSIM on Y (BT.601), RGB PSNR                                                                   | Both                     | `nss_metrics`                                                                                   |
| LPIPS-AlexNet                                                                                           | Both                     | [lpips](https://github.com/richzhang/PerceptualSimilarity) (BSD-2), with a torchvision fallback |
| LDR-FLIP                                                                                                | TSR (static convergence) | [flip-evaluator](https://github.com/NVlabs/flip) (BSD-3)                                        |
| Warping error along exact motion; temporal-change error                                                 | TSR                      | `nss_metrics`                                                                                   |
| tOF (RAFT) and tLP                                                                                      | Both                     | TecoGAN definitions                                                                             |
| Region PSNR: disocclusions, fine detail (top 15 percent gradients), particles                           | TSR                      | `nss_eval`                                                                                      |
| Static-scene convergence (PSNR against frame index)                                                     | TSR                      | Synthetic static suite                                                                          |
| Interpolation error, occluded-region PSNR                                                               | FG                       | `nss_eval`                                                                                      |
| UI-region PSNR, UI ghosting rate, cut detection rate, menu and cut correctness, camera-reversal quality | FG                       | `nss_eval` and synthetic suites                                                                 |
| Displayed frame rate, game and camera latency, content age                                              | FG                       | `simulate_latency`                                                                              |
| T4 latency per tier and resolution, GMAC per frame                                                      | Both                     | CUDA events, forward hooks                                                                      |

**Baselines.**

- TSR: bicubic, jitter-aware bilinear, and an analytic TAA upscaler (history reprojection, YCoCg 3x3 neighbourhood clamp, 10 percent blend).
- FG: frame repeat, average, motion-vector reprojection.
- FSR 3.1 itself needs engine captures; comparing against it is planned (section 17).

## 13. Hardware and runtime strategy

- **Arithmetic.** FP16 everywhere; FP32 only for warp coordinates and losses. INT8 (DP4a-friendly) quantisation-aware fine-tuning is planned for the `igpu` tiers.
- **Export.**
  - The fused networks export to ONNX (opset 17, static shapes per render resolution; `GridSample` needs opset 16+), with ONNX Runtime parity checked in the notebooks.
  - Deployment paths: [DirectML](https://github.com/microsoft/DirectML) on Windows (all vendors), Vulkan compute through [ncnn](https://github.com/Tencent/ncnn) (Windows, Linux, Android), Metal on Apple GPUs (to be evaluated), and TensorRT as the NVIDIA-only fast path.
  - The pre- and post-processing (dilation, reprojection, pixel shuffle, blend) belong in the engine's compute shaders.
- **Memory (estimate, 1080p, TSR 3x).**
  - Two FP16 RGBA history buffers at 1920x1080 take 33.2 MB.
  - TSR activations depend on the tier: the `low` tier's 24-channel ping-pong at 640x360 plus its 48-channel half-resolution level is about 28 MB.
  - FG activations range from about 12 MB (`low`, one level at 480x270) to about 70 MB (`mid`, a second 32-channel level at 960x540).
  - The previous plan's 100 MB pipeline budget remains the target.

## 14. Experiments and ablations

Run by the notebooks (short, equal budgets per ablation pair):

| ID         | Notebook | Change                                                       | Question                                        |
| :--------- | :------- | :----------------------------------------------------------- | :---------------------------------------------- |
| A-TSR-1    | TSR      | No history (single frame)                                    | Value of temporal accumulation                  |
| A-TSR-2    | TSR      | No G-buffers (colour only)                                   | Value of depth and motion vectors               |
| A-TSR-3    | TSR      | Bicubic degradation instead of jittered render-like sampling | Training/deployment domain gap (aliasing)       |
| A-TSR-4    | TSR      | No temporal loss                                             | Flicker and ghosting                            |
| A-TSR-5    | TSR      | No distillation                                              | Value of the teacher                            |
| A-FG-1     | FG       | No UI training                                               | UI correctness                                  |
| A-FG-2     | FG       | No motion priors                                             | Value of engine motion vectors and camera prior |
| A-FG-3 / 4 | FG       | Interpolation-only / extrapolation-only                      | Cost of one shared model                        |
| A-FG-5     | FG       | No distillation                                              | Value of the privileged teacher                 |

Next (not yet run):

- Tier sweeps (width, depth, U-Net level) for quality against GMAC.
- 1.5x scaling.
- Kernel-prediction output head.
- Adversarial / perceptual fine-tuning stage for texture.
- Native low-resolution training (the GameIR and `uhecoms` renders).
- INT8 quantisation.
- Multi-frame generation (2x, 3x, 4x).
- Engine-motion-vector forward splatting for FG.
- Learned versus analytic cut detection.

**Success criteria for this phase (Kaggle).**

1. On game-like suites, the 3x student beats the analytic TAA upscaler on both PSNR-Y and temporal-change error, and LPIPS is lower.
2. Static scenes converge above the single-frame ablation.
3. FG beats motion-vector reprojection on real suites in both modes.
4. UI ghosting rate is far below the reprojection baseline.
5. Fused-model parity is below 1e-3, and ONNX Runtime parity is checked.

All of these remain unmeasured until the notebooks have run.

## 15. Known limitations

1. **No real engine G-buffer data on Kaggle or Hugging Face with open terms** (section 5.2).
   - Lighting-dependent effects (specular aliasing, reflections, shadows, volumetrics, ray-traced noise) are under-represented.
   - The synthetic renderer covers geometry, motion, disocclusion, transparency without motion vectors, and UI only.
2. **Real sequences mostly lack exact motion.** Video motion comes from RAFT and carries its errors.
3. **GameIR frames are 10 rendered frames apart**, so its sequences are effectively high-motion.
4. **Simulated, not real, low-resolution renders.** Jittered single-sample reads of a finished high-resolution frame miss level-of-detail changes, mip selection and shading-rate effects of a real low-resolution render. GameIR native renders test this gap (2x only).
5. **Kaggle limits training** to a few hours per stage: models are under-trained compared with production systems; tiers above `mid` are not trained.
6. **T4 PyTorch latency is not in-engine latency on target GPUs**; the latency model is analytic.
7. **No head-to-head with FSR 3.1, DLSS or XeSS yet** (it needs engine captures with their SDKs).
8. **Perceptual deduplication is keyframe-based** with global thresholds (section 6).
9. **1.5x is not trained.** Fractional ratios need a different output head.

## 16. Licensing considerations

| Item                                                                  | Licence (as checked)                                                                                                                                                                                   | Policy                                                                                                                                                                                                               |
| :-------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ECBSR (RepConv idea)                                                  | Apache-2.0 ([repo](https://github.com/xindongzhang/ECBSR)); the previous plan said MIT, corrected                                                                                                      | Ideas re-implemented; attribution                                                                                                                                                                                    |
| RAFT (torchvision weights), LPIPS, FLIP, DINOv2, FAISS, PySceneDetect | BSD-3, BSD-2, BSD-3, Apache-2.0, MIT, BSD-3                                                                                                                                                            | Evaluation and data tooling; not shipped in runtime weights                                                                                                                                                          |
| RIFE / Practical-RIFE                                                 | MIT, including the released models per the README; the previous plan said non-commercial, corrected                                                                                                    | Reference only                                                                                                                                                                                                       |
| IFRNet                                                                | MIT on GitHub; the previous plan said Apache 2.0, corrected                                                                                                                                            | Reference only                                                                                                                                                                                                       |
| FidelityFX SDK (FSR 3.1)                                              | Open SDK on GPUOpen; the current repository has no SPDX licence detected (check the licence file per version)                                                                                          | Documentation informs the engine contract; no code copied                                                                                                                                                            |
| XeSS SDK                                                              | Custom Intel licence (not SPDX-classified on GitHub); closed models                                                                                                                                    | Reference only                                                                                                                                                                                                       |
| DLSS                                                                  | Proprietary                                                                                                                                                                                            | Reference only                                                                                                                                                                                                       |
| Data                                                                  | Section 5.2; DIV2K, Flickr2K, Vimeo-90K, Sintel (original terms), Vid4, game captures and Vimeo-1080p are research-use or unclear; REDS (CC BY 4.0), TartanAir (BSD-3) and GameIR (MIT) are permissive | **Research weights** may use everything listed. **Commercial weights** must be retrained on permissive sources (TartanAir, GameIR, REDS, project-owned renders) after each licence is confirmed on the original page |

## 17. Roadmap

| Phase                      | Deliverable                                                                                                                                                                               | Acceptance                                                                                            |
| :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| 0. Kaggle validation (now) | Run the three notebooks; publish metrics, plots and weights                                                                                                                               | Section 14 success criteria; data audit with zero embedding leaks                                     |
| 1. Data scale-up           | More TartanAir environments and trajectories (exact motion, depth), the native multi-resolution `uhecoms` renders, larger Vimeo-1080p subsets, optional Hugging Face token for throughput | Gains on held-out environments; native-LR gap measured                                                |
| 2. Engine captures         | Open-content UE5 / Unity / Godot scenes captured with jitter, motion vectors, depth, reactive masks, HUD-less colour and UI layers, plus FSR 3.1 outputs for the same frames              | First head-to-head with FSR 3.1 at 3x and 2x                                                          |
| 3. Quality                 | 1.5x head, perceptual / adversarial stage, multi-frame generation, INT8 for `igpu`                                                                                                        | Static convergence near native (PSNR / FLIP); UI ghosting under target                                |
| 4. Runtime                 | DirectML and Vulkan (ncnn) integrations, async-compute FG, pacing thread, PresentMon latency on an integrated GPU, an entry GPU and a mid GPU                                             | Measured latency within tier budgets; click-to-photon reported separately for game and camera latency |
| 5. Release                 | Weights trained only on permissive data; licences confirmed                                                                                                                               | Licence review complete                                                                               |

## 18. Repository layout and how to run

```text
src/neuralss/          single source of the library (embedded into the notebooks at build time)
  nss_common.py        run folders, logging, hardware report, pip helper, training launcher
  nss_data.py          source registry, discovery, Hugging Face range access, decoding, window / crop workers, manifest I/O
  nss_dedup.py         hashes, embeddings, calibration, GPU neighbour search, union-find split
  nss_pipeline.py      data-pipeline steps and the inline fallback
  nss_cache.py         memory-mapped caches and RAFT teacher motion
  nss_synth.py         synthetic game-like renderer and GPU batch builders
  nss_models.py        RepConv, TSR network, FG network, tiers, fusion, ONNX wrappers
  nss_metrics.py       losses, metrics, RAFT, analytic baselines, latency model
  nss_train.py         DDP harness
  nss_eval.py          test clips, runners, metric rows
  train_tsr.py, train_fg.py
notebooks/             percent-format notebook sources
tools/build_notebooks.py   builds the .ipynb files (python tools/build_notebooks.py [--check])
neural-data-pipeline/neural-data-pipeline.ipynb
neural-supersampling/neural-supersampling.ipynb
neural-frame-generation/neural-frame-generation.ipynb
neural-supersampling/outputs/   archived outputs of the superseded single-frame notebook (section 19)
```

**Run order on Kaggle** (GPU T4 x2, Internet on):

1. `neural-data-pipeline`: attach the seven Kaggle datasets; this produces the manifest and the Hugging Face subsets.
2. `neural-supersampling` and `neural-frame-generation`: attach the pipeline output and the same Kaggle datasets.

Each notebook can also run alone (inline manifest, smaller caps). `QUICK_RUN = True` gives a short smoke test of every cell.

## 19. Prior evidence

| Evidence                                                                               | Status                                                                                                                                                                                                                                                                                                                                     |
| :------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Single-frame Rep-TNSR v2 run (Kaggle, September 2026; `neural-supersampling/outputs`)  | _Measured_, superseded design. Set5 x3 PSNR-Y +2.29 dB and Set14 +1.67 dB over bicubic; fused 640x360 to 1080p in 4.2 ms on a T4. Validation plateaued after about 2 hours (capacity-bound), and 98 of the 100 "DIV2K" test images were also in its training split. These findings motivated the temporal, G-buffer, leakage-safe redesign |
| Rep-TNSR v1 (earlier project): 34.82 dB at 360p to 1080p, 2.72 ms on a GTX 1650 Mobile | _Reported by the earlier project, not reproduced here_; compared against FSR 1.0 only                                                                                                                                                                                                                                                      |

## 20. References

Industry: [DLSS 4 (NVIDIA research)](https://research.nvidia.com/labs/adlr/DLSS4) · [DLSS 4 integration (Streamline)](https://developer.nvidia.com/blog/how-to-integrate-nvidia-dlss-4-into-your-game-with-nvidia-streamline) · [Reflex 2 Frame Warp](https://www.nvidia.com/en-us/geforce/news/reflex-2-even-lower-latency-gameplay-with-frame-warp/) · [FSR 3 upscaler](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/) · [FSR frame interpolation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/frame-interpolation/) · [FSR frame-interpolation swap chain](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/frame-interpolation-swap-chain/) · [FSR Redstone for developers](https://gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering/) · [Intel XeSS SDK](https://github.com/intel/xess)

Research: [NSRR](https://research.facebook.com/blog/2020/7/introducing-neural-supersampling-for-real-time-rendering/) · [Thomas et al. 2022](https://diglib.eg.org/handle/10.1145/3543870) · [Mercier et al. 2023](https://arxiv.org/abs/2308.01483) · [FuseSR](https://arxiv.org/abs/2310.09726) · [RDG](https://github.com/sunny2109/RDG) · [ExtraNet](https://sites.cs.ucsb.edu/~lingqi/publications/paper_extranet.pdf) · [ExtraSS](https://doi.org/10.1145/3610548.3618224) · [STSS](https://arxiv.org/abs/2312.10890) · [GFFE](https://arxiv.org/abs/2406.18551) · [Mob-FGSR](https://sites.cs.ucsb.edu/~lingqi/publications/paper_sig24mobfgsr.pdf) · [RIFE](https://arxiv.org/abs/2011.06294) · [Discontinuity-aware VFI](https://arxiv.org/abs/2202.07291) · [FlashVSR](https://arxiv.org/abs/2510.12747) · [TecoGAN](https://arxiv.org/abs/1811.09393) · [GameIR](https://arxiv.org/abs/2408.16866) · [ECBSR](https://github.com/xindongzhang/ECBSR) · [SSCD copy detection](https://arxiv.org/abs/2202.10261)

Data: [TartanAir (HF)](https://huggingface.co/datasets/theairlabcmu/tartanair) · [GameIR (HF)](https://huggingface.co/datasets/LLLebin/GameIR) · [Game frames (HF)](https://huggingface.co/datasets/ericphann/video-game-super-resolution) · [Vimeo-1080p (HF)](https://huggingface.co/datasets/danjacobellis/vimeo1080p) · [MPI Sintel (Kaggle)](https://www.kaggle.com/datasets/artemmmtry/mpi-sintel-dataset) · [REDS (Kaggle)](https://www.kaggle.com/datasets/amithkesavmrajagiri/reds-dataset) · [Vimeo-90K septuplet (Kaggle)](https://www.kaggle.com/datasets/wangsally/vimeo-90k-7) · [Vimeo-90K triplet (Kaggle)](https://www.kaggle.com/datasets/chenshu123/vimeo-triplet) · [DIV2K (Kaggle)](https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images) · [Flickr2K (Kaggle)](https://www.kaggle.com/datasets/daehoyang/flickr2k) · [Vid4 (Kaggle)](https://www.kaggle.com/datasets/uom200647r/vid4-dataset) · [Origin Lab recordings](https://huggingface.co/datasets/originlab/game-recordings-v4) · [Generative World Renderer](https://huggingface.co/papers/2604.02329) · [QRISP](https://www.qualcomm.com/developer/software/qualcomm-rasterized-images-dataset) · [Inter4K](https://github.com/alexandrosstergiou/Inter4K) · [XVFI / X4K1000FPS](https://github.com/JihyongOh/XVFI)

Tools: [RAFT](https://github.com/princeton-vl/RAFT) · [LPIPS](https://github.com/richzhang/PerceptualSimilarity) · [FLIP](https://github.com/NVlabs/flip) · [DINOv2-small](https://huggingface.co/facebook/dinov2-small) · [FAISS](https://github.com/facebookresearch/faiss) · [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) · [SeedVR](https://github.com/ByteDance-Seed/SeedVR) · [IFRNet](https://github.com/ltkong218/IFRNet) · [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) · [FILM](https://github.com/google-research/frame-interpolation) · [DirectML](https://github.com/microsoft/DirectML) · [ncnn](https://github.com/Tencent/ncnn) · [ONNX Runtime](https://github.com/microsoft/onnxruntime)
