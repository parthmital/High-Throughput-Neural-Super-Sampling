# NeuralSS: Neural Super-Sampling and Frame Generation

Research plan for a GPU-agnostic, real-time neural upscaler (Rep-TNSR) and frame generator (NeuralFG) intended to measurably outperform AMD FSR 3.1 in image quality, temporal stability, latency and generalisation.

**Status (October 2026).** This repository contains this plan and two Kaggle notebooks that train and evaluate RGB-only versions of both networks on public datasets (section 13). No result in this document comes from the proposed G-buffer system yet. Every number is labelled as one of: a target, an arithmetic estimate, or a measurement carried over from the earlier Rep-TNSR v1 project (section 15).

## 1. Objective and Hypotheses

**Objective.** Build two small, decoupled networks, Rep-TNSR for upscaling and NeuralFG for frame interpolation. They consume engine G-buffers (colour, motion vectors, depth, exposure), run through vendor-neutral APIs (DirectML, Vulkan, plain compute shaders), and beat FSR 3.1 at the same scale factor on the same GPU.

**Core hypothesis.** A lightweight convolutional network conditioned on G-buffers produces sharper and more temporally stable upscaled frames, and more accurate interpolated frames, than FSR 3.1's hand-tuned heuristics, within the same latency and memory envelope and without vendor-specific hardware.

| ID  | Hypothesis                                                                                                                                    | Falsified if                                                                                                              |
| :-- | :-------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| H1  | Neural SR with G-buffer conditioning and structural reparameterisation gives higher perceptual quality than FSR 3.1 at the same scale factor. | Our LPIPS is not lower than FSR 3.1's on at least 3 of the 5 test sequences.                                              |
| H2  | Learned temporal accumulation with variance-clamped history reduces ghosting and flicker below FSR 3.1.                                       | Our warping error or tOF is not lower than FSR 3.1's.                                                                     |
| H3  | Learned intermediate flow with occlusion-aware blending produces fewer disocclusion artefacts than FSR 3.1 frame generation.                  | Human preference for ours is below 55 percent on the disocclusion test set.                                               |
| H4  | The full SR + FG pipeline fits FSR 3.1's latency envelope.                                                                                    | Our P95 latency exceeds FSR 3.1's by more than 20 percent on any tested GPU, or misses the absolute targets in section 2. |
| H5  | Training on diverse synthetic G-buffer data generalises to unseen games and engines without fine-tuning.                                      | LPIPS degrades by more than 15 percent on any held-out game set relative to in-distribution content.                      |

## 2. Success Criteria

Comparisons use the FSR 3.1 mode with the same scale factor. FSR 3.1 presets are Quality 1.5x, Balanced 1.7x, Performance 2.0x and Ultra Performance 3.0x per axis ([AMD FSR 3.1 manual](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/)). The primary configuration, 640x360 to 1920x1080, is 3x and is therefore compared against **Ultra Performance**, not Quality.

FSR 3.1 baseline values are not assumed. They are measured on the identical test sequences in experiment SR-001 and LAT-001, and the targets below are relative to those measurements.

| Metric                                    | Target                                                   | Measurement                                                     |
| :---------------------------------------- | :------------------------------------------------------- | :-------------------------------------------------------------- |
| PSNR, luma (dB)                           | at least +1.5 dB over FSR 3.1                            | Per-frame mean over the test sequences                          |
| SSIM                                      | at least +0.015 over FSR 3.1                             | Full RGB, per-frame mean                                        |
| LPIPS (AlexNet)                           | at least 20 percent lower                                | Per-frame mean                                                  |
| Warping error                             | at least 25 percent lower                                | Ground-truth motion-vector warp, valid pixels only (section 12) |
| tOF                                       | at least 20 percent lower                                | RAFT flow on outputs against RAFT flow on ground truth          |
| VMAF                                      | at least +5 points                                       | Per-sequence, Netflix VMAF via ffmpeg                           |
| FG PSNR (dB)                              | at least +2.0 dB over FSR 3.1 FG                         | Interpolated frame against the rendered middle frame            |
| FG LPIPS                                  | at least 25 percent lower                                | Interpolated frame                                              |
| SR latency, 1080p output, RTX 3060        | P95 at most 1.5 ms and at most 120 percent of FSR 3.1    | D3D12 timestamp queries                                         |
| FG latency, 1080p, RTX 3060               | P95 at most 2.5 ms and at most 120 percent of FSR 3.1 FG | D3D12 timestamp queries                                         |
| SR latency, 1080p output, GTX 1650 Mobile | P95 at most 3.0 ms (Performance or Quality tier)         | D3D12 timestamp queries                                         |
| Pipeline VRAM, 1080p                      | at most 100 MB                                           | Committed resources (section 10.3)                              |
| Parameters                                | SR at most 50K, FG at most 200K                          | Fused model parameter count                                     |

## 3. Baseline: AMD FSR 3.1

FSR 3.1 ([FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), MIT licence) is the open, cross-vendor baseline. Neither its upscaler nor its frame generator uses neural inference.

**Temporal upscaler.** Sub-pixel camera jitter from a Halton(2,3) sequence; reprojection of history with dilated motion vectors and depth; history rectification by colour-space variance clamping; Lanczos-style reconstruction of jittered samples; reactive and transparency masks to reduce history weight; sharpening (RCAS). Version 3.1 decoupled the upscaler from frame generation and reworked accumulation to reduce shimmering on thin geometry.

**Frame generation.** Hierarchical optical flow on luminance between frames t-1 and t, fused with dilated engine motion vectors; bidirectional warping to t-0.5; disocclusion hole filling; scene-cut detection by luminance change; UI composited separately through a swapchain proxy. Interpolation adds at least one frame of display latency, mitigated by latency-reduction SDKs (AMD Anti-Lag 2, NVIDIA Reflex).

**Required engine inputs.** LR colour, screen-space motion vectors (`R16G16_FLOAT`), depth (preferably inverted `R32_FLOAT`), exposure, optional reactive and transparency masks, and a separate HUD texture for frame generation.

**Known failure modes.** Smearing at disocclusions during fast camera motion, ghosting on thin geometry, double images when flow fails on fast rotation, HUD distortion without UI separation, particle and transparency smearing without reactive masks, and a sluggish feel below 50 to 60 FPS base rate.

**Other upscalers (reference only, not reproducible baselines).** NVIDIA DLSS 3.x: neural SR on Tensor Cores, frame generation on the RTX 40 optical flow accelerator, closed weights. Intel XeSS 1.3: neural SR on XMX with a DP4a fallback, open SDK (Apache 2.0), closed weights, no frame generation in that version. FSR 3.1 is the only fully open SR plus FG suite.

## 4. Hardware Targets and Compute Budget

Two hardware tiers are targeted. Estimates below assume 65 percent of peak arithmetic throughput, which matches the earlier v1 measurement (section 15.1: 7.96 GFLOP in 2.145 ms on a GTX 1650 Mobile, about 3.7 TFLOPS sustained).

| Quantity                           | GTX 1650 Mobile (TU117)                                   | RTX 3060 (GA106)                                                |
| :--------------------------------- | :-------------------------------------------------------- | :-------------------------------------------------------------- |
| Shader cores and clock             | 896 CUDA cores, 1.515 GHz boost (varies by laptop)        | 3,584 CUDA cores, 1.777 GHz boost                               |
| Peak FP32 (cores x 2 FLOP x clock) | 2.72 TFLOPS                                               | 12.74 TFLOPS                                                    |
| Peak FP16 without tensor cores     | 5.43 TFLOPS (dedicated FP16 units at twice the FP32 rate) | about 12.74 TFLOPS (same rate as FP32)                          |
| Sustained FP16 at 65 percent       | 3.53 TFLOPS                                               | 8.28 TFLOPS                                                     |
| Tensor cores                       | none                                                      | yes; usable only through DirectML meta-commands or vendor paths |
| Memory bandwidth                   | 128 GB/s (GDDR5) or 192 GB/s (GDDR6)                      | 360 GB/s                                                        |
| SR budget at 1080p, 60 FPS         | 3.0 ms of the 16.67 ms frame                              | 1.5 ms (success criterion)                                      |
| Compute in that budget             | 10.6 GFLOP, about 5.3 GMAC                                | 12.4 GFLOP, about 6.2 GMAC                                      |
| Data movable in that budget        | 384 MB at 128 GB/s                                        | 540 MB                                                          |

Design consequences: no attention, no multi-scale feature pyramids, no wide skip concatenations; all non-linear work happens at LR resolution; one plain 3x3 convolution chain at inference; FP16 storage and arithmetic.

## 5. Datasets

### 5.1 Strategy

1. **Priority 1: Kaggle-hosted public datasets.** Mounted read-only under `/kaggle/input`, used now by the two notebooks to validate architectures, training code and evaluation on standard benchmarks. RGB only.
2. **Priority 2: Kaggle G-buffer pack (planned).** 300 to 600 consecutive frames of the UE5 Bistro scene with full G-buffers, LR jittered renders and SSAA references, about 2.5 GB compressed, to be uploaded as a private Kaggle dataset. It does not exist yet.
3. **Priority 3: full UE5 captures and external datasets** on a workstation or cluster (sections 5.4 and 5.5).

### 5.2 Kaggle Datasets (Priority 1)

Contents were checked against the September 2026 notebook run, which listed 539,147 image files in total. Mirrors also contain degraded copies (LR, bicubic, blurred) and non-colour passes; the notebooks drop these by path token before use.

| Dataset             | Kaggle slug                                               | Observed content                                                                                | Used by                                                                               |
| :------------------ | :-------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| DIV2K               | `soumikrakshit/div2k-high-resolution-images`              | 900 images (800 train, 100 validation), 2K                                                      | SR training                                                                           |
| Flickr2K            | `daehoyang/flickr2k`                                      | 2,650 images, 2K                                                                                | SR training                                                                           |
| DF2K-OST            | `itzloghotxd/df2k-ost`                                    | 96,818 files; overlaps DIV2K and Flickr2K                                                       | SR training (duplicates removed by name)                                              |
| RealSR V3           | `yashchoudhary/realsr-v3`                                 | 3,130 files, paired camera LR/HR                                                                | SR training (HR side only)                                                            |
| REDS                | `amithkesavmrajagiri/reds-dataset`                        | 48,000 files, 720p sequences, including degraded copies                                         | SR and FG training                                                                    |
| REDS VSR toy        | `cookiemonsteryum/reds-video-superresolution-toy-dataset` | 48,000 small paired patches                                                                     | Not used (patches smaller than the training crops)                                    |
| Vimeo-90K septuplet | `wangsally/vimeo-90k-7`                                   | 65,009 frames (about 9,300 septuplets, a subset of the 91,701 in the original release), 448x256 | SR and FG training                                                                    |
| Vimeo-90K triplet   | `chenshu123/vimeo-triplet`                                | 219,573 frames (about 73,000 triplets), 448x256                                                 | FG training and test (official test list when present), SR training                   |
| MPI Sintel          | `artemmmtry/mpi-sintel-dataset`                           | 7,466 image files: clean and final passes plus flow visualisations and masks                    | SR and FG training (clean and final only)                                             |
| FlyingChairs        | `craljimenez/flyingchairs`                                | 22,872 image pairs with `.flo` flow                                                             | Not used by the notebooks (no middle frame); reserved for flow pre-training ablations |
| Vid4                | `uom200647r/vid4-dataset`                                 | 513 files: the 171 ground-truth frames of 4 sequences plus degraded copies                      | Test only (SR video metrics, FG zero-shot)                                            |
| SR benchmarks       | `jesucristo/super-resolution-benchmarks`                  | 1,344 files: Set5, Set14, B100, Urban100, Manga109 with LR copies                               | SR test only                                                                          |

**Mount paths.** Kaggle mounts a dataset at `/kaggle/input/<slug>` or, in the newer layout seen in the last run, at `/kaggle/input/datasets/<owner>/<slug>`. The notebooks look up datasets by slug under both layouts and never hard-code inner folder names.

### 5.3 Kaggle Data Pipeline

- **Discovery.** A randomised, capped directory walk per dataset (at most 40,000 to 60,000 files). A full recursive walk of the mounts took 15 minutes in the last run.
- **Filtering.** Path-token rules drop LR, bicubic, `x2`/`x3`/`x4`, blurred and compressed copies, other methods' outputs, and Sintel flow, occlusion, depth and albedo passes. For test sets, files marked HR or GT are preferred.
- **Splits.** A deterministic hash of the image, sequence or video id sends about 3 percent of groups to validation, so no sequence is split across train and validation. Benchmarks and Vid4 are test-only.
- **Caching.** Images are decoded once by all CPU cores into a uint8 patch cache in `/kaggle/working/cache`. The cache is memory-mapped and shared by both GPU processes, then deleted at the end. The last run was limited by JPEG and PNG decoding on 4 cores (about 210 images per second), not by the GPUs.
- **Degradation.** For SR, LR inputs are made on the GPU with anti-aliased bicubic downsampling to an exact size. This differs slightly from MATLAB `imresize`, so notebook numbers are compared against the bicubic baseline computed in the same way, not against published tables.

### 5.4 Custom UE5 Captures (Priority 3)

Twelve UE5 environments are captured with the pipeline in section 6. Scene roles are fixed in advance so that held-out scenes never enter training.

| Role                        | Scenes                                                                                                 |
| :-------------------------- | :----------------------------------------------------------------------------------------------------- |
| Training and validation (9) | Bistro, Sun Temple, Valley, City Park, SciFi Corridor, Medieval, Stylised Forest, Industrial, Interior |
| Held-out UE5 scenes (3)     | Desert Landscape, Underwater Reef, Cartoon Village                                                     |
| Held-out engine (2)         | Unity HDRP Urban Street, Unity HDRP Architectural Interior                                             |
| Direct FSR comparison       | FidelityFX SDK Bistro sample, frames dumped with RenderDoc                                             |

Storage estimate: 12 scenes x 10,800 frames x about 2 MB per frame is about 260 GB raw, and about 80 GB of cached training patches.

### 5.5 External Datasets (Priority 3)

| Dataset                                                               | Content                                                          | Use                        |
| :-------------------------------------------------------------------- | :--------------------------------------------------------------- | :------------------------- |
| [ExtraSS](https://github.com/NJU-3DV/ExtraSS)                         | Rendered sequences with G-buffers, jitter and disocclusion masks | Academic cross-validation  |
| [VIPER / Playing for Benchmarks](https://playing-for-benchmarks.org/) | GTA V sequences with flow and depth                              | Gaming generalisation      |
| [TartanAir](https://theairlab.org/tartanair-dataset/)                 | Synthetic UE4 sequences with flow, depth and camera poses        | Data-diversity ablation D2 |

### 5.6 Held-Out Test Sets

Never used for training or model selection: the 3 held-out UE5 scenes, the 2 Unity HDRP scenes, the FidelityFX SDK Bistro sequence, Vid4, and the SR benchmarks (Set5, Set14, B100, Urban100, Manga109).

### 5.7 Licences and Commercial Use

Licences must be confirmed on each source page before any release. The table records what is known and the resulting policy.

| Data                              | Licence as known                                                                                            | Commercial weights                        | Policy                                                        |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------- | :---------------------------------------- | :------------------------------------------------------------ |
| Custom UE5 captures               | Project-owned renders under the UE EULA                                                                     | Yes                                       | Primary commercial training data                              |
| REDS                              | CC BY 4.0                                                                                                   | Yes, with attribution                     | Allowed                                                       |
| MPI Sintel                        | Max Planck "PS:License 1.0" ([source](https://is.mpg.de/code/sintel-optical-flow-dataset)), not plain CC BY | Not until the licence terms are confirmed | Research only for now                                         |
| Vimeo-90K                         | Release code under MIT ([toflow](https://github.com/anchen1011/toflow)); source videos carry no licence     | No                                        | Research and prototyping only                                 |
| DIV2K, DF2K-OST, Flickr2K, RealSR | Academic use; Flickr2K images have mixed Flickr licences                                                    | No                                        | Research only                                                 |
| FlyingChairs                      | Research use                                                                                                | No                                        | Pre-training experiments only; excluded from released weights |
| Vid4, SR benchmarks               | Unclear or mixed                                                                                            | No                                        | Evaluation only                                               |

Released commercial weights are trained only on custom UE5 captures, REDS and any dataset whose licence has been confirmed to allow it.

## 6. Synthetic Data Capture (UE5)

Capture runs through an [UnrealCV](https://unrealcv.org/) plugin or a custom capture actor. For every frame:

- Inject sub-pixel jitter: take the Halton(2,3) point at index (t mod K) + 1, subtract 0.5, divide by the LR width and height, and add twice that offset to the projection matrix's third column. K = 9 phases for 3x, 8 for 2x.
- Render LR colour point-sampled with no TAA or post-processing, and the HR reference with 16x supersampling (2x2 spatial and 4x temporal).
- Export backward motion vectors (`R16G16_FLOAT`), forward motion vectors (needed for exact temporal-reversal augmentation), linear depth (`R32_FLOAT`), world normals and roughness, the auto-exposure value, and a JSON record of jitter offset, phase and camera matrices.
- Compute ground-truth disocclusion by warping the previous depth with the motion vectors and marking pixels whose depth disagrees by more than 1 percent, or whose source falls outside the screen.
- Export a reactive mask for particles, transparency and reflections, following the FSR 3.1 convention, and render UI to a separate overlay that never enters training.
- For frame generation, re-render the middle frame t-0.5 by interpolating the camera and animation state, with its motion vectors.

Each scene's camera paths cover static camera motion, skinned characters and vehicles, particles (fire, smoke, rain), alpha-tested foliage and fences, glass and water, dynamic lighting, screen-space and ray-traced reflections, and hard cuts.

## 7. Preprocessing, Augmentation and Storage

### 7.1 Preprocessing

1. Multiply linear HDR colour by the engine exposure.
2. Convert RGB to YCoCg (coefficients below). Compress luma with the reversible logarithmic curve used in v1: the log of one plus luma, divided by one plus that log. The curve maps [0, infinity) to [0, 1). Divide Co and Cg by the same denominator to keep chroma ratios in bright highlights. Convert back to RGB for the network.
3. Normalise motion vectors by the LR width and height.
4. Encode depth as inverse depth, normalised per frame to [0, 1].
5. Encode the jitter phase as phase / K.
6. Clamp every input to the FP16 safe range before the network.

| RGB to YCoCg |   R   |  G   |   B   |
| :----------- | :---: | :--: | :---: |
| Y            | 0.25  | 0.50 | 0.25  |
| Co           | 0.50  |  0   | -0.50 |
| Cg           | -0.25 | 0.50 | -0.25 |

| YCoCg to RGB |  Y  | Co  | Cg  |
| :----------- | :-: | :-: | :-: |
| R            |  1  |  1  | -1  |
| G            |  1  |  0  |  1  |
| B            |  1  | -1  | -1  |

The post-processing pass restores luma with the exact inverse of the compression curve.

### 7.2 Augmentation

| Augmentation                   | Probability | Detail                                                                                             |
| :----------------------------- | :---------- | :------------------------------------------------------------------------------------------------- |
| Horizontal flip                | 0.5         | Negate MV x                                                                                        |
| Vertical flip                  | 0.5         | Negate MV y                                                                                        |
| 90, 180 or 270 degree rotation | 0.5         | Rotate the MV field and its components                                                             |
| Random crop                    | 1.0         | 64x64 LR (192x192 HR at 3x); MVs, depth and masks cropped alike                                    |
| Temporal reversal              | 0.3         | Swap frame order and use the exported forward MVs (negating backward MVs is only an approximation) |
| Brightness jitter              | 0.2         | Multiply by a factor in [0.9, 1.1]                                                                 |
| MV noise                       | 0.3         | Gaussian, sigma 0.3 px                                                                             |
| Depth quantisation             | 0.1         | Reduce to 16-bit precision                                                                         |
| Exposure walk                  | 0.1         | Random walk within 0.3 EV                                                                          |

### 7.3 Storage

Raw frames are half-float OpenEXR files, one folder per scene, sequence and frame. Training uses pre-cropped FP16 patch caches (memory-mapped, NCHW) generated offline. The DataLoader uses pinned memory, persistent workers and prefetching.

## 8. Model Architecture

### 8.1 Two Decoupled Networks

SR and FG are separate networks with no shared backbone:

1. **Different queues.** SR is on the critical path of the graphics queue, before post-processing and UI. FG can run on an asynchronous compute queue, overlapping the next frame's G-buffer pass.
2. **Different domains.** SR works at LR resolution; FG must see display-resolution frames to avoid blurring detail.
3. **Different temporal structure.** SR accumulates history over many frames; FG combines exactly two anchor frames.
4. Separate networks can be ablated, shipped and mixed with other upscalers independently, as FSR 3.1 allows.

Presentation order with FG: the interpolated frame t-0.5 is presented first, then frame t, at half the base frame interval. FG is disabled automatically when the base frame time exceeds 25 ms (below 40 FPS), where linear motion assumptions fail and latency grows.

### 8.2 Rep-TNSR v2 (Super-Resolution)

**Block.** During training, each block sums six parallel linear branches and applies PReLU. The branches are a 3x3 convolution, a 1x1 convolution, an identity (when input and output widths match), and fixed Sobel-x, Sobel-y and Laplacian filters, each applied per channel and followed by a learnable 1x1 projection. This follows ECBSR ([xindongzhang/ECBSR](https://github.com/xindongzhang/ECBSR), MIT). Because every branch is linear, each block folds offline into a single 3x3 convolution and bias. Applying the fixed filters before the 1x1 projection keeps the fold exact, including at padded borders. The deployed network is a plain chain of 3x3 convolutions.

**Network.** N blocks at LR resolution, then a 3x3 projection to three channels per output sub-pixel (27 at 3x) and a pixel shuffle (depth-to-space) to HR. The current colour is repeated once per sub-pixel and added before the shuffle (nearest-neighbour residual, as in ECBSR), so the network learns only the correction. This add can be folded into the post-processing shader.

**Input tensor (12 channels at LR resolution).**

| Channels | Content                                                       |
| :------- | :------------------------------------------------------------ |
| 0 to 2   | Current colour, compressed                                    |
| 3 to 5   | Reprojected, variance-clamped history colour                  |
| 6, 7     | Dilated motion vectors, normalised                            |
| 8        | Disocclusion validity mask                                    |
| 9        | Linear depth, normalised                                      |
| 10       | Temporal confidence (exponential decay of depth disagreement) |
| 11       | Jitter phase / K                                              |

Channels 9 to 11 are new relative to v1, which used 9 channels. A reactive-mask channel is evaluated as ablation SA4b.

**Tiers** (fused parameters including PReLU slopes; cost per 640x360 input at 3x; latency estimates use the 65 percent assumption of section 4 and cover the trunk only):

| Tier                              | Blocks | Width | Params | GMAC | GFLOP | GTX 1650M estimate         | RTX 3060 estimate                |
| :-------------------------------- | :----- | :---- | :----- | :--- | :---- | :------------------------- | :------------------------------- |
| v1 (existing, 9-channel input)    | 4      | 20    | 17,467 | 3.98 | 7.96  | 2.26 ms (measured 2.15 ms) | 0.96 ms                          |
| Performance                       | 4      | 16    | 12,683 | 2.89 | 5.77  | 1.64 ms                    | 0.70 ms                          |
| Quality                           | 4      | 20    | 18,007 | 4.11 | 8.21  | 2.33 ms                    | 0.99 ms                          |
| Ultra                             | 6      | 24    | 34,659 | 7.91 | 15.83 | 4.48 ms (over budget)      | 1.91 ms (over the 1.5 ms target) |
| Ultra, RGB only (Kaggle notebook) | 6      | 24    | 32,715 | 7.47 | 14.93 | not a deployment target    | not a deployment target          |

Consequences: Performance and Quality fit the GTX 1650 Mobile 3 ms budget. Quality is the primary candidate for the RTX 3060 latency claim. Ultra meets it only if DirectML meta-commands put the convolutions on tensor cores, which is still to be measured.

Ultra parameter breakdown: first block 12 x 24 x 9 + 24 = 2,616; five 24-to-24 blocks 5 x (24 x 24 x 9 + 24) = 26,040; projection 24 x 27 x 9 + 27 = 5,859; PReLU 6 x 24 = 144; total 34,659. Cost: 34,344 multiply-accumulates per LR pixel, times 230,400 pixels at 640x360, is 7.91 GMAC.

### 8.3 NeuralFG (Frame Generation)

**Input** (display resolution): frames t-1 and t (6 channels), plus in the engine build the motion vectors from t-1 to t (2) and depth at t-1 and t (2), for 10 channels.

**Network.**

1. Average-pool the input by 4.
2. Apply three 3x3 convolutions of 32 channels with LeakyReLU(0.2), then a 3x3 head with 9 outputs: flow from t-0.5 to t-1, flow from t-0.5 to t, two blend logits, and an RGB residual.
3. Upsample bilinearly to full resolution and scale the flows by 4 into full-resolution pixels.
4. Backward-warp both frames with the flows, in FP32.
5. Blend the warped frames with softmax-normalised weights (a convex combination) and add the residual.
6. Zero-initialise the head, so training starts from the plain frame average.

| Variant                                      | Params | Cost at 1080p (convolutions at 480x270)                                                                   |
| :------------------------------------------- | :----- | :-------------------------------------------------------------------------------------------------------- |
| Engine build, 10-channel input               | 24,009 | 3.10 GMAC (6.2 GFLOP), about 0.75 ms on RTX 3060 at 65 percent, plus bandwidth-bound warping and blending |
| RGB build, 6-channel input (Kaggle notebook) | 22,857 | 2.95 GMAC                                                                                                 |

**Alternative design for ablation FA5 (engine-guided splatting).** Forward-splat half the engine motion vectors to t-0.5 with an atomic depth test, then let a small reparameterised trunk predict a blend map and a residual. This avoids learned flow where engine vectors are reliable, but a 24-channel, 4-stage trunk at full 1080p costs about 39 GMAC, far over budget. It would have to run at half or quarter resolution, and its cost must be fixed before it is compared.

### 8.4 Temporal State

Neither network holds a recurrent hidden state. For SR, temporal information enters through the reprojected, variance-clamped history buffer, managed by ping-pong buffers outside the network, as in v1 and FSR 3.1. FG is stateless per interpolation. Reasons: state is fragile across cuts, resolution changes, frame drops and pause or resume; external history management is proven in production; and recurrent operators complicate ONNX and DirectML export. A learned recurrent aggregator is kept only as a later ablation (SA7).

## 9. Loss Functions

### 9.1 Super-Resolution

| Term        | Definition                                                                                                               | Notes                                                                                                  |
| :---------- | :----------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| Charbonnier | Mean of the square root of (error squared + epsilon squared), epsilon = 1e-3                                             | Smooth L1 with a non-zero gradient at zero error                                                       |
| Edge        | Mean L1 difference of central-difference gradients in x and y between output and ground truth                            | Preserves sharp structure                                                                              |
| Perceptual  | Mean squared difference of frozen VGG-19 activations at `relu3_3` (`torchvision` `features[:16]`)                        | Index 14 is `conv3_3`, 15 is `relu3_3`                                                                 |
| Temporal    | Mean L1 difference between the current output and the previous output warped by the motion vectors, weighted by validity | Validity is zero at disocclusions and elsewhere decays exponentially with depth disagreement (rate 10) |
| Frequency   | Mean L1 difference between the 2D FFT magnitudes of output and ground-truth luma                                         | Keeps fine texture                                                                                     |

Loss weights by curriculum phase:

| Phase                          | Iterations   | Charbonnier | Edge | Perceptual | Temporal | Frequency |
| :----------------------------- | :----------- | :---------- | :--- | :--------- | :------- | :-------- |
| 1. Spatial warm-up             | 0 to 100K    | 1.0         | 0.3  | 0          | 0        | 0         |
| 2. Temporal integration        | 100K to 300K | 1.0         | 0.5  | 0.05       | 0.25     | 0.1       |
| 3. Fine-tuning, larger patches | 300K to 500K | 1.0         | 0.5  | 0.05       | 0.25     | 0.1       |

A differentiable FLIP term ([NVlabs/flip](https://github.com/NVlabs/flip)) is evaluated as ablation L8 rather than added by default.

### 9.2 Frame Generation

Charbonnier (weight 1.0) against the rendered middle frame, VGG-19 `relu3_3` perceptual loss (0.05), and a 7x7 soft ternary census loss (0.5). The census transform compares each pixel with its 7x7 neighbourhood on luma and passes each difference through a smooth sign function. Two transforms are compared with a saturating squared distance averaged over the window, using RIFE's constants of 0.81 and 0.1. Because only local orderings matter, the loss is robust to brightness changes between frames.

### 9.3 What the Kaggle Notebooks Use

SR: Charbonnier + 0.3 x edge (phase 1, single frame, no G-buffers). FG: Charbonnier + 0.5 x census. The VGG term is omitted because the notebooks run offline and torchvision's VGG weights must be downloaded.

## 10. Training and Runtime Pipeline

### 10.1 Training Configuration

| Item            | Setting                                                                                                                       |
| :-------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| Curriculum      | Phase 1: 64x64 LR patches, single frame. Phase 2: 64x64, frame pairs. Phase 3: 96x96, frame pairs.                            |
| Optimiser       | AdamW, learning rate 5e-4, betas (0.9, 0.999), weight decay 1e-4, gradient clipping at 1.0                                    |
| Schedule        | Cosine decay to 1e-6 over 500K iterations (SR) or 300K (FG), with a short warm-up                                             |
| Precision       | FP16 autocast with dynamic loss scaling; FP32 for warp coordinates and losses                                                 |
| Parallelism     | PyTorch DistributedDataParallel, one process per GPU, `find_unused_parameters=False`                                          |
| Batch           | 16 frame pairs per GPU (workstation); probed per GPU on Kaggle                                                                |
| Weights         | EMA of parameters (decay 0.999) used for validation and export                                                                |
| Checkpoints     | Every 10K iterations and on best validation: model, optimiser, scheduler, scaler, EMA, all RNG states, iteration, best metric |
| Reproducibility | Fixed seeds (42), `CUBLAS_WORKSPACE_CONFIG=:16:8` and deterministic algorithms for final runs                                 |

Compute planning estimate: SR about 20 hours on 4 GPUs of the RTX 4090 class (about 80 GPU-hours); FG about 14 hours (about 56 GPU-hours); ablations at 200K iterations on one GPU, about 20 GPU-hours each.

### 10.2 Runtime Passes (per frame, 1080p output)

1. **Pre-processing compute shader (LR resolution).** Dilate depth and motion vectors (nearest depth in 3x3), reproject the history, convert to YCoCg and compress luma, clamp history to the 3x3 neighbourhood mean plus or minus 1.25 standard deviations, compute disocclusion and temporal confidence, and pack the 12-channel FP16 input.
2. **SR trunk.** A DirectML compiled graph, or hand-written HLSL with weights in constant buffers.
3. **Post-processing compute shader.** Pixel shuffle, YCoCg to RGB, luma decompression, inverse exposure, write the HR output and update the history.
4. **Frame generation (optional, asynchronous compute queue).** Pool, flow, warp, blend; output t-0.5.
5. **Present.** UI is composited last, onto both real and generated frames.

Scene cuts: a tile-based colour histogram difference between frames resets history (SR runs in spatial-only mode for that frame) and bypasses FG.

### 10.3 Memory Budget (1080p output from 640x360)

| Resource                  | Format                              | Size        |
| :------------------------ | :---------------------------------- | :---------- |
| History ping and pong     | 2 x `R16G16B16A16_FLOAT`, 1920x1080 | 33.18 MB    |
| LR colour                 | `R16G16B16A16_FLOAT`, 640x360       | 1.84 MB     |
| Dilated motion vectors    | `R16G16_FLOAT`, 640x360             | 0.92 MB     |
| Depth                     | `R32_FLOAT`, 640x360                | 0.92 MB     |
| SR input                  | 12 channels FP16, 640x360           | 5.53 MB     |
| SR activations, ping-pong | 2 x 24 channels FP16, 640x360       | 22.12 MB    |
| SR projection output      | 27 channels FP16, 640x360           | 12.44 MB    |
| FG activations, ping-pong | 2 x 32 channels FP16, 480x270       | 16.59 MB    |
| SR and FG weights         | FP16                                | 0.12 MB     |
| **Total**                 |                                     | **93.7 MB** |

This is within the 100 MB criterion but above the earlier 61 MB estimate, which counted one SR activation buffer and omitted the projection output. Storing history as `R11G11B10_FLOAT` would save 16.6 MB (to about 77 MB) if quality allows. FG reads the HR frames through shader resource views, without copies.

Weights: SR Ultra is 34,659 x 2 bytes = 67.7 KiB, which needs two 64 KiB constant buffers in the HLSL path. FG is 24,009 x 2 bytes = 46.9 KiB and fits in one.

### 10.4 Deployment Paths

| Path                                                                 | Platforms               | Vendors                      | Notes                                                                                                                      |
| :------------------------------------------------------------------- | :---------------------- | :--------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| [DirectML](https://github.com/microsoft/DirectML) (primary)          | Windows                 | AMD, NVIDIA, Intel, Qualcomm | Records into the engine's D3D12 command list; Conv, PReLU and DepthToSpace supported                                       |
| Native HLSL compute (advanced)                                       | Windows D3D12           | All                          | Weights in constant buffers; pre- and post-processing fused with the first and last layers; DXC with `-enable-16bit-types` |
| [ncnn](https://github.com/Tencent/ncnn) Vulkan                       | Windows, Linux, Android | All                          | Export through PNNX; mature FP16 Vulkan backend                                                                            |
| TensorRT                                                             | Windows, Linux          | NVIDIA only                  | Fastest NVIDIA path; not cross-vendor                                                                                      |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) DirectML EP | Windows                 | AMD, NVIDIA, Intel           | Session dispatch and cross-queue synchronisation overhead; use for offline validation and parity tests only                |

Export flow: fuse the branches, export ONNX (opset 17, dynamic height and width), check parity in ONNX Runtime, then compile for DirectML, export ncnn through PNNX, build a TensorRT engine with FP16, or extract FP16 weights for HLSL. Opset 17 includes `GridSample`, needed for FG.

All pipeline resources are allocated once in a committed default heap at initialisation. History buffers swap roles each frame without copies, and passes are synchronised with fences or semaphores only.

## 11. Baselines

| Baseline                                                        | Type                      | Licence                      | Role                                           |
| :-------------------------------------------------------------- | :------------------------ | :--------------------------- | :--------------------------------------------- |
| FSR 3.1 (primary)                                               | Heuristic SR + FG         | MIT                          | Primary comparison at equal scale and hardware |
| FSR 1.0                                                         | Spatial upscaler          | MIT                          | Lower-bound spatial reference                  |
| Bicubic                                                         | Interpolation             | n/a                          | Trivial baseline                               |
| XeSS 1.3 (DP4a path)                                            | Neural SR, closed weights | SDK Apache 2.0               | Cross-vendor neural reference                  |
| DLSS 3.x                                                        | Neural SR + FG, closed    | Proprietary                  | Quality ceiling reference on RTX 40 only       |
| [RIFE](https://github.com/hzwer/Practical-RIFE) v4.x            | Neural interpolation      | Newer weights non-commercial | FG quality ceiling                             |
| [IFRNet](https://github.com/ltkong218/IFRNet)                   | Neural interpolation      | Apache 2.0                   | Commercially usable FG baseline                |
| [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus) | Neural video SR           | Apache 2.0                   | Offline SR quality ceiling                     |
| ECBSR                                                           | Lightweight SR            | MIT                          | Lightweight SR alternative                     |
| Rep-TNSR v1                                                     | Lightweight temporal SR   | Project-owned                | Ablation baseline                              |

## 12. Evaluation Protocol

### 12.1 Configurations

| Config | Input    | Output    | Scale | FG  | FSR 3.1 mode           | Purpose                  |
| :----- | :------- | :-------- | :---- | :-- | :--------------------- | :----------------------- |
| C1     | 640x360  | 1920x1080 | 3x    | No  | Ultra Performance      | Primary SR               |
| C2     | 960x540  | 1920x1080 | 2x    | No  | Performance            | Medium scale             |
| C3     | 1280x720 | 1920x1080 | 1.5x  | No  | Quality                | Low scale                |
| C4     | 640x360  | 1920x1080 | 3x    | Yes | Ultra Performance + FG | Full pipeline            |
| C5     | 1280x720 | 2560x1440 | 2x    | Yes | Performance + FG       | Higher output resolution |
| C6     | 1280x720 | 3840x2160 | 3x    | No  | Ultra Performance      | 4K output                |

### 12.2 Test Sequences

Five sequences of 300 frames each. The first three use held-out camera paths in training scenes (in-distribution). The last two come from held-out scenes (out-of-distribution).

| Sequence        | Scene                            | Stresses                                               |
| :-------------- | :------------------------------- | :----------------------------------------------------- |
| Bistro Interior | Bistro (held-out path)           | Static camera, fine detail, specular surfaces          |
| City Chase      | City Park (held-out path)        | Fast motion, disocclusions, vehicles, reflections      |
| SciFi Corridor  | SciFi Corridor (held-out path)   | Emissive materials, reflections, thin cables and pipes |
| Reef Swim       | Underwater Reef (held-out scene) | Particles, translucency, caustics                      |
| Village Walk    | Cartoon Village (held-out scene) | Stylised shading, outlines, foliage                    |

### 12.3 Metrics

| Category        | Metric                    | Implementation                                                                                                                                                                                                              |
| :-------------- | :------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fidelity        | PSNR on luma, SSIM        | [torchmetrics](https://github.com/Lightning-AI/torchmetrics)                                                                                                                                                                |
| Perceptual      | LPIPS (AlexNet)           | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)                                                                                                                                         |
| Perceptual      | FLIP                      | [NVlabs/flip](https://github.com/NVlabs/flip)                                                                                                                                                                               |
| Video quality   | VMAF                      | [Netflix/vmaf](https://github.com/Netflix/vmaf) via ffmpeg                                                                                                                                                                  |
| Temporal        | Warping error             | Mean L1 difference between each output frame and the previous output warped by ground-truth motion vectors, over valid (non-disoccluded) pixels, averaged over frames                                                       |
| Temporal        | tOF                       | Mean L1 difference between RAFT flow on consecutive outputs and RAFT flow on consecutive ground-truth frames ([RAFT](https://github.com/princeton-vl/RAFT), reference code in [TecoGAN](https://github.com/thunil/TecoGAN)) |
| Temporal        | tLP                       | LPIPS between consecutive warped frames, output against ground truth                                                                                                                                                        |
| Flicker         | Temporal difference error | Mean absolute difference between the output's frame-to-frame change and the ground truth's, on luma (flow-free; also used by the notebooks)                                                                                 |
| Ghosting        | Ghost ratio               | Share of disoccluded pixels whose warping error exceeds 0.05                                                                                                                                                                |
| Disocclusion    | Masked LPIPS and PSNR     | Inside ground-truth disocclusion masks                                                                                                                                                                                      |
| FG motion       | End-point error           | Predicted intermediate flow against ground-truth flow on Sintel                                                                                                                                                             |
| Latency         | GPU time                  | D3D12 timestamp queries or CUDA events; P50 and P95                                                                                                                                                                         |
| Display latency | Click-to-photon           | PresentMon, with and without FG                                                                                                                                                                                             |

### 12.4 Human Evaluation

A two-alternative forced choice study: 20 or more participants (gamers and non-gamers), 30 paired comparisons each (ours against FSR 3.1), 5-second clips at native resolution and refresh rate, randomised side, and double-blind. Each trial asks which clip has better detail, smoother motion and fewer artefacts. Report the preference rate with a 95 percent confidence interval and Bradley-Terry scores. Significance requires a preference above 60 percent on a one-sided binomial test, p < 0.05.

### 12.5 Generalisation and Robustness

| Test              | Condition                                                          | Pass criterion                                                  |
| :---------------- | :----------------------------------------------------------------- | :-------------------------------------------------------------- |
| Unseen content    | 3 held-out UE5 scenes                                              | LPIPS within 15 percent of in-distribution                      |
| Unseen engine     | 2 Unity HDRP scenes                                                | LPIPS within 15 percent                                         |
| Unseen style      | Cartoon, photorealistic, pixel art                                 | LPIPS within 20 percent                                         |
| Unseen resolution | 4K output, not trained at 4K                                       | PSNR drop below 1.0 dB                                          |
| Cross-GPU         | GTX 1650 Mobile, RTX 3060, RTX 4070, RX 6600, RX 7800 XT, Arc A770 | Latency within tier budget; outputs equal within FP16 tolerance |
| Degraded inputs   | Zeroed MVs, depth noise of 5 percent, exposure off by 1 EV         | No NaN or corruption; PSNR drop below 3 dB                      |

### 12.6 Statistics

Report mean and standard deviation per sequence. Use a paired t-test, or a Wilcoxon signed-rank test when Shapiro-Wilk rejects normality, for every metric against FSR 3.1. Apply a Bonferroni correction over the 13 compared metrics (alpha 0.05 / 13, about 0.0038). Give 95 percent bootstrap confidence intervals and Cohen's d for the primary comparisons. The sample is 5 sequences x 300 frames = 1,500 frame-level measurements per metric.

## 13. Kaggle Notebooks (Current Work)

Both notebooks run end to end on Kaggle **GPU T4 x2** with Internet off, attach only the datasets in section 5.2, and stay inside an **8 hour** budget:

| Stage                    | Budget                                                     |
| :----------------------- | :--------------------------------------------------------- |
| Setup, discovery and EDA | about 10 minutes                                           |
| Cache build              | at most 25 minutes                                         |
| Training                 | 5 hours, shortened automatically if earlier stages overrun |
| Evaluation and export    | about 30 minutes, with 55 minutes reserved                 |
| Total                    | about 6.5 hours, 8 hours hard ceiling                      |

| Notebook                        | What it trains                                    | Evaluation                                                                                                                                                                                                                                                              |
| :------------------------------ | :------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `neural-supersampling.ipynb`    | Rep-TNSR Ultra, RGB input, 3x, Charbonnier + edge | Set5, Set14, B100, Urban100, Manga109 and held-out frames (PSNR-Y and SSIM-Y with a 3-pixel shave, against bicubic); Vid4 per-frame and temporal difference error; fused-model parity; T4 latency fused against unfused; ONNX export                                    |
| `neural-frame-generation.ipynb` | NeuralFG, RGB input, Charbonnier + census         | Vimeo-90K test (official list when present) and held-out REDS and Sintel frames against frame-average and copy baselines (PSNR, SSIM, interpolation error); error by motion size; Vid4 zero-shot; flow and blend-weight maps; T4 latency at 540p and 1080p; ONNX export |

Shared design:

- DistributedDataParallel with one process per GPU, launched from the notebook.
- A memory-mapped uint8 cache so the 4 CPU cores never starve the GPUs, with augmentation and degradation on the GPU.
- A batch size probed for the available VRAM, FP16 autocast and EMA weights.
- A learning-rate schedule tied to wall-clock time, so the run always ends on time.
- Live progress and training curves, with every metric and plot saved, and a final `outputs.zip`.

What they establish: the architectures train stably, fusion is exact, and the RGB-only models beat bicubic and frame averaging on standard benchmarks. Results also show how throughput and VRAM scale on T4. What they cannot establish: anything about G-buffer conditioning, temporal accumulation or FSR 3.1, which needs the Priority 2 and 3 data.

The previous supersampling run stopped after about 78 minutes, an hour into the first training epoch, without a recorded error. The rewrite removes the causes found in that version:

- a 15-minute full directory walk;
- a CPU-bound loader at about 210 images per second;
- validation crops of different sizes, which cannot be batched;
- LR and degraded copies mixed into training and test data;
- DataParallel instead of DDP;
- no time-based stopping.

## 14. Ablations

| ID       | Ablation                                   | Variants                                                          | Question                            |
| :------- | :----------------------------------------- | :---------------------------------------------------------------- | :---------------------------------- |
| SA1      | SR width                                   | 16 / 20 / 24 / 32                                                 | Quality against latency             |
| SA2      | SR depth                                   | 3 / 4 / 5 / 6 / 8 blocks                                          | Depth against latency               |
| SA3      | Edge branches                              | With and without Sobel and Laplacian                              | Value of fixed operators            |
| SA4      | Inputs                                     | 9 (v1) / 12 (v2) / 12 + reactive mask (SA4b)                      | Value of extra signals              |
| SA5      | Activation                                 | PReLU / ReLU / SiLU                                               | Quality and speed                   |
| SA6      | Projection init                            | Default / ICNR                                                    | Checkerboard artefacts              |
| SA7      | Temporal model                             | External history / learned recurrent aggregator                   | Value of recurrence                 |
| FA1      | FG resolution                              | Pool by 2 / 4 / 8                                                 | Resolution against quality          |
| FA2      | Flow refinement                            | One level / two-level pyramid                                     | Large motion                        |
| FA3      | Engine MVs                                 | With / without                                                    | Value of engine motion              |
| FA4      | Occlusion                                  | Learned blend / forward-backward consistency                      | Disocclusion quality                |
| FA5      | Motion source                              | Learned flow / engine-MV forward splatting (section 8.3)          | Accuracy and cost                   |
| FA6      | Flow pre-training                          | None / FlyingChairs and Sintel flow                               | Value of synthetic flow supervision |
| L1 to L5 | SR loss build-up                           | Charbonnier; + edge; + perceptual; + temporal; + frequency (full) | Contribution of each term           |
| L6       | L1 instead of Charbonnier                  | Full set                                                          | Robust loss choice                  |
| L7       | LPIPS as training loss                     | Replaces VGG term                                                 | Perceptual loss choice              |
| L8       | FLIP as training loss                      | Added to full set                                                 | Perceptual loss choice              |
| D1       | UE5 only                                   | 9 training scenes                                                 | Baseline data                       |
| D2       | UE5 + Sintel + TartanAir                   | Mixed                                                             | Domain diversity                    |
| D3       | Native LR renders / bicubic-downsampled HR | Both                                                              | Realistic aliasing                  |
| D4       | Degradation augmentation                   | With / without                                                    | Robustness                          |
| D5       | Scene count                                | 5 / 9 scenes                                                      | Data scale                          |
| T1       | Curriculum                                 | None / three-phase                                                | Value of phasing                    |
| T2       | Patch size                                 | 64 / 128 LR throughout                                            | Context size                        |
| T3       | Optimiser                                  | Adam / AdamW / SGD with momentum                                  | Optimiser choice                    |
| T4       | Schedule                                   | Cosine / step / warm restarts                                     | Schedule choice                     |

About 30 ablation runs at 200K iterations on one GPU, about 20 GPU-hours each, about 600 GPU-hours in total.

## 15. Existing Evidence: Rep-TNSR v1

The measurements below come from the earlier Rep-TNSR v1 project. They are reported here as context and have not been reproduced in this repository. Note that v1 was compared against FSR 1.0 (spatial only), not FSR 3.1.

### 15.1 Latency on GTX 1650 Mobile

Hardware: TU117, 896 CUDA cores, 4 GB GDDR5 at 128 GB/s, 50 W, Windows 11, driver 550.x, D3D12 timestamp queries.

| Pass                            | Dispatch       | Memory traffic (read / write) | Work            | GPU time     |
| :------------------------------ | :------------- | :---------------------------- | :-------------- | :----------- |
| Dilation, reprojection, YCoCg   | 80 x 45 groups | 4.58 MB / 3.68 MB             | 0.052 GFLOP     | 0.282 ms     |
| Fused FP16 trunk                | DirectML graph | 8.55 MB / 8.55 MB             | 7.962 GFLOP     | 2.145 ms     |
| Pixel shuffle and decompression | 80 x 45 groups | 12.44 MB / 16.59 MB           | 0.015 GFLOP     | 0.184 ms     |
| Barriers and queue              | D3D12 timeline | negligible                    | negligible      | 0.112 ms     |
| **Total**                       |                | **54.39 MB**                  | **8.029 GFLOP** | **2.723 ms** |

Memory traffic is 14 percent of the 384 MB that can move in 3 ms at 128 GB/s.

### 15.2 Quality (360p to 1080p)

| Method                 | PSNR (dB) | SSIM      | IF-SSIM (v1 report) | Warping error (x 1e-3) | GTX 1650M time |
| :--------------------- | :-------- | :-------- | :------------------ | :--------------------- | :------------- |
| Bicubic                | 27.34     | 0.812     | 0.892               | 8.42                   | 0.08 ms        |
| FSR 1.0                | 28.12     | 0.835     | 0.901               | 7.91                   | 0.42 ms        |
| QuickSRNet-Medium      | 31.05     | 0.884     | 0.914               | 6.84                   | 2.21 ms        |
| **Rep-TNSR v1**        | **34.82** | **0.941** | **0.986**           | **1.72**               | 2.72 ms        |
| Native 1080p reference | n/a       | 1.000     | 0.994               | 1.15                   | n/a            |

v1 is 6.70 dB above FSR 1.0 and 3.77 dB above QuickSRNet-Medium. The gap to FSR 3.1, which accumulates jittered history, is unknown and is the subject of SR-001.

## 16. Failure Modes and Mitigations

| Failure                            | Cause                                           | Detection                                       | Mitigation                                                                                                  |
| :--------------------------------- | :---------------------------------------------- | :---------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| Ghosting on fast motion            | History clamp too loose                         | High warping error in motion regions            | Tighten clamp (1.25 to 1.0 standard deviations); raise depth sensitivity of validity (10 to 15)             |
| Checkerboard artefacts             | Pixel-shuffle initialisation                    | Pixel-level inspection                          | ICNR initialisation (SA6)                                                                                   |
| Flicker on thin geometry           | Detail below the LR sampling rate               | Flicker spikes on wire and fence sequences      | More jitter phases; tighter clamp at depth edges; temporal loss weight                                      |
| FG double images on fast rotation  | Motion beyond the quarter-resolution flow range | High end-point error on fast-rotation sequences | Two-level flow (FA2); engine MVs (FA3); cut detection bypass                                                |
| UI distortion in FG                | UI composited before FG                         | UI-heavy test scenes                            | Composite UI after FG; for engines that cannot separate UI, extract a UI mask from pre- and post-UI buffers |
| Particle and transparency smearing | No reliable MVs or depth                        | Reactive-mask coverage analysis                 | Zero temporal loss in reactive regions; favour the current frame there                                      |
| Colour shift in HDR                | Compression round-trip error                    | PSNR drop on bright regions                     | FP32 for compression and decompression                                                                      |
| Poor generalisation                | Training distribution mismatch                  | LPIPS degradation on held-out sets              | More scenes and engines; style and colour augmentation                                                      |
| NaN in FP16                        | Inputs beyond the FP16 range (65,504)           | Runtime NaN check                               | Clamp inputs; FP32 warp coordinates                                                                         |
| Bandwidth saturation               | Too many intermediate buffers                   | Profiler memory utilisation above 90 percent    | Fuse pre- and post-processing into the first and last layers                                                |
| Judder at low frame rate           | Linear motion assumption breaks                 | Frame time above 25 ms                          | Disable FG below 40 FPS base rate                                                                           |

## 17. Claims of Superiority over FSR 3.1

If supported at the significance levels of section 12.6, the following claims would establish meaningful superiority:

1. **Spatial quality.** PSNR at least 1.5 dB higher and LPIPS at least 20 percent lower, averaged over the 5 sequences at each configuration, p < 0.01.
2. **Temporal stability.** Warping error at least 25 percent lower and tOF at least 20 percent lower on the dynamic sequences (City Chase, SciFi Corridor, Reef Swim), p < 0.01.
3. **Frame generation.** PSNR at least 2.0 dB higher and LPIPS at least 25 percent lower than FSR 3.1 FG, p < 0.01.
4. **Latency.** Total pipeline time within 120 percent of FSR 3.1 on the same GPU, and within the absolute targets of section 2.
5. **Generalisation.** LPIPS degradation below 15 percent on held-out scenes and engines.
6. **Human preference.** At least 60 percent preference with at least 20 participants, p < 0.05, one-sided binomial.

## 18. Roadmap and Experiments

### 18.1 Phases

| Phase                | Weeks    | Deliverables                                                                                        | Acceptance                                                                       |
| :------------------- | :------- | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- |
| 0. Kaggle validation | 1 to 2   | Both notebooks run within 8 hours; RGB baselines on SR benchmarks, Vimeo-90K and Vid4               | Fusion parity below 1e-4; clear gains over bicubic and frame averaging           |
| 1. Falsification     | 2 to 4   | Bistro capture; FSR 3.1 Ultra Performance dumps; v1 retrained; SR-001 report                        | SR-001 does not falsify H1                                                       |
| 2. Rep-TNSR v2       | 4 to 8   | 12-channel pipeline; three tiers; full curriculum; 12-scene capture; SR ablations; temporal metrics | Section 2 SR targets met with significance                                       |
| 3. NeuralFG          | 9 to 13  | Middle-frame captures; FG training; FG ablations; comparison with FSR 3.1 FG, RIFE, IFRNet          | Section 2 FG targets; FG P95 at most 2.5 ms on RTX 3060                          |
| 4. Integration       | 14 to 18 | DirectML integration; ncnn and TensorRT exports; frame pacing; latency and VRAM profiling on 6 GPUs | Within 120 percent of FSR 3.1 latency; VRAM at most 100 MB; no NaN or corruption |
| 5. Generalisation    | 19 to 22 | Unity and held-out UE5 evaluation; style and degraded-input tests; human study; final report        | All section 17 claims tested with confidence intervals and effect sizes          |

### 18.2 Smallest Falsifying Experiment (SR-001)

1. Train Rep-TNSR v1 (17.5K parameters, 4 blocks, 20 channels) on Bistro training paths only: 100K iterations, Charbonnier + edge, one GPU, about 6 hours.
2. Capture FSR 3.1 **Ultra Performance** output (3x, 640x360 to 1920x1080) on a held-out 300-frame Bistro path with the FidelityFX SDK sample, dumping frames with RenderDoc.
3. Compute luma PSNR, RGB SSIM and LPIPS (AlexNet) per frame on identical inputs.

**Falsified if** FSR 3.1 has both higher PSNR and lower LPIPS. That would indicate that larger networks, a different architecture or more input signals are needed.

**Expectation, unverified.** v1 beat FSR 1.0 by 6.7 dB, but FSR 3.1 accumulates jittered history, so the gap may be much smaller. This experiment decides whether the project continues as planned.

### 18.3 Experiment Matrix

| ID             | Model                           | Data                          | Losses               | Scale    | Purpose                        |
| :------------- | :------------------------------ | :---------------------------- | :------------------- | :------- | :----------------------------- |
| KG-SR          | Rep-TNSR Ultra, RGB             | Kaggle (section 5.2)          | Charbonnier + edge   | 3x       | Pipeline validation (notebook) |
| KG-FG          | NeuralFG, RGB                   | Kaggle (section 5.2)          | Charbonnier + census | n/a      | Pipeline validation (notebook) |
| SR-001         | Rep-TNSR v1                     | Bistro                        | Charbonnier + edge   | 3x       | Falsification                  |
| SR-002         | Rep-TNSR v2 Ultra               | 9 training scenes             | Full curriculum      | 3x       | Primary SR quality result      |
| SR-003         | Rep-TNSR v2 Performance         | 9 scenes                      | Full                 | 3x       | GTX 1650 tier                  |
| SR-004         | Rep-TNSR v2 Quality             | 9 scenes                      | Full                 | 3x       | RTX 3060 latency claim         |
| SR-S1, SR-S2   | Rep-TNSR v2 Quality             | 9 scenes                      | Full                 | 2x, 1.5x | Other scale factors            |
| FG-001         | NeuralFG, 10-channel            | 9 scenes with middle frames   | Full                 | n/a      | Primary FG result              |
| GEN-001 to 003 | SR-004 + FG-001                 | Held-out UE5, Unity, stylised | n/a                  | 3x       | Generalisation                 |
| LAT-001        | All tiers and FSR 3.1           | Standard sequence             | n/a                  | 3x       | Latency on all GPUs            |
| HUM-001        | SR-004 + FG-001 against FSR 3.1 | 5 test sequences              | n/a                  | 3x       | Human study                    |

Ablation runs (section 14) are added to this matrix as SA, FA, L, D and T experiments.

## 19. Assumptions and Risks

### 19.1 Assumptions

| ID  | Assumption                                                       | If wrong                         | Mitigation                                                       |
| :-- | :--------------------------------------------------------------- | :------------------------------- | :--------------------------------------------------------------- |
| AS1 | A sub-50K-parameter network can beat FSR 3.1's temporal upscaler | Quality targets missed           | SR-001 first; tiers up to 50K; consider 50K to 100K if justified |
| AS2 | Engine MVs are accurate for geometric motion                     | FG errors on animated meshes     | Learned flow refines engine MVs (FA3)                            |
| AS3 | DirectML matches hand-written HLSL speed                         | Latency targets missed           | HLSL path is designed in                                         |
| AS4 | 9 training scenes give enough diversity                          | Generalisation fails             | More scenes; Sintel and TartanAir (D2)                           |
| AS5 | FP16 is sufficient for all intermediate values                   | NaN or quality loss              | FP32 for warps and luma compression                              |
| AS6 | Tensor cores are reachable through DirectML meta-commands        | Ultra misses the RTX 3060 target | Ship Quality tier as the default                                 |

### 19.2 Unavailable Dependencies

DLSS and XeSS weights are proprietary, so they are compared through captured frames only. DLSS frame generation needs RTX 40 hardware. Commercial game captures cannot be redistributed, so open UE5 scenes are used for all reproducible benchmarks. Unreal Engine cannot be redistributed; capture scripts and captured data can.

### 19.3 Risks

| Risk                                       | Severity | Likelihood         | Mitigation                                                        |
| :----------------------------------------- | :------- | :----------------- | :---------------------------------------------------------------- |
| Quality gap to FSR 3.1 too small           | High     | Medium             | More capacity, more inputs, bottleneck attention as a last resort |
| Ultra tier over budget on GTX 1650 class   | High     | High (by estimate) | Performance and Quality tiers                                     |
| Flicker worse than FSR 3.1                 | High     | Low                | Stronger temporal loss; tighter clamping                          |
| FG disocclusion quality worse than FSR 3.1 | Medium   | Medium             | Disocclusion-heavy data; inpainting head; two-level flow          |
| Poor generalisation                        | Medium   | Medium             | Data diversity; domain randomisation                              |
| Export regression (ONNX, DirectML)         | Low      | Low                | Automated parity tests at every export                            |

## 20. Planned Repository Layout

The repository currently holds `RESEARCH.md` and the two Kaggle notebooks. The planned layout:

```text
neuralss/
  configs/            YAML per tier, FG, FSR evaluation and each ablation (OmegaConf)
  neuralss/
    models/           repconv_block, sr_net, fg_net, fuse
    losses/           charbonnier, edge, perceptual, temporal, frequency, census
    data/             G-buffer dataset, augmentation with MV handling, preprocessing, patch cache
    metrics/          spatial, perceptual, temporal, VMAF wrapper, ghosting
    engine/           DDP trainer, evaluator, curriculum, checkpointing
    export/           ONNX, ncnn, TensorRT
  scripts/            dataset download with checksums, UE5 capture, cache generation, benchmark, ablation sweep, latency profiling
  deploy/
    directml/         SuperResolutionSystem (C++), PreProcessTemporal.hlsl, PostProcessReconstruct.hlsl
    ncnn_vulkan/      SR and FG pipelines
  notebooks/          Kaggle notebooks
  tests/              fusion parity, losses, dataset and MV augmentation, metrics, ONNX parity, latency
  train.py, evaluate.py, infer.py
```

## 21. Reproducibility Checklist

- [ ] Seeds fixed (Python, NumPy, PyTorch, CUDA) and recorded with each run.
- [ ] `CUBLAS_WORKSPACE_CONFIG=:16:8` and deterministic algorithms for final runs.
- [ ] Package versions pinned (PyTorch, torchvision, lpips, torchmetrics, OmegaConf, OpenEXR).
- [ ] Dataset downloads verified by SHA-256.
- [ ] One `torchrun` command plus one config file reproduces each training run.
- [ ] Checkpoints hold model, optimiser, scheduler, scaler, EMA, RNG states, iteration and best metric.
- [ ] Evaluation is deterministic for a given checkpoint and data.
- [ ] Metrics use the listed libraries, or documented reimplementations.
- [ ] GPU, driver, CUDA, PyTorch versions and git commit logged with every run.
- [ ] Unit tests: fused block and full-model parity, Charbonnier gradient at zero, temporal loss zero where the mask is zero, ONNX parity, latency within budget on target hardware.
- [ ] Runs tracked in [Weights & Biases](https://wandb.ai/) or [MLflow](https://github.com/mlflow/mlflow), with configuration hash, hardware, dataset version, and validation and test metrics.
