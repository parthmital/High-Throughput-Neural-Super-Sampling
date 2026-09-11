# NeuralSS: High-Throughput Neural Super-Sampling and Frame Generation

> GPU-agnostic neural upscaling and frame generation pipeline. Explicit objective: measurably outperform AMD FSR 3.1 in perceptual quality, temporal stability, end-to-end latency, and cross-game generalisation.

## 1. Research Objectives, Hypotheses, and Success Criteria

### 1.1 Core Hypothesis

A lightweight, temporally-recurrent neural architecture consuming engine G-buffers (colour, motion vectors, depth, exposure) can simultaneously produce spatially superior and temporally more stable upscaled frames, and generate higher-fidelity interpolated frames, compared to FSR 3.1's hand-crafted heuristic pipeline, while meeting the same latency and hardware-agnostic deployment constraints.

### 1.2 Sub-Hypotheses

| ID  | Hypothesis                                                                                                                                                             | Falsification Criterion                                                               |
| :-- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| H1  | Neural SR with G-buffer conditioning and structural reparameterisation produces higher perceptual quality than FSR 3.1 temporal upscaling at equivalent scale factors. | LPIPS(ours) >= LPIPS(FSR 3.1) on >= 3 of 5 test scenes.                               |
| H2  | Learned temporal accumulation with variance-guided gating reduces ghosting and flicker below FSR 3.1 levels.                                                           | $E_{\text{warp}}$(ours) >= $E_{\text{warp}}$(FSR 3.1) OR tOF(ours) >= tOF(FSR 3.1).   |
| H3  | Neural optical flow estimation plus learned warping produces fewer disocclusion artefacts than FSR 3.1's hierarchical compute-based flow.                              | Human preference rate < 55% vs FSR 3.1 frame generation on the disocclusion test set. |
| H4  | The full SR + FG pipeline can execute within FSR 3.1's latency envelope (SR: < 2.0 ms, FG: < 3.0 ms at 1080p on RTX 3060 class hardware).                              | End-to-end latency exceeds FSR 3.1 by > 20% on any tested GPU.                        |
| H5  | Training on diverse synthetic G-buffer data generalises to unseen games and engines without fine-tuning.                                                               | Quality degradation > 15% LPIPS on any of 3 held-out game datasets.                   |

### 1.3 Measurable Success Criteria (vs FSR 3.1 Quality Mode, 67% Render Scale)

| Metric                               | FSR 3.1 Baseline (estimated) | Target            | Measurement                                      |
| :----------------------------------- | :--------------------------- | :---------------- | :----------------------------------------------- |
| PSNR (dB)                            | 33.5                         | >= 35.0 (+1.5 dB) | Y-channel, per-frame average over test sequences |
| SSIM                                 | 0.935                        | >= 0.950          | Full RGB, per-frame average                      |
| LPIPS (AlexNet)                      | 0.085                        | <= 0.065 (-23%)   | Per-frame average, lower is better               |
| tOF ($\times 10^{-3}$)               | 3.8                          | <= 3.0            | RAFT-estimated flow on output vs GT              |
| $E_{\text{warp}}$ ($\times 10^{-3}$) | 2.4                          | <= 1.8            | GT motion-vector warped consistency              |
| VMAF                                 | 82                           | >= 87             | Per-sequence average via Netflix/vmaf            |
| Frame Gen PSNR (dB)                  | 28.5                         | >= 30.5 (+2.0 dB) | Interpolated frame vs GT mid-frame               |
| Frame Gen LPIPS                      | 0.110                        | <= 0.080          | Interpolated frame quality                       |
| SR Latency (1080p, RTX 3060)         | 1.2 ms                       | <= 1.5 ms         | D3D12 timestamp queries, P95                     |
| FG Latency (1080p, RTX 3060)         | 2.0 ms                       | <= 2.5 ms         | D3D12 timestamp queries, P95                     |
| VRAM (total pipeline)                | ~60 MB                       | <= 80 MB          | PIX/RenderDoc resource inspector                 |
| Parameters (SR)                      | 0 (heuristic)                | <= 50K            | `sum(p.numel())`                                 |
| Parameters (FG)                      | 0 (heuristic)                | <= 200K           | `sum(p.numel())`                                 |

### 1.4 FSR 3.1 Baseline Characterisation

FSR 3.1 ([GPUOpen-LibrariesAndSDKs/FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK), MIT licence) is the current cross-vendor baseline. Key architectural properties:

**Temporal upscaling pipeline** (heuristic, no neural inference):

- Sub-pixel camera jitter via Halton(2,3) low-discrepancy sequence.
- Temporal reprojection using 2D screen-space motion vectors and linear depth.
- History rectification via YCoCg colour-space variance bounding (mean $\pm$ $\gamma \cdot \sigma$).
- Lanczos-based spatial reconstruction for accumulating jittered samples.
- Rewritten accumulation weighting in v3.1 to reduce temporal fizziness on thin geometry.

**Optical flow frame generation pipeline** (compute-based, no neural inference):

- Hierarchical optical flow estimation on luminance delta between frames $t-1$ and $t$.
- Motion vector fusion: engine geometric MVs (dilated) merged with optical flow field.
- Bidirectional warping of frames $t-1$ and $t$ to synthesise frame $t-0.5$.
- Disocclusion hole-filling via adjacent valid sample blending.
- Scene-cut detection: luminance delta threshold bypass.
- HUD/UI decoupled via separate texture overlay within a custom swapchain proxy.

**Required engine inputs**: LR colour buffer, screen-space motion vectors (R16G16_FLOAT), depth buffer (preferably inverted R32_FLOAT), exposure value, reactive mask (optional), transparency and composition mask (optional), HUD/UI texture with alpha (required for FG).

**Performance characteristics** (Quality mode, mid-range GPU):

- Upscaling: 0.8 ms to 1.4 ms (1440p to 4K).
- Frame generation: 1.2 ms to 2.5 ms additional compute.
- Inherent latency: FG introduces at least one baseline frame time of delay (interpolation, not extrapolation). AMD Radeon Anti-Lag 2 SDK or NVIDIA Reflex mitigate input latency.

**Known failure modes**: disocclusion smearing on fast camera moves, ghosting on thin geometry (fences, wires) during rapid motion, optical flow mismatch on extreme camera rotation (double-image artefacts), HUD/UI distortion without proper UI separation, particle/transparency smearing without reactive masks, sluggish feel below 50-60 FPS base rate.

**Competitor landscape** (for reference only, not reproducible baselines):

- NVIDIA DLSS 3/3.7: neural SR (Tensor Cores) + OFA-based FG (RTX 40 series only). Closed source. ([github.com/NVIDIA/DLSS](https://github.com/NVIDIA/DLSS), API headers only.)
- Intel XeSS 1.3: neural SR (XMX on Arc, DP4a fallback). No integrated FG. SDK open source (Apache 2.0, [github.com/intel/xess](https://github.com/intel/xess)), model weights proprietary.
- FSR 3.1 is the only fully open-source, cross-vendor upscaling and frame generation suite.

## 2. Compute Boundaries and Hardware Execution Dynamics

Reconstructing low-resolution video and render streams from 360p ($640 \times 360$) to native 1080p ($1920 \times 1080$) at 60 FPS on entry-level mobile silicon requires strict adherence to arithmetic and memory bandwidth budgets. The target deployment hardware, exemplified by the NVIDIA GeForce GTX 1650 Mobile based on the TU117 Turing architecture, contains 896 active CUDA cores operating at a typical boost frequency of 1515 MHz. Unlike its higher-tier desktop or RTX-branded counterparts, the TU117 silicon contains zero dedicated matrix-acceleration engines (Tensor Cores). Consequently, all neural inference workloads must execute through standard Streaming Multiprocessor (SM) arithmetic logic units (ALUs).

The Turing SM microarchitecture features an independent half-precision datapath supporting dual-issue FP16 operations (FP16x2 packed math). The theoretical upper compute bounds are governed by the core configuration and operational clock rate:

$$\text{Peak}_{\text{FP32}} = 896\,\text{Cores} \times 2\,\frac{\text{FLOP}}{\text{Cycle}} \times 1.515\,\text{GHz} = 2.715\,\text{TFLOPS}$$

$$\text{Peak}_{\text{FP16}} = 2 \times \text{Peak}_{\text{FP32}} = 5.430\,\text{TFLOPS}$$

A 60 FPS runtime enforces a total target frame time of 16.67 ms. Allocating a maximum latency budget of 3.00 ms to the super-resolution reconstruction pass leaves approximately 13.67 ms for the host engine's primary graphics and compute pipelines. On real-world Turing SM architectures, when factoring in instruction cache thrashing, register allocation stalls, and thread divergence, standard compute shaders achieve an arithmetic efficiency of 60% to 65% of theoretical peak:

$$\text{Throughput}_{\text{Sustained FP16}} \approx 5.430\,\text{TFLOPS} \times 0.65 = 3.529\,\text{TFLOPS} = 3529\,\text{GFLOPS}$$

$$\text{FLOP Budget}_{(3.0\,\text{ms})} = 3529\,\text{GFLOPS} \times 0.003\,\text{s} \approx 10.587\,\text{GFLOPS} \equiv 5.293\,\text{GMACs}$$

Memory bandwidth imposes a co-equal boundary condition. The TU117 interfaces with 4 GB of memory across a 128-bit bus, providing $128\text{ GB/s}$ of peak bandwidth with GDDR5 or $192\text{ GB/s}$ with GDDR6. Over a 3.00 ms frame window, the maximum volume of data that can be moved across the GDDR5 physical interface is:

$$\text{Data Transfer Limit}_{(3.0\,\text{ms})} = 128\,\text{GB/s} \times 0.003\,\text{s} = 384\,\text{MB}$$

This constraint prohibits network designs that rely on multi-scale feature pyramids, widespread residual skip concatenations, or attention mechanisms. Such operators require extensive activation caching, generate fragmented global memory access patterns, and induce frequent L2 cache invalidation, which saturates memory bandwidth and increases execution latency.

| Architectural Parameter | Target Mobile GPU (TU117 / GTX 1650 Mobile)                              | Production Constraint (3.0 ms Budget)                                                                 |
| :---------------------- | :----------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| Compute Units           | 896 CUDA Cores, 0 Tensor Cores                                           | Must target generic FP16 vector instructions                                                          |
| Peak Throughput         | 2.715 TFLOPS FP32 / 5.430 TFLOPS FP16                                    | Maximum inference computation $\le 10.58\text{ GFLOPS}$                                               |
| VRAM Capacity & Bus     | 4 GB GDDR5/GDDR6, 128-bit bus                                            | Total pass memory footprint $\le 50\text{ MB}$ (< 1.25% VRAM)                                         |
| Memory Bandwidth        | 128 GB/s (GDDR5) to 192 GB/s (GDDR6)                                     | Total DRAM traffic per pass $\le 150\text{ MB}$ (< 39% interface capacity)                            |
| Spatial Transform       | $640 \times 360$ ($360\text{p}$) $\to 1920 \times 1080$ ($1080\text{p}$) | Scale factor $s = 3.0$ ($N_{\text{in}} = 230,400\text{ px} \to N_{\text{out}} = 2,073,600\text{ px}$) |

## 3. Datasets: Training, Validation, and Testing

### 3.1 Training Data Sources

| Dataset                           | Content                                                                                                                                            | Access                                                                                       | Licence                                               | G-Buffers Available                                    | Role                                                  |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- | :---------------------------------------------------- | :----------------------------------------------------- | :---------------------------------------------------- |
| **Custom UE5 Captures** (primary) | 8 to 12 Unreal Engine 5 scenes: Bistro, Sun Temple, Valley, City Park, SciFi Corridor, Medieval Game Environment, Stylised Forest, Industrial Zone | Self-generated via [UnrealCV](https://unrealcv.org/) plugin + custom capture actor           | Project-internal (UE5 EULA permits data distribution) | Colour, MV, Depth, Roughness, Normal, Exposure, Jitter | Primary training                                      |
| **ExtraSS Dataset**               | Multi-scene rendered sequences with G-buffers, Halton jitter, disocclusion masks                                                                   | [NJU-3DV/ExtraSS](https://github.com/NJU-3DV/ExtraSS)                                        | Academic research use                                 | Colour, MV, Depth, Disocclusion                        | Training + validation                                 |
| **MPI Sintel**                    | Animated film render passes with dense GT optical flow                                                                                             | [sintel.is.tue.mpg.de](http://sintel.is.tue.mpg.de/)                                         | CC BY 3.0                                             | Clean, Final, Flow, Depth, Occlusion                   | Validation + optical flow eval                        |
| **REDS**                          | 300 real-world dynamic video sequences at 720p                                                                                                     | [seungjunnah.github.io/Datasets/reds.html](https://seungjunnah.github.io/Datasets/reds.html) | CC BY 4.0                                             | None (colour only)                                     | Cross-domain generalisation                           |
| **Vimeo-90K Septuplet**           | 89,800 real-world video clips (448x256)                                                                                                            | [toflow.csail.mit.edu](https://toflow.csail.mit.edu/)                                        | Academic non-commercial                               | None (colour only)                                     | FG pretraining only (non-commercial constraint noted) |
| **TartanAir**                     | AirSim synthetic visual SLAM dataset                                                                                                               | [theairlab.org/tartanair](https://theairlab.org/tartanair/)                                  | Research                                              | Flow, Depth, Normals, Stereo                           | Supplementary optical flow training                   |
| **VIPER / Playing for Data**      | GTA V extracted sequences                                                                                                                          | [playing-for-benchmarks.org](https://playing-for-benchmarks.org/)                            | Research                                              | Flow, Depth, Semantics                                 | Supplementary training                                |

### 3.2 Held-Out Generalisation Test Sets (Excluded from Training)

| Dataset                                                                                    | Purpose                                              |
| :----------------------------------------------------------------------------------------- | :--------------------------------------------------- |
| 3 UE5 scenes with distinct art styles (Desert Landscape, Underwater Reef, Cartoon Village) | Unseen game content and art style generalisation     |
| 2 Unity HDRP captures (urban environment, interior architecture)                           | Unseen engine generalisation                         |
| FSR 3.1 SDK Bistro sample (frame dumps extracted via RenderDoc)                            | Direct FSR 3.1 comparison under identical conditions |

### 3.3 Dataset Construction and Splits

**Per-scene capture specification:**

- 60-second sequences at 60 FPS = 3,600 frames per sequence.
- Camera paths: 3 per scene (static orbit, dynamic chase, rapid cut montage).
- Total: 12 scenes $\times$ 3 paths $\times$ 3,600 frames $\approx$ 129,600 training frames.
- Split: 80% train / 10% validation / 10% test. Split by scene (not by frame) to prevent temporal leakage.

**Ground truth generation:**

- HR reference: native 1080p rendered with $16\times$ SSAA (temporal accumulation over 16 jittered sub-frames) to establish clean geometric edges.
- LR input: native 360p render (NOT downsampled from HR) with 9-phase Halton(2,3) sub-pixel camera jitter applied.
- Additional LR scales: 540p (for $2\times$ scale), 720p (for $1.5\times$ scale) renders.

**Per-frame data record:**

```
frame_NNNNNN/
  color_lr.exr            # R16G16B16A16_FLOAT, W_LR x H_LR
  color_hr.exr            # R16G16B16A16_FLOAT, W_HR x H_HR (GT)
  motion_vectors.exr       # R16G16_FLOAT, W_LR x H_LR (backward screen-space MV)
  depth.exr               # R32_FLOAT, W_LR x H_LR (linear camera depth)
  normal.exr              # R16G16B16_FLOAT, W_LR x H_LR (world-space normals)
  roughness.png           # R8, W_LR x H_LR (material roughness)
  exposure.json           # {"value": float, "jitter_x": float, "jitter_y": float, "frame_index": int, "phase_k": int}
```

### 3.4 Licensing Constraints

| Component           | Licence                 | Commercial Use         | Constraint                                         |
| :------------------ | :---------------------- | :--------------------- | :------------------------------------------------- |
| Custom UE5 captures | Project-internal        | Yes                    | UE5 EULA permits captured data distribution        |
| ExtraSS             | Academic research       | No                     | Training use restricted to non-commercial research |
| MPI Sintel          | CC BY 3.0               | Yes (with attribution) | Attribution required                               |
| REDS                | CC BY 4.0               | Yes (with attribution) | Attribution required                               |
| Vimeo-90K           | Academic non-commercial | No                     | Exclude if targeting commercial deployment         |
| TartanAir           | Research                | Restricted             | Check before commercial use                        |
| VIPER               | Research                | Restricted             | Check before commercial use                        |

For commercial deployment: train exclusively on custom UE5 captures + REDS + Sintel.

## 4. Synthetic Data Generation Pipeline

```
UnrealCV Plugin / Custom UE5 Capture Actor
│
├── Configure: Halton(2,3) jitter sequence (K=9 phases for s=3)
├── Configure: Camera path (keyframed or AI-driven random walk)
├── Per frame:
│   ├── Inject sub-pixel jitter into projection matrix (see Section 6.1)
│   ├── Render LR colour (360p, point-sampled, no TAA, no post-processing)
│   ├── Render HR reference (1080p, 16x SSAA temporal accumulation)
│   ├── Export screen-space backward MV buffer (R16G16_FLOAT)
│   ├── Export linear depth (R32_FLOAT, camera-space Z)
│   ├── Export world normals (R16G16B16_FLOAT) + roughness (R8)
│   ├── Record exposure value from auto-exposure luminance histogram
│   └── Save frame metadata JSON (jitter offset, phase index, camera matrix)
│
├── Motion variation injection (per-scene):
│   ├── Static objects with camera-only motion (architecture, terrain)
│   ├── Animated skeletal meshes (characters, vehicles, animals)
│   ├── Particle systems (fire, smoke, sparks, rain, snow)
│   ├── Alpha-tested foliage and chain-link fencing
│   ├── Transparent / translucent surfaces (glass, water, ice)
│   ├── Dynamic lighting changes (day/night cycle, explosions, muzzle flash)
│   ├── Screen-space reflections and ray-traced reflections
│   └── Camera cuts (hard scene transitions, teleports)
│
├── Disocclusion ground truth:
│   ├── Forward-warp frame t-1 depth to frame t coordinate space
│   ├── Mark pixels with depth inconsistency > threshold as disoccluded
│   └── Export binary disocclusion mask per frame
│
└── UI / Transparency handling:
    ├── UI elements rendered to separate overlay texture with alpha
    ├── Training data excludes UI compositing
    ├── Reactive mask channel marks pixels where temporal history
    │   should be discounted (transparent particles, reflections)
    └── Follows FSR 3.1 reactive mask convention for compatibility
```

**Frame generation training data extension:**

For each consecutive pair $(t-1, t)$, additionally render the ground-truth mid-frame $t-0.5$ by interpolating the camera matrix and re-rendering at the intermediate timestamp. This provides direct supervision for the frame generation network.

```
frame_NNNNNN/
  ... (standard buffers as above)
  color_hr_mid.exr         # GT frame at t-0.5 for FG training
  motion_vectors_mid.exr   # MV at t-0.5 (optional, for flow supervision)
```

## 5. Data Preprocessing, Augmentation, Normalisation, and Storage

### 5.1 Preprocessing Pipeline

```python
def preprocess_frame(
    color_lr: Tensor,   # [3, H, W], linear HDR RGB
    color_hr: Tensor,   # [3, sH, sW], linear HDR RGB
    mv: Tensor,         # [2, H, W], screen-space backward MV in pixels
    depth: Tensor,      # [1, H, W], linear camera depth
    exposure: float,    # scalar exposure multiplier from engine histogram
    jitter_xy: tuple,   # (jitter_x, jitter_y) in NDC
    phase_k: int,       # jitter phase index [0, K-1]
    W_lr: int, H_lr: int,
    K: int = 9,         # total jitter phases (s^2)
) -> dict:
    # 1. Exposure normalisation
    color_lr_exp = color_lr * exposure
    color_hr_exp = color_hr * exposure

    # 2. RGB -> YCoCg
    color_lr_ycocg = rgb_to_ycocg(color_lr_exp)
    color_hr_ycocg = rgb_to_ycocg(color_hr_exp)

    # 3. Luminance compression: Y_c = ln(1 + Y) / (1 + ln(1 + Y))
    color_lr_ycocg[0] = ln(1 + max(Y, 0)) / (1 + ln(1 + max(Y, 0)))
    color_hr_ycocg[0] = same

    # 4. Back to RGB (compressed domain)
    color_lr_norm = ycocg_to_rgb(color_lr_ycocg)
    color_hr_norm = ycocg_to_rgb(color_hr_ycocg)

    # 5. Motion vectors: normalise to [-1, 1] relative to LR resolution
    mv_norm = mv / tensor([W_lr, H_lr])

    # 6. Depth: inverse normalisation to [0, 1]
    depth_inv = 1.0 / (depth + 1e-6)
    depth_norm = (depth_inv - depth_inv.min()) / (depth_inv.max() - depth_inv.min() + 1e-6)

    # 7. Jitter phase encoding
    jitter_phase = phase_k / K  # scalar in [0, 1)

    return {
        "color_lr": color_lr_norm,       # [3, H, W]
        "color_hr": color_hr_norm,       # [3, sH, sW]
        "mv": mv_norm,                   # [2, H, W]
        "depth": depth_norm,             # [1, H, W]
        "jitter_phase": jitter_phase,    # scalar
        "exposure": exposure,            # scalar (for inverse at output)
    }
```

**YCoCg colour space transforms (exact definitions):**

$$\begin{bmatrix} Y \\ Co \\ Cg \end{bmatrix} = \begin{bmatrix} 0.25 & 0.50 & 0.25 \\ 0.50 & 0.00 & -0.50 \\ -0.25 & 0.50 & -0.25 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix}$$

$$\begin{bmatrix} R \\ G \\ B \end{bmatrix} = \begin{bmatrix} 1 & 1 & -1 \\ 1 & 0 & 1 \\ 1 & -1 & -1 \end{bmatrix} \begin{bmatrix} Y \\ Co \\ Cg \end{bmatrix}$$

**Luminance compression and decompression:**

$$Y_{\text{compressed}} = \frac{\ln(1.0 + Y)}{1.0 + \ln(1.0 + Y)}$$

$$Y_{\text{decompressed}} = \exp\!\left(\frac{Y_c}{1.0 - Y_c}\right) - 1.0, \quad Y_c \in [0, 1)$$

### 5.2 Augmentation (Applied Online During Training)

| Augmentation               | Probability | Parameters                          | MV Adjustment                  |
| :------------------------- | :---------- | :---------------------------------- | :----------------------------- |
| Random horizontal flip     | 0.5         | Mirror                              | Negate MV x-component          |
| Random vertical flip       | 0.5         | Mirror                              | Negate MV y-component          |
| Random 90/180/270 rotation | 0.5         | Rotate                              | Rotate MV accordingly          |
| Random crop (LR space)     | 1.0         | 64x64 LR patch (192x192 HR for s=3) | Crop MV, depth correspondingly |
| Temporal reverse           | 0.3         | Swap frame order                    | Negate all motion vectors      |
| Colour jitter (brightness) | 0.2         | $\pm 0.1$ multiplicative            | None                           |
| MV noise injection         | 0.3         | Gaussian $\sigma = 0.3$ px          | Applied to MV                  |
| Depth quantisation         | 0.1         | Reduce to 16-bit precision          | Applied to depth               |
| Exposure oscillation       | 0.1         | $\pm 0.3$ EV random walk            | Applied to exposure scalar     |

### 5.3 Storage Format and Caching

- **On-disk raw**: OpenEXR files (lossless, half-float), organised per-scene/per-sequence/per-frame.
- **Training cache**: Pre-cropped 64x64 LR patches stored as memory-mapped `.bin` files (FP16 tensors, NCHW layout). Generated offline by `scripts/generate_cache.py`.
- **DataLoader**: 8 workers, `pin_memory=True`, `prefetch_factor=4`, `persistent_workers=True`.
- **Estimated storage**: ~12 scenes $\times$ 10,800 frames $\times$ ~2 MB/frame $\approx$ 260 GB raw; ~80 GB cached patches.

## 6. Model Architecture

### 6.1 System Overview: Two Separate Networks

Super-resolution and frame generation are treated as **separate networks** with **no shared backbone**.

Rationale:

1. They operate on fundamentally different input configurations (LR G-buffers vs HR frame pairs).
2. They have different latency budgets (SR: < 2 ms; FG: < 3 ms).
3. Decoupling allows mixing with external upscalers (matching FSR 3.1's decoupled architecture introduced in v3.1).
4. Independent ablation and deployment.

```
Engine Render (LR) ──┬──> [Pre-Processing] ──> [SR Network] ──> [Post-Processing] ──> Upscaled Frame (HR)
                     │                                                                      │
                     │                                                                      ├──> Display (odd frames)
                     │                                                                      │
                     └──> [FG Network] ──────────────────────────────────────────────────────┴──> Interpolated Frame (HR)
                          (takes 2 upscaled                                                        │
                           frames + MVs + depth)                                                   └──> Display (even frames)
```

### 6.2 Super-Resolution Network: Rep-TNSR v2

#### 6.2.1 Structural Reparameterisation and Topology Formulation

To circumvent the latency penalties associated with multi-branch residual topologies while retaining the expressive capacity required to resolve high-frequency geometric boundaries, the architecture adopts a structurally reparameterisable, single-path convolutional backbone inspired by Edge-oriented Convolution Blocks (ECB, [xindongzhang/ECBSR](https://github.com/xindongzhang/ECBSR), MIT licence).

During the training phase, the network utilises an expanded multi-branch topology that simultaneously routes intermediate features through standard $3 \times 3$ convolutions, $1 \times 1$ point-wise projections, an identity connection, and a set of fixed, first-order and second-order differential spatial operators (Sobel and Laplacian). Prior to model compilation and engine deployment, every linear branch is collapsed into a single, homogeneous $3 \times 3$ convolution via associative linear transformations.

The feedforward representation of an intermediate feature block during training processes the activation tensor $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ through the following parallel operations:

$$Y = \text{Conv}_{3\times 3}(X) + \text{Conv}_{1\times 1}(X) + X + \text{Conv}_{1\times 1}^{\text{SobelX}}(K_{\text{SobelX}} * X) + \text{Conv}_{1\times 1}^{\text{SobelY}}(K_{\text{SobelY}} * X) + \text{Conv}_{1\times 1}^{\text{Lap}}(K_{\text{Lap}} * X)$$

Where $K_{\text{SobelX}}, K_{\text{SobelY}}, K_{\text{Lap}} \in \mathbb{R}^{1 \times 1 \times 3 \times 3}$ define non-trainable, discrete differential filters:

$$K_{\text{SobelX}} = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad K_{\text{SobelY}} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}, \quad K_{\text{Lap}} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

The unified inference kernel $W_{\text{fused}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times 3 \times 3}$ and its corresponding bias vector $B_{\text{fused}} \in \mathbb{R}^{C_{\text{out}}}$ are synthesised offline:

$$W_{\text{fused}} = W_{3\times 3} + W_{1\times 1 \to 3\times 3} + W_{\text{id}\to 3\times 3} + \sum_{p \in \{\text{SobelX}, \text{SobelY}, \text{Lap}\}} \left( W_{1\times 1}^{p} \otimes K_{p} \right)$$

$$B_{\text{fused}} = B_{3\times 3} + B_{1\times 1} + B_{\text{SobelX}} + B_{\text{SobelY}} + B_{\text{Lap}}$$

This conversion eliminates all runtime branching. The final deployed model consists of a strictly linear sequence of plain $3 \times 3$ convolutions, maximising GPU instruction-cache locality and avoiding the memory bandwidth overhead of intermediate skip concatenations.

#### 6.2.2 Design Evolution from Rep-TNSR v1

- **v1 (existing)**: 4 reparameterised blocks, 20 channels, 9-channel input, 17,387 parameters.
- **v2 (proposed)**: 6 reparameterised blocks, 24 channels, 12-channel input, ~34,659 parameters (within 50K budget).

Three quality tiers are provided for different hardware classes:

| Tier        | Blocks | Channels | Fused Params | GFLOPs (360p$\to$1080p) | Estimated Latency (GTX 1650 Mobile) | Estimated Latency (RTX 3060) | Target GPU Class    |
| :---------- | :----- | :------- | :----------- | :---------------------- | :---------------------------------- | :--------------------------- | :------------------ |
| Performance | 4      | 16       | ~11,000      | 4.2                     | ~1.2 ms                             | ~0.4 ms                      | GTX 1650, RX 570    |
| Quality     | 4      | 20       | ~17,387      | 7.96                    | ~2.3 ms                             | ~0.7 ms                      | RTX 2060, RX 5700   |
| Ultra       | 6      | 24       | ~34,659      | 15.83                   | ~4.5 ms (over budget)               | ~1.3 ms                      | RTX 3060+, RX 6700+ |

The Ultra tier exceeds the 3.0 ms budget on GTX 1650 Mobile class hardware. On that class, fall back to Quality or Performance tier. The Ultra tier targets RTX 3060+ class hardware where sustained FP16 throughput is ~12 TFLOPS.

#### 6.2.3 Input Tensor Specification

$X \in \mathbb{R}^{12 \times H_{\text{LR}} \times W_{\text{LR}}}$

| Channel Index | Content                                                      | Format | Source                |
| :------------ | :----------------------------------------------------------- | :----- | :-------------------- |
| 0, 1, 2       | Current colour (compressed RGB)                              | FP16   | Pre-processing shader |
| 3, 4, 5       | Reprojected history colour (variance-clamped)                | FP16   | Pre-processing shader |
| 6, 7          | Dilated motion vectors (normalised)                          | FP16   | Pre-processing shader |
| 8             | Disocclusion / validity mask                                 | FP16   | Pre-processing shader |
| 9             | Linear depth (normalised)                                    | FP16   | Depth buffer          |
| 10            | Temporal confidence (exponential decay of depth consistency) | FP16   | Pre-processing shader |
| 11            | Jitter phase index (encoded as float: phase/K)               | FP16   | CPU constant          |

Channels 9, 10, 11 are **new** relative to v1 (which used 9 channels). Depth provides geometric context for resolving edges at depth discontinuities. Temporal confidence provides a continuous reliability signal for the history buffer. Jitter phase allows the network to learn phase-specific reconstruction kernels.

#### 6.2.4 Architecture Specification

```
Input [12, H, W]
  │
  ├─ Layer 1: RepConvBlock(12, 24) + PReLU     → [24, H, W]
  ├─ Layer 2: RepConvBlock(24, 24) + PReLU     → [24, H, W]
  ├─ Layer 3: RepConvBlock(24, 24) + PReLU     → [24, H, W]
  ├─ Layer 4: RepConvBlock(24, 24) + PReLU     → [24, H, W]
  ├─ Layer 5: RepConvBlock(24, 24) + PReLU     → [24, H, W]
  ├─ Layer 6: RepConvBlock(24, 24) + PReLU     → [24, H, W]
  ├─ Layer 7: Conv2d(24, 3*s², 3, pad=1)       → [27, H, W]  (for s=3)
  └─ PixelShuffle(s)                            → [3, sH, sW]
```

All non-linear feature transformations execute in the low-resolution coordinate space. The final projection layer expands the channel depth to $C_{\text{out}} = 3 \times s^2$ prior to spatial rearrangement via sub-pixel convolution (depth-to-space pixel shuffle):

$$\mathcal{I}_{\text{SR}}(c, y \cdot s + d_y, x \cdot s + d_x) = \mathcal{T}(c \cdot s^2 + d_y \cdot s + d_x, y, x)$$

Where $c \in \{0, 1, 2\}$ denotes the target RGB channel, and $d_y, d_x \in \{0, \ldots, s-1\}$ represent the sub-pixel spatial offsets.

#### 6.2.5 Parameter Count (Ultra Tier, Fused, s=3)

| Layer                                         | Parameters                                           |
| :-------------------------------------------- | :--------------------------------------------------- |
| L1: Conv(12, 24, 3x3) + bias                  | 12 $\times$ 24 $\times$ 9 + 24 = 2,616               |
| L2 to L6: 5 $\times$ Conv(24, 24, 3x3) + bias | 5 $\times$ (24 $\times$ 24 $\times$ 9 + 24) = 26,040 |
| L7: Conv(24, 27, 3x3) + bias                  | 24 $\times$ 27 $\times$ 9 + 27 = 5,859               |
| PReLU: 6 $\times$ 24 learnable slopes         | 144                                                  |
| **Total**                                     | **34,659**                                           |

#### 6.2.6 FLOP Analysis (Ultra Tier, 640x360 Input, s=3)

$$\text{MACs} = N_{\text{pixels}} \times \sum_l (C_{\text{in}}^l \times C_{\text{out}}^l \times K^2)$$

$$= 230{,}400 \times (12 \times 24 \times 9 + 5 \times 24 \times 24 \times 9 + 24 \times 27 \times 9)$$

$$= 230{,}400 \times (2{,}592 + 25{,}920 + 5{,}832) = 230{,}400 \times 34{,}344$$

$$= 7.914\text{ GMACs} = 15.83\text{ GFLOPs}$$

$$\text{Estimated Kernel Latency (RTX 3060, ~12 TFLOPS sustained)} = \frac{15.83}{12{,}000} \times 1000 \approx 1.32\text{ ms}$$

#### 6.2.7 Reference Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReparameterizedConvBlock(nn.Module):
    """Multi-branch training block that fuses to a single 3x3 conv at deployment."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

        self.conv_sobel_x = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.conv_sobel_y = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.conv_laplacian = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        self.act = nn.PReLU(num_parameters=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_3x3 = self.conv3x3(x)
        out_1x1 = self.conv1x1(x)

        sx = F.conv2d(x, self.sobel_x.repeat(self.in_channels, 1, 1, 1), padding=1, groups=self.in_channels)
        sy = F.conv2d(x, self.sobel_y.repeat(self.in_channels, 1, 1, 1), padding=1, groups=self.in_channels)
        lap = F.conv2d(x, self.laplacian.repeat(self.in_channels, 1, 1, 1), padding=1, groups=self.in_channels)

        out_edge = self.conv_sobel_x(sx) + self.conv_sobel_y(sy) + self.conv_laplacian(lap)
        out_identity = x if self.in_channels == self.out_channels else 0.0

        return self.act(out_3x3 + out_1x1 + out_edge + out_identity)

    @torch.no_grad()
    def export_fused_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse all branches into a single 3x3 kernel + bias."""
        w_fused = self.conv3x3.weight.clone()
        b_fused = self.conv3x3.bias.clone()

        # Pad 1x1 to 3x3
        w_fused += F.pad(self.conv1x1.weight, (1, 1, 1, 1), mode="constant", value=0)
        b_fused += self.conv1x1.bias

        # Expand differential operators: W_1x1 outer-product K_operator
        w_sobel_x_fused = self.conv_sobel_x.weight * self.sobel_x
        w_sobel_y_fused = self.conv_sobel_y.weight * self.sobel_y
        w_lap_fused = self.conv_laplacian.weight * self.laplacian

        w_fused += (w_sobel_x_fused + w_sobel_y_fused + w_lap_fused)
        b_fused += (self.conv_sobel_x.bias + self.conv_sobel_y.bias + self.conv_laplacian.bias)

        # Identity branch
        if self.in_channels == self.out_channels:
            id_tensor = torch.zeros_like(w_fused)
            for i in range(self.in_channels):
                id_tensor[i, i, 1, 1] = 1.0
            w_fused += id_tensor

        return w_fused, b_fused


class RepTNSR(nn.Module):
    """Rep-TNSR v2: Reparameterisable Temporal Neural Super-Resolution.

    Quality tiers:
        Performance: in_channels=9,  base_channels=16, num_blocks=4 (~11K params)
        Quality:     in_channels=9,  base_channels=20, num_blocks=4 (~17K params)
        Ultra:       in_channels=12, base_channels=24, num_blocks=6 (~35K params)
    """

    def __init__(self, in_channels: int = 12, base_channels: int = 24,
                 num_blocks: int = 6, scale: int = 3):
        super().__init__()
        self.scale = scale
        self.stem = ReparameterizedConvBlock(in_channels, base_channels)
        self.blocks = nn.ModuleList([
            ReparameterizedConvBlock(base_channels, base_channels)
            for _ in range(num_blocks - 1)
        ])
        self.conv_out = nn.Conv2d(base_channels, 3 * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        for block in self.blocks:
            feat = block(feat)
        return self.pixel_shuffle(self.conv_out(feat))
```

### 6.3 Frame Generation Network: NeuralFG

Frame generation interpolates frame $t-0.5$ given upscaled frames $t-1$ and $t$ plus their engine-provided motion vectors and depth.

**Key difference from FSR 3.1 FG**: FSR 3.1 uses hand-crafted hierarchical optical flow on luminance. NeuralFG uses a lightweight learned intermediate flow estimator inspired by RIFE's IFNet ([hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)) but drastically compressed.

#### 6.3.1 Input Tensor Specification

$X_{\text{FG}} \in \mathbb{R}^{10 \times H_{\text{HR}} \times W_{\text{HR}}}$

| Channel Index | Content                                                        |
| :------------ | :------------------------------------------------------------- |
| 0, 1, 2       | Upscaled frame $t-1$ (compressed RGB)                          |
| 3, 4, 5       | Upscaled frame $t$ (compressed RGB)                            |
| 6, 7          | Engine motion vectors $t-1 \to t$ (upscaled to HR, normalised) |
| 8             | Depth $t$ (upscaled to HR, normalised)                         |
| 9             | Depth $t-1$ (upscaled to HR, normalised)                       |

#### 6.3.2 Architecture

The network operates at HR resolution but uses aggressive $4\times$ spatial downsampling to keep compute tractable. All flow estimation and occlusion reasoning happen at quarter resolution.

```
Input [10, H, W]
  │
  ├─ AvgPool2d(4)                                → [10, H/4, W/4]
  ├─ Conv2d(10, 32, 3, pad=1) + LeakyReLU(0.2)   → [32, H/4, W/4]
  ├─ Conv2d(32, 32, 3, pad=1) + LeakyReLU(0.2)   → [32, H/4, W/4]
  ├─ Conv2d(32, 32, 3, pad=1) + LeakyReLU(0.2)   → [32, H/4, W/4]
  ├─ Conv2d(32, 9, 3, pad=1)                      → [9, H/4, W/4]
  │   ├─ Channels 0, 1:  flow_{t-1 → t-0.5} (Δx, Δy)
  │   ├─ Channels 2, 3:  flow_{t → t-0.5} (Δx, Δy)
  │   ├─ Channels 4, 5:  occlusion weights (sigmoid-activated, per-direction)
  │   └─ Channels 6, 7, 8: residual colour correction (RGB)
  │
  ├─ F.interpolate(scale_factor=4, bilinear)      → [9, H, W]
  │
  ├─ Warp frame_{t-1} by flow_{0→0.5}            → [3, H, W]  (grid_sample, bilinear)
  ├─ Warp frame_{t}   by flow_{1→0.5}            → [3, H, W]  (grid_sample, bilinear)
  │
  ├─ Blend: out = σ(occ_0) * warp_0 + σ(occ_1) * warp_1 + residual
  └─ Output [3, H, W]
```

#### 6.3.3 Parameter Count

| Layer                               | Parameters                                           |
| :---------------------------------- | :--------------------------------------------------- |
| Conv(10, 32, 3x3) + bias            | 10 $\times$ 32 $\times$ 9 + 32 = 2,912               |
| 2 $\times$ Conv(32, 32, 3x3) + bias | 2 $\times$ (32 $\times$ 32 $\times$ 9 + 32) = 18,496 |
| Conv(32, 9, 3x3) + bias             | 32 $\times$ 9 $\times$ 9 + 9 = 2,601                 |
| **Total**                           | **24,009**                                           |

#### 6.3.4 FLOP Analysis (1080p Input, 4x Downsample to 270p Processing)

$$N_{\text{ds}} = 480 \times 270 = 129{,}600$$

$$\text{MACs} = 129{,}600 \times (10 \times 32 \times 9 + 2 \times 32 \times 32 \times 9 + 32 \times 9 \times 9)$$

$$= 129{,}600 \times (2{,}880 + 18{,}432 + 2{,}592) = 129{,}600 \times 23{,}904$$

$$= 3.098\text{ GMACs} = 6.196\text{ GFLOPs}$$

Plus bilinear warping (~0.5 GFLOPs): **total ~6.7 GFLOPs**.

$$\text{Estimated Latency (RTX 3060)} = \frac{6.7}{12{,}000} \times 1000 + 0.8\text{ ms (warp/blend)} \approx 1.4\text{ ms}$$

#### 6.3.5 Reference Implementation

```python
class NeuralFG(nn.Module):
    """Compact learned intermediate flow estimator for frame generation."""

    def __init__(self, in_channels: int = 10, mid_channels: int = 32, downsample: int = 4):
        super().__init__()
        self.downsample = downsample

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, 9, 3, padding=1),
        )

    def forward(self, frame_prev: torch.Tensor, frame_curr: torch.Tensor,
                mv: torch.Tensor, depth_prev: torch.Tensor,
                depth_curr: torch.Tensor) -> torch.Tensor:
        B, _, H, W = frame_prev.shape

        x = torch.cat([frame_prev, frame_curr, mv, depth_curr, depth_prev], dim=1)

        # Process at quarter resolution
        x_ds = F.avg_pool2d(x, self.downsample)
        pred = self.encoder(x_ds)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)

        # Unpack predictions
        flow_01 = pred[:, 0:2]   # flow from frame t-1 toward t-0.5
        flow_10 = pred[:, 2:4]   # flow from frame t toward t-0.5
        occ = torch.sigmoid(pred[:, 4:6])  # occlusion weights [2 channels]
        residual = pred[:, 6:9]  # colour residual

        # Warp both frames to t-0.5
        grid_01 = self._flow_to_grid(flow_01, H, W)
        grid_10 = self._flow_to_grid(flow_10, H, W)

        warp_0 = F.grid_sample(frame_prev, grid_01, mode="bilinear",
                               padding_mode="border", align_corners=False)
        warp_1 = F.grid_sample(frame_curr, grid_10, mode="bilinear",
                               padding_mode="border", align_corners=False)

        # Occlusion-weighted blend + residual
        out = occ[:, 0:1] * warp_0 + occ[:, 1:2] * warp_1 + residual
        return out

    @staticmethod
    def _flow_to_grid(flow: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert pixel-space flow to normalised grid for grid_sample."""
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=flow.device),
            torch.linspace(-1, 1, W, device=flow.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        flow_norm = torch.stack([
            2.0 * flow[:, 0] / W,
            2.0 * flow[:, 1] / H,
        ], dim=-1).unsqueeze(1) if flow.dim() == 3 else torch.stack([
            2.0 * flow[:, 0] / W,
            2.0 * flow[:, 1] / H,
        ], dim=-1)
        # Reshape flow_norm to [B, H, W, 2]
        flow_norm = flow.permute(0, 2, 3, 1)
        flow_norm = torch.stack([
            2.0 * flow_norm[..., 0] / W,
            2.0 * flow_norm[..., 1] / H,
        ], dim=-1)
        return grid.expand(flow.shape[0], -1, -1, -1) + flow_norm
```

### 6.4 Temporal Memory Strategy

**SR network**: No explicit recurrent hidden state. Temporal information enters via the reprojected + variance-clamped history buffer, managed externally via ping-pong double-buffering (identical to the existing Rep-TNSR v1 and FSR 3.1). The history buffer IS the temporal memory.

**FG network**: Stateless per-interpolation. Takes two completed frames and produces the intermediate. No hidden state.

**Rationale for no RNN/LSTM/ConvLSTM:**

1. Stateful networks require careful state management across scene cuts, resolution changes, and teleports.
2. FSR 3.1 manages temporal state externally (history buffer + variance clamping) and this approach is proven robust in production.
3. Eliminates ONNX/DirectML export complexity for recurrent operators.
4. Runtime state management for recurrent models across frame drops, resolution switches, and pause/resume is fragile.

### 6.5 Sub-Pixel Jittering and Sampling Mechanics

Reconstructing high-frequency geometric detail across a $3\times$ scaling factor without hallucinating artificial features requires accumulating phase-shifted spatial samples across consecutive frames. The camera projection matrix is jittered at each frame using a 2D Halton $(2, 3)$ low-discrepancy sequence. The sub-pixel offset vector $(\delta x_t, \delta y_t)$ is mapped into normalised device coordinates (NDC) using the dimensions of the low-resolution viewport $(W_{\text{LR}}, H_{\text{LR}})$:

$$\delta x_t = \frac{\text{Halton}(t \pmod K + 1, 2) - 0.5}{W_{\text{LR}}}, \quad \delta y_t = \frac{\text{Halton}(t \pmod K + 1, 3) - 0.5}{H_{\text{LR}}}$$

A phase sequence length of $K = 9$ (matching the spatial upsampling area $s^2 = 3^2$) provides uniform sample distribution across the reconstructed high-resolution pixel grid. The offset is incorporated directly into the camera's perspective projection matrix:

$$P_{\text{jittered}} = \begin{bmatrix} P_{00} & 0 & 2\,\delta x_t & 0 \\ 0 & P_{11} & 2\,\delta y_t & 0 \\ 0 & 0 & P_{22} & P_{23} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

## 7. Loss Functions

### 7.1 Super-Resolution Losses

$$\mathcal{L}_{\text{SR}} = \lambda_1 \mathcal{L}_{\text{char}} + \lambda_2 \mathcal{L}_{\text{edge}} + \lambda_3 \mathcal{L}_{\text{perc}} + \lambda_4 \mathcal{L}_{\text{temp}} + \lambda_5 \mathcal{L}_{\text{freq}}$$

**Charbonnier loss** (spatial fidelity, differentiable L1 approximation with non-vanishing gradient at zero):

$$\mathcal{L}_{\text{char}}(\hat{I}, I^{\text{GT}}) = \frac{1}{N}\sum_{i=1}^N \sqrt{(\hat{I}(i) - I^{\text{GT}}(i))^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

**Edge loss** (structural sharpness via discrete spatial gradients):

$$\mathcal{L}_{\text{edge}} = \frac{1}{N}\sum_{i=1}^N \left(|\nabla_x \hat{I}(i) - \nabla_x I^{\text{GT}}(i)| + |\nabla_y \hat{I}(i) - \nabla_y I^{\text{GT}}(i)|\right)$$

$$\nabla_x I(x, y) = I(x+1, y) - I(x-1, y), \quad \nabla_y I(x, y) = I(x, y+1) - I(x, y-1)$$

**Perceptual loss** (VGG-19 feature matching):

$$\mathcal{L}_{\text{perc}} = \frac{1}{C_j H_j W_j} \left\| \Phi_{\text{conv3-3}}(\hat{I}) - \Phi_{\text{conv3-3}}(I^{\text{GT}}) \right\|_2^2$$

$\Phi$: VGG-19 pretrained on ImageNet (from `torchvision.models.vgg19`, BSD 3-Clause, weights frozen). Feature extraction layer: `features[15]` (conv3_3). The network is frozen during SR training and used only for gradient computation through the feature space.

**Temporal consistency loss** (backward-warped consistency, masked by disocclusion):

$$\mathcal{L}_{\text{temp}} = \frac{1}{N}\sum_{i=1}^N M_{\text{valid}}(i) \cdot \left| \hat{I}_t(i) - \mathcal{W}(\hat{I}_{t-1}, V_{t \to t-1})(i) \right|$$

$$M_{\text{valid}}(p) = \exp\!\left(-\alpha \cdot \left| D_t(p) - \mathcal{W}(D_{t-1}, V_{t \to t-1})(p) \right|\right), \quad \alpha = 10.0$$

$\mathcal{W}(\cdot)$ represents bilinear sampling driven by backward motion vector field $V_{t \to t-1}$. $M_{\text{valid}} \in [0, 1]$ is a continuous disocclusion visibility mask using depth consistency. The exponential decay suppresses temporal loss gradients across disoccluded boundaries where past history is geometrically invalid.

**Frequency loss** (penalise high-frequency energy loss in Fourier domain):

$$\mathcal{L}_{\text{freq}} = \frac{1}{N}\sum_{i} \left|\text{FFT}(\hat{I})(i) - \text{FFT}(I^{\text{GT}})(i)\right|$$

Computed on the 2D DFT magnitude of the Y-channel (luminance). This encourages preservation of fine texture detail and sub-pixel edge information that Charbonnier alone would smooth away.

**Loss weight schedule:**

| Phase                                        | $\lambda_1$ (char) | $\lambda_2$ (edge) | $\lambda_3$ (perc) | $\lambda_4$ (temp) | $\lambda_5$ (freq) |
| :------------------------------------------- | :----------------- | :----------------- | :----------------- | :----------------- | :----------------- |
| Phase 1: Spatial warmup (0 to 100K iter)     | 1.0                | 0.3                | 0.0                | 0.0                | 0.0                |
| Phase 2: Temporal integration (100K to 300K) | 1.0                | 0.5                | 0.05               | 0.25               | 0.1                |
| Phase 3: Fine-tuning (300K to 500K)          | 1.0                | 0.5                | 0.05               | 0.25               | 0.1                |

### 7.2 Frame Generation Losses

$$\mathcal{L}_{\text{FG}} = \mu_1 \mathcal{L}_{\text{char}}^{\text{FG}} + \mu_2 \mathcal{L}_{\text{perc}}^{\text{FG}} + \mu_3 \mathcal{L}_{\text{census}}$$

**Charbonnier**: Same formulation as SR, applied to interpolated frame $\hat{I}_{t-0.5}$ vs ground-truth mid-frame $I_{t-0.5}^{\text{GT}}$.

**Perceptual**: Same VGG-19 conv3_3 formulation.

**Census transform loss** (robust to global illumination shifts during interpolation):

$$\mathcal{L}_{\text{census}} = \frac{1}{N}\sum_{i} \text{SoftHamming}\!\left(\text{Census}_{7 \times 7}(\hat{I}_{t-0.5})(i),\; \text{Census}_{7 \times 7}(I_{t-0.5}^{\text{GT}})(i)\right)$$

Census transform: binary comparison of each pixel against its $7 \times 7$ neighbourhood. SoftHamming uses $1 - \exp(-d^2)$ instead of hard Hamming distance for differentiability.

**Weights**: $\mu_1 = 1.0, \quad \mu_2 = 0.05, \quad \mu_3 = 0.5$

## 8. Training Configuration

### 8.1 Training Curriculum

| Phase                         | Iterations   | Focus                              | LR Patch Size | Temporal Frames per Sample    | Active Losses |
| :---------------------------- | :----------- | :--------------------------------- | :------------ | :---------------------------- | :------------ |
| Phase 1: Spatial warmup       | 0 to 100K    | Pixel-level spatial reconstruction | 64x64         | 1 (single frame, no temporal) | Char + Edge   |
| Phase 2: Temporal integration | 100K to 300K | Full multi-objective with temporal | 64x64         | 2 (consecutive pair)          | All 5 losses  |
| Phase 3: Fine-tune            | 300K to 500K | All losses, larger patches         | 96x96         | 2 (consecutive pair)          | All 5 losses  |

### 8.2 Optimiser and Learning Rate Schedule

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,           # eta_max
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=500_000,      # total iterations
    eta_min=1e-6,       # eta_min
)
```

### 8.3 Batch Strategy and Mixed Precision

```python
batch_size_per_gpu = 16  # 2-frame pairs, 64x64 LR patches

# PyTorch native AMP (automatic mixed precision)
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda", dtype=torch.float16):
    output = model(input_tensor)
    loss = compute_loss(output, target)
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

### 8.4 Distributed Training

```bash
# DDP with 2 to 4 GPUs
torchrun --nproc_per_node=4 train.py --config configs/sr_ultra.yaml --distributed
```

PyTorch DistributedDataParallel with `find_unused_parameters=False`. Each GPU processes `batch_size_per_gpu` samples. Effective batch size = `batch_size_per_gpu * num_gpus`.

### 8.5 Checkpointing and Reproducibility

- Checkpoint every 10K iterations.
- Save: model state dict, optimiser state dict, scheduler state dict, AMP scaler state dict, Python/NumPy/PyTorch/CUDA RNG states, current iteration counter, best validation metric value.
- Deterministic mode: `torch.use_deterministic_algorithms(True)` where supported.
- Fixed seeds: `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)`.
- `CUBLAS_WORKSPACE_CONFIG=:16:8` for deterministic cuBLAS.

### 8.6 Hardware Requirements and Training Cost Estimates

| Resource                     | Minimum (Functional) | Recommended (Full Pipeline)        |
| :--------------------------- | :------------------- | :--------------------------------- |
| GPUs                         | 1x RTX 3090 (24 GB)  | 4x RTX 4090 (24 GB each)           |
| SR training time (500K iter) | ~72 hours (1x 3090)  | ~20 hours (4x 4090)                |
| FG training time (300K iter) | ~48 hours (1x 3090)  | ~14 hours (4x 4090)                |
| Dataset storage              | 300 GB               | 500 GB (raw + cache + checkpoints) |
| System RAM                   | 32 GB                | 64 GB                              |

**Cloud cost estimate** (4x A100 40GB on Lambda Labs / RunPod at ~\$5/hr/GPU):

- Base training (SR + FG): ~34 GPU-hours $\times$ \$5 $\approx$ \$680.
- With hyperparameter sweeps (5 runs): ~\$3,400.
- Full ablation matrix (30 experiments): ~\$10,000.

## 9. Runtime Inference Pipeline

### 9.1 Execution Flow

```
Frame t render complete (360p)
│
├─ [Pass 1] Pre-Processing Compute Shader (0.2 to 0.3 ms)
│  ├─ Depth dilation (3x3 nearest-depth neighbourhood)
│  ├─ Motion vector dilation (use closest-depth MV)
│  ├─ Temporal reprojection (bilinear warp of history buffer)
│  ├─ RGB -> YCoCg conversion + luminance compression
│  ├─ 3x3 neighbourhood variance -> AABB min/max in YCoCg
│  ├─ History clamping to AABB (gamma threshold = 1.25)
│  ├─ Disocclusion mask evaluation (UV bounds + depth consistency)
│  ├─ Temporal confidence computation (exponential depth decay)
│  └─ Pack 12 channels -> flat FP16 structured buffer [12, H, W]
│
├─ [Pass 2] Neural SR Trunk via DirectML / Compute (1.5 to 2.5 ms)
│  ├─ IDMLCommandRecorder::RecordDispatch (DirectML path)
│  │   OR hand-coded HLSL compute dispatch (native path)
│  └─ Output: (3 * s^2)-channel feature map in LR coordinate space
│
├─ [Pass 3] Post-Processing Compute Shader (0.15 to 0.25 ms)
│  ├─ Depth-to-Space pixel shuffle rearrangement
│  ├─ YCoCg -> RGB conversion
│  ├─ Luminance decompression (inverse logarithmic transform)
│  ├─ Inverse exposure scaling
│  └─ Write to HR backbuffer + update history ping-pong buffer
│
├─ [Pass 4] Frame Generation (OPTIONAL, 1.2 to 2.0 ms)
│  ├─ Triggered every other frame or on demand
│  ├─ Input: HR frame t-1 + HR frame t (from SR) + upscaled MVs + depth
│  ├─ Downsample 4x -> flow estimation -> upsample 4x
│  ├─ Bidirectional warp + occlusion-weighted blend + residual
│  └─ Output: interpolated HR frame at t-0.5
│
└─ Present (HR frame OR interpolated frame, UI composited last)
```

### 9.2 Frame Pacing Strategy

**Without frame generation:**

```
Render(t) → SR(t) → Present(t) → Render(t+1) → SR(t+1) → Present(t+1)
|<------- 16.67 ms ------->|    |<------- 16.67 ms ------->|
Displayed FPS: equal to render rate
```

**With frame generation:**

```
Render(t) → SR(t) → FG(t-0.5) → Present(t-0.5) → Present(t) → Render(t+1) → ...
|<-------------- ~16.67 ms ------------->|
Displayed FPS: 2x base render rate
Input latency: base frame time + FG compute (~1.5 ms)
```

Frame generation presents the interpolated frame FIRST, then the real frame. This reduces perceived input latency compared to presenting the interpolated frame after the real frame. Combined with a frame queue depth of 1 (no pre-rendered frames), effective input latency overhead is limited to the FG compute time (~1.5 ms).

### 9.3 Memory Budget (1080p, s=3 from 360p)

| Resource                                 | Format                        | Size           |
| :--------------------------------------- | :---------------------------- | :------------- |
| History ping buffer                      | R16G16B16A16_FLOAT, 1920x1080 | 16.59 MB       |
| History pong buffer                      | R16G16B16A16_FLOAT, 1920x1080 | 16.59 MB       |
| LR colour input                          | R16G16B16A16_FLOAT, 640x360   | 1.84 MB        |
| Dilated MVs                              | R16G16_FLOAT, 640x360         | 0.92 MB        |
| Depth buffer                             | R32_FLOAT, 640x360            | 0.92 MB        |
| SR input (packed 12ch)                   | FP16, 12x360x640              | 5.53 MB        |
| SR activations (double-buffered)         | FP16, 24x360x640              | 10.62 MB       |
| SR model weights                         | FP16, 34,659 params           | 0.07 MB        |
| FG frame inputs (shared SRV, not copied) | Shared backbuffer references  | 0 MB (no copy) |
| FG activations (at 270p)                 | FP16, 32x270x480              | 7.91 MB        |
| FG model weights                         | FP16, 24,009 params           | 0.05 MB        |
| **Total**                                |                               | **~61 MB**     |

FG inputs share the existing HR backbuffer as shader resource views (SRV, read-only) rather than copying, saving ~33 MB.

### 9.4 Quantisation and Execution Backend

The inference pipeline is implemented using DirectML (DirectX 12) and native HLSL Compute Shaders (Shader Model 6.2+). DirectML optimises the structurally reparameterised neural trunk by compiling the linear convolutional sequence into an optimised hardware meta-command. Compiling compute passes with the DirectX Shader Compiler (DXC) using the `-enable-16bit-types` flag allows the hardware to execute packed FP16 arithmetic natively.

The fused model weights (34,659 parameters $\times$ 2 bytes FP16 = 69.3 KB for SR, 48.0 KB for FG) fit entirely within a single 64 KB GPU constant buffer (or two for FG). This eliminates global memory fetches for model weights during hand-coded HLSL execution.

### 9.5 Deployment Paths

| Path                               | Platform                | GPU Vendor Support           | Maturity   | Relative Latency  | Notes                                                                                                                                                                                                 |
| :--------------------------------- | :---------------------- | :--------------------------- | :--------- | :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DirectML C++ API** (primary)     | Windows 10/11           | AMD, NVIDIA, Intel, Qualcomm | Production | Optimal           | Records into D3D12 command list directly. [github.com/microsoft/DirectML](https://github.com/microsoft/DirectML), MIT. Operator coverage verified: Conv2D, PReLU, DepthToSpace all supported.         |
| **Native HLSL Compute** (advanced) | Windows (D3D12)         | All                          | Production | Lowest possible   | Weights in constant buffer. Fuse pre/post-processing with neural layers. Compile via DXC.                                                                                                             |
| **ncnn Vulkan**                    | Windows, Linux, Android | All                          | Production | Good              | [github.com/Tencent/ncnn](https://github.com/Tencent/ncnn), BSD 3-Clause. PNNX export from PyTorch. Mature FP16 Vulkan compute backend. CPU dispatch overhead < 0.1 ms.                               |
| **TensorRT** (NVIDIA fast path)    | Windows, Linux          | NVIDIA only                  | Production | Fastest on NVIDIA | [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt). Proprietary. Conv+PReLU fusion. Not cross-vendor.                                                                             |
| **ONNX Runtime DirectML EP**       | Windows                 | AMD, NVIDIA, Intel           | Production | Higher overhead   | [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime). CPU session dispatch adds 0.8 to 1.5 ms overhead. Likely exceeds budget for tight loops. Better for offline evaluation. |

ONNX Runtime is unsuitable for the sub-3 ms real-time target due to session dispatch overhead and D3D12 fence synchronisation cost between the game engine graphics queue and the ONNX Runtime session. Use it only for offline model validation and parity testing.

**Export pipeline:**

```
PyTorch model (training)
  ├─ Structural reparameterisation (fuse all branches offline)
  ├─ torch.onnx.export() → model_sr.onnx / model_fg.onnx
  │   ├─ opset_version=17
  │   ├─ dynamic_axes for H, W (channel dims fixed)
  │   └─ Verify numerically with onnxruntime.InferenceSession
  ├─ DirectML: Load ONNX → DML graph compilation → IDMLCompiledOperator
  ├─ ncnn: PNNX export (PyTorch → TorchScript → .param + .bin)
  ├─ TensorRT: trtexec --onnx=model.onnx --fp16 --workspace=256
  └─ Native HLSL: Extract fused FP16 weights → embed in constant buffer
```

### 9.6 Zero-Allocation Double-Buffered Memory Management

All memory targets required by the SR/FG pipeline are pre-allocated during engine initialisation within a single committed memory heap (`D3D12_HEAP_TYPE_DEFAULT`). No dynamic allocation occurs during the render loop.

Temporal history tracking uses a ping-pong double-buffering design. Two 1080p texture targets alternate roles: one serves as the history resource from frame $t-1$ (SRV), while the other acts as the unordered access view (UAV) write target for frame $t$. At frame completion, resource handles swap. No blit or copy operation is needed.

Synchronisation between passes uses direct execution fences (`ID3D12Fence` / `VkSemaphore`), eliminating CPU sync points and driver overhead.

### 9.7 Pipeline Integration (D3D12 C++)

```cpp
#pragma once
#include <d3d12.h>
#include <DirectML.h>
#include <wrl/client.h>
#include <cstdint>

using Microsoft::WRL::ComPtr;

struct SuperResConstants
{
    uint32_t LRWidth;
    uint32_t LRHeight;
    uint32_t HRWidth;
    uint32_t HRHeight;
    float JitterOffsetX;
    float JitterOffsetY;
    float ExposureMultiplier;
    float InvExposureMultiplier;
    float GammaThreshold;
    float TemporalConfidenceDecay;
    float JitterPhaseNorm;
    float Padding;
};

class SuperResolutionSystem
{
public:
    void Dispatch(
        ID3D12GraphicsCommandList4* cmdList,
        ID3D12Resource* currentLRColor,
        ID3D12Resource* motionVectors,
        ID3D12Resource* depthBuffer,
        ID3D12Resource* outputHRBuffer,
        float jitterX, float jitterY,
        float currentExposure,
        float jitterPhaseNorm)
    {
        D3D12_RESOURCE_BARRIER initialBarriers[4] = {};
        initialBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            currentLRColor, D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        initialBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            motionVectors, D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        initialBarriers[2] = CD3DX12_RESOURCE_BARRIER::Transition(
            m_historyBufferPing.Get(), D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        initialBarriers[3] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputHRBuffer, D3D12_RESOURCE_STATE_PRESENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cmdList->ResourceBarrier(4, initialBarriers);

        SuperResConstants cb = {};
        cb.LRWidth = 640;
        cb.LRHeight = 360;
        cb.HRWidth = 1920;
        cb.HRHeight = 1080;
        cb.JitterOffsetX = jitterX;
        cb.JitterOffsetY = jitterY;
        cb.ExposureMultiplier = currentExposure;
        cb.InvExposureMultiplier = 1.0f / (currentExposure > 1e-5f ? currentExposure : 1.0f);
        cb.GammaThreshold = 1.25f;
        cb.TemporalConfidenceDecay = 10.0f;
        cb.JitterPhaseNorm = jitterPhaseNorm;

        cmdList->SetComputeRootSignature(m_rootSignature.Get());
        cmdList->SetComputeRoot32BitConstants(0, sizeof(SuperResConstants) / 4, &cb, 0);

        // Pass 1: Pre-processing (dilation, reprojection, variance clamping)
        cmdList->SetPipelineState(m_preProcessPSO.Get());
        cmdList->Dispatch((640 + 7) / 8, (360 + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER midBarrier =
            CD3DX12_RESOURCE_BARRIER::UAV(m_fusedInputActivationBuffer.Get());
        cmdList->ResourceBarrier(1, &midBarrier);

        // Pass 2: DirectML neural trunk
        ID3D12DescriptorHeap* descriptorHeaps[] = { m_dmlDescriptorHeap.Get() };
        cmdList->SetDescriptorHeaps(1, descriptorHeaps);
        m_dmlCommandRecorder->RecordDispatch(
            cmdList, m_dmlCompiledModel.Get(), m_dmlBindingTable.Get());

        D3D12_RESOURCE_BARRIER dmlBarrier =
            CD3DX12_RESOURCE_BARRIER::UAV(m_fusedExpandedActivationBuffer.Get());
        cmdList->ResourceBarrier(1, &dmlBarrier);

        // Pass 3: Post-processing (pixel shuffle, colour inversion)
        cmdList->SetPipelineState(m_reconstructPSO.Get());
        cmdList->Dispatch((640 + 7) / 8, (360 + 7) / 8, 1);

        D3D12_RESOURCE_BARRIER finalBarriers[2] = {};
        finalBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(
            outputHRBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PRESENT);
        finalBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(
            m_historyBufferPing.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
        cmdList->ResourceBarrier(2, finalBarriers);

        m_historyBufferPing.Swap(m_historyBufferPong);
    }

private:
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_preProcessPSO;
    ComPtr<ID3D12PipelineState> m_reconstructPSO;
    ComPtr<ID3D12DescriptorHeap> m_dmlDescriptorHeap;
    ComPtr<IDMLCompiledOperator> m_dmlCompiledModel;
    ComPtr<IDMLCommandRecorder> m_dmlCommandRecorder;
    ComPtr<IDMLBindingTable> m_dmlBindingTable;
    ComPtr<ID3D12Resource> m_historyBufferPing;
    ComPtr<ID3D12Resource> m_historyBufferPong;
    ComPtr<ID3D12Resource> m_fusedInputActivationBuffer;
    ComPtr<ID3D12Resource> m_fusedExpandedActivationBuffer;
};
```

### 9.8 Pre-Processing Compute Shader (HLSL)

```hlsl
// PreProcessTemporal.hlsl
// Execution: Compute Shader (8x8 Thread Group), dispatched over 640x360
// Produces 12-channel packed FP16 input for neural SR trunk

cbuffer SuperResConstants : register(b0)
{
    uint2  g_LRResolution;
    uint2  g_HRResolution;
    float2 g_JitterOffset;
    float  g_ExposureMultiplier;
    float  g_InvExposureMultiplier;
    float  g_GammaThreshold;
    float  g_TemporalConfidenceDecay;
    float  g_JitterPhaseNorm;
    float  g_Padding;
};

Texture2D<float4> g_CurrentColorTexture   : register(t0);
Texture2D<float2> g_MotionVectorTexture   : register(t1);
Texture2D<float>  g_LinearDepthTexture    : register(t2);
Texture2D<float4> g_HistoryColorTexture   : register(t3);
Texture2D<float>  g_HistoryDepthTexture   : register(t4);

SamplerState g_LinearSampler : register(s0);
SamplerState g_PointSampler  : register(s1);

RWStructuredBuffer<float16_t> g_FusedNeuralInputBuffer : register(u0);

float3 RGB_to_YCoCg(float3 c)
{
    return float3(
        0.25f * c.r + 0.50f * c.g + 0.25f * c.b,
        0.50f * c.r + 0.00f * c.g - 0.50f * c.b,
       -0.25f * c.r + 0.50f * c.g - 0.25f * c.b
    );
}

float3 YCoCg_to_RGB(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

float CompressLuminance(float y)
{
    return log(1.0f + max(y, 0.0f)) / (1.0f + log(1.0f + max(y, 0.0f)));
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= g_LRResolution.x || dtid.y >= g_LRResolution.y) return;

    int2 coord = int2(dtid.xy);
    float2 invRes = 1.0f / float2(g_LRResolution);
    float2 centerUV = (float2(coord) + 0.5f) * invRes;

    // 1. Dilated MV fetch (3x3 nearest-depth)
    float nearestDepth = 1e8f;
    int2 bestOff = int2(0, 0);
    [unroll] for (int dy = -1; dy <= 1; ++dy)
    [unroll] for (int dx = -1; dx <= 1; ++dx)
    {
        int2 sc = clamp(coord + int2(dx, dy), int2(0,0), int2(g_LRResolution) - 1);
        float d = g_LinearDepthTexture.Load(int3(sc, 0)).r;
        if (d < nearestDepth) { nearestDepth = d; bestOff = int2(dx, dy); }
    }
    float2 dilatedMV = g_MotionVectorTexture.Load(int3(coord + bestOff, 0)).xy;

    // 2. Reprojected coordinates
    float2 historyUV = centerUV - dilatedMV - g_JitterOffset;

    // 3. Local 3x3 moments in YCoCg
    float3 m1 = 0, m2 = 0;
    float3 centerRGB = 0;
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int x = -1; x <= 1; ++x)
    {
        int2 sc = clamp(coord + int2(x, y), int2(0,0), int2(g_LRResolution) - 1);
        float3 s = g_CurrentColorTexture.Load(int3(sc, 0)).rgb * g_ExposureMultiplier;
        float3 ycocg = RGB_to_YCoCg(s);
        if (x == 0 && y == 0) centerRGB = s;
        m1 += ycocg; m2 += ycocg * ycocg;
    }
    float3 mean = m1 / 9.0f;
    float3 stdDev = sqrt(abs(m2 / 9.0f - mean * mean));
    float3 aabbMin = mean - g_GammaThreshold * stdDev;
    float3 aabbMax = mean + g_GammaThreshold * stdDev;

    // 4. Sample and clamp history
    float3 rawHist = g_HistoryColorTexture.SampleLevel(g_LinearSampler, historyUV, 0).rgb
                     * g_ExposureMultiplier;
    float3 histYCoCg = clamp(RGB_to_YCoCg(rawHist), aabbMin, aabbMax);
    float3 clampedHist = YCoCg_to_RGB(histYCoCg);

    // 5. Disocclusion mask
    float disoccMask = 1.0f;
    if (historyUV.x < 0 || historyUV.x > 1 || historyUV.y < 0 || historyUV.y > 1)
    {
        disoccMask = 0.0f;
        clampedHist = centerRGB;
    }

    // 6. Temporal confidence (depth-based)
    float histDepth = g_HistoryDepthTexture.SampleLevel(g_LinearSampler, historyUV, 0).r;
    float depthDiff = abs(nearestDepth - histDepth);
    float temporalConf = exp(-g_TemporalConfidenceDecay * depthDiff) * disoccMask;

    // 7. Normalise depth
    float depthNorm = 1.0f / (nearestDepth + 1e-6f);
    // Note: per-frame min/max normalisation done at CPU side; here use raw inverse

    // 8. Luminance compression
    float3 curYCoCg = RGB_to_YCoCg(centerRGB);
    curYCoCg.x = CompressLuminance(curYCoCg.x);
    float3 normCur = YCoCg_to_RGB(curYCoCg);

    float3 hstYCoCg = RGB_to_YCoCg(clampedHist);
    hstYCoCg.x = CompressLuminance(hstYCoCg.x);
    float3 normHist = YCoCg_to_RGB(hstYCoCg);

    // 9. Write 12-channel planar buffer [12, H, W]
    uint si = coord.y * g_LRResolution.x + coord.x;
    uint ps = g_LRResolution.x * g_LRResolution.y;

    g_FusedNeuralInputBuffer[ 0 * ps + si] = float16_t(normCur.r);
    g_FusedNeuralInputBuffer[ 1 * ps + si] = float16_t(normCur.g);
    g_FusedNeuralInputBuffer[ 2 * ps + si] = float16_t(normCur.b);
    g_FusedNeuralInputBuffer[ 3 * ps + si] = float16_t(normHist.r);
    g_FusedNeuralInputBuffer[ 4 * ps + si] = float16_t(normHist.g);
    g_FusedNeuralInputBuffer[ 5 * ps + si] = float16_t(normHist.b);
    g_FusedNeuralInputBuffer[ 6 * ps + si] = float16_t(dilatedMV.x);
    g_FusedNeuralInputBuffer[ 7 * ps + si] = float16_t(dilatedMV.y);
    g_FusedNeuralInputBuffer[ 8 * ps + si] = float16_t(disoccMask);
    g_FusedNeuralInputBuffer[ 9 * ps + si] = float16_t(depthNorm);
    g_FusedNeuralInputBuffer[10 * ps + si] = float16_t(temporalConf);
    g_FusedNeuralInputBuffer[11 * ps + si] = float16_t(g_JitterPhaseNorm);
}
```

### 9.9 Post-Processing Reconstruction Shader (HLSL)

```hlsl
// PostProcessReconstruct.hlsl
// Dispatched across 640x360 to output 1920x1080 (3x scaling)

#define UPSCALE_FACTOR 3

cbuffer SuperResConstants : register(b0)
{
    uint2  g_LRResolution;
    uint2  g_HRResolution;
    float2 g_JitterOffset;
    float  g_ExposureMultiplier;
    float  g_InvExposureMultiplier;
    float  g_GammaThreshold;
    float  g_TemporalConfidenceDecay;
    float  g_JitterPhaseNorm;
    float  g_Padding;
};

StructuredBuffer<float16_t> g_ExpandedActivationFeatures : register(t0);
RWTexture2D<float4> g_FinalOutputTarget : register(u0);

float3 RGB_to_YCoCg(float3 c)
{
    return float3(
        0.25f * c.r + 0.50f * c.g + 0.25f * c.b,
        0.50f * c.r + 0.00f * c.g - 0.50f * c.b,
       -0.25f * c.r + 0.50f * c.g - 0.25f * c.b
    );
}

float3 YCoCg_to_RGB(float3 c)
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

float DecompressLuminance(float cy)
{
    float y = clamp(cy, 0.0f, 0.999f);
    return exp(y / (1.0f - y)) - 1.0f;
}

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID)
{
    uint2 lrCoord = dtid.xy;
    if (lrCoord.x >= g_LRResolution.x || lrCoord.y >= g_LRResolution.y) return;

    uint si = lrCoord.y * g_LRResolution.x + lrCoord.x;
    uint ps = g_LRResolution.x * g_LRResolution.y;

    float16_t rawChannels[27];
    [unroll] for (int c = 0; c < 27; ++c)
        rawChannels[c] = g_ExpandedActivationFeatures[c * ps + si];

    [unroll] for (int dy = 0; dy < UPSCALE_FACTOR; ++dy)
    [unroll] for (int dx = 0; dx < UPSCALE_FACTOR; ++dx)
    {
        uint2 hrCoord = lrCoord * UPSCALE_FACTOR + uint2(dx, dy);
        if (hrCoord.x < g_HRResolution.x && hrCoord.y < g_HRResolution.y)
        {
            uint spi = dy * UPSCALE_FACTOR + dx;
            float3 rgb;
            rgb.r = float(rawChannels[0 * 9 + spi]);
            rgb.g = float(rawChannels[1 * 9 + spi]);
            rgb.b = float(rawChannels[2 * 9 + spi]);

            float3 ycocg = RGB_to_YCoCg(rgb);
            ycocg.x = DecompressLuminance(ycocg.x);
            rgb = YCoCg_to_RGB(ycocg);
            rgb *= g_InvExposureMultiplier;

            g_FinalOutputTarget[hrCoord] = float4(max(rgb, 0.0f), 1.0f);
        }
    }
}
```

## 10. Baselines

| Baseline                   | Type                       | Source                                                                                                | Licence                        | Reproducible                        | Role                              |
| :------------------------- | :------------------------- | :---------------------------------------------------------------------------------------------------- | :----------------------------- | :---------------------------------- | :-------------------------------- |
| **FSR 3.1** (primary)      | Heuristic SR + FG          | [GPUOpen-LibrariesAndSDKs/FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) | MIT                            | Yes, fully open source              | Primary comparison target         |
| **FSR 1.0** (spatial only) | Spatial filter             | Same repo                                                                                             | MIT                            | Yes                                 | Lower bound spatial reference     |
| **Bicubic**                | Interpolation              | OpenCV / PyTorch `F.interpolate`                                                                      | BSD                            | Yes                                 | Trivial baseline                  |
| **RIFE v4.x**              | Neural frame interpolation | [hzwer/Practical-RIFE](https://github.com/hzwer/Practical-RIFE)                                       | Non-commercial (newer weights) | Yes (weights available)             | FG quality upper bound            |
| **IFRNet**                 | Neural frame interpolation | [ltkong218/IFRNet](https://github.com/ltkong218/IFRNet)                                               | Apache 2.0                     | Yes                                 | FG baseline (commercially usable) |
| **BasicVSR++**             | Neural video SR            | [ckkelvinchan/BasicVSR_PlusPlus](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)                   | Apache 2.0                     | Yes (offline only, too slow for RT) | SR quality upper bound            |
| **ECBSR**                  | Lightweight SR             | [xindongzhang/ECBSR](https://github.com/xindongzhang/ECBSR)                                           | MIT                            | Yes                                 | Lightweight SR alternative        |
| **Rep-TNSR v1**            | Lightweight temporal SR    | This project (existing architecture)                                                                  | Project-internal               | Yes                                 | Ablation baseline                 |

BasicVSR++ and RIFE are offline quality-ceiling references. The primary latency-matched comparison is FSR 3.1 under identical input conditions and hardware.

## 11. Evaluation Protocol

### 11.1 Test Configurations

| Config | Input Resolution | Output Resolution | Scale Factor | Frame Generation | Purpose                  |
| :----- | :--------------- | :---------------- | :----------- | :--------------- | :----------------------- |
| C1     | 640x360          | 1920x1080         | 3x           | No               | Primary SR evaluation    |
| C2     | 960x540          | 1920x1080         | 2x           | No               | Medium scale SR          |
| C3     | 1280x720         | 1920x1080         | 1.5x         | No               | Quality mode SR          |
| C4     | 640x360          | 1920x1080         | 3x           | Yes              | Full pipeline            |
| C5     | 1280x720         | 2560x1440         | 2x           | Yes              | Higher target resolution |
| C6     | 1280x720         | 3840x2160         | 3x           | No               | 4K output test           |

### 11.2 Test Sequences (5 Scenes, 300 Frames Each)

1. **Bistro Interior**: Static camera, fine detail, specular surfaces, complex lighting.
2. **City Chase**: Fast camera motion, many disocclusions, moving vehicles, reflections.
3. **Forest Walk**: Alpha-tested foliage, particle effects (fireflies), subsurface scattering.
4. **SciFi Corridor**: Emissive materials, screen-space reflections, thin geometry (cables, pipes).
5. **Stylised Village**: Non-photorealistic rendering, flat shading, cel-shaded outlines, cartoon palette.

### 11.3 Objective Metrics

| Category             | Metric                      | Implementation Source                                                                                                                                                                      | Measures                                       |
| :------------------- | :-------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------- |
| Spatial fidelity     | PSNR (Y-channel)            | [Lightning-AI/torchmetrics](https://github.com/Lightning-AI/torchmetrics) `PeakSignalNoiseRatio` (Apache 2.0)                                                                              | Pixel-level reconstruction accuracy            |
| Spatial fidelity     | SSIM                        | `torchmetrics.StructuralSimilarityIndexMeasure`                                                                                                                                            | Structural preservation                        |
| Perceptual quality   | LPIPS (AlexNet)             | [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) `lpips.LPIPS(net='alex')` (BSD 2-Clause)                                                               | Learned perceptual distance                    |
| Perceptual quality   | VMAF                        | [Netflix/vmaf](https://github.com/Netflix/vmaf) via `ffmpeg -filter_complex libvmaf` (BSD+Patent)                                                                                          | Video multi-method quality fusion              |
| Temporal stability   | $E_{\text{warp}}$           | Custom: GT MV warped L1, masked by valid pixels                                                                                                                                            | Temporal consistency                           |
| Temporal stability   | tOF                         | RAFT-estimated flow comparison. RAFT: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) (BSD 3-Clause). Reference metric code: [thunil/TecoGAN](https://github.com/thunil/TecoGAN) | Optical flow temporal fidelity                 |
| Temporal stability   | tLP                         | LPIPS between consecutive warped frames. Reference: TecoGAN and BasicVSR++ evaluation scripts.                                                                                             | Temporal perceptual stability                  |
| Temporal flicker     | MAFD                        | Custom: `mean(abs(I_t - I_{t-1}))` on Y-channel                                                                                                                                            | Raw flicker magnitude                          |
| Ghosting             | Ghost ratio                 | Custom: percentage of pixels where $E_{\text{warp}} > \tau$ in disoccluded regions ($\tau = 0.05$)                                                                                         | Ghosting severity                              |
| Disocclusion quality | LPIPS (disoccluded regions) | Masked LPIPS using GT occlusion maps                                                                                                                                                       | Reconstruction quality in newly revealed areas |
| FG motion accuracy   | EPE (End-Point Error)       | Compare estimated intermediate flow vs GT flow on Sintel                                                                                                                                   | Frame generation motion reconstruction         |
| Latency              | GPU execution time (ms)     | D3D12 `ID3D12QueryHeap` timestamp queries / `cudaEventElapsedTime`                                                                                                                         | Per-pass and total pipeline latency            |
| Throughput           | Effective FPS               | $1000 / \text{total\_pipeline\_ms}$                                                                                                                                                        | Rendering throughput                           |

**$E_{\text{warp}}$ exact definition:**

$$E_{\text{warp}} = \frac{1}{T-1}\sum_{t=2}^T \frac{\sum_i M_t(i) \cdot \left\| \hat{I}_t(i) - \mathcal{W}(\hat{I}_{t-1}, V_{t \to t-1})(i) \right\|_1}{\sum_i M_t(i)}$$

Where $M_t$ is the valid-pixel mask (non-disoccluded region).

**tOF exact definition:**

$$\text{tOF} = \frac{1}{T-1}\sum_{t=2}^T \left\| \text{Flow}_{\text{RAFT}}(\hat{I}_t, \hat{I}_{t-1}) - \text{Flow}_{\text{RAFT}}(I_t^{\text{GT}}, I_{t-1}^{\text{GT}}) \right\|_1$$

### 11.4 Human/Perceptual Evaluation

**Protocol**: Two-alternative forced choice (2AFC).

- 20 participants (balanced mix of experienced gamers and non-gamers).
- 30 paired comparisons per participant (Ours vs FSR 3.1).
- 5-second video clips at native playback speed, displayed at native monitor resolution.
- Randomised left/right placement per trial, double-blind (neither participant nor experimenter knows which is which during viewing).
- Questions per trial: "Which video has (a) better detail? (b) smoother motion? (c) fewer artefacts?"
- Analysis: preference rate $\pm$ 95% CI, Bradley-Terry model scores for overall ranking.

**Minimum significance**: preference rate > 60% on a one-sided binomial test, $p < 0.05$.

### 11.5 Generalisation Tests

| Test                | Condition                                                                               | Pass Criterion                                                                 |
| :------------------ | :-------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------- |
| Unseen game content | 3 held-out UE5 scenes                                                                   | LPIPS degradation < 15% relative to training-distribution content              |
| Unseen engine       | 2 Unity HDRP scenes                                                                     | LPIPS degradation < 15%                                                        |
| Unseen art style    | Cartoon/cel-shaded, photorealistic, pixel art                                           | LPIPS degradation < 20% (wider tolerance for extreme style shift)              |
| Unseen resolution   | 4K output (not trained at 4K)                                                           | PSNR degradation < 1.0 dB                                                      |
| Cross-GPU           | GTX 1650, RTX 3060, RTX 4070, RX 6700 XT, RX 7800 XT, Arc A770                          | Latency within budget per GPU tier, output bit-identical within FP16 tolerance |
| Degraded inputs     | Missing MV (zero-filled), noisy depth ($\pm 5\%$ Gaussian), wrong exposure ($\pm 1$ EV) | Graceful degradation, no NaN/corruption, PSNR drop < 3 dB                      |

### 11.6 Statistical Significance and Confidence Intervals

- Report: mean $\pm$ standard deviation across test sequences for all metrics.
- Paired t-test (or Wilcoxon signed-rank test if normality assumption fails via Shapiro-Wilk) for each metric comparing Ours vs FSR 3.1.
- Bonferroni correction for multiple comparisons (13 metrics $\Rightarrow$ corrected $\alpha = 0.05 / 13 \approx 0.0038$).
- 95% confidence intervals for all reported metric differences.
- Effect size: Cohen's $d$ for primary comparisons.
- Minimum sample: 5 test scenes $\times$ 300 frames = 1,500 frame-level measurements per metric.

## 12. Ablation Matrix

### 12.1 Architecture Ablations (SR)

| ID  | Ablation            | Variants                                            | Expected Insight                                         |
| :-- | :------------------ | :-------------------------------------------------- | :------------------------------------------------------- |
| A1  | Channel width       | 16 / 20 / 24 / 32                                   | Quality vs latency Pareto frontier                       |
| A2  | Block count         | 3 / 4 / 5 / 6 / 8                                   | Depth vs compute tradeoff                                |
| A3  | Edge operators      | With vs without Sobel/Laplacian branches            | Contribution of differential operators to edge sharpness |
| A4  | Input channels      | 9ch (v1) vs 12ch (v2: +depth, +confidence, +jitter) | Marginal value of additional input signals               |
| A5  | Activation function | PReLU vs ReLU vs GELU vs SiLU                       | Activation choice impact on quality and inference speed  |
| A6  | Upsampling init     | Random init vs ICNR PixelShuffle init               | Checkerboard artefact reduction                          |

### 12.2 Architecture Ablations (FG)

| ID  | Ablation              | Variants                                                      | Expected Insight                         |
| :-- | :-------------------- | :------------------------------------------------------------ | :--------------------------------------- |
| B1  | Processing resolution | 2x / 4x / 8x downsample                                       | Resolution vs quality tradeoff           |
| B2  | Flow refinement       | Single-scale vs 2-level multi-scale pyramid                   | Large motion handling capability         |
| B3  | Engine MV usage       | With engine MVs vs optical-flow-only                          | Value of engine geometric motion vectors |
| B4  | Occlusion method      | Learned sigmoid weights vs forward-backward consistency check | Disocclusion quality                     |

### 12.3 Loss Ablations

| ID  | Configuration            | Losses Active                                                                                                                   |
| :-- | :----------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| C1  | Charbonnier only         | $\mathcal{L}_{\text{char}}$                                                                                                     |
| C2  | + Edge                   | $\mathcal{L}_{\text{char}} + \mathcal{L}_{\text{edge}}$                                                                         |
| C3  | + Perceptual             | $\mathcal{L}_{\text{char}} + \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{perc}}$                                             |
| C4  | + Temporal               | $\mathcal{L}_{\text{char}} + \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{perc}} + \mathcal{L}_{\text{temp}}$                 |
| C5  | + Frequency (full)       | All 5 losses (proposed configuration)                                                                                           |
| C6  | L1 replacing Charbonnier | $\mathcal{L}_1 + \mathcal{L}_{\text{edge}} + \mathcal{L}_{\text{perc}} + \mathcal{L}_{\text{temp}} + \mathcal{L}_{\text{freq}}$ |
| C7  | LPIPS as training loss   | Replace VGG perceptual with LPIPS                                                                                               |

### 12.4 Data Ablations

| ID  | Configuration                                                           | Purpose                                           |
| :-- | :---------------------------------------------------------------------- | :------------------------------------------------ |
| D1  | UE5 data only (12 scenes)                                               | Baseline data quality                             |
| D2  | UE5 + Sintel + TartanAir                                                | Cross-domain diversity impact                     |
| D3  | Native LR renders vs bicubic-downsampled HR                             | Importance of realistic aliasing in training data |
| D4  | With vs without degradation augmentation (MV noise, depth quantisation) | Robustness impact of degradation simulation       |
| D5  | 6 scenes vs 12 scenes                                                   | Data scale requirements                           |

### 12.5 Training Strategy Ablations

| ID  | Configuration                                   | Purpose                   |
| :-- | :---------------------------------------------- | :------------------------ |
| E1  | No curriculum (all losses from iteration 0)     | Value of phased training  |
| E2  | Curriculum (proposed 3-phase)                   | Proposed approach         |
| E3  | Larger patches (128x128 LR) throughout          | Patch context size impact |
| E4  | Adam vs AdamW vs SGD with momentum              | Optimiser comparison      |
| E5  | Cosine annealing vs step decay vs warm restarts | Schedule comparison       |

**Total ablations: ~30 experiments.** Each ~20 hours on 4x RTX 4090. Total ablation compute: ~600 GPU-hours.

## 13. Failure Modes and Mitigation Strategies

| Failure Mode                          | Root Cause                                                | Detection Method                                         | Mitigation Strategy                                                                                                                |
| :------------------------------------ | :-------------------------------------------------------- | :------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| **Ghosting on fast motion**           | History clamping AABB too permissive                      | High $E_{\text{warp}}$ in motion regions                 | Tighten $\gamma$ (1.25 $\to$ 1.0); increase disocclusion mask sensitivity ($\alpha$ 10 $\to$ 15)                                   |
| **Checkerboard artefacts**            | PixelShuffle random initialisation bias                   | Visual inspection of HR output at pixel level            | ICNR weight initialisation for final conv layer (ref: [ablation A6])                                                               |
| **Temporal flicker on thin geometry** | Aliased LR input below Nyquist sampling frequency         | MAFD spikes on wire/fence test sequences                 | Increase jitter phase count $K$; add anti-flicker temporal post-filter                                                             |
| **FG double-image on fast rotation**  | Optical flow quarter-resolution search range exceeded     | EPE > threshold on fast-rotation validation sequences    | Multi-scale pyramid refinement (2 levels) [ablation B2]; scene-cut detection bypass                                                |
| **HUD/UI distortion in FG**           | UI composited before FG pass warps static elements        | Visual inspection with UI-heavy scenes                   | Enforce UI separation: separate overlay texture composited AFTER FG                                                                |
| **Particle/transparency smearing**    | Missing MV for non-depth-writing geometry                 | Reactive mask coverage analysis on particle-heavy scenes | Train with reactive-mask-aware loss weighting (zero temporal loss in masked regions); fall back to current frame in masked regions |
| **Colour shift in HDR scenes**        | Luminance compression roundtrip numerical error           | PSNR drop > 1 dB on HDR-bright test regions              | Higher precision $\epsilon$ in compress/decompress; FP32 for compress/decompress path                                              |
| **Generalisation failure**            | Training data distribution mismatch with target game      | LPIPS degradation > 15% on held-out test set             | Expand training data diversity (more scenes, more engines); add style augmentation (colour transfer, contrast randomisation)       |
| **NaN propagation in FP16**           | Unbounded input values exceeding FP16 range ($\pm 65504$) | Runtime NaN check on output tensor                       | Clamp all inputs to FP16 safe range before neural trunk; use FP32 for warp grid coordinates in FG                                  |
| **Memory bandwidth saturation**       | Too many intermediate buffer reads/writes per frame       | GPU profiler memory utilisation > 90%                    | Fuse pre/post-processing with first/last neural layers in HLSL; eliminate intermediate buffer roundtrips                           |

## 14. Claims for Meaningful Superiority over FSR 3.1

The following claims, if supported by experimental evidence meeting the statistical requirements in Section 11.6, would constitute meaningful and defensible superiority:

1. **Spatial quality**: PSNR improvement $\geq$ 1.5 dB AND LPIPS improvement $\geq$ 20% averaged across all 5 test scenes at Quality mode (67% scale), with $p < 0.01$ on paired test.

2. **Temporal stability**: $E_{\text{warp}}$ reduction $\geq$ 25% AND tOF reduction $\geq$ 20% averaged across all dynamic test scenes (City Chase, Forest Walk, SciFi Corridor), with $p < 0.01$.

3. **Frame generation quality**: Interpolated frame PSNR improvement $\geq$ 2.0 dB AND LPIPS improvement $\geq$ 25% vs FSR 3.1 FG, with $p < 0.01$.

4. **Latency parity**: Total pipeline latency within 120% of FSR 3.1 on the same hardware (allowing up to 20% overhead for neural inference cost).

5. **Generalisation**: Quality metrics degrade by < 15% (relative LPIPS) on held-out unseen game content compared to training-distribution content.

6. **Human preference**: $\geq$ 60% preference rate in 2AFC study with $\geq$ 20 participants, $p < 0.05$ (one-sided binomial).

## 15. Reproducibility Checklist and Experiment Tracking

### 15.1 Reproducibility Checklist

- [ ] All random seeds fixed and documented (Python `random.seed(42)`, NumPy `np.random.seed(42)`, PyTorch `torch.manual_seed(42)`, CUDA `torch.cuda.manual_seed_all(42)`).
- [ ] `CUBLAS_WORKSPACE_CONFIG=:16:8` set for deterministic cuBLAS operations.
- [ ] Exact package versions pinned in `requirements.txt` (PyTorch, torchvision, lpips, torchmetrics, omegaconf, wandb, OpenEXR).
- [ ] Dataset download scripts automated with SHA-256 checksum verification.
- [ ] Training command reproducible with a single `torchrun` invocation and config file.
- [ ] Checkpoint saves full state: model, optimiser, scheduler, AMP scaler, all RNG states, iteration counter, best metric.
- [ ] Evaluation scripts produce deterministic results given identical model checkpoint and test data.
- [ ] All metrics computed using specified library versions (not custom reimplementations unless documented).
- [ ] Hardware specifications logged: GPU model, driver version, CUDA version, PyTorch version.
- [ ] Git commit hash logged with each experiment run.

### 15.2 Experiment Tracking Schema

```yaml
experiment:
  id: "exp_YYYYMMDD_HHMMSS_{short_hash}"
  git_commit: "abc123def456"
  config_hash: "sha256:..."
  config_path: "configs/sr_ultra.yaml"
  hardware:
    gpus: ["NVIDIA RTX 4090 x4"]
    driver: "560.35.03"
    cuda: "12.4"
    pytorch: "2.4.0"
    os: "Ubuntu 22.04"
  training:
    dataset: "ue5_12scene_v2"
    model_variant: "sr_ultra_24ch_6blk"
    total_iterations: 500000
    batch_size_per_gpu: 16
    effective_batch_size: 64
    lr_initial: 5e-4
    loss_config: "full_5loss_curriculum"
    wall_clock_hours: 20.5
    gpu_hours: 82.0
  validation: # best validation result
    iteration: 420000
    psnr: 35.2
    ssim: 0.953
    lpips: 0.062
    checkpoint: "ckpt_iter_420000.pth"
  test: # final test set evaluation
    psnr_mean: 35.1
    psnr_std: 1.2
    ssim_mean: 0.952
    lpips_mean: 0.063
    lpips_std: 0.008
    ewarp_mean: 0.00178
    tof_mean: 0.00285
    vmaf_mean: 87.3
    latency_p95_ms: 1.35
```

**Tracking tool**: [Weights & Biases](https://wandb.ai/) (free tier for personal/academic) or [MLflow](https://github.com/mlflow/mlflow) (Apache 2.0, self-hosted). W&B preferred for experiment comparison dashboards.

## 16. Project Directory Structure

```
neuralss/
├── configs/
│   ├── sr_performance.yaml          # 4 blocks, 16 channels, s=3
│   ├── sr_quality.yaml              # 4 blocks, 20 channels, s=3
│   ├── sr_ultra.yaml                # 6 blocks, 24 channels, s=3
│   ├── fg_default.yaml              # Frame generation default
│   ├── eval_fsr31.yaml              # FSR 3.1 baseline eval config
│   └── ablations/                   # One YAML per ablation experiment
│       ├── sr_a1_16ch.yaml
│       ├── sr_a1_32ch.yaml
│       ├── sr_c1_char_only.yaml
│       └── ...
├── neuralss/                        # Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── repconv_block.py         # ReparameterizedConvBlock
│   │   ├── sr_net.py                # RepTNSR v2 (all tiers)
│   │   ├── fg_net.py                # NeuralFG
│   │   └── fuse.py                  # Structural reparameterisation (branch collapse)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── charbonnier.py
│   │   ├── edge.py
│   │   ├── perceptual.py            # VGG-19 feature extractor (frozen)
│   │   ├── temporal.py              # Warp consistency loss with disocclusion mask
│   │   ├── frequency.py             # FFT magnitude loss
│   │   └── census.py                # Census transform loss (for FG)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py               # GBufferDataset (EXR/binary loader)
│   │   ├── augmentation.py          # Spatial + temporal augmentations with MV adjustment
│   │   ├── preprocessing.py         # YCoCg, exposure, luminance compression
│   │   └── cache.py                 # Memory-mapped pre-cropped patch cache
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── spatial.py               # PSNR, SSIM wrappers
│   │   ├── perceptual.py            # LPIPS wrapper
│   │   ├── temporal.py              # E_warp, tOF, tLP, MAFD
│   │   ├── vmaf.py                  # VMAF subprocess wrapper (calls ffmpeg)
│   │   └── ghosting.py              # Ghost ratio metric
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop (DDP-aware, AMP, curriculum)
│   │   ├── evaluator.py             # Evaluation harness (all metrics)
│   │   ├── curriculum.py            # Loss weight scheduling
│   │   └── checkpoint.py            # Save/load with full RNG state
│   └── export/
│       ├── __init__.py
│       ├── onnx_export.py           # Fused model -> ONNX
│       ├── ncnn_export.py           # PNNX -> ncnn .param/.bin
│       └── tensorrt_export.py       # ONNX -> TensorRT .engine
├── scripts/
│   ├── download_datasets.py         # Automated download + SHA-256 verify
│   ├── capture_ue5.py               # UnrealCV G-buffer capture automation
│   ├── generate_cache.py            # Pre-crop training patches to .bin
│   ├── benchmark.py                 # All metrics, all baselines, all configs
│   ├── ablation_sweep.py            # Launch ablation experiments (multi-GPU)
│   ├── export_all.py                # Export to ONNX, ncnn, TRT
│   └── profile_latency.py           # D3D12/CUDA latency profiling
├── deploy/
│   ├── directml/
│   │   ├── SuperResolutionSystem.h
│   │   ├── SuperResolutionSystem.cpp
│   │   ├── PreProcessTemporal.hlsl
│   │   ├── PostProcessReconstruct.hlsl
│   │   └── CMakeLists.txt
│   ├── ncnn_vulkan/
│   │   ├── sr_pipeline.cpp
│   │   ├── fg_pipeline.cpp
│   │   └── CMakeLists.txt
│   └── weights/                     # Exported model files
│       ├── sr_quality_fused.onnx
│       ├── sr_ultra_fused.onnx
│       ├── fg_default.onnx
│       ├── sr_quality.param / .bin  # ncnn format
│       └── sr_quality_fp16.engine   # TensorRT format
├── tests/
│   ├── test_repconv_block.py        # Reparameterisation correctness
│   ├── test_fuse.py                 # Fused == multi-branch numerical parity
│   ├── test_losses.py               # Loss computation correctness
│   ├── test_dataset.py              # Data loading, augmentation, MV adjustment
│   ├── test_metrics.py              # Metric implementations vs reference
│   ├── test_onnx_export.py          # ONNX export + inference parity
│   ├── test_inference_parity.py     # PyTorch vs ONNX vs DirectML output match
│   └── test_latency.py             # Sub-budget latency verification on target HW
├── train.py                         # Entry: python train.py --config configs/sr_ultra.yaml
├── evaluate.py                      # Entry: python evaluate.py --checkpoint ... --test_dir ...
├── infer.py                         # Entry: python infer.py --model ... --input ...
├── requirements.txt
├── pyproject.toml
├── RESEARCH.md                      # This document
└── LICENSE
```

## 17. Configuration System

YAML-based with [OmegaConf](https://github.com/omry/omegaconf) (BSD 3-Clause) for hierarchical config merging and CLI overrides.

```yaml
# configs/sr_ultra.yaml
model:
  type: "RepTNSR"
  in_channels: 12
  base_channels: 24
  num_blocks: 6
  scale: 3
  activation: "prelu"

data:
  train_dir: "data/ue5_train"
  val_dir: "data/ue5_val"
  patch_size_lr: 64
  batch_size: 16
  num_workers: 8
  temporal_frames: 2
  augmentation:
    flip_h: true
    flip_v: true
    rotate: true
    temporal_reverse: true
    mv_noise_sigma: 0.3
    brightness_jitter: 0.1

loss:
  charbonnier: { weight: 1.0, epsilon: 1.0e-3 }
  edge: { weight: 0.5 }
  perceptual: { weight: 0.05, layer: "conv3_3", network: "vgg19" }
  temporal: { weight: 0.25, alpha: 10.0 }
  frequency: { weight: 0.1 }

training:
  total_iterations: 500000
  optimizer:
    { type: "adamw", lr: 5.0e-4, betas: [0.9, 0.999], weight_decay: 1.0e-4 }
  scheduler: { type: "cosine", T_max: 500000, eta_min: 1.0e-6 }
  grad_clip: 1.0
  mixed_precision: true
  curriculum:
    phase1_end: 100000
    phase2_end: 300000
    phase3_end: 500000
    phase3_patch_size_lr: 96

checkpoint:
  save_interval: 10000
  keep_last: 5
  save_best: true
  best_metric: "val/lpips"
  best_mode: "min"

logging:
  backend: "wandb"
  project: "neuralss"
  log_interval: 100
  val_interval: 5000
```

CLI override example: `python train.py --config configs/sr_ultra.yaml model.base_channels=32 training.total_iterations=200000`

## 18. Commands

```bash
# Dataset download (Sintel, TartanAir, REDS)
python scripts/download_datasets.py --datasets sintel tartanair reds --output data/

# UE5 G-buffer capture (requires UnrealCV plugin installed)
python scripts/capture_ue5.py \
    --project /path/to/UE5/project \
    --scenes bistro city_chase forest scifi_corridor stylised_village \
    --output data/ue5_train/ \
    --fps 60 --duration 60 --jitter_phases 9

# Generate training cache (pre-crop patches)
python scripts/generate_cache.py \
    --input data/ue5_train/ --output data/cache/ \
    --patch_size 64 --scale 3 --format fp16

# Train SR (single GPU)
python train.py --config configs/sr_ultra.yaml

# Train SR (multi-GPU DDP, 4 GPUs)
torchrun --nproc_per_node=4 train.py --config configs/sr_ultra.yaml --distributed

# Train Frame Generation
torchrun --nproc_per_node=4 train.py --config configs/fg_default.yaml --distributed

# Evaluate SR
python evaluate.py \
    --checkpoint checkpoints/sr_ultra_best.pth \
    --config configs/sr_ultra.yaml \
    --test_dir data/ue5_test/ \
    --output results/sr_ultra/

# Full benchmark (all methods, all metrics)
python scripts/benchmark.py \
    --methods ours fsr31 fsr1 bicubic ecbsr rife ifrnet \
    --test_dir data/ue5_test/ \
    --output results/benchmark/ \
    --configs C1 C2 C3 C4

# Export to deployment formats
python scripts/export_all.py \
    --checkpoint checkpoints/sr_ultra_best.pth \
    --config configs/sr_ultra.yaml \
    --formats onnx ncnn tensorrt

# Run ablation sweep
python scripts/ablation_sweep.py \
    --ablation_dir configs/ablations/ \
    --gpus 0,1,2,3 \
    --iterations 200000

# Unit and integration tests
pytest tests/ -v --tb=short

# Latency profiling
python scripts/profile_latency.py \
    --model weights/sr_quality_fused.onnx \
    --input_size 640 360 \
    --backend directml \
    --warmup 100 --iterations 1000
```

## 19. Unit and Integration Tests

```python
# tests/test_repconv_block.py
def test_reparameterisation_equivalence():
    """Fused 3x3 conv must produce identical output to multi-branch."""
    block = ReparameterizedConvBlock(20, 20)
    block.eval()
    x = torch.randn(1, 20, 64, 64)
    out_multi = block(x)
    w_fused, b_fused = block.export_fused_weight()
    out_fused = F.conv2d(x, w_fused, b_fused, padding=1)
    out_fused = block.act(out_fused)
    assert torch.allclose(out_multi, out_fused, atol=1e-5)

# tests/test_fuse.py
def test_full_model_fuse_parity():
    """Full model output before and after fusion must match."""
    model = RepTNSR(in_channels=12, base_channels=24, num_blocks=6, scale=3)
    model.eval()
    x = torch.randn(1, 12, 64, 64)
    out_before = model(x)
    fused_model = fuse_model(model)
    out_after = fused_model(x)
    assert torch.allclose(out_before, out_after, atol=1e-4)

# tests/test_losses.py
def test_charbonnier_gradient_at_zero():
    """Charbonnier must have non-zero gradient when pred == target."""
    pred = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 3, 8, 8)
    loss = charbonnier_loss(pred, target, eps=1e-3)
    loss.backward()
    assert pred.grad.abs().sum() > 0

def test_temporal_loss_zero_in_disoccluded():
    """Temporal loss must be zero where disocclusion mask is zero."""
    # Construct mask that is zero everywhere
    mask = torch.zeros(1, 1, 8, 8)
    loss = temporal_consistency_loss(
        pred=torch.randn(1, 3, 8, 8),
        warped_prev=torch.randn(1, 3, 8, 8),
        mask=mask
    )
    assert loss.item() == 0.0

# tests/test_onnx_export.py
def test_onnx_pytorch_parity():
    """ONNX model output matches PyTorch within FP16 tolerance."""
    model = RepTNSR(in_channels=12, base_channels=24, num_blocks=6, scale=3)
    fused_model = fuse_model(model)
    fused_model.eval()
    x = torch.randn(1, 12, 64, 64)
    pt_out = fused_model(x).detach().numpy()
    # Export
    torch.onnx.export(fused_model, x, "test_sr.onnx", opset_version=17)
    # Verify
    sess = ort.InferenceSession("test_sr.onnx")
    ort_out = sess.run(None, {"input": x.numpy()})[0]
    assert np.allclose(pt_out, ort_out, atol=1e-3)

# tests/test_latency.py (integration, requires target GPU)
def test_sr_latency_within_budget():
    """SR inference must complete within 3 ms on target GPU."""
    model_path = "weights/sr_quality_fused.onnx"
    x = torch.randn(1, 9, 360, 640).half().cuda()  # Quality tier
    # ... load and benchmark ...
    # Assert P95 < 3.0 ms
```

## 20. Performance Characterisation (Existing Rep-TNSR v1 Profiling Data)

Microarchitectural profiling was conducted on an NVIDIA GeForce GTX 1650 Mobile (TU117, 896 CUDA cores, 4 GB GDDR5 at 128 GB/s, 50W TGP) running under Windows 11 with driver branch 550.x. GPU execution timeline measured using D3D12 timestamp queries through `ID3D12QueryHeap`.

| Execution Stage                            | Dispatch Dimensions          | Memory Traffic         | Arithmetic Workload | GPU Latency (TU117) |
| :----------------------------------------- | :--------------------------- | :--------------------- | :------------------ | :------------------ |
| Pass 1: Dilation, Temporal & YCoCg         | 80 $\times$ 45 thread groups | 4.58 MB / 3.68 MB      | 0.052 GFLOPs        | 0.282 ms            |
| Pass 2: Fused FP16 Neural Trunk (L1 to L5) | DirectML compiled pipeline   | 8.55 MB / 8.55 MB      | 7.962 GFLOPs        | 2.145 ms            |
| Pass 3: Sub-Pixel Shuffle & Decompression  | 80 $\times$ 45 thread groups | 12.44 MB / 16.59 MB    | 0.015 GFLOPs        | 0.184 ms            |
| Pass 4: Resource Barriers & Command Queue  | D3D12 timeline               | Negligible             | Negligible          | 0.112 ms            |
| **Complete SR System**                     |                              | **54.39 MB aggregate** | **8.029 GFLOPs**    | **2.723 ms**        |

The complete v1 pipeline executes in 2.723 ms, within the 3.0 ms budget. Memory traffic of 54.39 MB corresponds to 14.16% of the 384 MB transfer limit in a 3.0 ms window on a 128 GB/s interface.

Reconstruction quality (existing v1 evaluation against spatial-only and single-frame methods):

| Method                 | Input            | PSNR (dB) | SSIM      | IF-SSIM   | $E_{\text{warp}}$ ($\times 10^{-3}$) | GTX 1650 Runtime |
| :--------------------- | :--------------- | :-------- | :-------- | :-------- | :----------------------------------- | :--------------- |
| Bicubic Interpolation  | 360p $\to$ 1080p | 27.34     | 0.812     | 0.892     | 8.42                                 | 0.08 ms          |
| AMD FSR 1.0 (Spatial)  | 360p $\to$ 1080p | 28.12     | 0.835     | 0.901     | 7.91                                 | 0.42 ms          |
| QuickSRNet-Medium      | 360p $\to$ 1080p | 31.05     | 0.884     | 0.914     | 6.84                                 | 2.21 ms          |
| **Rep-TNSR v1**        | 360p $\to$ 1080p | **34.82** | **0.941** | **0.986** | **1.72**                             | 2.72 ms          |
| Native 1080p Reference | Native SSAA      | $\infty$  | 1.000     | 0.994     | 1.15                                 | 16.67 ms         |

Rep-TNSR v1 improves by +6.70 dB over FSR 1.0 and +3.77 dB over QuickSRNet-Medium. These results are against FSR 1.0 (spatial only); the comparison against FSR 3.1 (temporal) is pending and is the core objective of this project.

## 21. Implementation Roadmap

### Phase 0: Minimum Viable Prototype and Hypothesis Falsification (Weeks 1 to 3)

**Goal**: Validate H1 (neural SR beats FSR 3.1 spatial quality) with minimum effort.

**Deliverables**:

- [ ] Existing Rep-TNSR v1 (17.4K params) trained on 3 UE5 scenes (Bistro, Valley, City Chase).
- [ ] Evaluation harness computing PSNR, SSIM, LPIPS.
- [ ] FSR 3.1 baseline frame dumps from FidelityFX SDK sample app (Bistro scene, Quality mode).
- [ ] Side-by-side quantitative comparison report.

**Acceptance criteria**:

- PSNR(ours) > PSNR(FSR 3.1) by $\geq$ 1.0 dB on $\geq$ 2/3 test scenes.
- LPIPS(ours) < LPIPS(FSR 3.1) on $\geq$ 2/3 test scenes.
- All evaluation code automated and reproducible.

### Phase 1: Enhanced SR (Weeks 4 to 8)

**Goal**: Rep-TNSR v2 with expanded inputs and full multi-objective training.

**Deliverables**:

- [ ] 12-channel input pipeline (add depth, temporal confidence, jitter phase).
- [ ] Rep-TNSR v2 architecture (3 quality tiers implemented).
- [ ] Full 5-loss training with curriculum.
- [ ] Frequency loss implementation.
- [ ] 12-scene UE5 dataset captured and cached.
- [ ] Complete architecture + loss ablation matrix.
- [ ] Temporal stability metrics ($E_{\text{warp}}$, tOF, tLP).

**Acceptance criteria**: SR meets all target metrics from Section 1.3. Ablations complete with statistical significance.

### Phase 2: Frame Generation (Weeks 9 to 13)

**Goal**: NeuralFG producing higher-quality interpolated frames than FSR 3.1 FG.

**Deliverables**:

- [ ] NeuralFG architecture implemented and trained.
- [ ] Census transform loss.
- [ ] FG-specific training data (GT mid-frames).
- [ ] FG ablation matrix.
- [ ] Comparison vs FSR 3.1 FG, RIFE v4, IFRNet.

**Acceptance criteria**: FG meets target metrics. FG latency < 2.5 ms on RTX 3060.

### Phase 3: Integration and Deployment (Weeks 14 to 18)

**Goal**: Complete SR + FG pipeline deployable via DirectML, ncnn, and native HLSL.

**Deliverables**:

- [ ] DirectML C++ integration (updated from existing code for v2 + FG).
- [ ] ncnn Vulkan export and benchmark.
- [ ] TensorRT fast-path for NVIDIA.
- [ ] Frame pacing implementation.
- [ ] Multi-GPU latency profiling (6+ GPU models).
- [ ] VRAM usage profiling.
- [ ] Complete benchmark report (all configs, all baselines, all GPUs).

**Acceptance criteria**: Latency within 120% of FSR 3.1 on all tested GPUs. VRAM < 100 MB. No NaN or visual corruption.

### Phase 4: Generalisation and Hardening (Weeks 19 to 22)

**Goal**: Cross-game/engine generalisation and robustness validation.

**Deliverables**:

- [ ] Unity HDRP test captures and evaluation.
- [ ] Held-out UE5 scenes evaluation.
- [ ] Art style diversity testing.
- [ ] Degraded input robustness testing.
- [ ] Human perceptual evaluation study (2AFC, $n \geq 20$).
- [ ] Final benchmark report with confidence intervals and effect sizes.

**Acceptance criteria**: All claims from Section 14 supported with $p < 0.05$.

## 22. Smallest Experiment to Falsify the Core Hypothesis

**Experiment SR-001:**

1. Train existing Rep-TNSR v1 (17.4K params, 4 blocks, 20 channels) on the Bistro scene only.
2. 100K iterations, Charbonnier + Edge loss only, single GPU, ~6 hours.
3. Capture FSR 3.1 Quality mode output on the identical Bistro test sequence (300 frames, 360p $\to$ 1080p, $3\times$ scale) using the FidelityFX SDK sample app. Extract frame dumps via RenderDoc.
4. Compute PSNR (Y-channel), SSIM (RGB), LPIPS (AlexNet) per-frame.

**Falsification criterion**: If FSR 3.1 achieves higher PSNR AND lower LPIPS than Rep-TNSR v1 on this sequence, the hypothesis that a lightweight neural network can outperform FSR 3.1's hand-crafted temporal upscaler at this parameter scale is falsified. This would indicate the need for: (a) significantly larger networks (possibly contradicting real-time constraints), (b) fundamentally different architecture (attention-based), or (c) additional input signals not currently provided.

**Expected outcome**: Based on existing v1 results (34.82 dB PSNR vs FSR 1.0's 28.12 dB), and FSR 3.1's estimated ~33.5 dB at Quality mode (significant improvement over FSR 1.0 due to temporal accumulation), we expect the neural approach to still win on both metrics. However, this has NOT been verified against FSR 3.1 specifically. The gap may be narrower than against FSR 1.0.

**Required resources**: 1x RTX 3090, ~10 GB data, ~8 hours total (training + evaluation). This is the minimum cost to determine whether the project direction is viable.

## 23. Exact Experiment Matrix

| Exp ID      | Model              | Config              | Dataset                | Loss                | Scale | Metric Focus                   |
| :---------- | :----------------- | :------------------ | :--------------------- | :------------------ | :---- | :----------------------------- |
| **SR-001**  | RepTNSR v1 (17K)   | Quality (20ch/4blk) | Bistro only            | Char+Edge           | 3x    | **Falsification test**         |
| SR-002      | RepTNSR v2 (35K)   | Ultra (24ch/6blk)   | 12 scenes              | Full 5-loss         | 3x    | Primary SR result              |
| SR-003      | RepTNSR v2 (11K)   | Perf (16ch/4blk)    | 12 scenes              | Full 5-loss         | 3x    | Latency-constrained            |
| SR-004      | RepTNSR v2 (17K)   | Quality (20ch/4blk) | 12 scenes              | Full 5-loss         | 3x    | Balanced tier                  |
| SR-A1a      | RepTNSR v2         | 16ch/6blk           | 12 scenes              | Full                | 3x    | Ablation: width                |
| SR-A1b      | RepTNSR v2         | 32ch/6blk           | 12 scenes              | Full                | 3x    | Ablation: width                |
| SR-A2a      | RepTNSR v2         | 24ch/3blk           | 12 scenes              | Full                | 3x    | Ablation: depth                |
| SR-A2b      | RepTNSR v2         | 24ch/8blk           | 12 scenes              | Full                | 3x    | Ablation: depth                |
| SR-A3       | RepTNSR v2 no-edge | 24ch/6blk           | 12 scenes              | Full (no Sobel/Lap) | 3x    | Ablation: edge ops             |
| SR-A4       | RepTNSR v2 9ch     | 24ch/6blk, 9ch in   | 12 scenes              | Full                | 3x    | Ablation: inputs               |
| SR-A5a      | RepTNSR v2 ReLU    | 24ch/6blk           | 12 scenes              | Full                | 3x    | Ablation: activation           |
| SR-C1       | RepTNSR v2         | Ultra               | 12 scenes              | Char only           | 3x    | Ablation: loss baseline        |
| SR-C2       | RepTNSR v2         | Ultra               | 12 scenes              | Char+Edge           | 3x    | Ablation: +edge                |
| SR-C3       | RepTNSR v2         | Ultra               | 12 scenes              | Char+Edge+Perc      | 3x    | Ablation: +perc                |
| SR-C4       | RepTNSR v2         | Ultra               | 12 scenes              | Char+Edge+Perc+Temp | 3x    | Ablation: +temp                |
| SR-D3       | RepTNSR v2         | Ultra               | 12 scenes (bicubic LR) | Full                | 3x    | Ablation: LR generation method |
| SR-S1       | RepTNSR v2         | Ultra               | 12 scenes              | Full                | 2x    | Scale factor test              |
| SR-S2       | RepTNSR v2         | Ultra               | 12 scenes              | Full                | 1.5x  | Scale factor test              |
| **FG-001**  | NeuralFG (24K)     | Default             | 12 scenes              | Full 3-loss         | N/A   | Primary FG result              |
| FG-B1a      | NeuralFG           | 2x downsample       | 12 scenes              | Full                | N/A   | Ablation: resolution           |
| FG-B1b      | NeuralFG           | 8x downsample       | 12 scenes              | Full                | N/A   | Ablation: resolution           |
| FG-B2       | NeuralFG 2-level   | Multi-scale pyramid | 12 scenes              | Full                | N/A   | Ablation: refinement           |
| FG-B3       | NeuralFG no-MV     | No engine MVs       | 12 scenes              | Full                | N/A   | Ablation: MV contribution      |
| **GEN-001** | SR-002 + FG-001    | Full pipeline       | Held-out UE5           | N/A                 | 3x    | Generalisation                 |
| **GEN-002** | SR-002 + FG-001    | Full pipeline       | Unity HDRP             | N/A                 | 3x    | Cross-engine                   |
| **GEN-003** | SR-002 + FG-001    | Full pipeline       | Cartoon scenes         | N/A                 | 3x    | Art style                      |
| **LAT-001** | All tiers          | All GPUs            | Standard seq           | N/A                 | 3x    | Latency profiling              |
| **HUM-001** | SR-002 vs FSR 3.1  | 2AFC study          | 5 test scenes          | N/A                 | 3x    | Human evaluation               |

**Total: ~30 experiments.** Estimated compute: ~600 GPU-hours $\approx$ \$3,000 cloud (4x A100).

## 24. Assumptions, Risks, and Constraints

### 24.1 Assumptions

| #   | Assumption                                                                | Impact if Wrong                                   | Mitigation                                                  |
| :-- | :------------------------------------------------------------------------ | :------------------------------------------------ | :---------------------------------------------------------- |
| A1  | FSR 3.1 Quality mode achieves ~33.5 dB PSNR at 3x scale.                  | Target metrics may need recalibration.            | Run SR-001 falsification experiment first.                  |
| A2  | 35K parameters is sufficient for competitive quality vs FSR 3.1 temporal. | Need to increase to 50K to 100K params.           | Quality tiers already provide scaling path.                 |
| A3  | Engine-provided motion vectors are accurate for geometric motion.         | FG quality degrades on animated meshes.           | Fall back to optical flow for non-MV regions.               |
| A4  | DirectML compiled operator performance matches hand-coded HLSL.           | May need full HLSL compute shader implementation. | HLSL path already designed as "advanced" deployment option. |
| A5  | 12 UE5 scenes provide sufficient data diversity.                          | Generalisation failure on held-out data.          | Expand to 20+ scenes; add Sintel/TartanAir/VIPER.           |
| A6  | FP16 precision is sufficient for all intermediate computations.           | NaN or quality regression in warp coordinates.    | Mixed FP16/FP32: FP32 for warp grid_sample coordinates.     |

### 24.2 Unavailable/Proprietary Dependencies

| Dependency                                    | Status                                       | Impact                                                | Mitigation                                                                 |
| :-------------------------------------------- | :------------------------------------------- | :---------------------------------------------------- | :------------------------------------------------------------------------- |
| DLSS 3 network weights                        | Proprietary (NVIDIA)                         | Cannot reproduce or directly compare architecturally. | Compare quality only via game captures (not architecture).                 |
| DLSS 3 frame generation                       | Requires RTX 40 series OFA hardware.         | Cannot benchmark on non-RTX 40 hardware.              | Compare via published benchmarks and captured frames.                      |
| XeSS network weights                          | Proprietary (Intel SDK open, models closed). | Cannot retrain or inspect.                            | Use SDK binary for baseline comparison on supported HW.                    |
| Game-specific captures from commercial titles | Requires game licences.                      | Cannot distribute test data.                          | Use open UE5 scenes + Sintel for all reproducible benchmarks.              |
| Unreal Engine 5                               | Free for development (royalty-based EULA).   | Engine itself is not distributable.                   | Capture data IS distributable. Provide capture scripts, not engine builds. |

### 24.3 Research Risks

| Risk                                                  | Severity | Likelihood              | Mitigation                                                                                       |
| :---------------------------------------------------- | :------- | :---------------------- | :----------------------------------------------------------------------------------------------- |
| Quality gap insufficient vs FSR 3.1 temporal upscaler | High     | Medium                  | Increase network capacity; add lightweight attention at bottleneck layer; expand input channels. |
| Latency exceeds budget on GTX 1650 class              | High     | Medium (for Ultra tier) | Provide Performance/Quality tiers; optimise HLSL compute shaders.                                |
| Temporal flickering worse than FSR 3.1                | High     | Low                     | Strengthen temporal loss weight; add anti-flicker post-filter; tighten variance clamping.        |
| FG disocclusion quality worse than FSR 3.1            | Medium   | Medium                  | Train with diverse disocclusion scenarios; add inpainting refinement head; multi-scale pyramid.  |
| Poor generalisation to unseen content                 | Medium   | Medium                  | Increase training data diversity; add domain randomisation augmentation.                         |
| ONNX/DirectML export quality regression               | Low      | Low                     | Exhaustive parity testing; bit-level comparison; mixed precision profiling.                      |
