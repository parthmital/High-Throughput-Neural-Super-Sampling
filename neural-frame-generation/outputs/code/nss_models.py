"""Models: NeuralSS temporal super-resolution (TSR) and low-latency frame generation (FG).

Motion comes from an intermediate-flow network (IFNet) with the design of RIFE / Practical-RIFE 4.26 (MIT): a
coarse-to-fine chain of flow blocks that estimates motion from the colour frames alone, so both tasks work on
ordinary video and on games that expose no engine data. Engine motion vectors, depth and the camera are optional
extra input channels whose weights start at zero, so a model initialised from the pre-trained RIFE weights starts
as exactly RIFE and learns during fine-tuning how much to trust each engine signal.

The TSR reconstruction trunk is built from RepConv blocks: during training each block is a sum of linear branches
(3x3, 1x1, identity and fixed Sobel-x, Sobel-y, Laplacian filters with a learnable 1x1 projection, after ECBSR).
The branches are folded into one 3x3 kernel on every forward pass ("online re-parameterisation"), so training
costs one convolution per block and the deployed network is a plain chain of 3x3 convolutions. Every operator
(convolution, transposed convolution, pixel (un)shuffle, bilinear resize, grid sampling) maps to any GPU API
(DirectML, Vulkan / compute shaders, Metal, ONNX Runtime).
"""

import copy
import hashlib
import io
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
LAPLACIAN = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]


class RepConv(nn.Module):
    """3x3 + 1x1 + identity + fixed edge filters, folded into one 3x3 convolution on every call.

    The 1x1 and edge-projection weights start at zero (they still train): with PyTorch's default initialisation
    every branch adds gain, activations grow about 1.9x per block, and the deep teacher tiers overflow FP16
    before the first step (the first TSR teacher run was NaN from step 10). At zero, a block starts as
    3x3 + identity and activations stay of order 1 at any depth."""

    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.conv3 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv1 = nn.Conv2d(cin, cout, 1)
        self.edge_pw = nn.Conv2d(3 * cin, cout, 1)
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.edge_pw.weight)
        self.register_buffer(
            "edges", torch.tensor([SOBEL_X, SOBEL_Y, LAPLACIAN]), persistent=False
        )
        self.act = nn.PReLU(cout) if act else nn.Identity()

    def weight_bias(self):
        w = self.conv3.weight + F.pad(self.conv1.weight, [1, 1, 1, 1])
        pw = self.edge_pw.weight[:, :, 0, 0].reshape(self.cout, self.cin, 3)
        w = w + torch.einsum("oik,khw->oihw", pw, self.edges.to(pw.dtype))
        if self.cin == self.cout:
            w = w + torch.eye(self.cin, device=w.device, dtype=w.dtype)[
                :, :, None, None
            ] * F.pad(
                torch.ones(1, 1, 1, 1, device=w.device, dtype=w.dtype), [1, 1, 1, 1]
            )
        return w, self.conv3.bias + self.conv1.bias + self.edge_pw.bias

    def forward(self, x):
        w, b = self.weight_bias()
        return self.act(F.conv2d(x, w, b, padding=1))

    @torch.no_grad()
    def fused(self):
        w, b = self.weight_bias()
        conv = nn.Conv2d(self.cin, self.cout, 3, padding=1).to(w.device, w.dtype)
        conv.weight.copy_(w)
        conv.bias.copy_(b)
        return nn.Sequential(conv, copy.deepcopy(self.act))


def fuse_model(model):
    """Deep copy with every RepConv replaced by its fused Conv2d (+ activation)."""
    fused = copy.deepcopy(model).eval()

    def swap(module):
        for name, child in module.named_children():
            if isinstance(child, RepConv):
                setattr(module, name, child.fused())
            else:
                swap(child)

    swap(fused)
    return fused


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def backward_warp(img, flow):
    """Sample img at (x + flow) for every output pixel; flow (N, 2, H, W) in pixels of img. Written exactly
    as RIFE's warp (align_corners=True grid), so pre-trained RIFE weights reproduce RIFE's output bit for bit.
    """
    n, _, h, w = img.shape
    # arange with float scales stays float32 in traced ONNX graphs (linspace is exported in float64)
    sx, sy = 2.0 / (w - 1.0), 2.0 / (h - 1.0)
    f32 = dict(device=img.device, dtype=torch.float32)
    gx = (torch.arange(w, **f32) * sx - 1.0).view(1, w)
    gy = (torch.arange(h, **f32) * sy - 1.0).view(h, 1)
    grid = torch.stack(
        [gx[None] + flow[:, 0].float() * sx, gy[None] + flow[:, 1].float() * sy], -1
    )
    return F.grid_sample(
        img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).to(img.dtype)


def upsample_flow(flow, scale):
    return (
        F.interpolate(
            flow.float(), scale_factor=scale, mode="bilinear", align_corners=False
        )
        * scale
    )


def jitter_upsample(color, jitter, scale, mode="bilinear"):
    """Reconstruction at output pixel centres that undoes the sub-pixel jitter of the low-resolution samples:
    color (N, 3, h, w), jitter (N, 2) in low-resolution pixels; mode is a grid_sample mode (bilinear, bicubic).
    """
    n, _, h, w = color.shape
    H, W = h * scale, w * scale
    ys = (torch.arange(H, device=color.device, dtype=torch.float32) + 0.5) / scale
    xs = (torch.arange(W, device=color.device, dtype=torch.float32) + 0.5) / scale
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    jx, jy = jitter[:, 0, None, None].float(), jitter[:, 1, None, None].float()
    grid = torch.stack(
        [(gx[None] - jx) * (2.0 / w) - 1.0, (gy[None] - jy) * (2.0 / h) - 1.0], -1
    )
    return F.grid_sample(
        color.float(), grid, mode=mode, padding_mode="border", align_corners=False
    )


def pad_to(x, multiple):
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    return (F.pad(x, (0, pw, 0, ph), mode="replicate") if ph or pw else x), (h, w)


class Trunk(nn.Module):
    """Plain RepConv chain with an optional half-resolution U-Net level (strided conv down, pixel-shuffle up)."""

    def __init__(self, cin, width, blocks, unet_blocks):
        super().__init__()
        self.head = RepConv(cin, width)
        n_a = blocks // 2
        self.a = nn.Sequential(*[RepConv(width, width) for _ in range(n_a)])
        self.unet = unet_blocks > 0
        if self.unet:
            self.down = nn.Sequential(
                nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.PReLU(2 * width)
            )
            self.mid = nn.Sequential(
                *[RepConv(2 * width, 2 * width) for _ in range(unet_blocks)]
            )
            self.up = nn.Sequential(
                nn.Conv2d(2 * width, 4 * width, 1), nn.PixelShuffle(2)
            )
        self.b = nn.Sequential(*[RepConv(width, width) for _ in range(blocks - n_a)])

    def forward(self, x):
        x = self.a(self.head(x))
        if self.unet:
            x = x + self.up(self.mid(self.down(x)))
        return self.b(x)


def count_macs(model, run):
    """Multiply-accumulates of every Conv2d and ConvTranspose2d executed by run(model), per batch sample,
    counted with forward hooks."""
    total = [0]

    def hook(mod, inp, out):
        k = mod.kernel_size[0] * mod.kernel_size[1] // mod.groups
        if isinstance(
            mod, nn.ConvTranspose2d
        ):  # every input pixel scatters a k x k x cout kernel
            total[0] += inp[0][0].numel() * mod.out_channels * k
        else:
            total[0] += out[0].numel() * mod.in_channels * k

    hooks = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    ]
    try:
        with torch.no_grad():
            run(model)
    finally:
        for hk in hooks:
            hk.remove()
    return total[0]


# ---------------------------------------------------------------------------------------------------------------
# Motion: intermediate-flow network (IFNet), the design of RIFE / Practical-RIFE 4.26 (MIT, Megvii / hzwer)
# ---------------------------------------------------------------------------------------------------------------
# Pre-trained RIFE 4.26 weights from the author's Hugging Face repository; both hashes were checked on
# 3 November 2026, and the checkpoint is a plain state dict, so it loads with torch.load(weights_only=True).
RIFE_URL = "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip"
RIFE_ZIP_SHA256 = "1fa9b9cda3d9b8c3e301359e2595960902f97bf926c08598b0e9957a3f3f760e"
RIFE_PKL_SHA256 = "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f"
RIFE_WIDTHS = (192, 128, 96, 64, 32)
RIFE_SCALES = (16, 8, 4, 2, 1)


def _conv_lrelu(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.LeakyReLU(0.2, True))


class IFHead(nn.Module):
    """Per-frame 4-channel feature encoder (RIFE's Head). It depends on one frame only, so in a running
    pipeline it is computed once per real frame and reused by every generated frame (the per-frame half of
    DLSS 4's split frame-generation network)."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """One flow level: at 1/scale of the input, two stride-2 convolutions, n_res residual convolutions and a
    transposed convolution + pixel shuffle back to 1/scale; outputs a flow update (target -> frame 0, target ->
    frame 1), a blend logit and 8 feature channels. n_hint extra input channels (engine data) are appended after
    the RIFE inputs, with zero weights at initialisation; the first n_hint_flow of them are flows in pixels and
    are rescaled with the resolution like the flow itself."""

    def __init__(self, cin, c, n_res, n_hint=0, n_hint_flow=0):
        super().__init__()
        self.cin, self.n_hint, self.n_hint_flow = cin, n_hint, n_hint_flow
        self.conv0 = nn.Sequential(
            _conv_lrelu(cin + n_hint, c // 2, 2), _conv_lrelu(c // 2, c, 2)
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(n_res)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2)
        )
        if n_hint:
            with torch.no_grad():
                self.conv0[0][0].weight[:, cin:].zero_()

    def forward(self, x, flow=None, hint=None, scale=1.0):
        down = lambda t: (  # noqa: E731
            t
            if scale == 1
            else F.interpolate(
                t, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )
        )
        x = down(x)
        if flow is not None:
            x = torch.cat((x, down(flow) / scale), 1)
        if hint is not None:
            h = down(hint)
            if self.n_hint_flow:
                h = torch.cat(
                    (h[:, : self.n_hint_flow] / scale, h[:, self.n_hint_flow :]), 1
                )
            x = torch.cat((x, h.to(x.dtype)), 1)
        tmp = self.lastconv(self.convblock(self.conv0(x)))
        if scale != 1:
            tmp = F.interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    """Coarse-to-fine intermediate-flow estimator between frames img0 and img1 for a target time t (a
    per-sample scalar; t = 0.5 interpolates, t = 1.5 extrapolates, t = 1 with img0 = previous frame gives the
    motion vector of the current frame). Block k runs at 1/scales[k] of the input. With widths RIFE_WIDTHS and
    n_res 8 the parameter names and shapes are those of RIFE 4.26's IFNet, so its weights load directly;
    narrower variants inherit leading channel slices (load_ifnet_weights). Returns the flows after every block
    (each (N, 4, H, W): target -> img0, target -> img1, in pixels), the final blend logit and features, and the
    two warped frames."""

    def __init__(self, widths=RIFE_WIDTHS, n_res=8, n_hint=0, n_hint_flow=0):
        super().__init__()
        self.encode = IFHead()
        self.n_blocks = len(widths)
        for k, c in enumerate(widths):
            cin = 7 + 8 if k == 0 else 8 + 4 + 8 + 8
            setattr(self, f"block{k}", IFBlock(cin, c, n_res, n_hint, n_hint_flow))

    def blocks(self):
        return [getattr(self, f"block{k}") for k in range(self.n_blocks)]

    def forward(self, img0, img1, timestep, scales, hint=None, feats=None):
        n, _, h, w = img0.shape
        t = timestep.view(n, 1, 1, 1).to(img0.dtype).expand(n, 1, h, w)
        f0, f1 = feats if feats is not None else (self.encode(img0), self.encode(img1))
        flow = mask = feat = None
        w0, w1, flows = img0, img1, []
        for blk, scale in zip(self.blocks(), scales):
            if flow is None:
                flow, mask, feat = blk(
                    torch.cat((img0, img1, f0, f1, t), 1), None, hint, scale
                )
            else:
                wf0 = backward_warp(f0, flow[:, :2])
                wf1 = backward_warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(
                    torch.cat((w0, w1, wf0, wf1, t, mask, feat), 1), flow, hint, scale
                )
                flow = flow + fd
            flows.append(flow)
            w0 = backward_warp(img0, flow[:, :2])
            w1 = backward_warp(img1, flow[:, 2:4])
        return {"flows": flows, "mask": mask, "feat": feat, "w0": w0, "w1": w1}


def load_ifnet_weights(ifnet, state):
    """Copy pre-trained IFNet weights (a RIFE state dict, with or without the DDP 'module.' prefix) into
    ifnet. Matching names are copied whole; where ifnet is narrower (fewer channels or residual blocks) the
    leading slices are copied, a weight-inheritance initialisation; extra engine-hint input channels keep their
    zero weights. Returns counts of exact, sliced and missing tensors."""
    src = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
        if torch.is_tensor(v)
    }
    report = {"exact": 0, "sliced": 0, "missing": 0}
    with torch.no_grad():
        for k, p in ifnet.state_dict().items():  # shares storage with the parameters
            s = src.get(k)
            if s is None or s.dim() != p.dim():
                report["missing"] += 1
                continue
            sl = tuple(slice(0, min(a, b)) for a, b in zip(p.shape, s.shape))
            p[sl].copy_(s[sl].to(p.dtype))
            report["exact" if p.shape == s.shape else "sliced"] += 1
    return report


def fetch_rife_weights(dest_dir, log=print):
    """Download RIFE 4.26 (once) into dest_dir, check both SHA-256 hashes and return its state dict, loaded
    with weights_only=True so the checkpoint cannot run code. Returns None (logged) when it is unreachable.
    """
    from pathlib import Path
    from urllib.request import urlopen

    path = Path(dest_dir) / "RIFEv4.26_0921.zip"
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(RIFE_URL, timeout=120) as r:
                path.write_bytes(r.read())
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != RIFE_ZIP_SHA256:
            raise ValueError("RIFE zip hash mismatch")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(
                m
                for m in z.namelist()
                if m.endswith("flownet.pkl") and "__MACOSX" not in m
            )
            pkl = z.read(name)
        if hashlib.sha256(pkl).hexdigest() != RIFE_PKL_SHA256:
            raise ValueError("RIFE flownet.pkl hash mismatch")
        return torch.load(io.BytesIO(pkl), map_location="cpu", weights_only=True)
    except Exception as exc:
        log(
            f"RIFE 4.26 weights unavailable ({exc!r}): models start from random weights"
        )
        return None


# ---------------------------------------------------------------------------------------------------------------
# Temporal super-resolution
# ---------------------------------------------------------------------------------------------------------------
# motion: the built-in IFNet that estimates low-resolution motion when the input has no motion vectors
# (ordinary video, colour-only captures); hr_refine: width of a two-layer convolution at output resolution
# that removes the pixel-shuffle pattern of the low-resolution head (0 = off).
TSR_TIERS = {
    "igpu": {
        "width": 16,
        "blocks": 3,
        "unet_blocks": 0,
        "hidden": 8,
        "motion": ((32, 24, 16), 2),
        "hr_refine": 0,
    },
    "low": {
        "width": 24,
        "blocks": 4,
        "unet_blocks": 2,
        "hidden": 8,
        "motion": ((48, 32, 24), 3),
        "hr_refine": 8,
    },
    "mid": {
        "width": 32,
        "blocks": 4,
        "unet_blocks": 3,
        "hidden": 16,
        "motion": ((64, 48, 32), 4),
        "hr_refine": 8,
    },
    "high": {
        "width": 48,
        "blocks": 6,
        "unet_blocks": 4,
        "hidden": 16,
        "motion": ((96, 64, 48), 4),
        "hr_refine": 12,
    },
    "teacher": {
        "width": 64,
        "blocks": 8,
        "unet_blocks": 6,
        "hidden": 32,
        "motion": ((128, 96, 64), 6),
        "hr_refine": 16,
    },
}
TSR_MOTION_SCALES = (4, 2, 1)
EST_MV_QUALITY = (
    0.25  # mv_quality channel value when the built-in estimator supplies the motion
)


def normals_from_depth(d):
    """View-space normal estimate from normalised depth (N, 1, h, w) by central differences."""
    dp = F.pad(d, (1, 1, 1, 1), mode="replicate")
    gx = (dp[..., 1:-1, 2:] - dp[..., 1:-1, :-2]) * 8.0
    gy = (dp[..., 2:, 1:-1] - dp[..., :-2, 1:-1]) * 8.0
    n = torch.cat([-gx, -gy, torch.ones_like(d)], 1)
    return n / n.norm(dim=1, keepdim=True)


class TSRNet(nn.Module):
    """Recurrent temporal super-resolution at low resolution.

    Per frame inputs (low resolution h x w): colour (jittered render samples, or a downscaled video frame),
    depth (+ normals derived from it), motion vectors (current -> previous, low-res pixels), a depth-based
    disocclusion cue, reactive mask, exposure, jitter offset, availability flags and the input kind (1 =
    jittered render, 0 = video); plus the previous output warped to the current frame at full resolution and
    folded to low resolution (space-to-depth), and a warped recurrent hidden state.

    Motion: samples with mv_quality > 0 use the given motion vectors (exact engine vectors, or estimated ones);
    the others use the built-in IFNet estimate from the previous and current colour frames, so the network runs
    on ordinary video with no side information.

    Output: full-resolution colour = a * warped_history + (1 - a) * jitter-aware upsample + residual, where the
    per-pixel history weight a and the residual are predicted at low resolution and unfolded (pixel shuffle),
    followed by an optional small refinement at output resolution.
    """

    def __init__(
        self, scale, width, blocks, unet_blocks, hidden, motion=None, hr_refine=0
    ):
        super().__init__()
        self.scale, self.hidden = scale, hidden
        s2 = scale * scale
        cin = 3 + 1 + 3 + 2 + 1 + 1 + 1 + 2 + 2 + 1 + 3 * s2 + 3 + hidden
        self.trunk = Trunk(cin, width, blocks, unet_blocks)
        self.out = nn.Conv2d(width, 4 * s2, 3, padding=1)
        self.to_hidden = nn.Conv2d(width, hidden, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.motion = None
        if (
            motion is not None
        ):  # hints: jitter difference to the previous frame (2), input kind (1)
            widths, n_res = motion
            self.motion = IFNet(widths, n_res, n_hint=3)
        self.refine = None
        if hr_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(6, hr_refine, 3, padding=1),
                nn.PReLU(hr_refine),
                nn.Conv2d(hr_refine, 3, 3, padding=1),
            )
            nn.init.zeros_(self.refine[2].weight)
            nn.init.zeros_(self.refine[2].bias)

    def base_upsample(self, color, jitter):
        """Bilinear reconstruction at output pixel centres that undoes the sub-pixel jitter of the samples."""
        return jitter_upsample(color, jitter, self.scale)

    def estimate_motion(self, prev_color, color, jitter_delta, kind):
        """Motion vectors (current -> previous, low-resolution pixels) from two colour frames."""
        n = color.shape[0]
        x0, (h0, w0) = pad_to(prev_color, 4 * TSR_MOTION_SCALES[0])
        x1 = pad_to(color, 4 * TSR_MOTION_SCALES[0])[0]
        hp, wp = x0.shape[-2:]
        hint = torch.cat(
            [
                jitter_delta.view(n, 2, 1, 1).expand(n, 2, hp, wp),
                kind.view(n, 1, 1, 1).expand(n, 1, hp, wp),
            ],
            1,
        ).to(x0.dtype)
        r = self.motion(
            x0, x1, torch.ones(n, device=color.device), TSR_MOTION_SCALES, hint=hint
        )
        return r["flows"][-1][:, :2, :h0, :w0].float()

    def forward(self, f, state=None, use_history=True):
        """f: dict with color (N,3,h,w), depth (N,1,h,w), mv (N,2,h,w), reactive (N,1,h,w), exposure (N,),
        jitter (N,2), has_depth (N,), mv_quality (N,), kind (N,; optional, default 1). Training only:
        mv_label (N,2,h,w) and mv_force (N,) bool, which warps history of forced samples with the label motion
        instead of the estimate while the estimator is still poor (teacher forcing). Returns (output (N,3,H,W),
        new state); the state holds the estimated motion ("mv_est", None without an estimate).
        """
        color, depth, mv, jitter = f["color"], f["depth"], f["mv"], f["jitter"]
        n, _, h, w = color.shape
        s = self.scale
        kind = f.get("kind")
        kind = color.new_ones(n) if kind is None else kind
        quality = f["mv_quality"].float()
        base = self.base_upsample(color, jitter)
        mv_est = None
        if state is None or not use_history:
            hist, hidden, d_prev, first = (
                base,
                color.new_zeros(n, self.hidden, h, w),
                depth,
                True,
            )
        else:
            # training estimates motion for every sample (all feed the motion loss); inference only when a
            # sample has no motion vectors
            if self.motion is not None and (
                self.training or bool((quality <= 0).any())
            ):
                mv_est = self.estimate_motion(
                    state["color"], color, jitter - state["jitter"], kind
                )
                est = (quality <= 0).view(n, 1, 1, 1)
                used = mv_est.to(mv.dtype)
                if "mv_force" in f:
                    force = f["mv_force"].view(n, 1, 1, 1)
                    used = torch.where(force, f["mv_label"].to(mv.dtype), used)
                mv = torch.where(est, used, mv)
                quality = torch.where(
                    quality <= 0, torch.full_like(quality, EST_MV_QUALITY), quality
                )
            mv_hr = upsample_flow(mv, s)
            hist = backward_warp(state["out"], mv_hr)
            hidden = backward_warp(state["hidden"], mv)
            d_prev, first = backward_warp(state["depth"], mv), False
        has_d = f["has_depth"].view(n, 1, 1, 1).to(color.dtype)
        disocc = ((depth - d_prev).abs() / (depth.abs() + 1e-3)).clamp(0, 1) * has_d
        hist_lr = F.avg_pool2d(hist, s)
        feats = [
            color,
            depth * has_d,
            normals_from_depth(depth) * has_d,
            mv * 0.125,
            disocc,
            f["reactive"],
            f["exposure"].view(n, 1, 1, 1).expand(n, 1, h, w).to(color.dtype),
            jitter.view(n, 2, 1, 1).expand(n, 2, h, w).to(color.dtype),
            has_d.expand(n, 1, h, w),
            quality.view(n, 1, 1, 1).expand(n, 1, h, w).to(color.dtype),
            kind.view(n, 1, 1, 1).expand(n, 1, h, w).to(color.dtype),
            F.pixel_unshuffle(hist.to(color.dtype), s) * (0.0 if first else 1.0),
            (hist_lr.to(color.dtype) - color).abs() * (0.0 if first else 1.0),
            hidden.to(color.dtype),
        ]
        x, (h0, w0) = pad_to(torch.cat(feats, 1), 2)
        y = self.trunk(x)
        o = self.out(y)[..., :h0, :w0]
        new_hidden = torch.tanh(self.to_hidden(y))[..., :h0, :w0]
        o = self.shuffle(o.float())
        residual, a = o[:, :3], torch.sigmoid(o[:, 3:4] + 2.0)
        if first:
            a = torch.zeros_like(a)
        out = a * hist + (1 - a) * base + residual
        if self.refine is not None:
            out = out + self.refine(torch.cat([out, base], 1).to(color.dtype)).float()
        return out, {
            "out": out,
            "hidden": new_hidden,
            "depth": depth,
            "color": color,
            "jitter": jitter,
            "first": False,
            "alpha": a,
            "mv_est": mv_est,
        }


def build_tsr(tier, scale, motion_net=True):
    spec = dict(TSR_TIERS[tier])
    if not motion_net:
        spec["motion"] = None
    return TSRNet(scale, **spec)


def tsr_inputs(n, h, w, dev="cpu"):
    """Random single-frame TSR inputs without motion vectors (so the built-in motion estimator runs)."""
    return {
        "color": torch.rand(n, 3, h, w, device=dev),
        "depth": torch.rand(n, 1, h, w, device=dev),
        "mv": torch.zeros(n, 2, h, w, device=dev),
        "reactive": torch.zeros(n, 1, h, w, device=dev),
        "exposure": torch.zeros(n, device=dev),
        "jitter": torch.zeros(n, 2, device=dev),
        "has_depth": torch.ones(n, device=dev),
        "mv_quality": torch.zeros(n, device=dev),
        "kind": torch.ones(n, device=dev),
    }


def tsr_macs_per_lr_pixel(model, sample_hw=(64, 64)):
    """Multiply-accumulates per low-resolution pixel of one steady-state frame (with history and, when the
    model has one, the built-in motion estimator), counted with forward hooks on a fused copy.
    """
    fused = fuse_model(model).float().cpu()
    h, w = sample_hw
    f = tsr_inputs(1, h, w)
    with torch.no_grad():
        _, state = fused(f, None)
    macs = count_macs(fused, lambda m: m(f, state))
    return macs / (h * w)


# ---------------------------------------------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------------------------------------------
# widths / n_res: the IFNet flow blocks (the high tier is RIFE 4.26's network); refine: width of the half-resolution
# head that predicts the residual and the UI mask; flow_lines: the most image lines the flow estimation runs at.
# Taller frames are halved until they fit (RIFE's scale argument, chosen per resolution as RIFE recommends; 0.5
# is about a quarter of the flow cost), so a 1080p frame runs the igpu and low tiers' flow at 540 lines, while
# 256-pixel training crops and video up to 576 lines keep full resolution. The run of 2 to 3 November 2026 used a fixed
# scale of 0.5 for igpu and low, so they trained with flow at 128 pixels and ended no better than frame
# averaging on Middlebury (28.4 dB against 28.0; mid 33.7, high 34.3).
FG_TIERS = {
    "igpu": {
        "widths": (64, 48, 32, 24, 16),
        "n_res": 3,
        "refine": 12,
        "flow_lines": 576,
    },
    "low": {
        "widths": (96, 64, 48, 32, 16),
        "n_res": 4,
        "refine": 16,
        "flow_lines": 576,
    },
    "mid": {
        "widths": (128, 96, 64, 48, 24),
        "n_res": 6,
        "refine": 24,
        "flow_lines": 1152,
    },
    "high": {"widths": RIFE_WIDTHS, "n_res": 8, "refine": 32, "flow_lines": 1152},
}
FG_TIERS["teacher"] = FG_TIERS[
    "high"
]  # the fine-tuned RIFE-width model teaches the smaller tiers
FG_HINT_FLOW = 6  # tau * M1, (tau - 1) * M1, camera prior (target -> I1)
FG_HINT = FG_HINT_FLOW + 4  # + mv_avail, depth, depth_avail, cam_avail


class FGRefine(nn.Module):
    """Half-resolution head on the warped frames, blend weight and IFNet features: a colour residual and the
    UI / static-overlay logit. The last layer starts at zero (UI logit at -4), so a fresh model outputs RIFE's
    blend."""

    def __init__(self, c):
        super().__init__()
        self.down = _conv_lrelu(3 * 4 + 1 + 8, c, 2)
        self.body = nn.Sequential(ResConv(c), ResConv(c))
        self.up = nn.ConvTranspose2d(c, 4, 4, 2, 1)
        nn.init.zeros_(self.up.weight)
        with torch.no_grad():
            self.up.bias.copy_(torch.tensor([0.0, 0.0, 0.0, -4.0]))

    def forward(self, x):
        return self.up(self.body(self.down(x)))


class FGNet(nn.Module):
    """Frame generator for interpolation (0 < tau < 1, any tau, so 2x to 6x multi-frame generation) and
    extrapolation (tau > 1, holds no real frame) from real frames I0 (t = 0) and I1 (t = 1).

    Motion is estimated by an IFNet from the colour frames alone, so the generator works on ordinary video.
    Engine data are optional hint channels: the motion vector of I1 (I1 -> I0) as linear-motion priors
    F(tau->0) = tau * M1 and F(tau->1) = (tau - 1) * M1, the camera-only flow target -> I1 from the latest input
    (late latching, for extrapolation), depth of I1, and availability flags. The output is
        out = u * I1 + (1 - u) * (m * warp(I0) + (1 - m) * warp(I1) + residual)
    so HUD, subtitles and menus are taken from the newest real frame instead of being interpolated.
    """

    def __init__(self, widths, n_res, refine, flow_lines=1152):
        super().__init__()
        self.flow_lines = flow_lines
        self.scales = RIFE_SCALES[-len(widths) :]
        self.ifnet = IFNet(widths, n_res, n_hint=FG_HINT, n_hint_flow=FG_HINT_FLOW)
        self.refine = FGRefine(refine)

    def flow_scale(self, height):
        """Resolution of the flow estimation relative to a frame of the given height: 1, 1/2, 1/4, ..."""
        s = 1.0
        while height * s > self.flow_lines:
            s *= 0.5
        return s

    @staticmethod
    def priors(f):
        n = f["i0"].shape[0]
        tau = f["tau"].view(n, 1, 1, 1)
        m1 = f["mv1"] * f["mv_avail"].view(n, 1, 1, 1)
        return tau * m1, (tau - 1.0) * m1

    def hints(self, f):
        n = f["i0"].shape[0]
        av = lambda k: f[k].view(n, 1, 1, 1).to(f["i0"].dtype)  # noqa: E731
        p0, p1 = self.priors(f)
        h, w = f["i0"].shape[-2:]
        return torch.cat(
            [
                p0,
                p1,
                f["cam_prior"] * av("cam_avail"),
                av("mv_avail").expand(n, 1, h, w),
                f["depth1"] * av("depth_avail"),
                av("depth_avail").expand(n, 1, h, w),
                av("cam_avail").expand(n, 1, h, w),
            ],
            1,
        )

    def forward(self, f):
        i0, i1 = f["i0"], f["i1"]
        s = self.flow_scale(i0.shape[-2])
        mult = int(4 * self.scales[0] / s)
        i0p, (H0, W0) = pad_to(i0, mult)
        i1p = pad_to(i1, mult)[0]
        hint = pad_to(self.hints(f), mult)[0]
        if s == 1:
            r = self.ifnet(i0p, i1p, f["tau"], self.scales, hint=hint)
            flow, mask, feat = r["flows"][-1], r["mask"], r["feat"]
            w0, w1 = r["w0"], r["w1"]
        else:  # the whole flow network runs on frames downscaled by s; flows are upsampled for the warp
            down = lambda t: F.interpolate(  # noqa: E731
                t, scale_factor=s, mode="bilinear", align_corners=False
            )
            up = lambda t: F.interpolate(  # noqa: E731
                t, scale_factor=1.0 / s, mode="bilinear", align_corners=False
            )
            hs = down(hint)
            hs = torch.cat([hs[:, :FG_HINT_FLOW] * s, hs[:, FG_HINT_FLOW:]], 1)
            r = self.ifnet(down(i0p), down(i1p), f["tau"], self.scales, hint=hs)
            flow = upsample_flow(r["flows"][-1], 1.0 / s).to(i0p.dtype)
            mask, feat = up(r["mask"]), up(r["feat"])
            w0, w1 = backward_warp(i0p, flow[:, :2]), backward_warp(i1p, flow[:, 2:4])
        m = torch.sigmoid(mask.float())
        y = self.refine(torch.cat([i0p, i1p, w0, w1, m.to(i0p.dtype), feat], 1)).float()
        w0, w1 = w0.float(), w1.float()
        gen = m * w0 + (1 - m) * w1 + y[:, :3]
        u = torch.sigmoid(y[:, 3:4])
        out = u * i1p.float() + (1 - u) * gen
        crop = (..., slice(0, H0), slice(0, W0))
        return out[crop], {
            "flow0": flow[:, :2].float()[crop],
            "flow1": flow[:, 2:4].float()[crop],
            "blend": m[crop],
            "ui": u[crop],
            "ui_logit": y[:, 3:4][crop],
            "gen": gen[crop],
            "merged": (m * w0 + (1 - m) * w1)[crop],
        }


def build_fg(tier):
    return FGNet(**FG_TIERS[tier])


def fg_inputs(n, h, w, dev="cpu", tau=1.5):
    """Random frame-generation inputs with every engine signal available."""
    return {
        "i0": torch.rand(n, 3, h, w, device=dev),
        "i1": torch.rand(n, 3, h, w, device=dev),
        "mv1": torch.randn(n, 2, h, w, device=dev),
        "depth1": torch.rand(n, 1, h, w, device=dev),
        "cam_prior": torch.randn(n, 2, h, w, device=dev),
        "tau": torch.full((n,), tau, device=dev),
        "extrap": torch.full((n,), float(tau > 1), device=dev),
        "mv_avail": torch.ones(n, device=dev),
        "cam_avail": torch.ones(n, device=dev),
        "depth_avail": torch.ones(n, device=dev),
        "target": torch.rand(n, 3, h, w, device=dev),
    }


class TSRExport(nn.Module):
    """ONNX wrapper of one fused TSR step with explicit recurrent state tensors. flags: has_depth, mv_quality
    (0 = no motion vectors: the built-in estimator runs), kind (1 = jittered render, 0 = video). Returns the
    output and the new hidden state; the caller keeps this frame's colour, depth and jitter as the next
    frame's prev_* inputs (pass-through outputs would rename the graph inputs)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(
        self,
        color,
        depth,
        mv,
        reactive,
        exposure,
        jitter,
        flags,
        prev_out,
        prev_hidden,
        prev_depth,
        prev_color,
        prev_jitter,
    ):
        f = {
            "color": color,
            "depth": depth,
            "mv": mv,
            "reactive": reactive,
            "exposure": exposure,
            "jitter": jitter,
            "has_depth": flags[:, 0],
            "mv_quality": flags[:, 1],
            "kind": flags[:, 2],
        }
        out, st = self.net(
            f,
            {
                "out": prev_out,
                "hidden": prev_hidden,
                "depth": prev_depth,
                "color": prev_color,
                "jitter": prev_jitter,
                "first": False,
            },
        )
        return out, st["hidden"]


class FGExport(nn.Module):
    """ONNX wrapper of the fused frame generator."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, i0, i1, mv1, depth1, cam_prior, tau, avail):
        f = {
            "i0": i0,
            "i1": i1,
            "mv1": mv1,
            "depth1": depth1,
            "cam_prior": cam_prior,
            "tau": tau,
            "mv_avail": avail[:, 0],
            "cam_avail": avail[:, 1],
            "depth_avail": avail[:, 2],
        }
        return self.net(f)[0]


def export_onnx(wrapper, args, path, input_names, output_names, opset=17):
    """TorchScript-based ONNX export with static shapes on any PyTorch 2.x. From 2.5 export() takes a dynamo flag
    (later the default, which needs onnxscript); older versions do not accept it."""
    import inspect
    import warnings

    extra = (
        {"dynamo": False}
        if "dynamo" in inspect.signature(torch.onnx.export).parameters
        else {}
    )
    # static shapes make pad_to's size test a constant, which the tracer and constant folding warn about
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", message="Constant folding")
        torch.onnx.export(
            wrapper,
            args,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            **extra,
        )
