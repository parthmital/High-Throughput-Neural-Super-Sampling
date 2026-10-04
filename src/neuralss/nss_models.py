"""Models: NeuralSS temporal super-resolution (TSR) and low-latency frame generation (FG).

Both are built from RepConv blocks: during training each block is a sum of linear branches (3x3, 1x1, identity
and fixed Sobel-x, Sobel-y, Laplacian filters with a learnable 1x1 projection, after ECBSR). The branches are
folded into one 3x3 kernel on every forward pass ("online re-parameterisation"), so training costs one
convolution per block and the deployed network is a plain chain of 3x3 convolutions, which maps to any GPU API
(DirectML, Vulkan / compute shaders, Metal, ONNX Runtime).
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
LAPLACIAN = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]


class RepConv(nn.Module):
    """3x3 + 1x1 + identity + fixed edge filters, folded into one 3x3 convolution on every call."""

    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.cin, self.cout = cin, cout
        self.conv3 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv1 = nn.Conv2d(cin, cout, 1)
        self.edge_pw = nn.Conv2d(3 * cin, cout, 1)
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
    """Sample img at (x + flow) for every output pixel; flow (N, 2, H, W) in pixels of img."""
    n, _, h, w = img.shape
    ys = torch.arange(h, device=img.device, dtype=torch.float32)
    xs = torch.arange(w, device=img.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    fx, fy = flow[:, 0].float(), flow[:, 1].float()
    grid = torch.stack(
        [
            (gx[None] + fx + 0.5) * (2.0 / w) - 1.0,
            (gy[None] + fy + 0.5) * (2.0 / h) - 1.0,
        ],
        -1,
    )
    return F.grid_sample(
        img.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).to(img.dtype)


def upsample_flow(flow, scale):
    return (
        F.interpolate(
            flow.float(), scale_factor=scale, mode="bilinear", align_corners=False
        )
        * scale
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


# ---------------------------------------------------------------------------------------------------------------
# Temporal super-resolution
# ---------------------------------------------------------------------------------------------------------------
TSR_TIERS = {
    "igpu": {"width": 16, "blocks": 3, "unet_blocks": 0, "hidden": 8},
    "low": {"width": 24, "blocks": 4, "unet_blocks": 2, "hidden": 8},
    "mid": {"width": 32, "blocks": 4, "unet_blocks": 3, "hidden": 16},
    "high": {"width": 48, "blocks": 6, "unet_blocks": 4, "hidden": 16},
    "teacher": {"width": 64, "blocks": 8, "unet_blocks": 6, "hidden": 32},
}


def normals_from_depth(d):
    """View-space normal estimate from normalised depth (N, 1, h, w) by central differences."""
    dp = F.pad(d, (1, 1, 1, 1), mode="replicate")
    gx = (dp[..., 1:-1, 2:] - dp[..., 1:-1, :-2]) * 8.0
    gy = (dp[..., 2:, 1:-1] - dp[..., :-2, 1:-1]) * 8.0
    n = torch.cat([-gx, -gy, torch.ones_like(d)], 1)
    return n / n.norm(dim=1, keepdim=True)


class TSRNet(nn.Module):
    """Recurrent temporal super-resolution at low resolution.

    Per frame inputs (low resolution h x w): jittered colour, depth (+ normals derived from it), motion vectors
    (current -> previous, low-res pixels), a depth-based disocclusion cue, reactive mask, exposure, jitter offset
    and availability flags; plus the previous output warped to the current frame at full resolution and folded to
    low resolution (space-to-depth), and a warped recurrent hidden state.

    Output: full-resolution colour = a * warped_history + (1 - a) * jitter-aware upsample + residual, where the
    per-pixel history weight a and the residual are predicted at low resolution and unfolded (pixel shuffle).
    """

    def __init__(self, scale, width, blocks, unet_blocks, hidden):
        super().__init__()
        self.scale, self.hidden = scale, hidden
        s2 = scale * scale
        cin = 3 + 1 + 3 + 2 + 1 + 1 + 1 + 2 + 2 + 3 * s2 + 3 + hidden
        self.trunk = Trunk(cin, width, blocks, unet_blocks)
        self.out = nn.Conv2d(width, 4 * s2, 3, padding=1)
        self.to_hidden = nn.Conv2d(width, hidden, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def base_upsample(self, color, jitter):
        """Bilinear reconstruction at output pixel centres that undoes the sub-pixel jitter of the samples."""
        n, _, h, w = color.shape
        s = self.scale
        H, W = h * s, w * s
        ys = (torch.arange(H, device=color.device, dtype=torch.float32) + 0.5) / s
        xs = (torch.arange(W, device=color.device, dtype=torch.float32) + 0.5) / s
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        jx, jy = jitter[:, 0, None, None].float(), jitter[:, 1, None, None].float()
        grid = torch.stack(
            [(gx[None] - jx) * (2.0 / w) - 1.0, (gy[None] - jy) * (2.0 / h) - 1.0], -1
        )
        return F.grid_sample(
            color.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def forward(self, f, state=None, use_history=True):
        """f: dict with color (N,3,h,w), depth (N,1,h,w), mv (N,2,h,w), reactive (N,1,h,w), exposure (N,),
        jitter (N,2), has_depth (N,), mv_quality (N,). Returns (output (N,3,H,W), new state).
        """
        color, depth, mv, jitter = f["color"], f["depth"], f["mv"], f["jitter"]
        n, _, h, w = color.shape
        s = self.scale
        base = self.base_upsample(color, jitter)
        if state is None or not use_history:
            hist, hidden, d_prev, first = (
                base,
                color.new_zeros(n, self.hidden, h, w),
                depth,
                True,
            )
        else:
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
            f["mv_quality"].view(n, 1, 1, 1).expand(n, 1, h, w).to(color.dtype),
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
        return out, {
            "out": out,
            "hidden": new_hidden,
            "depth": depth,
            "first": False,
            "alpha": a,
        }


def build_tsr(tier, scale):
    return TSRNet(scale, **TSR_TIERS[tier])


def tsr_macs_per_lr_pixel(model, sample_hw=(64, 64)):
    """Multiply-accumulates per low-resolution pixel, counted with forward hooks on a fused copy."""
    fused = fuse_model(model).float().cpu()
    total = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total[0] += (
                out.numel()
                // out.shape[0]
                * mod.in_channels
                // mod.groups
                * mod.kernel_size[0]
                * mod.kernel_size[1]
            )

    hooks = [
        m.register_forward_hook(hook)
        for m in fused.modules()
        if isinstance(m, nn.Conv2d)
    ]
    h, w = sample_hw
    f = {
        "color": torch.rand(1, 3, h, w),
        "depth": torch.rand(1, 1, h, w),
        "mv": torch.zeros(1, 2, h, w),
        "reactive": torch.zeros(1, 1, h, w),
        "exposure": torch.zeros(1),
        "jitter": torch.zeros(1, 2),
        "has_depth": torch.ones(1),
        "mv_quality": torch.ones(1),
    }
    with torch.no_grad():
        fused(f, None)
    for hk in hooks:
        hk.remove()
    return total[0] / (h * w)


# ---------------------------------------------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------------------------------------------
FG_TIERS = {
    "igpu": {"width": 16, "levels": 1, "blocks": 3},
    "low": {"width": 24, "levels": 1, "blocks": 4},
    "mid": {"width": 32, "levels": 2, "blocks": 4},
    "high": {"width": 48, "levels": 2, "blocks": 5},
    "teacher": {"width": 64, "levels": 2, "blocks": 6},
}


class FGNet(nn.Module):
    """Coarse-to-fine intermediate-flow frame generator for interpolation (tau = 0.5) and extrapolation
    (tau = 1.5) from real frames I0 and I1.

    Motion priors: the engine motion vector of I1 (I1 -> I0) gives linear-motion flows from the target time,
    F(tau->0) = tau * M1 and F(tau->1) = (tau - 1) * M1; for extrapolation the camera-only flow of the latest input
    (target -> I1) is also given. Each level (1/4, then 1/2 resolution) predicts flow corrections, a blend
    weight, a residual and a UI/static-overlay probability. The output is
        out = u * I1 + (1 - u) * (m * warp(I0) + (1 - m) * warp(I1) + residual)
    so text, HUD and menus are taken from the newest real frame instead of being interpolated.
    The privileged teacher additionally sees the true target frame (training only)."""

    def __init__(self, width, levels, blocks, privileged=False):
        super().__init__()
        self.levels = [4, 2][:levels] if levels > 1 else [4]
        self.privileged = privileged
        cin = 3 * 4 + 2 * 3 + 1 + 2 + 3 + (3 if privileged else 0)
        self.nets = nn.ModuleList()
        for _ in self.levels:
            trunk = Trunk(cin + 2, width, blocks, 0)
            head = nn.Conv2d(width, 2 + 2 + 1 + 1 + 3, 3, padding=1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.bias[5] = -4.0  # UI probability starts near zero
            self.nets.append(nn.ModuleDict({"trunk": trunk, "head": head}))

    @staticmethod
    def priors(f):
        n = f["i0"].shape[0]
        tau = f["tau"].view(n, 1, 1, 1)
        m1 = f["mv1"] * f["mv_avail"].view(n, 1, 1, 1)
        f0, f1 = tau * m1, (tau - 1.0) * m1
        cam = f["cam_prior"] * f["cam_avail"].view(n, 1, 1, 1)
        use_cam = (f["cam_avail"] > 0) & (f["mv_avail"] <= 0)
        f1 = torch.where(use_cam.view(n, 1, 1, 1), cam, f1)
        return f0, f1

    def forward(self, f):
        i0, i1 = f["i0"], f["i1"]
        n, _, H, W = i0.shape
        i0p, (H0, W0) = pad_to(i0, 8)
        i1p = pad_to(i1, 8)[0]
        f0, f1 = (pad_to(x, 8)[0] for x in self.priors(f))
        cam = pad_to(f["cam_prior"] * f["cam_avail"].view(n, 1, 1, 1), 8)[0]
        depth = pad_to(f["depth1"] * f["depth_avail"].view(n, 1, 1, 1), 8)[0]
        gt = pad_to(f["target"], 8)[0] if self.privileged else None
        Hp, Wp = i0p.shape[-2:]
        blend = torch.zeros(n, 1, Hp, Wp, device=i0.device)
        ui = torch.full((n, 1, Hp, Wp), -4.0, device=i0.device)
        residual = torch.zeros(n, 3, Hp, Wp, device=i0.device)
        consts = torch.stack(
            [
                f["tau"] - 1.0,
                f["extrap"],
                f["mv_avail"],
                f["cam_avail"],
                f["depth_avail"],
            ],
            1,
        ).view(n, 5, 1, 1)
        for d, net in zip(self.levels, self.nets):
            down = lambda x: F.avg_pool2d(x.float(), d)  # noqa: E731
            w0, w1 = backward_warp(i0p, f0), backward_warp(i1p, f1)
            x = [
                down(i0p),
                down(i1p),
                down(w0),
                down(w1),
                down(f0) / (d * 8),
                down(f1) / (d * 8),
                down(cam) / (d * 8),
                down(depth),
                consts[:, :2].expand(n, 2, Hp // d, Wp // d),
                consts[:, 2:].expand(n, 3, Hp // d, Wp // d),
            ]
            if gt is not None:
                x.append(down(gt))
            x += [F.avg_pool2d(blend, d), F.avg_pool2d(ui, d)]
            y = net["head"](net["trunk"](torch.cat(x, 1))).float()
            up = lambda t: F.interpolate(
                t, scale_factor=d, mode="bilinear", align_corners=False
            )  # noqa: E731
            f0 = f0 + up(y[:, 0:2]) * d
            f1 = f1 + up(y[:, 2:4]) * d
            blend = blend + up(y[:, 4:5])
            ui = ui + up(y[:, 5:6])
            residual = residual + up(y[:, 6:9])
        w0, w1 = backward_warp(i0p, f0), backward_warp(i1p, f1)
        m, u = torch.sigmoid(blend), torch.sigmoid(ui)
        gen = m * w0 + (1 - m) * w1 + residual
        out = u * i1p.float() + (1 - u) * gen
        crop = (..., slice(0, H0), slice(0, W0))
        return out[crop], {
            "flow0": f0[crop],
            "flow1": f1[crop],
            "blend": m[crop],
            "ui": u[crop],
            "ui_logit": ui[crop],
            "gen": gen[crop],
        }


def build_fg(tier):
    spec = dict(FG_TIERS[tier])
    return FGNet(
        spec["width"], spec["levels"], spec["blocks"], privileged=(tier == "teacher")
    )


class TSRExport(nn.Module):
    """ONNX wrapper of one fused TSR step with explicit recurrent state tensors."""

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
        }
        out, st = self.net(
            f,
            {
                "out": prev_out,
                "hidden": prev_hidden,
                "depth": prev_depth,
                "first": False,
            },
        )
        return out, st["hidden"], depth


class FGExport(nn.Module):
    """ONNX wrapper of the fused frame generator."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, i0, i1, mv1, depth1, cam_prior, tau, extrap, avail):
        f = {
            "i0": i0,
            "i1": i1,
            "mv1": mv1,
            "depth1": depth1,
            "cam_prior": cam_prior,
            "tau": tau,
            "extrap": extrap,
            "mv_avail": avail[:, 0],
            "cam_avail": avail[:, 1],
            "depth_avail": avail[:, 2],
        }
        return self.net(f)[0]
