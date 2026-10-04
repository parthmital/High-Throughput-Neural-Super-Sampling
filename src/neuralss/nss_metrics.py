"""Losses, image / perceptual / temporal metrics, optical-flow teacher, analytic baselines and the frame-pacing
latency simulator shared by the supersampling and frame-generation notebooks."""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nss_models import backward_warp


# ---------------------------------------------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------------------------------------------
def charbonnier(pred, target, mask=None, eps=1e-3):
    err = torch.sqrt((pred - target) ** 2 + eps * eps)
    if mask is None:
        return err.mean()
    return (err * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)


def gradient_loss(pred, target):
    def g(x):
        return x[..., :, 2:] - x[..., :, :-2], x[..., 2:, :] - x[..., :-2, :]

    (px, py), (tx, ty) = g(pred), g(target)
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def fft_loss(pred, target):
    """L1 distance of 2D spectrum magnitudes (keeps fine texture energy; Fourier-space loss)."""
    fp = torch.fft.rfft2(pred.float(), norm="ortho")
    ft = torch.fft.rfft2(target.float(), norm="ortho")
    return (fp.abs() - ft.abs()).abs().mean()


def census_loss(pred, target, patch=7):
    """Soft ternary census distance (robust to brightness changes; as used for frame interpolation)."""

    def census(img):
        grey = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        eye = torch.eye(patch * patch, device=img.device, dtype=img.dtype).reshape(
            patch * patch, 1, patch, patch
        )
        diff = F.conv2d(grey, eye, padding=patch // 2) - grey
        return diff / torch.sqrt(0.81 + diff * diff)

    d = (census(pred.float()) - census(target.float())) ** 2
    dist = (d / (0.1 + d)).mean(dim=1, keepdim=True)
    b = patch // 2
    return dist[..., b:-b, b:-b].mean()


def temporal_loss(out_t, out_prev, gt_t, gt_prev, mv_hr, valid):
    """Match the temporal change of the output to that of the ground truth along exact motion (flicker and
    ghosting penalty): |(o_t - W o_{t-1}) - (g_t - W g_{t-1})| over pixels visible in both frames.
    """
    d_out = out_t - backward_warp(out_prev, mv_hr)
    d_gt = gt_t - backward_warp(gt_prev, mv_hr)
    return ((d_out - d_gt).abs() * valid).sum() / (valid.sum() * 3 + 1e-6)


# ---------------------------------------------------------------------------------------------------------------
# Image metrics
# ---------------------------------------------------------------------------------------------------------------
def rgb_to_y(x):
    return (
        16.0 + 65.481 * x[:, 0:1] + 128.553 * x[:, 1:2] + 24.966 * x[:, 2:3]
    ) / 255.0


def quantize(x):
    return (x.clamp(0, 1) * 255.0).round() / 255.0


def psnr(a, b, mask=None, border=0):
    if border:
        a, b = (
            a[..., border:-border, border:-border],
            b[..., border:-border, border:-border],
        )
        mask = mask[..., border:-border, border:-border] if mask is not None else None
    se = (a.float() - b.float()) ** 2
    if mask is None:
        mse = se.mean(dim=(1, 2, 3))
    else:
        m = mask.float().expand_as(se)
        mse = (se * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp_min(1)
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def interpolation_error(a, b):
    """Middlebury interpolation error: RMS difference on the 0-255 scale."""
    return ((a.float() - b.float()) ** 2).mean(dim=(1, 2, 3)).sqrt() * 255.0


def ssim(a, b, border=0):
    if border:
        a, b = (
            a[..., border:-border, border:-border],
            b[..., border:-border, border:-border],
        )
    a, b = a.float(), b.float()
    c = a.shape[1]
    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(c, 1, 11, 11).contiguous()
    mu_a, mu_b = F.conv2d(a, win, groups=c), F.conv2d(b, win, groups=c)
    va = F.conv2d(a * a, win, groups=c) - mu_a**2
    vb = F.conv2d(b * b, win, groups=c) - mu_b**2
    cov = F.conv2d(a * b, win, groups=c) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    )
    return s.mean(dim=(1, 2, 3))


class LPIPSAlex(nn.Module):
    """LPIPS (AlexNet, v0.1) built from torchvision AlexNet weights and the linear calibration weights shipped
    with the lpips package; used when lpips.LPIPS itself cannot be constructed."""

    def __init__(self, lin_path):
        super().__init__()
        from torchvision.models import AlexNet_Weights, alexnet

        feats = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList(
            [feats[0:2], feats[2:5], feats[5:8], feats[8:10], feats[10:12]]
        )
        chans = [64, 192, 384, 256, 256]
        self.lins = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in chans])
        state = torch.load(lin_path, map_location="cpu")
        for k, lin in enumerate(self.lins):
            lin.weight.data.copy_(state[f"lin{k}.model.1.weight"])
        self.register_buffer(
            "shift", torch.tensor([-0.030, -0.088, -0.188]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "scale", torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1)
        )
        self.requires_grad_(False)

    def forward(self, a, b):
        xa, xb = (2 * a - 1 - self.shift) / self.scale, (
            2 * b - 1 - self.shift
        ) / self.scale
        total = 0
        for sl, lin in zip(self.slices, self.lins):
            xa, xb = sl(xa), sl(xb)
            na = xa / (xa.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            nb = xb / (xb.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            total = total + lin((na - nb) ** 2).mean(dim=(2, 3))
        return total.view(-1)


def load_lpips(device, log=print):
    """Callable lpips(a, b) for images in [0, 1] (N, 3, H, W) returning (N,), or None when unavailable."""
    try:
        import lpips

        try:
            net = (
                lpips.LPIPS(net="alex", verbose=False)
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            return lambda a, b: net(a.float(), b.float(), normalize=True).view(-1)
        except Exception as exc:
            log(
                f"lpips.LPIPS failed ({exc}); building LPIPS-Alex from torchvision weights"
            )
            lin = os.path.join(
                os.path.dirname(lpips.__file__), "weights", "v0.1", "alex.pth"
            )
            net = LPIPSAlex(lin).to(device).eval()
            return lambda a, b: net(a.float(), b.float())
    except Exception as exc:
        log(f"LPIPS unavailable ({exc})")
        return None


def flip_error(ref, test):
    """Mean LDR-FLIP error (NVIDIA, BSD-3) of two (3, H, W) tensors in [0, 1], or None if unavailable."""
    try:
        import flip_evaluator as flip

        r = ref.permute(1, 2, 0).float().cpu().numpy().astype(np.float32)
        t = test.permute(1, 2, 0).float().cpu().numpy().astype(np.float32)
        return float(flip.evaluate(r, t, "LDR")[1])
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------------------
# Optical-flow teacher (torchvision RAFT, BSD-3)
# ---------------------------------------------------------------------------------------------------------------
def load_raft(device, log=print, large=False):
    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        net = (
            raft_large(weights=Raft_Large_Weights.DEFAULT)
            if large
            else raft_small(weights=Raft_Small_Weights.DEFAULT)
        )
        return net.to(device).eval().requires_grad_(False)
    except Exception as exc:
        log(
            f"RAFT unavailable ({exc}); estimated motion vectors and RAFT metrics are skipped"
        )
        return None


@torch.no_grad()
def raft_flow(net, a, b, iters=12):
    """Flow from a to b (for each pixel of a, its displacement in b), pixels; a, b in [0, 1] (N, 3, H, W)."""
    h, w = a.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    pa = F.pad(a.float() * 2 - 1, (0, pw, 0, ph), mode="replicate")
    pb = F.pad(b.float() * 2 - 1, (0, pw, 0, ph), mode="replicate")
    return net(pa, pb, num_flow_updates=iters)[-1][..., :h, :w]


@torch.no_grad()
def raft_mv_and_valid(net, cur, prev, tol=1.0):
    """Motion vector cur -> prev with a forward-backward consistency validity mask."""
    fwd = raft_flow(net, cur, prev)
    bwd = raft_flow(net, prev, cur)
    back = backward_warp(bwd, fwd)
    err = (fwd + back).norm(dim=1, keepdim=True)
    valid = (err < tol + 0.05 * fwd.norm(dim=1, keepdim=True)).float()
    return fwd, valid


# ---------------------------------------------------------------------------------------------------------------
# Temporal metrics
# ---------------------------------------------------------------------------------------------------------------
def warping_error(out_t, out_prev, mv, valid):
    """Mean absolute difference between a frame and the previous frame warped by ground-truth motion."""
    diff = (out_t - backward_warp(out_prev, mv)).abs().mean(1, keepdim=True)
    return (diff * valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3)).clamp_min(1)


def temporal_change_error(out_t, out_prev, gt_t, gt_prev, mv, valid):
    d = (out_t - backward_warp(out_prev, mv)) - (gt_t - backward_warp(gt_prev, mv))
    diff = d.abs().mean(1, keepdim=True)
    return (diff * valid).sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3)).clamp_min(1)


@torch.no_grad()
def tof(raft, out_t, out_prev, gt_t, gt_prev):
    """tOF (TecoGAN): mean |flow(out) - flow(gt)| between consecutive frames, pixels."""
    if raft is None:
        return None
    return (
        (raft_flow(raft, out_t, out_prev) - raft_flow(raft, gt_t, gt_prev))
        .norm(dim=1)
        .mean(dim=(1, 2))
    )


@torch.no_grad()
def tlp(lpips_fn, out_t, out_prev, gt_t, gt_prev):
    """tLP (TecoGAN): |LPIPS(out_t, out_prev) - LPIPS(gt_t, gt_prev)|."""
    if lpips_fn is None:
        return None
    return (lpips_fn(out_t, out_prev) - lpips_fn(gt_t, gt_prev)).abs()


# ---------------------------------------------------------------------------------------------------------------
# Analytic baselines
# ---------------------------------------------------------------------------------------------------------------
def rgb_to_ycocg(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat(
        [
            0.25 * r + 0.5 * g + 0.25 * b,
            0.5 * r - 0.5 * b,
            -0.25 * r + 0.5 * g - 0.25 * b,
        ],
        1,
    )


def ycocg_to_rgb(x):
    y, co, cg = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([y + co - cg, y + cg, y - co - cg], 1)


class AnalyticTAAU:
    """Hand-tuned temporal anti-aliasing upscaler: reproject history with upsampled motion vectors, clamp it to the
    3x3 YCoCg neighbourhood of the current jitter-aware upsample, reset on invalid history, blend 10 percent of the
    new frame. A stand-in for analytic TAAU / FSR 2-style accumulation without its proprietary heuristics.
    """

    def __init__(self, base_upsample, scale, blend=0.1):
        self.base_upsample, self.scale, self.blend = base_upsample, scale, blend

    def __call__(self, color, jitter, mv, state):
        base = self.base_upsample(color, jitter)
        if state is None:
            return base, base
        hist = backward_warp(
            state,
            F.interpolate(
                mv, scale_factor=self.scale, mode="bilinear", align_corners=False
            )
            * self.scale,
        )
        yb = rgb_to_ycocg(base)
        lo = -F.max_pool2d(-yb, 3, 1, 1)
        hi = F.max_pool2d(yb, 3, 1, 1)
        hc = ycocg_to_rgb(torch.max(torch.min(rgb_to_ycocg(hist), hi), lo))
        out = hc + self.blend * (base - hc)
        return out, out


def fg_analytic(i0, i1, f0, f1):
    """Motion-vector reprojection baseline: warp both frames along linear-motion flows and keep, per pixel, the
    warp that agrees better with the other (occlusion heuristic); falls back to I1 where both disagree strongly.
    """
    w0, w1 = backward_warp(i0, f0), backward_warp(i1, f1)
    agree = (w0 - w1).abs().mean(1, keepdim=True)
    m = torch.sigmoid((0.1 - agree) * 40.0)
    return m * 0.5 * (w0 + w1) + (1 - m) * w1


# ---------------------------------------------------------------------------------------------------------------
# Frame-pacing latency simulator
# ---------------------------------------------------------------------------------------------------------------
def simulate_latency(
    mode,
    render_ms,
    sr_ms=0.0,
    fg_ms=0.0,
    cpu_ms=4.0,
    frames=2000,
    cv=0.08,
    present_ms=1.0,
    seed=0,
):
    """Discrete-event model of a GPU-bound game loop with a just-in-time low-latency scheduler (CPU work starts so
    that the GPU never queues frames) on a variable-refresh display.

    modes: 'native' (no generation), 'interp' (generated frame between real frames k-1 and k; real frame k is held
    until after the generated one), 'extrap' (generated frame after k from k-1 and k; nothing held), and
    'extrap_late' (extrapolation whose camera is re-sampled from input just before generation).

    Returns per-mode statistics: displayed frame rate, game input latency (input sample -> first photons of a
    real frame reflecting it), camera latency (input sample -> first displayed frame, real or generated, whose
    camera reflects it) and content age (time since the newest real input in each displayed frame).
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    real_lat, cam_lat, shown, ages = [], [], [], []
    for _ in range(frames):
        r = render_ms * float(rng.lognormal(0.0, cv))
        gpu = r + sr_ms + (fg_ms if mode != "native" else 0.0)
        period = max(cpu_ms, gpu)
        t_input = t
        ready = t_input + cpu_ms + r + sr_ms
        if mode == "native":
            p_real = ready + present_ms
            shown.append(p_real)
            real_lat.append(p_real - t_input)
            cam_lat.append(p_real - t_input)
            ages.append(p_real - t_input)
        elif mode == "interp":
            p_gen = ready + fg_ms + present_ms
            p_real = p_gen + period / 2
            shown += [p_gen, p_real]
            real_lat.append(p_real - t_input)
            cam_lat.append(
                p_gen - t_input
            )  # the generated frame already moves toward frame k's camera (half way)
            ages += [p_gen - t_input, p_real - t_input]
        else:
            p_real = ready + present_ms
            p_gen = p_real + period / 2
            shown += [p_real, p_gen]
            real_lat.append(p_real - t_input)
            if mode == "extrap_late":
                t_cam = (
                    p_gen - fg_ms - present_ms
                )  # camera re-sampled from input just before generation
                cam_lat += [p_real - t_input, p_gen - t_cam]
            else:
                cam_lat += [p_real - t_input, p_gen - t_input]
            ages += [p_real - t_input, p_gen - t_input]
        t += period
    shown = np.array(shown)
    gaps = np.diff(shown[len(shown) // 10 :])
    stats = lambda x: (float(np.mean(x)), float(np.percentile(x, 95)))  # noqa: E731
    rl, cl, ag = stats(real_lat), stats(cam_lat), stats(ages)
    return {
        "mode": mode,
        "render_ms": render_ms,
        "sr_ms": sr_ms,
        "fg_ms": fg_ms,
        "displayed_fps": float(1000.0 / gaps.mean()),
        "frame_gap_cv": float(gaps.std() / gaps.mean()),
        "game_latency_ms": rl[0],
        "game_latency_p95_ms": rl[1],
        "camera_latency_ms": cl[0],
        "camera_latency_p95_ms": cl[1],
        "content_age_ms": ag[0],
    }


def estimate_ms(gmac, sustained_tflops):
    """Arithmetic latency estimate: 2 FLOP per MAC at a sustained throughput (FP16 TFLOPS)."""
    return (
        2.0 * gmac / (sustained_tflops * 1e3) * 1e3
        if sustained_tflops > 0
        else math.inf
    )
