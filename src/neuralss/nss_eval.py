"""Evaluation helpers for both training notebooks: full-resolution test clips, input construction, method
runners (networks and analytic baselines) and per-frame metric rows."""

import numpy as np
import torch
import torch.nn.functional as F

import nss_data as D
import nss_metrics as M
import nss_synth as S
from nss_models import backward_warp


# ---------------------------------------------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------------------------------------------
def load_clip(data_root, item, start, T, crop_hw, max_side=None):
    """Full-resolution test clip: frames uint8 (T, H, W, 3) center-cropped to crop_hw (after an optional
    down-scale so the shorter side is at most max_side), plus exact motion vectors (T, H, W, 2) and validity
    (T, H, W) in network time when the source has exact flow, depth (T, H, W) when available, and native
    low-resolution frames when the source provides them. Exact-forward sources are played backwards.
    """
    res = D.Resolver(data_root)
    idx = list(range(start, start + T))
    frames = D.load_rgb_many(res, [item["frames"][i] for i in idx])
    h, w = frames[0].shape[:2]
    r = min(1.0, max_side / min(h, w)) if max_side else 1.0
    W, H = int(round(w * r)), int(round(h * r))
    ch, cw = crop_hw
    if H < ch or W < cw:
        return None
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    out = {
        "rgb": np.stack([D._resize(f, (W, H), "image")[sl] for f in frames]),
        "exact": False,
    }
    if item.get("mv") == "exact_forward":
        flows = [
            D._resize(D.load_flow(res, item["flows"][i]), (W, H), "flow")[sl]
            for i in idx
        ]
        valid = [
            D._resize(
                D.load_valid(res, item["valid"][i], item.get("valid_kind", "valid")),
                (W, H),
                "nearest",
            )[sl]
            for i in idx
        ]
        mv, va = np.stack(flows)[::-1].copy(), np.stack(valid)[::-1].copy()
        mv[0], va[0] = 0, 0
        out.update(rgb=out["rgb"][::-1].copy(), mv=mv, valid=va, exact=True)
    if item.get("depth"):
        d = np.stack(
            [
                D.normalise_depth(
                    D._resize(D.load_depth(res, item["depth"][i]), (W, H), "depth")[sl]
                )
                for i in idx
            ]
        )
        out["depth"] = d[::-1].copy() if out["exact"] else d
    if item.get("native_lr"):
        lr = D.load_rgb_many(res, [item["native_lr"][i] for i in idx])
        f = lr[0].shape[0] / h
        lsl = (
            slice(int(y0 * f), int(y0 * f) + int(ch * f)),
            slice(int(x0 * f), int(x0 * f) + int(cw * f)),
        )
        out["native_lr"] = np.stack(
            [
                D._resize(x, (int(round(w * r * f)), int(round(h * r * f))), "image")[
                    lsl
                ]
                for x in lr
            ]
        )
    return out


def sr_inputs(clip, scale, device, raft=None, jitter=True, native=False, phases=16):
    """Batch (N = 1) for the TSR network from a clip: jittered single-sample low-resolution frames (or the
    native low-resolution render), motion vectors (exact, or RAFT on consecutive low-resolution frames),
    validity, depth, full-resolution targets."""
    hr = (
        torch.from_numpy(clip["rgb"]).to(device).permute(0, 3, 1, 2).float()[None]
        / 255.0
    )  # 1, T, 3, H, W
    _, T, _, H, W = hr.shape
    h, w = H // scale, W // scale
    seq = (
        S.jitter_sequence(T, phases).to(device)
        if jitter and not native
        else torch.zeros(T, 2, device=device)
    )
    lrs, mv_lr, va_lr, d_lr, mv_hr, va_hr = [], [], [], [], [], []
    for t in range(T):
        if native:
            lr = (
                torch.from_numpy(clip["native_lr"][t])
                .to(device)
                .permute(2, 0, 1)
                .float()[None]
                / 255.0
            )
            lr = (
                F.interpolate(
                    lr,
                    size=(h, w),
                    mode="bilinear",
                    antialias=True,
                    align_corners=False,
                )
                if lr.shape[-2:] != (h, w)
                else lr
            )
        else:
            pts = S.pixel_grid(1, h, w, device, seq[t : t + 1])
            # non-square frames: pixel_grid normalises x and y independently, which matches grid_sample
            lr = F.grid_sample(
                hr[:, t],
                pts,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        lrs.append(lr)
    lr_all = torch.stack(lrs, 1)
    quality = 0.0
    for t in range(T):
        if clip.get("exact"):
            m = (
                torch.from_numpy(clip["mv"][t])
                .to(device)
                .permute(2, 0, 1)
                .float()[None]
            )
            v = torch.from_numpy(clip["valid"][t]).to(device).float()[None, None]
            pts = S.pixel_grid(1, h, w, device, seq[t : t + 1])
            mv_lr.append(
                F.grid_sample(
                    m, pts, mode="bilinear", padding_mode="border", align_corners=False
                )
                / scale
            )
            va_lr.append(
                F.grid_sample(
                    v, pts, mode="nearest", padding_mode="border", align_corners=False
                )
            )
            mv_hr.append(m)
            va_hr.append(v)
            quality = 1.0
        elif raft is not None and t > 0:
            m, v = M.raft_mv_and_valid(raft, lr_all[:, t], lr_all[:, t - 1])
            mv_lr.append(m)
            va_lr.append(v)
            mv_hr.append(
                F.interpolate(
                    m, scale_factor=scale, mode="bilinear", align_corners=False
                )
                * scale
            )
            va_hr.append(F.interpolate(v, scale_factor=scale, mode="nearest"))
            quality = 0.5
        else:
            mv_lr.append(torch.zeros(1, 2, h, w, device=device))
            va_lr.append(torch.zeros(1, 1, h, w, device=device))
            mv_hr.append(torch.zeros(1, 2, H, W, device=device))
            va_hr.append(torch.zeros(1, 1, H, W, device=device))
        if "depth" in clip:
            d = torch.from_numpy(clip["depth"][t]).to(device).float()[None, None]
            d_lr.append(
                F.interpolate(d, size=(h, w), mode="bilinear", align_corners=False)
            )
        else:
            d_lr.append(torch.zeros(1, 1, h, w, device=device))
    st = lambda xs: torch.stack(xs, 1)  # noqa: E731
    return {
        "lr": lr_all,
        "hr": hr,
        "mv_lr": st(mv_lr),
        "valid_lr": st(va_lr),
        "mv_hr": st(mv_hr),
        "valid_hr": st(va_hr),
        "depth_lr": st(d_lr),
        "reactive_lr": torch.zeros(1, T, 1, h, w, device=device),
        "jitter": seq[None],
        "exposure": torch.zeros(1, T, device=device),
        "has_depth": torch.tensor([1.0 if "depth" in clip else 0.0], device=device),
        "mv_quality": torch.tensor([quality], device=device),
        "speed_class": torch.full((1,), -1, device=device),
    }


def synthetic_suite(stills, scale, T, P, motion_class, seed, device, text_bank=None):
    """Synthetic game-like test sequences with every sample in one motion class (static / slow / medium / fast).
    The camera path is only kept inside the texture for 6 frames so long sequences keep their speed class.
    """
    prof = dict(S.MOTION_PROFILE)
    prof["probs"] = [
        1.0 if i == motion_class else 0.0 for i in range(len(prof["probs"]))
    ]
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return S.sr_batch(
        stills, scale, T, P, g, profile=prof, text_bank=text_bank, room_frames=min(T, 6)
    )


# ---------------------------------------------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------------------------------------------
def frame_inputs(b, t):
    return {
        "color": b["lr"][:, t],
        "depth": b["depth_lr"][:, t],
        "mv": b["mv_lr"][:, t],
        "reactive": b["reactive_lr"][:, t],
        "exposure": b["exposure"][:, t],
        "jitter": b["jitter"][:, t],
        "has_depth": b["has_depth"],
        "mv_quality": b["mv_quality"],
    }


@torch.no_grad()
def run_tsr(
    net, b, use_history=True, ablate_gbuffers=False, keep_alpha=False, amp=True
):
    outs, alphas, state = [], [], None
    for t in range(b["lr"].shape[1]):
        f = frame_inputs(b, t)
        if ablate_gbuffers:
            f.update(
                depth=torch.zeros_like(f["depth"]),
                mv=torch.zeros_like(f["mv"]),
                has_depth=torch.zeros_like(f["has_depth"]),
                mv_quality=torch.zeros_like(f["mv_quality"]),
            )
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out, state = net(f, state, use_history=use_history)
        outs.append(out.float().clamp(0, 1))
        if keep_alpha:
            alphas.append(state["alpha"].float())
    return (
        (torch.stack(outs, 1), torch.stack(alphas, 1))
        if keep_alpha
        else torch.stack(outs, 1)
    )


@torch.no_grad()
def run_bilinear(net, b):
    return torch.stack(
        [
            net.base_upsample(b["lr"][:, t], b["jitter"][:, t]).clamp(0, 1)
            for t in range(b["lr"].shape[1])
        ],
        1,
    )


@torch.no_grad()
def run_bicubic(b, scale):
    n, T, _, h, w = b["lr"].shape
    x = F.interpolate(
        b["lr"].flatten(0, 1), scale_factor=scale, mode="bicubic", align_corners=False
    ).clamp(0, 1)
    return x.view(n, T, 3, h * scale, w * scale)


@torch.no_grad()
def run_taau(net, b, scale):
    taau = M.AnalyticTAAU(net.base_upsample, scale)
    outs, state = [], None
    for t in range(b["lr"].shape[1]):
        out, state = taau(b["lr"][:, t], b["jitter"][:, t], b["mv_lr"][:, t], state)
        outs.append(out.clamp(0, 1))
    return torch.stack(outs, 1)


def hf_mask(hr, frac=0.15):
    """Fine-detail mask: pixels in the top frac of ground-truth gradient magnitude."""
    y = M.rgb_to_y(hr)
    gx = F.pad(y[..., :, 2:] - y[..., :, :-2], (1, 1, 0, 0))
    gy = F.pad(y[..., 2:, :] - y[..., :-2, :], (0, 0, 1, 1))
    mag = (gx**2 + gy**2).sqrt().flatten(1)
    thr = torch.quantile(mag.float(), 1 - frac, dim=1)
    return (mag >= thr[:, None]).view_as(y).float()


@torch.no_grad()
def sr_frame_metrics(
    out, b, scale, lpips_fn=None, raft=None, lpips_every=4, skip_first=1
):
    """Per (sample, frame) metric rows: PSNR / SSIM on Y, PSNR in disoccluded, fine-detail and particle regions,
    LPIPS (every lpips_every frames), warping and temporal-change errors along ground-truth motion when exact,
    tOF with RAFT."""
    hr = b["hr"]
    n, T = out.shape[:2]
    rows = []
    exact = b["mv_quality"] >= 1.0
    for t in range(skip_first, T):
        o, g = out[:, t], hr[:, t]
        yo, yg = M.rgb_to_y(o), M.rgb_to_y(g)
        p = M.psnr(yo, yg, border=scale)
        sm = M.ssim(yo, yg, border=scale)
        disocc = (1 - b["valid_hr"][:, t]) if t > 0 else torch.zeros_like(yo)
        hfm = hf_mask(g)
        rea = (
            F.interpolate(b["reactive_lr"][:, t], scale_factor=scale, mode="nearest")
            > 0.1
        )
        p_dis = M.psnr(yo, yg, mask=disocc) if disocc.sum() > 50 else None
        p_hf = M.psnr(yo, yg, mask=hfm)
        p_rea = M.psnr(yo, yg, mask=rea.float()) if rea.sum() > 50 else None
        lp = lpips_fn(o, g) if (lpips_fn is not None and t % lpips_every == 0) else None
        mv_px = (
            b["mv_hr"][:, t].norm(dim=1).mean(dim=(1, 2))
        )  # actual motion magnitude, output pixels per frame
        we = tce = tof = None
        if t > 0:
            we = M.warping_error(
                o, out[:, t - 1], b["mv_hr"][:, t], b["valid_hr"][:, t]
            )
            tce = M.temporal_change_error(
                o, out[:, t - 1], g, hr[:, t - 1], b["mv_hr"][:, t], b["valid_hr"][:, t]
            )
            if raft is not None and t % lpips_every == 0:
                tof = M.tof(raft, o, out[:, t - 1], g, hr[:, t - 1])
        for i in range(n):
            rows.append(
                {
                    "sample": i,
                    "frame": t,
                    "psnr_y": p[i].item(),
                    "ssim_y": sm[i].item(),
                    "psnr_disocc": None if p_dis is None else p_dis[i].item(),
                    "psnr_detail": p_hf[i].item(),
                    "psnr_particles": None if p_rea is None else p_rea[i].item(),
                    "lpips": None if lp is None else lp[i].item(),
                    "warp_err": None if (we is None or not exact[i]) else we[i].item(),
                    "temporal_err": (
                        None
                        if (tce is None or b["mv_quality"][i] <= 0)
                        else tce[i].item()
                    ),
                    "tof": None if tof is None else tof[i].item(),
                    "mv_px": mv_px[i].item(),
                    "speed_class": int(b["speed_class"][i].item()),
                }
            )
    return rows


# ---------------------------------------------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------------------------------------------
def fg_suite(
    stills,
    P,
    seed,
    device,
    tiles=None,
    extrap=False,
    motion_class=None,
    p_ui=0.0,
    p_cut=0.0,
    p_menu=0.0,
    p_flip=0.0,
):
    prof = dict(S.MOTION_PROFILE)
    if motion_class is not None:
        prof["probs"] = [
            1.0 if i == motion_class else 0.0 for i in range(len(prof["probs"]))
        ]
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    b = S.fg_batch(
        stills,
        P,
        g,
        tiles=tiles,
        p_extrap=1.0 if extrap else 0.0,
        p_ui=p_ui,
        p_cut=p_cut,
        p_menu=p_menu,
        p_flip=p_flip,
        profile=prof,
    )
    n = b["i0"].shape[0]
    b["mv_avail"] = torch.ones(n, device=device)
    b["depth_avail"] = torch.ones(n, device=device)
    b["cam_avail"] = b["extrap"].clone()
    return b


@torch.no_grad()
def run_fg(net, b, hudless=False):
    f = dict(b)
    if hudless and "hudless_i0" in b:
        f["i0"], f["i1"] = b["hudless_i0"], b["hudless_i1"]
    with torch.autocast("cuda", dtype=torch.float16):
        out, aux = net(f)
    out = out.float().clamp(0, 1)
    if (
        hudless and "ui_rgb" in b
    ):  # engine path: generate the scene without UI, composite the latest UI on top
        out = out * (1 - b["ui_alpha"]) + b["ui_rgb"]
    return out, aux


def fg_baselines(b):
    """Frame repeat (latest real frame), plain average, and motion-vector reprojection."""
    n = b["i0"].shape[0]
    tau = b["tau"].view(n, 1, 1, 1)
    m1 = b["mv1"] * b["mv_avail"].view(n, 1, 1, 1)
    return {
        "repeat": b["i1"],
        "average": 0.5 * (b["i0"] + b["i1"]),
        "mv_reprojection": M.fg_analytic(
            b["i0"], b["i1"], tau * m1, (tau - 1) * m1
        ).clamp(0, 1),
    }


@torch.no_grad()
def fg_metrics(out, b, lpips_fn=None):
    tgt = b["target"]
    n = tgt.shape[0]
    p = M.psnr(out, tgt)
    sm = M.ssim(out, tgt)
    ie = M.interpolation_error(out, tgt)
    lp = lpips_fn(out, tgt) if lpips_fn is not None else None
    ui = (b["ui_mask"] > 0.5).float()
    has_ui = ui.flatten(1).amax(1) > 0
    p_ui = M.psnr(out, tgt, mask=ui)
    occl = (1 - b["valid_t1"]) if "valid_t1" in b else torch.zeros_like(ui)
    has_occ = occl.flatten(1).sum(1) > 50
    p_occ = M.psnr(out, tgt, mask=occl)
    # UI ghosting: fraction of UI pixels whose error exceeds 10 percent (half-interpolated text / HUD)
    ghost = (((out - tgt).abs().amax(1, keepdim=True) > 0.1).float() * ui).flatten(
        1
    ).sum(1) / ui.flatten(1).sum(1).clamp_min(1)
    mv_px = (
        b["mv1"].norm(dim=1).mean(dim=(1, 2))
    )  # motion between the real frames, pixels
    rows = []
    for i in range(n):
        rows.append(
            {
                "psnr": p[i].item(),
                "ssim": sm[i].item(),
                "ie": ie[i].item(),
                "lpips": None if lp is None else lp[i].item(),
                "psnr_ui": p_ui[i].item() if has_ui[i] else None,
                "ui_ghost_rate": ghost[i].item() if has_ui[i] else None,
                "psnr_occluded": p_occ[i].item() if has_occ[i] else None,
                "mv_px": mv_px[i].item(),
                "extrap": bool(b["extrap"][i] > 0.5),
                "speed_class": int(b["speed_class"][i].item()),
                "cut": bool(b["cut"][i] > 0.5),
                "menu": bool(b["menu"][i] > 0.5),
                "flip": bool(b["flip"][i] > 0.5),
            }
        )
    return rows


@torch.no_grad()
def cut_score(i0, i1, mv1=None):
    """Scene-cut score: luma-histogram distance plus motion-compensated difference (higher = more likely a cut)."""
    y0, y1 = M.rgb_to_y(i0), M.rgb_to_y(i1)
    h0 = torch.stack([torch.histc(y, bins=32, min=0, max=1) for y in y0.flatten(1)])
    h1 = torch.stack([torch.histc(y, bins=32, min=0, max=1) for y in y1.flatten(1)])
    h0, h1 = h0 / h0.sum(1, keepdim=True), h1 / h1.sum(1, keepdim=True)
    hist = 0.5 * (h0 - h1).abs().sum(1)
    w = backward_warp(i0, mv1) if mv1 is not None else i0
    diff = (i1 - w).abs().mean(dim=(1, 2, 3))
    return hist + 2.0 * diff
