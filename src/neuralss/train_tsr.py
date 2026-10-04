"""DDP training of the temporal super-resolution network (one process per GPU, launched by the notebook with
torch.distributed.run; ablations run as single-GPU processes, one per GPU in parallel).

Each step mixes synthetic layered game-like sequences rendered on the GPU from cached stills (exact motion,
disocclusion, thin geometry, particles) with cached real sequences (exact motion from TartanAir / Sintel,
teacher-estimated motion from video). Loss per frame: Charbonnier + gradient + Fourier-magnitude terms,
LPIPS on the last frames, a temporal-change term along exact motion, and optional distillation toward a frozen
teacher network unrolled on the same inputs."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_metrics import (
    charbonnier,
    fft_loss,
    gradient_loss,
    load_lpips,
    psnr,
    rgb_to_y,
    temporal_change_error,
    temporal_loss,
)  # noqa: E402
from nss_models import build_tsr  # noqa: E402
from nss_synth import make_text_bank, sr_batch, sr_batch_from_frames  # noqa: E402
from nss_train import emit, make_loader, setup_ddp, train_loop  # noqa: E402


def frame_inputs(b, t, ablate_gbuffers=False):
    n = b["lr"].shape[0]
    f = {
        "color": b["lr"][:, t],
        "depth": b["depth_lr"][:, t],
        "mv": b["mv_lr"][:, t],
        "reactive": b["reactive_lr"][:, t],
        "exposure": b["exposure"][:, t],
        "jitter": b["jitter"][:, t],
        "has_depth": b["has_depth"],
        "mv_quality": b["mv_quality"],
    }
    if (
        ablate_gbuffers
    ):  # colour-only input: no depth, no motion vectors (history warped with zero motion)
        f["depth"], f["mv"] = torch.zeros_like(f["depth"]), torch.zeros_like(f["mv"])
        f["has_depth"], f["mv_quality"] = torch.zeros(
            n, device=f["color"].device
        ), torch.zeros(n, device=f["color"].device)
    return f


def unroll(model, b, use_history=True, ablate_gbuffers=False):
    outs, state = [], None
    for t in range(b["lr"].shape[1]):
        out, state = model(
            frame_inputs(b, t, ablate_gbuffers), state, use_history=use_history
        )
        outs.append(out)
    return torch.stack(outs, 1)


class Sequence(torch.nn.Module):
    """Wraps the recurrent unroll in one module call, so DDP sees one forward per backward."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, b, use_history=True, ablate_gbuffers=False):
        return unroll(self.net, b, use_history, ablate_gbuffers)


def bicubic_inputs(b, scale):
    """Ablation: classic image-SR degradation (anti-aliased bicubic, no jitter) instead of render-like sampling."""
    n, T, _, H, W = b["hr"].shape
    hr = b["hr"].flatten(0, 1)
    lr = F.interpolate(
        hr,
        size=(H // scale, W // scale),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).clamp(0, 1)
    b = dict(b)
    b["lr"] = lr.view(n, T, 3, H // scale, W // scale)
    b["jitter"] = torch.zeros_like(b["jitter"])
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.load(open(parser.parse_args().config))
    rank, world, device = setup_ddp()
    torch.manual_seed(cfg["seed"] + rank)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"] * 100 + rank)
    s, T, P = cfg["scale"], cfg["seq_len"], cfg["patch"]
    model = Sequence(build_tsr(cfg["tier"], s)).to(device)
    teacher = None
    if cfg.get("teacher_ckpt"):
        teacher = Sequence(build_tsr(cfg["teacher_tier"], s)).to(device)
        teacher.load_state_dict(
            torch.load(cfg["teacher_ckpt"], map_location=device)["ema"]
        )
        teacher.eval().requires_grad_(False)
    lpips_fn = (
        load_lpips(device, log=lambda m: emit("LOG", {"msg": m}))
        if cfg["w_lpips"] > 0
        else None
    )
    tiles = (
        torch.from_numpy(make_text_bank(64, cfg["seed"], height=24, width=128)).to(
            device
        )
        if cfg.get("world_text", True)
        else None
    )
    caches = {k: v for k, v in cfg["caches"].items() if v["count"] > 0}
    b_real = int(round(cfg["batch_per_gpu"] * cfg["p_real"])) if "seq" in caches else 0
    sizes = {"still": cfg["batch_per_gpu"] - b_real, "seq": b_real}
    loader = make_loader(
        caches, sizes, cfg["seed"] * 1000 + rank * 10**7, cfg["workers_per_rank"]
    )

    def make_batch(raw, g, n_synth_override=None):
        parts = []
        if "still/rgb" in raw and raw["still/rgb"].shape[0] > 0:
            parts.append(sr_batch(raw["still/rgb"], s, T, P, g, text_bank=tiles))
        if "seq/rgb" in raw and raw["seq/rgb"].shape[0] > 0:
            meta = raw["seq/meta"].float()
            parts.append(
                sr_batch_from_frames(
                    raw["seq/rgb"],
                    raw["seq/mv"],
                    raw["seq/valid"],
                    raw["seq/depth"],
                    meta[:, 0],
                    meta[:, 1],
                    s,
                    g,
                )
            )
        b = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
        return bicubic_inputs(b, s) if cfg.get("degradation") == "bicubic" else b

    def losses(out, b, teacher_out=None):
        n, T_ = out.shape[:2]
        hr = b["hr"]
        w_t = torch.ones(T_, device=out.device)
        w_t[0] = 0.25  # the first frame has no history
        o, g = out.flatten(0, 1), hr.flatten(0, 1)
        per = torch.stack([charbonnier(out[:, t], hr[:, t]) for t in range(T_)])
        l_pix = (per * w_t).sum() / w_t.sum()
        l_grad = gradient_loss(o, g)
        l_fft = fft_loss(o, g)
        l_tmp = out.new_zeros(())
        q = b["mv_quality"].view(n, 1, 1, 1)
        for t in range(1, T_):
            l_tmp = l_tmp + temporal_loss(
                out[:, t],
                out[:, t - 1],
                hr[:, t],
                hr[:, t - 1],
                b["mv_hr"][:, t],
                b["valid_hr"][:, t] * q,
            )
        l_tmp = l_tmp / max(T_ - 1, 1)
        l_lp = (
            lpips_fn(out[:, -1].clamp(0, 1), hr[:, -1]).mean()
            if lpips_fn is not None
            else out.new_zeros(())
        )
        l_kd = (
            (out[:, 1:] - teacher_out[:, 1:]).abs().mean()
            if teacher_out is not None
            else out.new_zeros(())
        )
        total = (
            l_pix
            + cfg["w_grad"] * l_grad
            + cfg["w_fft"] * l_fft
            + cfg["w_temporal"] * l_tmp
            + cfg["w_lpips"] * l_lp
            + cfg["w_distill"] * l_kd
        )
        return total, {
            "loss": total.detach(),
            "pix": l_pix.detach(),
            "temporal": l_tmp.detach(),
            "lpips": l_lp.detach(),
            "distill": l_kd.detach(),
        }

    def step_fn(net, raw, step):
        b = make_batch(raw, gen)
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(
                b, cfg.get("use_history", True), cfg.get("ablate_gbuffers", False)
            )
            t_out = None
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(b)
        loss, logs = losses(
            out.float(), b, t_out.float() if t_out is not None else None
        )
        return loss, logs, b["lr"].shape[0]

    # fixed validation batches (same for every round): synthetic from validation stills + real validation windows
    val_batches = []
    if rank == 0:
        vg = torch.Generator(device=device)
        vg.manual_seed(12345)
        vc = cfg["val_caches"]
        for k in range(cfg["val_batches"]):
            raw = {}
            for name in ("still", "seq"):
                if vc.get(name, {}).get("count", 0) > 0:
                    arrs = {
                        key: np.load(p, mmap_mode="r")
                        for key, p in vc[name]["arrays"].items()
                    }
                    idx = (
                        np.arange(k * cfg["val_bs"], (k + 1) * cfg["val_bs"])
                        % vc[name]["count"]
                    )
                    for key, a in arrs.items():
                        raw[f"{name}/{key}"] = torch.from_numpy(
                            np.ascontiguousarray(a[np.sort(idx)])
                        ).to(device)
            if raw:
                val_batches.append(
                    {k2: v.cpu() for k2, v in make_batch(raw, vg).items()}
                )

    @torch.no_grad()
    def val_fn(ema_model):
        rows = {"psnr_y": [], "psnr_y_last": [], "tce": []}
        for vb in val_batches:
            b = {k: v.to(device) for k, v in vb.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = (
                    ema_model(
                        b,
                        cfg.get("use_history", True),
                        cfg.get("ablate_gbuffers", False),
                    )
                    .float()
                    .clamp(0, 1)
                )
            hr = b["hr"]
            T_ = out.shape[1]
            ps = torch.stack(
                [
                    psnr(rgb_to_y(out[:, t]), rgb_to_y(hr[:, t]), border=s)
                    for t in range(1, T_)
                ],
                1,
            )
            rows["psnr_y"].append(ps.mean(1))
            rows["psnr_y_last"].append(ps[:, -1])
            tce = torch.stack(
                [
                    temporal_change_error(
                        out[:, t],
                        out[:, t - 1],
                        hr[:, t],
                        hr[:, t - 1],
                        b["mv_hr"][:, t],
                        b["valid_hr"][:, t],
                    )
                    for t in range(1, T_)
                ],
                1,
            ).mean(1)
            rows["tce"].append(tce)
        return {k: torch.cat(v).mean().item() for k, v in rows.items() if v}

    train_loop(cfg, model, step_fn, val_fn, rank, world, device, loader)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
