"""DDP training of the temporal super-resolution network (one process per GPU, launched by the notebook with
torch.distributed.run; ablations run as single-GPU processes, one per GPU in parallel).

Each step mixes synthetic layered game-like sequences rendered on the GPU from cached stills (exact motion,
disocclusion, thin geometry, particles) with cached real sequences (exact motion from TartanAir / Sintel,
teacher-estimated motion from video). Two input kinds are trained together: jittered render-like samples with
engine data (motion vectors and depth, each randomly withheld) and ordinary video (anti-aliased downscaling,
blur, noise, no jitter and no side information), so one model serves engine integrations and plain video.
Without motion vectors the network's built-in IFNet estimates them, supervised by the exact or RAFT motion;
during the first mv_warmup_frac of the run those samples warp history with the label motion at a falling rate
(teacher forcing), so the reconstruction trunk does not learn to distrust history while the estimator is poor.
Every long_every-th step unrolls a long synthetic sequence so the recurrence learns to keep converging on
static content past the short training horizon.

Loss per frame: Charbonnier + gradient + Fourier-magnitude terms, LPIPS on the last frame, a temporal-change term
along known motion, motion-estimation error, and optional distillation toward a frozen teacher unrolled on the
same inputs."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

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
from nss_models import build_tsr, load_ifnet_weights  # noqa: E402
from nss_synth import (  # noqa: E402
    make_text_bank,
    sr_batch,
    sr_batch_from_frames,
    video_degrade,
)
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
        "kind": b["kind"],
    }
    if "mv_force" in b:  # teacher forcing of estimated motion (see TSRNet.forward)
        f["mv_force"], f["mv_label"] = b["mv_force"], b["mv_lr"][:, t]
    # colour-only input: no depth and no motion vectors (the built-in estimator supplies motion, if present)
    if ablate_gbuffers:
        z = torch.zeros(n, device=f["color"].device)
        f["depth"], f["mv"] = torch.zeros_like(f["depth"]), torch.zeros_like(f["mv"])
        f["has_depth"], f["mv_quality"] = z, z
    return f


def unroll(model, b, use_history=True, ablate_gbuffers=False):
    """Outputs (N, T, 3, H, W) and the estimated motion of frames 1..T-1 (N, T-1, 2, h, w), or None when the
    model has no estimator or runs without history."""
    outs, ests, state = [], [], None
    for t in range(b["lr"].shape[1]):
        out, state = model(
            frame_inputs(b, t, ablate_gbuffers), state, use_history=use_history
        )
        outs.append(out)
        if state["mv_est"] is not None:
            ests.append(state["mv_est"])
    return torch.stack(outs, 1), (torch.stack(ests, 1) if ests else None)


class Sequence(torch.nn.Module):
    """Wraps the recurrent unroll in one module call, so DDP sees one forward per backward."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, b, use_history=True, ablate_gbuffers=False):
        return unroll(self.net, b, use_history, ablate_gbuffers)


def withhold_engine_data(b, g, p_mv, p_depth):
    """Every sample of the batch (render and video kinds alike) loses its motion vectors (share p_mv: the estimator
    runs on the jittered frames, as for colour-only captures) or its depth (share p_depth). Video samples have no
    depth already, and those that keep motion vectors can lose them here too."""
    n, dev = b["lr"].shape[0], b["lr"].device
    b = dict(b)
    drop_mv = torch.rand(n, device=dev, generator=g) < p_mv
    drop_d = torch.rand(n, device=dev, generator=g) < p_depth
    b["mv_quality"] = torch.where(
        drop_mv, torch.zeros_like(b["mv_quality"]), b["mv_quality"]
    )
    b["has_depth"] = torch.where(
        drop_d, torch.zeros_like(b["has_depth"]), b["has_depth"]
    )
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
    log = lambda m: emit("LOG", {"msg": m}) if rank == 0 else None  # noqa: E731
    net = build_tsr(cfg["tier"], s, motion_net=cfg.get("motion_net", True))
    if net.motion is not None and cfg.get("rife_weights"):
        rep = load_ifnet_weights(
            net.motion, torch.load(cfg["rife_weights"], weights_only=True)
        )
        log(f"motion estimator initialised from RIFE 4.26: {rep}")
    model = Sequence(net).to(device)
    teacher = None
    if cfg.get("teacher_ckpt"):
        teacher = Sequence(build_tsr(cfg["teacher_tier"], s)).to(device)
        teacher.load_state_dict(
            torch.load(cfg["teacher_ckpt"], map_location=device)["ema"]
        )
        teacher.eval().requires_grad_(False)
    lpips_fn = load_lpips(device, log=log) if cfg["w_lpips"] > 0 else None
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

    def make_batch(raw, g, p_video=None, long=False):
        """p_video overrides both video shares (validation builds a render and a video copy)."""
        pv_s = cfg["p_video_synth"] if p_video is None else p_video
        pv_r = cfg["p_video_real"] if p_video is None else p_video
        parts = []
        if "still/rgb" in raw and raw["still/rgb"].shape[0] > 0:
            if long:
                b = sr_batch(
                    raw["still/rgb"],
                    s,
                    cfg["long_T"],
                    cfg["long_patch"],
                    g,
                    text_bank=tiles,
                )
            else:
                b = sr_batch(raw["still/rgb"], s, T, P, g, text_bank=tiles)
            parts.append(video_degrade(b, s, g, pv_s, cfg["p_video_mv"]))
        if not long and "seq/rgb" in raw and raw["seq/rgb"].shape[0] > 0:
            meta = raw["seq/meta"].float()
            b = sr_batch_from_frames(
                raw["seq/rgb"],
                raw["seq/mv"],
                raw["seq/valid"],
                raw["seq/depth"],
                meta[:, 0],
                meta[:, 1],
                s,
                g,
            )
            parts.append(video_degrade(b, s, g, pv_r, cfg["p_video_mv"]))
        b = {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
        if (
            p_video is None
        ):  # training: engine data are sometimes withheld from render samples
            b = withhold_engine_data(b, g, cfg["p_drop_mv"], cfg["p_drop_depth"])
        return b

    def losses(out, est, b, teacher_out=None):
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
        q = b["label_quality"].view(n, 1, 1, 1)
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
        l_flow = out.new_zeros(())
        if (
            est is not None
        ):  # motion vectors in low-resolution pixels, weighted by label quality
            m = b["valid_lr"][:, 1:] * b["label_quality"].view(n, 1, 1, 1, 1)
            err = (est.float() - b["mv_lr"][:, 1:]).abs().sum(2, keepdim=True)
            l_flow = (err * m).sum() / (m.sum() + 1.0)
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
            + cfg["w_flow"] * l_flow
            + cfg["w_lpips"] * l_lp
            + cfg["w_distill"] * l_kd
        )
        return total, {
            "loss": total.detach(),
            "pix": l_pix.detach(),
            "temporal": l_tmp.detach(),
            "flow": l_flow.detach(),
            "lpips": l_lp.detach(),
            "distill": l_kd.detach(),
        }

    use_hist, ablate = cfg.get("use_history", True), cfg.get("ablate_gbuffers", False)
    t_start = []

    def step_fn(net, raw, step):
        long = cfg.get("long_every", 0) > 0 and step % cfg["long_every"] == 0
        b = make_batch(raw, gen, long=long)
        # motion warm-up: while the inherited estimator is still poor, samples without motion vectors warp
        # history with the label motion with a probability falling from 1 to 0 over mv_warmup_frac of the run
        if not t_start:
            t_start.append(time.time())
        frac = (time.time() - t_start[0]) / cfg["train_seconds"]
        p_force = (
            max(0.0, 1.0 - frac / cfg["mv_warmup_frac"])
            if cfg.get("mv_warmup_frac")
            else 0.0
        )
        if p_force > 0:
            n = b["lr"].shape[0]
            b["mv_force"] = (torch.rand(n, device=device, generator=gen) < p_force) & (
                b["label_quality"] > 0
            )
        with torch.autocast("cuda", dtype=torch.float16):
            out, est = net(b, use_hist, ablate)
            t_out = None
            if teacher is not None and not long:
                with torch.no_grad():
                    t_out = teacher(b)[0]
        loss, logs = losses(
            out.float(), est, b, t_out.float() if t_out is not None else None
        )
        return loss, logs, b["lr"].shape[0]

    # fixed validation batches (same for every round): synthetic from validation stills + real validation windows,
    # each as a render copy (engine data given) and a plain-video copy (no side information)
    val_batches = []
    if rank == 0:
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
                for kind, pv in (("render", 0.0), ("video", 1.0)):
                    vg = torch.Generator(device=device)
                    vg.manual_seed(12345 + k)
                    vb = make_batch(raw, vg, p_video=pv)
                    val_batches.append((kind, {k2: v.cpu() for k2, v in vb.items()}))

    @torch.no_grad()
    def val_fn(ema_model):
        rows = {"psnr_y_render": [], "psnr_y_video": [], "psnr_y_last": [], "tce": []}
        for kind, vb in val_batches:
            b = {k: v.to(device) for k, v in vb.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = ema_model(b, use_hist, ablate)[0].float().clamp(0, 1)
            hr = b["hr"]
            T_ = out.shape[1]
            ps = torch.stack(
                [
                    psnr(rgb_to_y(out[:, t]), rgb_to_y(hr[:, t]), border=s)
                    for t in range(1, T_)
                ],
                1,
            )
            rows[f"psnr_y_{kind}"].append(ps.mean(1))
            rows["psnr_y_last"].append(ps[:, -1])
            if kind == "render":
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
        res = {k: torch.cat(v).mean().item() for k, v in rows.items() if v}
        res["psnr_y"] = float(
            np.mean([res[k] for k in ("psnr_y_render", "psnr_y_video") if k in res])
        )
        return res

    train_loop(cfg, model, step_fn, val_fn, rank, world, device, loader)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
