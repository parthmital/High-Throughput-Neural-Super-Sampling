"""DDP training of the frame generator (interpolation at any in-between time, input-aware extrapolation,
UI-aware).

The generator's IFNet starts from the pre-trained RIFE 4.26 weights (the high tier loads them whole, smaller
tiers inherit leading channel slices), so training is fine-tuning: it adds game content, UI handling,
extrapolation and the use of engine data to a model that already estimates motion from colour alone.

Each step mixes synthetic layered scenes rendered on the GPU (exact intermediate flows, disocclusions, thin
geometry, particles, HUD / subtitles / menus, scene cuts, rapid camera reversals, random in-between times) with
cached real quads of video frames (teacher optical flow from RAFT; some samples are 3x multi-frame generation
from frames k and k+3). Availability of engine motion vectors, depth and the camera prior is randomly dropped,
also on real video, so one model serves engines that provide them and plain video that does not.

Loss: Charbonnier + census + LPIPS on the generated frame, flow supervision (exact flows for synthetic data, RAFT
flows for video at a quarter weight), binary cross-entropy on the UI / static-overlay mask, and distillation of
flows and output toward a larger fine-tuned generator when one is given."""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nss_metrics import census_loss, charbonnier, load_lpips, psnr  # noqa: E402
from nss_models import build_fg, load_ifnet_weights  # noqa: E402
from nss_synth import fg_batch, fg_batch_from_frames, make_text_bank  # noqa: E402
from nss_train import emit, make_loader, setup_ddp, train_loop  # noqa: E402

KEYS = [
    "i0",
    "i1",
    "target",
    "tau",
    "extrap",
    "mv1",
    "mv1_valid",
    "depth1",
    "cam_prior",
    "flow_t0",
    "flow_t1",
    "valid_t0",
    "valid_t1",
    "speed_class",
    "flip",
    "cut",
    "menu",
    "ui_mask",
]
RAFT_FLOW_WEIGHT = (
    0.25  # RAFT labels on video are less reliable than the renderer's exact flows
)


def set_availability(b, synthetic, g, drop_mv, drop_depth=0.3, ablate_prior=False):
    """Which auxiliary inputs the generator may use. Synthetic data has exact motion, depth and camera; video
    has only estimated motion vectors. Each is withheld at random (drop_mv for motion vectors) so the model
    also works without them."""
    n, dev = b["i0"].shape[0], b["i0"].device
    keep_mv = (torch.rand(n, device=dev, generator=g) >= drop_mv).float()
    if synthetic:
        b["mv_avail"] = keep_mv
        b["depth_avail"] = (
            torch.rand(n, device=dev, generator=g) >= drop_depth
        ).float()
        b["cam_avail"] = (
            b["extrap"] * (torch.rand(n, device=dev, generator=g) >= 0.2).float()
        )
        b["flow_w"] = torch.ones(n, device=dev)
    else:
        b["mv_avail"] = (b["mv1_valid"].flatten(1).amax(1) > 0).float() * keep_mv
        b["depth_avail"] = torch.zeros(n, device=dev)
        b["cam_avail"] = torch.zeros(n, device=dev)
        b["flow_w"] = torch.full((n,), RAFT_FLOW_WEIGHT, device=dev)
    if ablate_prior:
        b["mv_avail"], b["cam_avail"] = torch.zeros(n, device=dev), torch.zeros(
            n, device=dev
        )
    return b


def plain(b):
    """The same batch without any engine data (ordinary video)."""
    z = torch.zeros_like(b["mv_avail"])
    return dict(b, mv_avail=z, depth_avail=z, cam_avail=z)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.load(open(parser.parse_args().config))
    rank, world, device = setup_ddp()
    torch.manual_seed(cfg["seed"] + rank)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"] * 100 + rank)
    P = cfg["patch"]
    log = lambda m: emit("LOG", {"msg": m}) if rank == 0 else None  # noqa: E731
    model = build_fg(cfg["tier"])
    if cfg.get("rife_weights"):
        rep = load_ifnet_weights(
            model.ifnet, torch.load(cfg["rife_weights"], weights_only=True)
        )
        log(f"IFNet initialised from RIFE 4.26: {rep}")
    model = model.to(device)
    teacher = None
    if cfg.get("teacher_ckpt"):
        teacher = build_fg("teacher").to(device)
        teacher.load_state_dict(
            torch.load(cfg["teacher_ckpt"], map_location=device)["ema"]
        )
        teacher.eval().requires_grad_(False)
    lpips_fn = load_lpips(device, log=log) if cfg["w_lpips"] > 0 else None
    tiles = None
    if cfg.get("ui_training", True):
        tiles = (
            torch.from_numpy(make_text_bank(256, cfg["seed"], height=28, width=160))
            .to(device)
            .float()
            / 255.0
        )
    caches = {k: v for k, v in cfg["caches"].items() if v["count"] > 0}
    b_real = int(round(cfg["batch_per_gpu"] * cfg["p_real"])) if "quad" in caches else 0
    sizes = {"still": cfg["batch_per_gpu"] - b_real, "quad": b_real}
    loader = make_loader(
        caches, sizes, cfg["seed"] * 1000 + rank * 10**7, cfg["workers_per_rank"]
    )
    mode_p = {"interp": 0.0, "extrap": 1.0}.get(cfg.get("mode"), cfg["p_extrap"])
    ablate = cfg.get("ablate_prior", False)

    def make_batch(raw, g):
        parts = []
        if "still/rgb" in raw and raw["still/rgb"].shape[0] > 0:
            b = fg_batch(
                raw["still/rgb"],
                P,
                g,
                tiles=tiles,
                p_extrap=mode_p,
                p_ui=cfg["p_ui"],
                p_cut=cfg["p_cut"],
                p_menu=cfg["p_menu"],
                p_flip=cfg["p_flip"],
                p_tau=cfg["p_tau"],
            )
            parts.append(
                set_availability(
                    {k: b[k] for k in KEYS},
                    True,
                    g,
                    cfg["p_drop_mv"],
                    ablate_prior=ablate,
                )
            )
        if "quad/rgb" in raw and raw["quad/rgb"].shape[0] > 0:
            b = fg_batch_from_frames(
                raw["quad/rgb"],
                raw["quad/flows"],
                raw["quad/meta"][:, 0].float(),
                g,
                tiles=tiles,
                p_extrap=mode_p,
                p_ui=cfg["p_ui"],
                p_mfg=cfg["p_mfg"],
            )
            parts.append(
                set_availability(
                    {k: b[k] for k in KEYS},
                    False,
                    g,
                    cfg["p_drop_mv_real"],
                    ablate_prior=ablate,
                )
            )
        return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}

    def losses(out, aux, b, t_out=None, t_aux=None):
        tgt = b["target"]
        l_pix = charbonnier(out, tgt)
        l_cen = census_loss(out, tgt)
        l_lp = (
            lpips_fn(out.clamp(0, 1), tgt).mean()
            if lpips_fn is not None
            else out.new_zeros(())
        )
        fw = b["flow_w"].view(-1, 1, 1, 1)
        v0, v1 = b["valid_t0"] * fw, b["valid_t1"] * fw
        l_flow = ((aux["flow0"] - b["flow_t0"]).abs() * v0).sum() / (
            b["valid_t0"].sum() * 2 + 1
        ) + ((aux["flow1"] - b["flow_t1"]).abs() * v1).sum() / (
            b["valid_t1"].sum() * 2 + 1
        )
        ui_t = (b["ui_mask"] > 0.5).float()
        l_ui = F.binary_cross_entropy_with_logits(
            aux["ui_logit"], ui_t, pos_weight=torch.tensor(4.0, device=out.device)
        )
        l_kd = out.new_zeros(())
        if t_out is not None:
            l_kd = (out - t_out).abs().mean() + 0.01 * (
                (aux["flow0"] - t_aux["flow0"]).abs().mean()
                + (aux["flow1"] - t_aux["flow1"]).abs().mean()
            )
        total = (
            l_pix
            + cfg["w_census"] * l_cen
            + cfg["w_lpips"] * l_lp
            + cfg["w_flow"] * l_flow
            + cfg["w_ui"] * l_ui
            + cfg["w_distill"] * l_kd
        )
        return total, {
            "loss": total.detach(),
            "pix": l_pix.detach(),
            "census": l_cen.detach(),
            "flow": l_flow.detach(),
            "ui": l_ui.detach(),
            "distill": l_kd.detach(),
        }

    def step_fn(net, raw, step):
        b = make_batch(raw, gen)
        with torch.autocast("cuda", dtype=torch.float16):
            out, aux = net(b)
            t_out = t_aux = None
            if teacher is not None:
                with torch.no_grad():
                    t_out, t_aux = teacher(b)
        loss, logs = losses(
            out.float(),
            {k: v.float() for k, v in aux.items()},
            b,
            t_out.float() if t_out is not None else None,
            {k: v.float() for k, v in t_aux.items()} if t_aux is not None else None,
        )
        return loss, logs, b["i0"].shape[0]

    val_batches = []
    if rank == 0:
        vg = torch.Generator(device=device)
        vg.manual_seed(54321)
        vc = cfg["val_caches"]
        for k in range(cfg["val_batches"]):
            raw = {}
            for name in ("still", "quad"):
                if vc.get(name, {}).get("count", 0) > 0:
                    arrs = {
                        key: np.load(p, mmap_mode="r")
                        for key, p in vc[name]["arrays"].items()
                    }
                    idx = np.sort(
                        np.arange(k * cfg["val_bs"], (k + 1) * cfg["val_bs"])
                        % vc[name]["count"]
                    )
                    for key, a in arrs.items():
                        raw[f"{name}/{key}"] = torch.from_numpy(
                            np.ascontiguousarray(a[idx])
                        ).to(device)
            if raw:
                val_batches.append(
                    {k2: v.cpu() for k2, v in make_batch(raw, vg).items()}
                )

    @torch.no_grad()
    def val_fn(ema_model):
        """psnr_engine: engine data as drawn for the batch; psnr_plain: the same samples without any engine
        data; psnr (the selection metric) is their mean."""
        rows = {
            k: []
            for k in (
                "psnr_engine",
                "psnr_plain",
                "psnr_interp",
                "psnr_extrap",
                "psnr_ui",
            )
        }
        for vb in val_batches:
            b = {k: v.to(device) for k, v in vb.items()}
            for key, bb in (("psnr_engine", b), ("psnr_plain", plain(b))):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = ema_model(bb)[0].float().clamp(0, 1)
                p = psnr(out, b["target"])
                rows[key].append(p)
                if key == "psnr_plain":
                    continue
                rows["psnr_interp"].append(p[b["extrap"] < 0.5])
                rows["psnr_extrap"].append(p[b["extrap"] > 0.5])
                has_ui = b["ui_mask"].flatten(1).amax(1) > 0.5
                if has_ui.any():
                    rows["psnr_ui"].append(
                        psnr(
                            out[has_ui],
                            b["target"][has_ui],
                            mask=(b["ui_mask"][has_ui] > 0.5),
                        )
                    )
        res = {
            k: torch.cat(v).mean().item()
            for k, v in rows.items()
            if v and torch.cat(v).numel()
        }
        res["psnr"] = 0.5 * (res["psnr_engine"] + res["psnr_plain"])
        return res

    train_loop(cfg, model, step_fn, val_fn, rank, world, device, loader)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
