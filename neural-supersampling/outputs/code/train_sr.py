"""DDP training entry point for Rep-TNSR, launched by the notebook through torch.distributed.run.

Rank 0 prints machine-readable lines that the notebook turns into a live progress bar and plots:
PROGRESS {json} every few seconds, ROUND {json} after each validation round, DONE {json} at the end.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_lib import (
    bicubic_down,
    bicubic_up,
    build_model,
    charbonnier,
    edge_loss,
    psnr,  # noqa: E402
    quantize,
    random_dihedral,
    rgb_to_y,
    ssim,
)


class PatchCache(Dataset):
    """Returns whole batches of uint8 patches from a memory-mapped .npy cache, opened lazily per worker."""

    def __init__(self, path, count):
        self.path, self.count, self.data = path, count, None

    def __len__(self):
        return self.count

    def __getitem__(self, indices):
        if self.data is None:
            self.data = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(
            np.ascontiguousarray(self.data[np.sort(np.asarray(indices))])
        )


def lr_at(progress, cfg):
    """Linear warm-up then cosine decay, driven by the fraction of the wall-clock budget used."""
    if progress < cfg["warmup_frac"]:
        return cfg["lr"] * max(progress / cfg["warmup_frac"], 0.01)
    t = min(1.0, (progress - cfg["warmup_frac"]) / (1.0 - cfg["warmup_frac"]))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (
        1.0 + math.cos(math.pi * t)
    )


def to_hr(batch, device, crop):
    """uint8 NHWC batch -> random crop (one offset per batch) -> per-sample dihedral -> float NCHW."""
    x = batch.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
    size = x.shape[-1]
    if crop < size:
        oy, ox = np.random.randint(0, size - crop + 1, size=2)
        x = x[..., oy : oy + crop, ox : ox + crop]
    return random_dihedral(x).contiguous(memory_format=torch.channels_last)


def gpu_utilisation():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return [float(v) for v in out.split()]
    except Exception:
        return []


@torch.no_grad()
def validate(model, val_patches, device, scale, batch):
    model.eval()
    rows = {"psnr_y": [], "ssim_y": [], "bicubic_psnr_y": []}
    for i in range(0, val_patches.shape[0], batch):
        hr = (
            torch.from_numpy(val_patches[i : i + batch])
            .to(device)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255.0)
        )
        lr = bicubic_down(hr, scale)
        with torch.autocast("cuda", dtype=torch.float16):
            sr = model(lr.contiguous(memory_format=torch.channels_last))
        sr_y, hr_y = rgb_to_y(quantize(sr.float())), rgb_to_y(hr)
        rows["psnr_y"].append(psnr(sr_y, hr_y, scale))
        rows["ssim_y"].append(ssim(sr_y, hr_y, scale))
        rows["bicubic_psnr_y"].append(
            psnr(rgb_to_y(quantize(bicubic_up(lr, scale))), hr_y, scale)
        )
    return {k: torch.cat(v).mean().item() for k, v in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.load(open(parser.parse_args().config))

    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True

    model = (
        build_model(cfg["tier"], cfg["scale"])
        .to(device)
        .to(memory_format=torch.channels_last)
    )
    ema = copy.deepcopy(model).eval().requires_grad_(False) if rank == 0 else None
    ddp = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    val_patches = np.load(cfg["val_cache"])[: cfg["val_count"]] if rank == 0 else None

    sampler = DistributedSampler(
        range(cfg["train_count"]),
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=cfg["seed"],
        drop_last=True,
    )
    loader = DataLoader(
        PatchCache(cfg["train_cache"], cfg["train_count"]),
        sampler=BatchSampler(sampler, cfg["batch_per_gpu"], drop_last=True),
        batch_size=None,
        num_workers=cfg["workers_per_rank"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    start = time.time()
    last_val = last_progress = round_start = start
    sync = torch.zeros(2 + world, device=device)
    train_acc = torch.zeros(2, device=device)
    step = epoch = round_steps = round_samples = 0
    best, history, done, lr = -1.0, [], False, lr_at(0.0, cfg)
    while not done:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step % cfg["sync_every"] == 0:
                # One collective agrees on time, validation and per-GPU memory, so every rank
                # uses the same learning rate and stops at the same step.
                sync.zero_()
                if rank == 0:
                    elapsed = time.time() - start
                    sync[0] = elapsed
                    sync[1] = float(
                        elapsed - (last_val - start) >= cfg["val_interval_s"]
                        or elapsed >= cfg["train_seconds"]
                    )
                sync[2 + rank] = torch.cuda.max_memory_allocated(device) / 2**30
                dist.all_reduce(sync)
                elapsed, progress = sync[0].item(), min(
                    1.0, sync[0].item() / cfg["train_seconds"]
                )
                mem_gb = [round(v, 2) for v in sync[2:].tolist()]
                lr = lr_at(progress, cfg)
                for group in opt.param_groups:
                    group["lr"] = lr
                if sync[1].item() > 0.5 and round_steps > 0:
                    dist.all_reduce(train_acc)
                    if rank == 0:
                        now = time.time()
                        row = {
                            "round": len(history),
                            "elapsed_min": round(elapsed / 60, 2),
                            "step": step,
                            "epoch": epoch,
                            "train_loss": train_acc[0].item() / (round_steps * world),
                            "train_psnr_rgb": train_acc[1].item()
                            / (round_steps * world),
                            "lr": lr,
                            "img_per_s": round_samples * world / (now - round_start),
                            "mem_gb": mem_gb,
                            "util_pct": gpu_utilisation(),
                        }
                        row.update(
                            {
                                "val_" + k: v
                                for k, v in validate(
                                    ema,
                                    val_patches,
                                    device,
                                    cfg["scale"],
                                    cfg["val_batch"],
                                ).items()
                            }
                        )
                        history.append(row)
                        state = {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": opt.state_dict(),
                            "scaler": scaler.state_dict(),
                            "step": step,
                            "elapsed_s": elapsed,
                            "cfg": cfg,
                        }
                        torch.save(
                            state, os.path.join(cfg["weights_dir"], "sr_last.pt")
                        )
                        if row["val_psnr_y"] > best:
                            best = row["val_psnr_y"]
                            torch.save(
                                {
                                    "ema": ema.state_dict(),
                                    "round": row["round"],
                                    "val_psnr_y": best,
                                    "cfg": cfg,
                                },
                                os.path.join(cfg["weights_dir"], "sr_best_ema.pt"),
                            )
                        row["best_val_psnr_y"] = best
                        print("ROUND " + json.dumps(row), flush=True)
                        last_val = time.time()
                    dist.barrier()
                    train_acc.zero_()
                    round_steps = round_samples = 0
                    round_start = time.time()
                if progress >= 1.0:
                    done = True
                    break
                if (
                    rank == 0
                    and time.time() - last_progress >= cfg["progress_interval_s"]
                ):
                    print(
                        "PROGRESS "
                        + json.dumps(
                            {
                                "progress": progress,
                                "step": step,
                                "elapsed_min": round(elapsed / 60, 2),
                                "lr": lr,
                                "loss": train_acc[0].item() / max(round_steps, 1),
                                "img_per_s": round_samples
                                * world
                                / max(time.time() - round_start, 1e-6),
                                "mem_gb": mem_gb,
                            }
                        ),
                        flush=True,
                    )
                    last_progress = time.time()

            hr = to_hr(batch, device, cfg["hr_patch"])
            lr_img = bicubic_down(hr, cfg["scale"]).contiguous(
                memory_format=torch.channels_last
            )
            with torch.autocast("cuda", dtype=torch.float16):
                sr = ddp(lr_img)
            sr = sr.float()
            loss = charbonnier(sr, hr) + cfg["w_edge"] * edge_loss(sr, hr)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if rank == 0:
                decay = min(cfg["ema_decay"], (1 + step) / (10 + step))
                with torch.no_grad():
                    for e, p in zip(ema.parameters(), model.parameters()):
                        e.lerp_(p, 1.0 - decay)
            with torch.no_grad():
                train_acc[0] += loss.detach()
                train_acc[1] += psnr(sr.detach().clamp(0, 1), hr).mean()
            step += 1
            round_steps += 1
            round_samples += hr.shape[0]
        epoch += 1

    if rank == 0:
        with open(os.path.join(cfg["metrics_dir"], "train_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(
            "DONE "
            + json.dumps(
                {
                    "steps": step,
                    "epochs": epoch,
                    "train_seconds": round(time.time() - start, 1),
                    "best_val_psnr_y": best,
                }
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
