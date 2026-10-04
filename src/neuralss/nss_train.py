"""Shared DDP training harness: process group setup, random-batch streaming from memory-mapped caches,
wall-clock learning-rate schedule, EMA, checkpointing and the machine-readable progress protocol
(PROGRESS / ROUND / DONE JSON lines on rank 0, parsed by the notebooks)."""

import copy
import json
import math
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank, world, local = (
            dist.get_rank(),
            dist.get_world_size(),
            int(os.environ["LOCAL_RANK"]),
        )
    else:
        rank, world, local = 0, 1, int(os.environ.get("CUDA_DEVICE", 0))
    torch.cuda.set_device(local)
    torch.backends.cudnn.benchmark = True
    return rank, world, torch.device("cuda", local)


def is_dist():
    return dist.is_available() and dist.is_initialized()


class BatchStream(Dataset):
    """Each item is a whole random batch: for every cache {name: {"arrays": {key: npy path}, "count": n}} and
    per-cache batch size, sorted random indices are read from the memory maps (opened lazily per worker).
    """

    def __init__(self, caches, sizes, seed, steps=10**9):
        self.caches, self.sizes, self.seed, self.steps, self.maps = (
            caches,
            sizes,
            seed,
            steps,
            None,
        )

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        if self.maps is None:
            self.maps = {
                name: {k: np.load(p, mmap_mode="r") for k, p in c["arrays"].items()}
                for name, c in self.caches.items()
            }
        rng = np.random.default_rng(self.seed + i * 7919)
        out = {}
        for name, c in self.caches.items():
            b = self.sizes.get(name, 0)
            if b <= 0 or c["count"] <= 0:
                continue
            idx = np.sort(rng.choice(c["count"], size=b, replace=c["count"] < b))
            for k, arr in self.maps[name].items():
                out[f"{name}/{k}"] = torch.from_numpy(np.ascontiguousarray(arr[idx]))
        return out


def make_loader(caches, sizes, seed, workers):
    return DataLoader(
        BatchStream(caches, sizes, seed),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def lr_at(progress, cfg):
    if progress < cfg["warmup_frac"]:
        return cfg["lr"] * max(progress / cfg["warmup_frac"], 0.01)
    t = min(1.0, (progress - cfg["warmup_frac"]) / (1.0 - cfg["warmup_frac"]))
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (
        1.0 + math.cos(math.pi * t)
    )


class EMA:
    def __init__(self, model, decay):
        self.model, self.decay, self.step = (
            copy.deepcopy(model).eval().requires_grad_(False),
            decay,
            0,
        )

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.model.parameters(), model.parameters()):
            e.lerp_(p.detach(), 1.0 - d)


def emit(kind, payload):
    print(f"{kind} " + json.dumps(payload, default=float), flush=True)


def train_loop(cfg, model, step_fn, val_fn, rank, world, device, loader):
    """Generic loop. step_fn(batch, step) -> (loss tensor, logs dict of floats); val_fn(ema_model) -> metrics dict
    (rank 0 only). The schedule follows the fraction of cfg['train_seconds'] already used, agreed across ranks.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    ddp = (
        DDP(
            model,
            device_ids=[device.index],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get("find_unused", False),
        )
        if world > 1
        else model
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.99),
        weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, cfg["ema_decay"]) if rank == 0 else None
    start = last_val = last_prog = time.time()
    sync = torch.zeros(3 + world, device=device)
    acc, n_acc, seen, round_start = {}, 0, 0, time.time()
    history, best, step, done = [], -1e9, 0, False
    it = iter(loader)
    while not done:
        if step % cfg["sync_every"] == 0:
            sync.zero_()
            if rank == 0:
                el = time.time() - start
                sync[0] = el
                sync[1] = float(
                    el - (last_val - start) >= cfg["val_interval_s"]
                    or el >= cfg["train_seconds"]
                )
            sync[3 + rank] = torch.cuda.max_memory_allocated(device) / 2**30
            if world > 1:
                dist.all_reduce(sync)
            el = sync[0].item()
            progress = min(1.0, el / cfg["train_seconds"])
            lr = lr_at(progress, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            mem = [round(v, 2) for v in sync[3:].tolist()]
            if sync[1].item() > 0.5 and n_acc > 0:
                if rank == 0:
                    row = {
                        "round": len(history),
                        "elapsed_min": round(el / 60, 2),
                        "step": step,
                        "lr": lr,
                        "samples_per_s": seen
                        * world
                        / max(time.time() - round_start, 1e-6),
                        "mem_gb": mem,
                        **{f"train_{k}": float(v) / n_acc for k, v in acc.items()},
                    }
                    row.update({f"val_{k}": v for k, v in val_fn(ema.model).items()})
                    history.append(row)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.model.state_dict(),
                            "step": step,
                            "cfg": cfg,
                        },
                        os.path.join(cfg["weights_dir"], f"{cfg['name']}_last.pt"),
                    )
                    score = row.get(f"val_{cfg['select_metric']}", -1e9)
                    if score > best:
                        best = score
                        torch.save(
                            {
                                "ema": ema.model.state_dict(),
                                "round": row["round"],
                                "score": best,
                                "cfg": cfg,
                            },
                            os.path.join(cfg["weights_dir"], f"{cfg['name']}_best.pt"),
                        )
                    row["best"] = best
                    emit("ROUND", row)
                    last_val = time.time()
                if world > 1:
                    dist.barrier()
                acc, n_acc, seen, round_start = {}, 0, 0, time.time()
            if progress >= 1.0:
                break
            if rank == 0 and time.time() - last_prog >= cfg["progress_interval_s"]:
                emit(
                    "PROGRESS",
                    {
                        "progress": progress,
                        "step": step,
                        "elapsed_min": round(el / 60, 2),
                        "lr": lr,
                        "loss": float(acc.get("loss", 0.0)) / max(n_acc, 1),
                        "samples_per_s": seen
                        * world
                        / max(time.time() - round_start, 1e-6),
                        "mem_gb": mem,
                    },
                )
                last_prog = time.time()
        batch = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
        loss, logs, n_samples = step_fn(ddp, batch, step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(opt)
        scaler.update()
        if ema is not None:
            ema.update(model)
        for k, v in logs.items():  # kept on the GPU; converted when a line is printed
            acc[k] = acc.get(k, 0.0) + (
                v.detach().float() if torch.is_tensor(v) else float(v)
            )
        n_acc += 1
        seen += n_samples
        step += 1
    if rank == 0:
        with open(
            os.path.join(cfg["metrics_dir"], f"{cfg['name']}_history.json"), "w"
        ) as f:
            json.dump(history, f, indent=2, default=float)
        emit(
            "DONE",
            {
                "steps": step,
                "train_seconds": round(time.time() - start, 1),
                "best": best,
            },
        )
    if world > 1:
        dist.barrier()
