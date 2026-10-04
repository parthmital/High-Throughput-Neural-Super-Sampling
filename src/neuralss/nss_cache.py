"""Training caches: memory-mapped uint8 still crops (textures for the synthetic renderer) and fixed-size
sequence windows with motion / validity / depth, plus RAFT teacher motion for windows without exact motion.

Arrays per window cache prefix:
  rgb (N, T, P, P, 3) uint8 | mv (N, T, P/2, P/2, 2) fp16 (P-grid pixels, frame -> previous frame)
  valid (N, T, P/2, P/2) uint8 | depth (N, T, P/2, P/2) fp16 | meta (N, 2) fp32 [has_depth, mv_quality]
Quad caches (frame generation) add flows (N, 4, P/2, P/2, 2) fp16 = teacher flows [2->0, 1->0, 1->2, 3->2].
"""

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

import nss_data as D
import nss_metrics as M

# down-scale range per source before cropping (detail density; removes compression noise from large frames)
SCALE_RANGES = {
    "div2k": (0.35, 0.9),
    "flickr2k": (0.35, 0.9),
    "game_stills": (0.5, 1.0),
    "gameir": (0.4, 0.8),
    "vimeo1080p": (0.35, 0.7),
    "reds": (0.5, 1.0),
    "vimeo_septuplet": (1.0, 1.0),
    "tartanair": (0.6, 1.0),
    "sintel": (0.6, 1.0),
    "vid4": (1.0, 1.0),
    "vimeo_triplet": (1.0, 1.0),
}


def items_for(manifest, split, sources=None, kinds=None, min_frames=1):
    df = manifest[manifest["split"] == split]
    if sources is not None:
        df = df[df["source"].isin(sources)]
    if kinds is not None:
        df = df[df["kind"].isin(kinds)]
    df = df[df["n_frames"] >= min_frames]
    return D.items_from_frame(df)


def still_tasks(items, data_root, capacity, S, crops_per_image, seed, min_std=8.0):
    """One decode task per (item, frame): stills directly, sequences contribute a few random frames."""
    rng = random.Random(seed)
    tasks = []
    for it in items:
        frames = (
            it["frames"]
            if it["n_frames"] == 1
            else rng.sample(it["frames"], min(3, it["n_frames"]))
        )
        for uri in frames:
            tasks.append(
                (
                    str(data_root),
                    uri,
                    crops_per_image,
                    S,
                    SCALE_RANGES.get(it["source"], (0.5, 1.0)),
                    rng.randrange(1 << 30),
                    min_std,
                )
            )
    rng.shuffle(tasks)
    need = math.ceil(capacity / crops_per_image) * 2
    return tasks[:need]


def build_still_cache(tasks, path, capacity, S, workers, deadline_s, log=print):
    from tqdm.auto import tqdm

    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(capacity, S, S, 3)
    )
    filled, failed, t0 = 0, 0, time.time()
    bar = tqdm(total=capacity, desc=f"Still crops -> {path.name}", unit="crop")
    with Pool(workers) as pool:
        for _, crops in pool.imap_unordered(D.still_crop, tasks, chunksize=2):
            if crops is None:
                failed += 1
                continue
            take = min(len(crops), capacity - filled)
            arr[filled : filled + take] = crops[:take]
            filled += take
            bar.update(take)
            if filled >= capacity or time.time() - t0 > deadline_s:
                break
    bar.close()
    arr.flush()
    del arr
    log(f"{path.name}: {filled} crops ({failed} decode failures)")
    return filled


def open_window_arrays(prefix, capacity, T, P, flows=False):
    half = P // 2
    spec = {
        "rgb": ((capacity, T, P, P, 3), np.uint8),
        "mv": ((capacity, T, half, half, 2), np.float16),
        "valid": ((capacity, T, half, half), np.uint8),
        "depth": ((capacity, T, half, half), np.float16),
        "meta": ((capacity, 2), np.float32),
    }
    if flows:
        spec["flows"] = ((capacity, 4, half, half, 2), np.float16)
    return {
        k: np.lib.format.open_memmap(
            f"{prefix}_{k}.npy", mode="w+", dtype=dt, shape=shape
        )
        for k, (shape, dt) in spec.items()
    }


def build_window_cache(
    items,
    data_root,
    prefix,
    capacity,
    T,
    P,
    workers,
    deadline_s,
    seed,
    flows=False,
    log=print,
):
    """Decode capacity windows of T frames into memory maps (round-robin over items so the cache spans many
    sequences). Returns (count, per-source counts, indices of windows that still need estimated motion).
    """
    from tqdm.auto import tqdm

    plans = D.plan_windows(items, T, int(capacity * 1.3), seed)
    tasks = [
        (
            str(data_root),
            it,
            start,
            T,
            P,
            SCALE_RANGES.get(it["source"], (0.5, 1.0)),
            seed * 7 + k,
        )
        for k, (it, start) in enumerate(plans)
    ]
    src_of = {seed * 7 + k: it["source"] for k, (it, _) in enumerate(plans)}
    arrs = open_window_arrays(prefix, capacity, T, P, flows=flows)
    filled, failed, need_raft, counts, t0 = 0, 0, [], {}, time.time()
    bar = tqdm(total=capacity, desc=f"Windows -> {prefix.name}", unit="win")
    with Pool(workers) as pool:
        for key, w in pool.imap_unordered(D.extract_window, tasks, chunksize=1):
            if w is None:
                failed += 1
                continue
            i = filled
            arrs["rgb"][i], arrs["mv"][i], arrs["valid"][i], arrs["depth"][i] = (
                w["rgb"],
                w["mv"],
                w["valid"],
                w["depth"],
            )
            arrs["meta"][i] = (
                (0.0, 0.0) if flows else (float(w["has_depth"]), float(w["mv_quality"]))
            )
            if w["mv_quality"] < 1.0 or flows:
                need_raft.append(i)
            counts[src_of[key]] = counts.get(src_of[key], 0) + 1
            filled += 1
            bar.update(1)
            if filled >= capacity or time.time() - t0 > deadline_s:
                break
    bar.close()
    for a in arrs.values():
        a.flush()
    log(f"{prefix.name}: {filled} windows ({failed} failures)")
    return filled, counts, need_raft


def fill_estimated_motion(
    prefix, indices, rafts, devices, batch=16, quads=False, log=print
):
    """RAFT teacher motion for cached windows: motion vectors frame -> previous frame with forward-backward
    validity (windows without exact motion), or the four frame-generation flows of each quad. Work is split
    across GPUs with one thread per GPU; each thread writes disjoint rows of the memory maps.
    """
    from tqdm.auto import tqdm

    if not indices or not rafts:
        if indices:
            log(
                f"{prefix.name}: no RAFT; {len(indices)} windows keep zero motion (mv_quality 0)"
            )
        return
    rgb = np.load(f"{prefix}_rgb.npy", mmap_mode="r")
    mv = np.load(f"{prefix}_mv.npy", mmap_mode="r+")
    valid = np.load(f"{prefix}_valid.npy", mmap_mode="r+")
    meta = np.load(f"{prefix}_meta.npy", mmap_mode="r+")
    fl = np.load(f"{prefix}_flows.npy", mmap_mode="r+") if quads else None
    chunks = [indices[i : i + batch] for i in range(0, len(indices), batch)]
    bar = tqdm(total=len(indices), desc=f"RAFT motion -> {prefix.name}", unit="win")

    def half(x):
        return F.avg_pool2d(x, 2)  # values stay in P-grid pixels

    def work(rank):
        dev, net = devices[rank], rafts[rank]
        for c in chunks[rank :: len(devices)]:
            idx = np.array(c)
            x = (
                torch.from_numpy(np.ascontiguousarray(rgb[idx]))
                .to(dev)
                .permute(0, 1, 4, 2, 3)
                .float()
                / 255.0
            )
            n, T = x.shape[:2]
            if quads:
                pairs = [(2, 0), (1, 0), (1, 2), (3, 2)]
                out = [half(M.raft_flow(net, x[:, a], x[:, b])) for a, b in pairs]
                fl[idx] = (
                    torch.stack(out, 1).permute(0, 1, 3, 4, 2).half().cpu().numpy()
                )
                meta[idx, 0] = 1.0  # quad caches: meta = [flows available, unused]
            else:
                m_all = np.zeros((n, T) + mv.shape[2:], np.float16)
                v_all = np.zeros((n, T) + valid.shape[2:], np.uint8)
                for t in range(1, T):
                    m, v = M.raft_mv_and_valid(net, x[:, t], x[:, t - 1])
                    m_all[:, t] = half(m).permute(0, 2, 3, 1).half().cpu().numpy()
                    v_all[:, t] = (half(v)[:, 0] > 0.5).to(torch.uint8).cpu().numpy()
                mv[idx], valid[idx] = m_all, v_all
                meta[idx, 1] = 0.5
            bar.update(len(c))

    with ThreadPoolExecutor(len(devices)) as pool:
        list(pool.map(work, range(len(devices))))
    bar.close()
    for a in (mv, valid, meta) + ((fl,) if quads else ()):
        a.flush()


def cache_spec(prefix, count, keys):
    return {"arrays": {k: f"{prefix}_{k}.npy" for k in keys}, "count": int(count)}
