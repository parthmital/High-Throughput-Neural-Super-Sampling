"""Dataset discovery, filtering, splitting and patch extraction for the Kaggle SR notebook."""

import hashlib
import os
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}
# Path tokens that mark degraded copies (LR, blurred, compressed) or non-colour passes.
LOW_QUALITY_TOKEN = re.compile(
    r"^(lr\d*|lq|low|lowres|bicubic|bix\d|bdx\d|lrbi|lrbd|x\d|blur|blurred|noisy|compressed|jpeg)$"
)
NON_COLOUR_TOKENS = {
    "occlusions",
    "occlusion",
    "invalid",
    "flow",
    "viz",
    "mask",
    "masks",
    "depth",
    "albedo",
    "shading",
    "disparities",
    "segmentation",
    "camdata",
    "edges",
    "normals",
}
# Benchmark files from SelfExSR-style mirrors also contain other methods' outputs.
METHOD_TOKENS = {
    "glasner",
    "scsr",
    "selfexsr",
    "kim",
    "srcnn",
    "aplus",
    "a+",
    "nearest",
}
HR_TOKENS = {"hr", "gt", "original", "groundtruth"}


def tokens(text):
    return [t for t in re.split(r"[_\-\s.]+", text.lower()) if t]


def path_tokens(path, root):
    rel = Path(path).relative_to(root)
    parts = list(rel.parts[:-1]) + [Path(rel.parts[-1]).stem]
    return [t for part in parts for t in tokens(part)]


def is_clean_colour(path, root):
    toks = path_tokens(path, root)
    return not any(
        LOW_QUALITY_TOKEN.match(t) or t in NON_COLOUR_TOKENS or t in METHOD_TOKENS
        for t in toks
    )


def has_hr_marker(path, root):
    return any(t in HR_TOKENS for t in path_tokens(path, root))


def find_dataset_root(input_root, owner_slug):
    """Return the mount folder of a Kaggle dataset under either supported layout, or None."""
    owner, slug = owner_slug.split("/")
    input_root = Path(input_root)
    for candidate in (
        input_root / slug,
        input_root / "datasets" / owner / slug,
        input_root / "datasets" / slug,
    ):
        if candidate.is_dir():
            return candidate
    for depth_one in input_root.iterdir() if input_root.is_dir() else []:
        if depth_one.is_dir():
            for depth_two in [depth_one] + [
                p for p in depth_one.iterdir() if p.is_dir()
            ]:
                if depth_two.name == slug:
                    return depth_two
    return None


def sampled_walk(root, max_files, seed):
    """List image files, expanding a random directory from the frontier each step.

    Random frontier expansion samples many sequences early instead of exhausting one branch,
    so a capped listing stays diverse while touching only a fraction of a large mount.
    """
    rng = random.Random(seed)
    frontier, files, dirs_listed = [str(root)], [], 0
    while frontier and len(files) < max_files:
        i = rng.randrange(len(frontier))
        frontier[i], frontier[-1] = frontier[-1], frontier[i]
        current = frontier.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        dirs_listed += 1
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                frontier.append(entry.path)
            elif os.path.splitext(entry.name)[1].lower() in IMG_EXTS:
                files.append(entry.path)
    return sorted(files), dirs_listed, len(frontier) == 0


def sequence_name(path):
    """Sequence folder of a video frame, skipping a trailing HR/GT marker folder (calendar/GT/x.png)."""
    parent = Path(path).parent
    return parent.parent.name if parent.name.lower() in HR_TOKENS else parent.name


def canonical_stem(path):
    """Image identity across SelfExSR-style copies: img_001_SRF_3_HR -> img_001."""
    stem = re.sub(r"_srf_\d+", "", Path(path).stem.lower())
    return re.sub(r"_(hr|gt)$", "", stem)


def group_key(path, level):
    """Split unit: 0 = the file itself, 1 = parent folder (sequence), 2 = grandparent (video)."""
    p = Path(path)
    return p.stem if level == 0 else p.parents[level - 1].name


def in_holdout(dataset, key, permille):
    digest = hashlib.md5(f"{dataset}/{key}".encode()).hexdigest()
    return int(digest, 16) % 1000 < permille


def interleave_by_group(files, groups, seed):
    """Round-robin over shuffled groups so a capped selection spans many sequences."""
    rng = random.Random(seed)
    buckets = {}
    for f, g in zip(files, groups):
        buckets.setdefault(g, []).append(f)
    order = list(buckets)
    rng.shuffle(order)
    for g in order:
        rng.shuffle(buckets[g])
    out, depth = [], 0
    while len(out) < len(files):
        for g in order:
            if depth < len(buckets[g]):
                out.append(buckets[g][depth])
        depth += 1
    return out


def image_size(path):
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def extract_patches(task):
    """Worker: decode one image and return up to n random textured uint8 patches of size p."""
    path, n, p, seed, min_std = task
    try:
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))
    except Exception:
        return path, None
    h, w = img.shape[:2]
    if h < p or w < p:
        return path, None
    rng = np.random.default_rng(seed)
    patches = []
    for _ in range(n * 3):
        y, x = int(rng.integers(0, h - p + 1)), int(rng.integers(0, w - p + 1))
        patch = img[y : y + p, x : x + p]
        if patch.mean(axis=2).std() >= min_std:
            patches.append(patch)
            if len(patches) == n:
                break
    return path, np.stack(patches) if patches else None


def load_rgb_modcrop(path, scale):
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"))
    h, w = img.shape[:2]
    return img[: h - h % scale, : w - w % scale]
