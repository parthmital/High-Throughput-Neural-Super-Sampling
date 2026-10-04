"""Unified dataset layer: source registry (Kaggle + Hugging Face), discovery, filtering, materialisation of
Hugging Face subsets, sequence building, decoding of colour / depth / flow / masks, and window extraction.

URI scheme used in the manifest (so it stays valid across notebooks and machines):
  kaggle:<owner/slug>|<path inside the dataset>   resolved under /kaggle/input
  data:<relative path>                            resolved under the data root (materialised Hugging Face subsets)
  mp4:<relative path>#<frame index>               a frame of a video file under the data root
"""

import hashlib
import io
import json
import os
import random
import re
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".ppm"}
# Path tokens that mark degraded copies or non-colour passes; such files are never used as targets.
LOW_QUALITY_TOKEN = re.compile(
    r"^(lr\d*|lq|low|lowres|bicubic|bix\d|bdx\d|lrbi|lrbd|x\d|blur|blurred|noisy|compressed|jpeg)$"
)
NON_COLOUR_TOKENS = {
    "occlusions",
    "occlusion",
    "invalid",
    "flow",
    "flow_viz",
    "viz",
    "mask",
    "masks",
    "depth",
    "albedo",
    "shading",
    "disparities",
    "segmentation",
    "camdata",
    "camdata_left",
    "edges",
    "normals",
    "seg",
}
HR_TOKENS = {"hr", "gt", "original", "groundtruth"}
HF = "https://huggingface.co"

# ---------------------------------------------------------------------------------------------------------------
# Source registry. Every source states where it lives, what it provides and how it may be used.
# role: train sources are split into train/val by leakage group; test sources (or test parts) are pinned to test.
# ---------------------------------------------------------------------------------------------------------------
SOURCES = [
    {
        "name": "div2k",
        "host": "kaggle",
        "ref": "soumikrakshit/div2k-high-resolution-images",
        "kind": "still",
        "test_dirs": ["DIV2K_valid_HR"],
        "licence": "DIV2K: academic research use",
        "content": "photo",
    },
    {
        "name": "flickr2k",
        "host": "kaggle",
        "ref": "daehoyang/flickr2k",
        "kind": "still",
        "licence": "Flickr2K: research use (Flickr images, original licences)",
        "content": "photo",
    },
    {
        "name": "reds",
        "host": "kaggle",
        "ref": "amithkesavmrajagiri/reds-dataset",
        "kind": "video",
        "group_level": 1,
        "licence": "REDS: CC BY 4.0",
        "content": "video",
    },
    {
        "name": "vimeo_septuplet",
        "host": "kaggle",
        "ref": "wangsally/vimeo-90k-7",
        "kind": "video",
        "group_level": 2,
        "licence": "Vimeo-90K: research use (Vimeo uploads)",
        "content": "video",
    },
    {
        "name": "vimeo_triplet",
        "host": "kaggle",
        "ref": "chenshu123/vimeo-triplet",
        "kind": "video",
        "group_level": 2,
        "test_list": "tri_testlist.txt",
        "test_only": True,
        "list_cap": 6000,
        "max_items": 1000,
        "licence": "Vimeo-90K: research use",
        "content": "video",
    },
    {
        "name": "sintel",
        "host": "kaggle",
        "ref": "artemmmtry/mpi-sintel-dataset",
        "kind": "render_seq",
        "test_groups": ["market", "cave"],
        "licence": "MPI Sintel: CC BY 3.0 (Kaggle mirror)",
        "content": "rendered",
    },
    {
        "name": "vid4",
        "host": "kaggle",
        "ref": "uom200647r/vid4-dataset",
        "kind": "video",
        "group_level": 1,
        "test_only": True,
        "licence": "Vid4: research benchmark (Kaggle mirror Apache 2.0)",
        "content": "video",
    },
    {
        "name": "game_stills",
        "host": "hf",
        "repo": "ericphann/video-game-super-resolution",
        "kind": "still",
        "train_dir": "training-hr",
        "test_dir": "test-hr",
        "licence": "HF tag Apache 2.0 (game captures; check per use)",
        "content": "game",
    },
    {
        "name": "tartanair",
        "host": "hf",
        "repo": "theairlabcmu/tartanair",
        "kind": "render_seq",
        "train_envs": [
            "abandonedfactory",
            "amusement",
            "carwelding",
            "endofworld",
            "gascola",
            "hospital",
            "neighborhood",
            "oldtown",
            "seasonsforest",
            "soulcity",
            "westerndesert",
            "ocean",
        ],
        "test_envs": ["japanesealley", "seasidetown"],
        "licence": "TartanAir: BSD-3-Clause (HF tag and tools)",
        "content": "rendered",
    },
    {
        "name": "gameir",
        "host": "hf",
        "repo": "LLLebin/GameIR",
        "kind": "render_pairs",
        "train_tars": ["mini_dataset/train/GameIR-SR/GameIR-SR-000000.tar"],
        "test_tars": ["mini_dataset/test/GameIR-SR/GameIR-SR-000000.tar"],
        "licence": "GameIR: MIT",
        "content": "game",
    },
    {
        "name": "vimeo1080p",
        "host": "hf",
        "repo": "danjacobellis/vimeo1080p",
        "kind": "video_file",
        "train_shards": [
            "data/train-00000-of-00045.parquet",
            "data/train-00001-of-00045.parquet",
        ],
        "val_shards": ["data/validation-00000-of-00003.parquet"],
        "licence": "unspecified on HF (Vimeo uploads); research use",
        "content": "video",
    },
]


def source(name):
    return next(s for s in SOURCES if s["name"] == name)


# ---------------------------------------------------------------------------------------------------------------
# Path utilities (Kaggle mounts)
# ---------------------------------------------------------------------------------------------------------------
def tokens(text):
    return [t for t in re.split(r"[_\-\s.]+", text.lower()) if t]


def path_tokens(path, root):
    rel = Path(path).relative_to(root)
    parts = list(rel.parts[:-1]) + [Path(rel.parts[-1]).stem]
    return [t for part in parts for t in tokens(part)]


def is_clean_colour(path, root):
    return not any(
        LOW_QUALITY_TOKEN.match(t) or t in NON_COLOUR_TOKENS
        for t in path_tokens(path, root)
    )


def find_dataset_root(input_root, owner_slug):
    """Mount folder of a Kaggle dataset under either supported layout, or None."""
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


def sampled_walk(root, max_files, seed, exts=IMG_EXTS):
    """List files, expanding a random directory from the frontier each step, so a capped listing spans many
    sequences. Each directory is listed completely, so a listed sequence folder is never cut in half.
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
            elif os.path.splitext(entry.name)[1].lower() in exts:
                files.append(entry.path)
    return sorted(files), dirs_listed, len(frontier) == 0


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(path).lower())]


def stable_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest(), 16)


# ---------------------------------------------------------------------------------------------------------------
# Hugging Face access: plain HTTPS with retries and range requests (no local Hub cache, works for Xet-backed files)
# ---------------------------------------------------------------------------------------------------------------
def _headers():
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_url(repo, path):
    return f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"


def http_get(url, headers=None, retries=4, timeout=120):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**_headers(), **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), r.headers
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def http_download(url, dest, retries=4):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def hf_list(repo, path=""):
    """All entries under a dataset folder (follows pagination)."""
    url = f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}"
    out = []
    while url:
        body, headers = http_get(url)
        out.extend(json.loads(body))
        link = headers.get("Link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return out


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, so zip members can be read without downloading archives."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD", headers=_headers())
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        data, _ = http_get(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}
        )
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))


# ---------------------------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------------------------
class Resolver:
    """Maps manifest URIs to local paths."""

    def __init__(self, data_root, input_root="/kaggle/input"):
        self.data_root, self.input_root, self.roots = (
            Path(data_root),
            Path(input_root),
            {},
        )

    def kaggle_root(self, ref):
        if ref not in self.roots:
            self.roots[ref] = find_dataset_root(self.input_root, ref)
        return self.roots[ref]

    def path(self, uri):
        if uri.startswith("kaggle:"):
            ref, rel = uri[7:].split("|", 1)
            root = self.kaggle_root(ref)
            if root is None:
                raise FileNotFoundError(f"Kaggle dataset {ref} is not attached")
            return root / rel
        if uri.startswith("data:"):
            return self.data_root / uri[5:]
        if uri.startswith("mp4:"):
            return self.data_root / uri[4:].split("#")[0]
        raise ValueError(uri)


def kaggle_uri(ref, root, path):
    return f"kaggle:{ref}|{Path(path).relative_to(root).as_posix()}"


def read_video_frames(path, indices):
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames, want = {}, sorted(set(indices))
    if want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, want[0])
        idx = want[0]
        while idx <= want[-1]:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    cap.release()
    missing = [i for i in indices if i not in frames]
    if missing:
        raise IOError(f"could not decode frames {missing[:3]} of {path}")
    return [frames[i] for i in indices]


def video_frame_count(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def load_rgb(resolver, uri):
    if uri.startswith("mp4:"):
        return read_video_frames(resolver.path(uri), [int(uri.split("#")[1])])[0]
    with Image.open(resolver.path(uri)) as im:
        return np.asarray(im.convert("RGB"))


def load_rgb_many(resolver, uris):
    """Decode a list of frames, reading each video file once."""
    if (
        uris
        and all(u.startswith("mp4:") for u in uris)
        and len({u.split("#")[0] for u in uris}) == 1
    ):
        return read_video_frames(
            resolver.path(uris[0]), [int(u.split("#")[1]) for u in uris]
        )
    return [load_rgb(resolver, u) for u in uris]


def read_flo(path):
    """Middlebury .flo (Sintel): float32 (H, W, 2)."""
    with open(path, "rb") as f:
        magic = np.frombuffer(f.read(4), np.float32)[0]
        assert abs(magic - 202021.25) < 1e-3, f"bad .flo magic in {path}"
        w, h = np.frombuffer(f.read(8), np.int32)
        return np.frombuffer(f.read(), np.float32).reshape(h, w, 2).copy()


def load_flow(resolver, uri):
    p = resolver.path(uri)
    return read_flo(p) if p.suffix == ".flo" else np.load(p).astype(np.float32)


def load_valid(resolver, uri, kind):
    """Validity of the motion vector of each pixel (1 = the pixel is visible in the previous frame)."""
    p = resolver.path(uri)
    with Image.open(p) as im:
        a = np.asarray(im.convert("L"))
    return (
        (a < 128).astype(np.uint8)
        if kind == "occlusion"
        else (a >= 128).astype(np.uint8)
    )


def decode_carla_depth(rgba):
    """CARLA depth PNG (R + 256 G + 65536 B) / (2^24 - 1) * 1000 m."""
    rgb = rgba[..., :3].astype(np.float64)
    norm = (rgb[..., 0] + rgb[..., 1] * 256.0 + rgb[..., 2] * 65536.0) / (256.0**3 - 1)
    return (norm * 1000.0).astype(np.float32)


def load_depth(resolver, uri):
    p = resolver.path(uri)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float32)
    with Image.open(p) as im:
        return decode_carla_depth(np.asarray(im.convert("RGBA")))


def normalise_depth(d):
    """Unit-free inverse-depth style encoding in [0, 1] (1 = near): 1 / (1 + d / median)."""
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    med = np.nanmedian(d) if np.isfinite(d).any() else 1.0
    out = 1.0 / (1.0 + np.nan_to_num(d, nan=1e9) / max(med, 1e-6))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------
# Catalogue builders: each returns a list of item dicts (one per still image or per sequence)
# ---------------------------------------------------------------------------------------------------------------
def _item(src, kind, group, frames, role, **extra):
    item_id = (
        f"{src['name']}/{group}/{hashlib.md5(frames[0].encode()).hexdigest()[:10]}"
    )
    return {
        "item_id": item_id,
        "source": src["name"],
        "kind": kind,
        "group": f"{src['name']}:{group}",
        "role": role,
        "frames": frames,
        "n_frames": len(frames),
        "content": src.get("content", ""),
        **extra,
    }


def catalogue_kaggle(src, input_root, cap, seed):
    """Stills and frame folders of a Kaggle-mounted source, with the degraded-copy filter."""
    root = find_dataset_root(input_root, src["ref"])
    if root is None:
        return [], {"found": False}
    files, dirs_listed, complete = sampled_walk(root, src.get("list_cap", cap), seed)
    kept = [f for f in files if is_clean_colour(f, root)]
    if src["kind"] in ("video",) and any(
        any(t in HR_TOKENS for t in path_tokens(f, root)) for f in kept
    ):
        kept = [f for f in kept if any(t in HR_TOKENS for t in path_tokens(f, root))]
    items = []
    if src["kind"] == "still":
        for f in kept:
            rel = Path(f).relative_to(root)
            role = (
                "test"
                if any(d in rel.parts for d in src.get("test_dirs", []))
                else "train"
            )
            items.append(
                _item(
                    src, "still", Path(f).stem, [kaggle_uri(src["ref"], root, f)], role
                )
            )
    elif src["kind"] == "video":
        test_set = set()
        if src.get("test_list"):
            lists = sorted(Path(root).rglob(src["test_list"]))
            if lists:
                test_set = {
                    line.strip()
                    for line in lists[0].read_text().splitlines()
                    if line.strip()
                }
        folders = {}
        for f in kept:
            folders.setdefault(str(Path(f).parent), []).append(f)
        for folder, frames in folders.items():
            frames = sorted(frames, key=natural_key)
            if len(frames) < 3:
                continue
            p = Path(folder)
            if p.name.lower() in HR_TOKENS:
                p = p.parent
            level = src.get("group_level", 1)
            group = p.name if level == 1 else p.parent.name
            clip_key = f"{p.parent.name}/{p.name}"
            if src.get("test_only"):
                if test_set and clip_key not in test_set:
                    continue
                role = "test"
            else:
                role = "train"
            items.append(
                _item(
                    src,
                    "seq",
                    group,
                    [kaggle_uri(src["ref"], root, f) for f in frames],
                    role,
                    mv="none",
                )
            )
    elif src["name"] == "sintel":
        items = catalogue_sintel(src, root)
    if src.get("max_items") and len(items) > src["max_items"]:
        items = random.Random(seed).sample(items, src["max_items"])
    return items, {
        "found": True,
        "listed": len(files),
        "kept_files": len(kept),
        "dirs_listed": dirs_listed,
        "listing_complete": complete,
    }


def catalogue_sintel(src, root):
    """Sintel training scenes with exact forward flow and occlusion masks. The leakage group is the scene prefix
    (alley, ambush, market, ...) because numbered scenes share sets and characters."""
    items = []
    for pass_dir in sorted(Path(root).rglob("training/clean")) + sorted(
        Path(root).rglob("training/final")
    ):
        train_root = pass_dir.parent
        for scene in sorted(p for p in pass_dir.iterdir() if p.is_dir()):
            frames = sorted(scene.glob("*.png"), key=natural_key)
            flows = [
                train_root / "flow" / scene.name / (f.stem + ".flo") for f in frames
            ]
            occ = [train_root / "occlusions" / scene.name / f.name for f in frames]
            if len(frames) < 4 or not flows[0].exists():
                continue
            prefix = scene.name.split("_")[0]
            role = "test" if prefix in src.get("test_groups", []) else "train"
            n = len(frames) - 1  # the last frame has no forward flow
            items.append(
                _item(
                    src,
                    "seq",
                    prefix,
                    [kaggle_uri(src["ref"], root, f) for f in frames[:n]],
                    role,
                    mv="exact_forward",
                    flows=[kaggle_uri(src["ref"], root, f) for f in flows[:n]],
                    valid=[kaggle_uri(src["ref"], root, f) for f in occ[:n]],
                    valid_kind="occlusion",
                    variant=pass_dir.name,
                )
            )
    return items


# ---------------------------------------------------------------------------------------------------------------
# Hugging Face materialisation (download only what is needed into the data root)
# ---------------------------------------------------------------------------------------------------------------
def materialise_game_stills(src, data_root, n_train, n_test, seed, log=print):
    rng = random.Random(seed)
    items = []
    for role, folder, n in (
        ("train", src["train_dir"], n_train),
        ("test", src["test_dir"], n_test),
    ):
        names = sorted(
            e["path"]
            for e in hf_list(src["repo"], folder)
            if e["type"] == "file" and e["path"].endswith(".png")
        )
        chosen = sorted(rng.sample(names, min(n, len(names))))
        dests = [Path(data_root) / "game_stills" / p for p in chosen]
        with ThreadPoolExecutor(8) as pool:
            list(
                pool.map(
                    lambda pd: http_download(hf_url(src["repo"], pd[0]), pd[1]),
                    zip(chosen, dests),
                )
            )
        for p in chosen:
            items.append(
                _item(src, "still", Path(p).stem, [f"data:game_stills/{p}"], role)
            )
        log(f"game_stills {role}: {len(chosen)} of {len(names)} images")
    return items


def _tartanair_env(src, data_root, env, role, frames_per_env, seq_len, difficulty):
    """One environment: read a contiguous run of frames of the first trajectory from the remote zips (range
    requests only) and store RGB PNG, depth (fp16 npy), forward flow (fp16 npy) and validity (PNG, 255 = valid).
    TartanAir mask values: 0 = valid correspondence, non-zero = occluded or out of view.
    """
    base = f"{env}/{difficulty}"
    zips = {
        k: remote_zip(hf_url(src["repo"], f"{base}/{k}.zip"))
        for k in ("image_left", "depth_left", "flow_flow", "flow_mask")
    }
    trajs = sorted(
        {n.split("/")[2] for n in zips["image_left"].namelist() if n.count("/") >= 4}
    )
    traj = trajs[0]
    names = {
        k: {
            Path(n).name: n
            for n in z.namelist()
            if f"/{traj}/" in n and not n.endswith("/")
        }
        for k, z in zips.items()
    }
    frame_ids = sorted(
        int(n[:6]) for n in names["image_left"] if n.endswith("_left.png")
    )
    start = (
        frame_ids[len(frame_ids) // 4]
        if len(frame_ids) > 2 * frames_per_env
        else frame_ids[0]
    )
    run = [i for i in frame_ids if start <= i < start + frames_per_env]
    out_dir = Path(data_root) / "tartanair" / env / traj
    for sub in ("image", "depth", "flow"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    have = []
    for i in run:
        fid, nid = f"{i:06d}", f"{i + 1:06d}"
        f_key, m_key = f"{fid}_{nid}_flow.npy", f"{fid}_{nid}_mask.npy"
        if f_key not in names["flow_flow"] or m_key not in names["flow_mask"]:
            break  # keep the run contiguous
        if not (out_dir / "flow" / f"{fid}_valid.png").exists():
            (out_dir / "image" / f"{fid}.png").write_bytes(
                zips["image_left"].read(names["image_left"][f"{fid}_left.png"])
            )
            d = np.load(
                io.BytesIO(
                    zips["depth_left"].read(
                        names["depth_left"][f"{fid}_left_depth.npy"]
                    )
                )
            )
            np.save(
                out_dir / "depth" / f"{fid}.npy", np.minimum(d, 1e4).astype(np.float16)
            )
            fl = np.load(io.BytesIO(zips["flow_flow"].read(names["flow_flow"][f_key])))
            np.save(out_dir / "flow" / f"{fid}.npy", fl.astype(np.float16))
            m = np.load(io.BytesIO(zips["flow_mask"].read(names["flow_mask"][m_key])))
            Image.fromarray(np.where(m == 0, 255, 0).astype(np.uint8)).save(
                out_dir / "flow" / f"{fid}_valid.png"
            )
        have.append(i)
    items, rel = [], f"tartanair/{env}/{traj}"
    for k in range(0, len(have) - seq_len + 1, seq_len):
        seg = have[k : k + seq_len]
        items.append(
            _item(
                src,
                "seq",
                env,
                [f"data:{rel}/image/{i:06d}.png" for i in seg],
                role,
                mv="exact_forward",
                flows=[f"data:{rel}/flow/{i:06d}.npy" for i in seg],
                valid=[f"data:{rel}/flow/{i:06d}_valid.png" for i in seg],
                valid_kind="valid",
                depth=[f"data:{rel}/depth/{i:06d}.npy" for i in seg],
            )
        )
    return items, len(have), traj


def materialise_tartanair(
    src,
    data_root,
    frames_per_env,
    envs_train,
    envs_test,
    seq_len,
    difficulty="Easy",
    workers=6,
    log=print,
):
    """Environments are fetched in parallel threads (one zip set per thread; zip readers are not shared)."""
    jobs = [
        (env, "test" if env in envs_test else "train") for env in envs_train + envs_test
    ]

    def run(job):
        env, role = job
        try:
            return (
                env,
                role,
                _tartanair_env(
                    src, data_root, env, role, frames_per_env, seq_len, difficulty
                ),
                None,
            )
        except Exception as exc:
            return env, role, None, exc

    items = []
    with ThreadPoolExecutor(workers) as pool:
        for env, role, result, exc in pool.map(run, jobs):
            if exc is not None:
                log(f"tartanair {env}: skipped ({exc})")
                continue
            env_items, n, traj = result
            items.extend(env_items)
            log(
                f"tartanair {env} ({role}): {n} frames from {traj}, {len(env_items)} sequences"
            )
    return items


def materialise_gameir(src, data_root, max_clips_train, max_clips_test, log=print):
    """Stream GameIR-SR tars (sequential read, stops early) keeping native 720p and 1440p RGB and depth.
    Frames of a clip are sampled every 10 rendered frames; the leakage group is the town.
    """
    items = []
    for role, tars, cap in (
        ("train", src["train_tars"], max_clips_train),
        ("test", src["test_tars"], max_clips_test),
    ):
        clips = {}
        for tar_path in tars:
            try:
                stream = io.BufferedReader(
                    HttpRangeFile(hf_url(src["repo"], tar_path)), buffer_size=1 << 22
                )
                tf = tarfile.open(fileobj=stream, mode="r|")
            except Exception as exc:
                log(f"gameir {tar_path}: skipped ({exc})")
                continue
            for m in tf:
                parts = m.name.split("/")
                if (
                    len(parts) < 5
                    or not m.isfile()
                    or not (
                        m.name.endswith(".rgb.png") or m.name.endswith(".depth.png")
                    )
                ):
                    continue
                town, clip, res = parts[1], parts[2], parts[3]
                key = (town, clip)
                if key not in clips and len(clips) >= cap:
                    break
                clips.setdefault(key, set()).add(res)
                dst = Path(data_root) / "gameir" / town / clip / res / parts[4]
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(tf.extractfile(m).read())
            tf.close()
        for (town, clip), res in sorted(clips.items()):
            folder = Path(data_root) / "gameir" / town / clip
            hr = sorted((folder / "1440p").glob("*.rgb.png"), key=natural_key)
            lr = sorted((folder / "720p").glob("*.rgb.png"), key=natural_key)
            if len(hr) < 3 or len(lr) != len(hr):
                continue
            rel = f"gameir/{town}/{clip}"
            items.append(
                _item(
                    src,
                    "seq",
                    town.split("_")[-1],
                    [f"data:{rel}/1440p/{p.name}" for p in hr],
                    role,
                    mv="none",
                    native_lr=[f"data:{rel}/720p/{p.name}" for p in lr],
                    depth=[
                        f"data:{rel}/1440p/{p.name.replace('.rgb.png', '.depth.png')}"
                        for p in hr
                    ],
                    frame_stride=10,
                )
            )
        log(f"gameir {role}: {sum(1 for i in items if i['role'] == role)} clips")
    return items


def materialise_vimeo1080p(src, data_root, n_train, n_val, log=print):
    """Download parquet shards, write each embedded MP4 to disk and register it as a sequence of frames."""
    import pyarrow.parquet as pq

    items = []
    for role, shards, cap in (
        ("train", src["train_shards"], n_train),
        ("val_pinned", src["val_shards"], n_val),
    ):
        written = 0
        for shard in shards:
            if written >= cap:
                break
            local = http_download(
                hf_url(src["repo"], shard), Path(data_root) / "_downloads" / shard
            )
            table = pq.ParquetFile(local)
            for batch in table.iter_batches(
                batch_size=16, columns=["video_file", "video"]
            ):
                for name, blob in zip(
                    batch.column(0).to_pylist(), batch.column(1).to_pylist()
                ):
                    if written >= cap:
                        break
                    stem = Path(name).stem
                    rel = f"vimeo1080p/{role}/{stem}.mp4"
                    dst = Path(data_root) / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(blob)
                    n = video_frame_count(dst)
                    if n < 8:
                        dst.unlink()
                        continue
                    items.append(
                        _item(
                            src,
                            "seq",
                            stem,
                            [f"mp4:{rel}#{i}" for i in range(n)],
                            "test" if role == "val_pinned" else "train",
                            mv="none",
                        )
                    )
                    written += 1
            local.unlink()  # the shard is only a container
        log(f"vimeo1080p {role}: {written} videos")
    return items


# ---------------------------------------------------------------------------------------------------------------
# Item statistics for filtering (run in worker processes)
# ---------------------------------------------------------------------------------------------------------------
def keyframes(item):
    """Frames used to represent an item in deduplication: the still itself, the middle frame of a short clip,
    or first, middle and last frame of a long sequence (scene content can change within it).
    """
    frames, n = item["frames"], item["n_frames"]
    if n == 1:
        return [frames[0]]
    if n >= 24:
        return [frames[0], frames[n // 2], frames[-1]]
    return [frames[n // 2]]


def thumb_and_stats(task):
    """Worker: decode one frame, return a 256x256 thumbnail and quality statistics."""
    data_root, uri = task
    resolver = Resolver(data_root)
    try:
        img = load_rgb(resolver, uri)
    except Exception as exc:
        return uri, None, {"error": str(exc)[:200]}
    h, w = img.shape[:2]
    grey = img.mean(axis=2)
    rows = grey.mean(axis=1)
    dark = rows < 8
    letterbox = (
        (np.argmax(~dark) + np.argmax(~dark[::-1])) / h if (~dark).any() else 1.0
    )
    small = np.asarray(Image.fromarray(img).resize((256, 256), Image.BILINEAR))
    g = small.mean(axis=2).astype(np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    rg, yb = (
        small[..., 0].astype(np.float32) - small[..., 1],
        0.5 * (small[..., 0].astype(np.float32) + small[..., 1]) - small[..., 2],
    )
    stats = {
        "width": w,
        "height": h,
        "mean": float(grey.mean()),
        "std": float(grey.std()),
        "sharpness": float(lap.var()),
        "letterbox": float(letterbox),
        "colourfulness": float(
            np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
        ),
    }
    return uri, small, stats


# ---------------------------------------------------------------------------------------------------------------
# Window extraction for training caches (run in worker processes)
# ---------------------------------------------------------------------------------------------------------------
def _resize(a, size, mode):
    """Resize an array to size = (W, H). Modes: 'image' (uint8 HxWx3, Lanczos), 'nearest' (uint8 HxW),
    'depth' (float HxW, bilinear), 'flow' (float HxWx2, bilinear, vectors rescaled to the new pixel units).
    """
    h, w = a.shape[:2]
    W, H = size
    if (W, H) == (w, h):
        return a
    if mode == "image":
        return np.asarray(Image.fromarray(a).resize((W, H), Image.LANCZOS))
    if mode == "nearest":
        return np.asarray(Image.fromarray(a).resize((W, H), Image.NEAREST))
    if mode == "depth":
        return np.asarray(
            Image.fromarray(a.astype(np.float32), mode="F").resize(
                (W, H), Image.BILINEAR
            )
        )
    if mode == "flow":
        chans = [
            np.asarray(
                Image.fromarray(a[..., c].astype(np.float32), mode="F").resize(
                    (W, H), Image.BILINEAR
                )
            )
            for c in range(2)
        ]
        return np.stack(chans, -1) * np.array([W / w, H / h], np.float32)
    raise ValueError(mode)


def extract_window(task):
    """Worker: decode a window of T frames (plus exact motion, validity and depth when the source has them),
    down-scale by a random factor, take one P x P crop at the same place in all frames, and return motion,
    validity and depth at half resolution.

    Convention: mv[k] is, for every pixel of network frame k, the displacement to its position in network frame
    k-1, in full-resolution (P-grid) pixels (a game engine's motion vector). Sources with exact forward flow
    (TartanAir, Sintel) are played backwards so that their forward flow becomes exactly this vector.
    """
    data_root, item, start, T, P, scale_range, seed = task
    rng = np.random.default_rng(seed)
    resolver = Resolver(data_root)
    exact = item.get("mv") == "exact_forward"
    try:
        idx = list(range(start, start + T))
        frames = load_rgb_many(resolver, [item["frames"][i] for i in idx])
        h, w = frames[0].shape[:2]
        if any(f.shape[:2] != (h, w) for f in frames):
            return seed, None
        flows = [load_flow(resolver, item["flows"][i]) for i in idx] if exact else None
        valids = (
            [
                load_valid(resolver, item["valid"][i], item.get("valid_kind", "valid"))
                for i in idx
            ]
            if exact
            else None
        )
        depths = (
            [load_depth(resolver, item["depth"][i]) for i in idx]
            if item.get("depth")
            else None
        )
    except Exception:
        return seed, None
    r = min(1.0, max(float(rng.uniform(*scale_range)), P / min(h, w) * 1.02))
    W, H = int(round(w * r)), int(round(h * r))
    if min(W, H) < P:
        return seed, None
    frames = [_resize(f, (W, H), "image") for f in frames]
    y, x = int(rng.integers(0, H - P + 1)), int(rng.integers(0, W - P + 1))
    crop = (slice(y, y + P), slice(x, x + P))
    half = (P // 2, P // 2)
    out = {
        "rgb": np.stack([f[crop] for f in frames]),
        "mv": np.zeros((T, P // 2, P // 2, 2), np.float16),
        "valid": np.zeros((T, P // 2, P // 2), np.uint8),
        "depth": np.zeros((T, P // 2, P // 2), np.float16),
        "has_depth": depths is not None,
        "mv_quality": 0.0,
    }
    if depths is not None:
        depths = [_resize(d, (W, H), "depth") for d in depths]
        out["depth"] = np.stack(
            [_resize(normalise_depth(d[crop]), half, "depth") for d in depths]
        ).astype(np.float16)
    if exact:
        flows = [_resize(f, (W, H), "flow")[crop] for f in flows]  # P-grid pixel units
        valids = [_resize(v, (W, H), "nearest")[crop] for v in valids]
        # resampling to half resolution must keep P-grid units, so rescale the half-res vectors back by 2
        mv = np.stack([_resize(f, half, "flow") * 2.0 for f in flows])
        va = np.stack([_resize(v, half, "nearest") for v in valids])
        # reverse time: network frame k holds source frame T-1-k; its previous network frame (k-1) is source
        # frame T-k, which is exactly where the forward flow of source frame T-1-k points
        out["rgb"], out["depth"] = out["rgb"][::-1].copy(), out["depth"][::-1].copy()
        mv, va = mv[::-1].copy(), va[::-1].copy()
        mv[0], va[0] = 0.0, 0  # network frame 0 has no previous frame in the window
        out["mv"], out["valid"], out["mv_quality"] = mv.astype(np.float16), va, 1.0
    return seed, out


def plan_windows(items, T, n, seed, stride=None):
    """Sample n (item, start) windows, spread evenly across items (round-robin over shuffled items)."""
    rng = random.Random(seed)
    usable = [it for it in items if it["n_frames"] >= T]
    rng.shuffle(usable)
    plans = []
    while usable and len(plans) < n:
        for it in usable:
            starts = range(0, it["n_frames"] - T + 1, stride or max(1, T // 2))
            plans.append((it, rng.choice(list(starts))))
            if len(plans) >= n:
                break
    return plans


def still_crop(task):
    """Worker: decode one still or frame and return k random S x S crops after a random down-scale."""
    data_root, uri, k, S, scale_range, seed, min_std = task
    rng = np.random.default_rng(seed)
    try:
        img = load_rgb(Resolver(data_root), uri)
    except Exception:
        return seed, None
    h, w = img.shape[:2]
    r = max(float(rng.uniform(*scale_range)), S / min(h, w) * 1.02)
    if r < 1.0:
        img = _resize(img, (int(round(w * r)), int(round(h * r))), "image")
    h, w = img.shape[:2]
    if min(h, w) < S:
        return seed, None
    crops = []
    for _ in range(k * 3):
        y, x = int(rng.integers(0, h - S + 1)), int(rng.integers(0, w - S + 1))
        c = img[y : y + S, x : x + S]
        if c.mean(axis=2).std() >= min_std:
            crops.append(c)
            if len(crops) == k:
                break
    return seed, np.stack(crops) if crops else None


# ---------------------------------------------------------------------------------------------------------------
# Manifest I/O and discovery of an attached data-pipeline output
# ---------------------------------------------------------------------------------------------------------------
LIST_COLUMNS = ["frames", "flows", "valid", "depth", "native_lr"]


def save_manifest(items, path):
    import pandas as pd

    df = pd.DataFrame(items)
    for c in LIST_COLUMNS:
        if c in df:
            df[c] = df[c].map(lambda v: json.dumps(v) if isinstance(v, list) else None)
    df.to_parquet(path, index=False)
    return df


def load_manifest(path):
    import pandas as pd

    df = pd.read_parquet(path)
    for c in LIST_COLUMNS:
        if c in df:
            df[c] = df[c].map(lambda v: json.loads(v) if isinstance(v, str) else None)
    return df


def items_from_frame(df):
    return [
        {k: v for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))}
        for row in df.to_dict("records")
    ]


def locate_data_root(
    input_root="/kaggle/input", name="nss_manifest.parquet", max_depth=5
):
    """Folder holding the manifest produced by the data-pipeline notebook, if that output is attached."""
    input_root = Path(input_root)
    if not input_root.exists():
        return None
    frontier = [(input_root, 0)]
    while frontier:
        folder, depth = frontier.pop()
        if (folder / name).exists():
            return folder
        if depth < max_depth:
            try:
                frontier.extend((p, depth + 1) for p in folder.iterdir() if p.is_dir())
            except OSError:
                pass
    return None
