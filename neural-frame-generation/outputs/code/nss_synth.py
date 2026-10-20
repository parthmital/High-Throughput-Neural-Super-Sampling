"""GPU renderer of synthetic game-like sequences with exact ground truth.

A scene is a 2.5D layered world built from cached high-resolution stills: a background plane and a foreground
layer with a procedural alpha mask (solid blobs, thin wires and fences), each moving with its own continuous
affine camera trajectory, plus alpha-blended particles that write no motion vectors (as in real engines),
optional world-space text stamped into the background, and optional screen-space HUD/UI overlays.

Because every pixel's layer and trajectory are known, the renderer returns exact motion vectors, disocclusion
masks, per-layer depth, intermediate-time optical flow, reactive (transparency) masks and UI masks.

Low-resolution frames are rendered like a game renders at reduced resolution: one sample per pixel at a sub-pixel
jittered position (Halton 2,3), sampling the full-resolution texture without mip filtering, which produces real
aliasing on thin geometry and fine texture. Targets are rendered at output resolution with 4 samples per pixel
(rotated grid), an anti-aliased "native" reference.

Coordinates: screen positions are normalised to [-1, 1] (pixel centres at (2i + 1) / W - 1); texture
coordinates are normalised to [-1, 1]; a layer maps screen to texture with tex = s R(theta) x + o.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

SSAA_OFFSETS = [
    (-0.375, -0.125),
    (0.125, -0.375),
    (0.375, 0.125),
    (-0.125, 0.375),
]  # rotated grid, pixel units


def halton(index, base):
    f, r = 1.0, 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r


def jitter_sequence(n, phase_count=16):
    """Halton(2,3) sub-pixel offsets in [-0.5, 0.5) for frames 0..n-1 (cycle of phase_count)."""
    return torch.tensor(
        [
            [halton(i % phase_count + 1, 2) - 0.5, halton(i % phase_count + 1, 3) - 0.5]
            for i in range(n)
        ]
    )


def pixel_grid(n, h, w, device, offset=None):
    """Normalised screen positions of pixel centres, (n, h, w, 2) as (x, y). offset: (n, 2) in pixels."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([gx, gy], -1)[None].expand(n, h, w, 2)
    if offset is not None:
        grid = (
            grid
            + (offset * torch.tensor([2.0 / w, 2.0 / h], device=device))[
                :, None, None, :
            ]
        )
    return grid


def srgb_to_linear(x):
    return torch.where(
        x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp_min(0) ** 2.4
    )


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(
        x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055
    )


# ---------------------------------------------------------------------------------------------------------------
# Layer trajectories
# ---------------------------------------------------------------------------------------------------------------
def layer_matrix(p, t):
    """p: dict of (N, ...) tensors; t: (N,) times. Returns (N, 2, 3) screen -> texture affine."""
    s = p["s0"] * torch.exp(p["zeta"] * t)
    th = p["th0"] + p["omega"] * t
    flip = torch.relu(t - p["t_flip"])[:, None] * p["flip"][:, None]
    o = (
        p["o0"]
        + p["u"] * t[:, None]
        + 0.5 * p["a"] * (t**2)[:, None]
        - 2.0 * p["u"] * flip
    )
    c, s_ = torch.cos(th), torch.sin(th)
    row0 = torch.stack([s * c, -s * s_, o[:, 0]], -1)
    row1 = torch.stack([s * s_, s * c, o[:, 1]], -1)
    return torch.stack([row0, row1], 1)


def apply_affine(m, pts):
    return torch.einsum("nij,nhwj->nhwi", m[:, :, :2], pts) + m[:, None, None, :, 2]


def invert_affine(m):
    a_inv = torch.linalg.inv(m[:, :, :2])
    return torch.cat([a_inv, -(a_inv @ m[:, :, 2:])], 2)


def _speed_px(n, g, profile, device):
    """Screen speed in output pixels per frame from a motion profile: static / slow / medium / fast mixture."""
    probs = torch.tensor(profile["probs"], device=device)
    k = torch.multinomial(probs, n, replacement=True, generator=g)
    lo = torch.tensor([b[0] for b in profile["bins"]], device=device)[k]
    hi = torch.tensor([b[1] for b in profile["bins"]], device=device)[k]
    return lo + (hi - lo) * torch.rand(n, device=device, generator=g), k


MOTION_PROFILE = {
    "names": ["static", "slow", "medium", "fast"],
    "probs": [0.15, 0.45, 0.25, 0.15],
    "bins": [(0.0, 0.0), (0.3, 3.0), (3.0, 10.0), (10.0, 28.0)],
}


def random_layer(
    n, g, device, out_px, tex_px, duration, profile, centre_screen=None, rot_scale=1.0
):
    """Trajectory parameters for n layers whose view of out_px pixels samples a texture of tex_px texels.
    The path is centred so the view stays inside the texture for the whole duration where possible.
    """
    rnd = lambda *shape: torch.rand(*shape, device=device, generator=g)  # noqa: E731
    k = 1.0 + 0.45 * rnd(
        n
    )  # texels per output pixel (> 1: minification, aliasing at low resolution)
    s0 = out_px * k / tex_px
    speed, cls = _speed_px(n, g, profile, device)
    ang = 2 * math.pi * rnd(n)
    u_tex = speed * (2.0 / out_px) * s0  # texture units per frame
    room = (1.0 - 1.42 * s0 - 0.02).clamp_min(0.0)
    u_tex = torch.minimum(u_tex, 2 * room / max(duration, 1e-3))
    u = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * u_tex[:, None]
    o0 = (
        -u * duration / 2
        + (rnd(n, 2) * 2 - 1) * (room - u_tex * duration / 2).clamp_min(0)[:, None]
    )
    p = {
        "s0": s0,
        "zeta": torch.randn(n, device=device, generator=g) * 0.006 * (cls > 0),
        "th0": (rnd(n) * 2 - 1) * 0.2 * rot_scale,
        "omega": torch.randn(n, device=device, generator=g)
        * 0.008
        * rot_scale
        * (cls > 0),
        "o0": o0,
        "u": u,
        "a": torch.randn(n, 2, device=device, generator=g) * 0.1 * u_tex[:, None],
        "flip": torch.zeros(n, device=device),
        "t_flip": torch.full((n,), 1e9, device=device),
        "speed_class": cls,
    }
    if (
        centre_screen is not None
    ):  # place a texture point (the object centre) at a screen position at t = 0
        m0 = layer_matrix(p, torch.zeros(n, device=device))
        tex_at = (
            torch.einsum("nij,nj->ni", m0[:, :, :2], centre_screen[1]) + m0[:, :, 2]
        )
        p["o0"] = p["o0"] + centre_screen[0] - tex_at
    return p


# ---------------------------------------------------------------------------------------------------------------
# Procedural foreground masks and text atlas
# ---------------------------------------------------------------------------------------------------------------
def foreground_alpha(n, size, g, device, centres):
    """Binary-ish alpha (n, 1, S, S): one superellipse blob around centres (n, 2) plus thin wires and, sometimes,
    a fence of thin bars (1-2 texels wide) to create sub-pixel geometry."""
    rnd = lambda *shape: torch.rand(*shape, device=device, generator=g)  # noqa: E731
    coords = (torch.arange(size, device=device, dtype=torch.float32) + 0.5) * (
        2.0 / size
    ) - 1.0
    v, u = torch.meshgrid(coords, coords, indexing="ij")
    u, v = u[None], v[None]
    ang = (rnd(n) * math.pi)[:, None, None]
    du, dv = u - centres[:, 0, None, None], v - centres[:, 1, None, None]
    ru, rv = du * torch.cos(ang) + dv * torch.sin(ang), -du * torch.sin(
        ang
    ) + dv * torch.cos(ang)
    rx, ry = (0.12 + 0.3 * rnd(n))[:, None, None], (0.12 + 0.3 * rnd(n))[:, None, None]
    pw = (1.5 + 2.5 * rnd(n))[:, None, None]
    alpha = ((ru / rx).abs() ** pw + (rv / ry).abs() ** pw < 1.0).float()
    texel = 2.0 / size
    for _ in range(4):  # thin wires, present with probability 0.6 each
        on = (rnd(n) < 0.6)[:, None, None]
        px, py = (rnd(n) * 1.6 - 0.8)[:, None, None], (rnd(n) * 1.6 - 0.8)[
            :, None, None
        ]
        phi = (rnd(n) * math.pi)[:, None, None]
        width = ((0.6 + 1.4 * rnd(n)) * texel)[:, None, None]
        dist = ((u - px) * torch.sin(phi) - (v - py) * torch.cos(phi)).abs()
        alpha = torch.maximum(alpha, ((dist < width / 2) & on).float())
    fence = (rnd(n) < 0.35)[:, None, None]
    period = ((6 + 14 * rnd(n)) * texel)[:, None, None]
    bar = ((1.0 + rnd(n)) * texel)[:, None, None]
    phi = (rnd(n) * math.pi)[:, None, None]
    proj = u * torch.cos(phi) + v * torch.sin(phi)
    frac = torch.remainder(proj, period)
    region = ((u - centres[:, 0, None, None]).abs() < 0.6) & (
        (v - centres[:, 1, None, None]).abs() < 0.6
    )
    alpha = torch.maximum(alpha, ((frac < bar) & fence & region).float())
    return alpha[:, None]


def make_text_bank(n, seed, height=40, width=320):
    """RGBA uint8 tiles (n, 4, height, width) of HUD-like text, bars and icons rendered with PIL (CPU, once)."""
    import matplotlib.font_manager as fm
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    font_paths = sorted(
        {
            fm.findfont("DejaVu Sans"),
            *[f for f in fm.findSystemFonts() if f.lower().endswith(".ttf")],
        }
    )
    words = [
        "HP",
        "AMMO",
        "SCORE",
        "LVL",
        "QUEST",
        "Objective:",
        "Press E to interact",
        "Paused",
        "Settings",
        "Resume",
        "Mission complete",
        "Reload",
        "Map",
        "FPS",
        "Health",
        "Shield",
        "XP",
        "Gold",
        "Wave",
        "Where are you going?",
        "We have to move now.",
        "Checkpoint reached",
        "Inventory full",
    ]
    tiles = []
    for _ in range(n):
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        kind = rng.integers(0, 4)
        col = tuple(int(c) for c in rng.integers(150, 256, 3)) + (255,)
        if (
            kind == 0 or kind == 3
        ):  # text, optionally with a translucent plate (subtitle style)
            if kind == 3:
                d.rectangle(
                    [0, 0, width - 1, height - 1],
                    fill=(0, 0, 0, int(rng.integers(90, 180))),
                )
            text = " ".join(rng.choice(words, size=int(rng.integers(1, 4))))
            if rng.random() < 0.5:
                text += f" {int(rng.integers(0, 999))}"
            fp = font_paths[int(rng.integers(0, len(font_paths)))]
            try:
                font = (
                    ImageFont.truetype(fp, int(rng.integers(14, 30)))
                    if fp
                    else ImageFont.load_default()
                )
            except Exception:
                font = ImageFont.load_default()
            d.text(
                (6, 4),
                text,
                font=font,
                fill=col,
                stroke_width=1,
                stroke_fill=(0, 0, 0, 255),
            )
        elif kind == 1:  # health / progress bar
            frac = rng.uniform(0.1, 1.0)
            d.rectangle(
                [4, 10, width - 5, height - 11], outline=(255, 255, 255, 255), width=2
            )
            d.rectangle([7, 13, 7 + int((width - 15) * frac), height - 14], fill=col)
        else:  # crosshair / minimap frame
            cx, cy = width // 2, height // 2
            d.line([cx - 14, cy, cx + 14, cy], fill=col, width=2)
            d.line([cx, cy - 14, cx, cy + 14], fill=col, width=2)
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=col, width=1)
        tiles.append(np.asarray(img).transpose(2, 0, 1))
    return np.stack(tiles)


def stamp_tiles(canvas, alpha, tiles, idx, xy):
    """Alpha-blend tiles (K, 4, th, tw) float [0,1] into canvas (N, 3, H, W) / alpha (N, 1, H, W) at integer
    positions xy (N, 2) for tile indices idx (N,). Rows with idx < 0 are skipped."""
    th, tw = tiles.shape[-2:]
    H, W = canvas.shape[-2:]
    for i in range(canvas.shape[0]):
        k = int(idx[i])
        if k < 0:
            continue
        x, y = int(xy[i, 0]), int(xy[i, 1])
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + tw, W), min(y + th, H)
        if x1 <= x0 or y1 <= y0:
            continue
        t = tiles[k, :, y0 - y : y1 - y, x0 - x : x1 - x]
        a = t[3:4]
        canvas[i, :, y0:y1, x0:x1] = canvas[i, :, y0:y1, x0:x1] * (1 - a) + t[:3] * a
        alpha[i, :, y0:y1, x0:x1] = alpha[i, :, y0:y1, x0:x1] * (1 - a) + a


# ---------------------------------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------------------------------
class Scene:
    """A batch of layered scenes. stills: uint8 (N, S, S, 3) tensor on the target device."""

    def __init__(
        self,
        stills,
        out_px,
        duration,
        g,
        profile=MOTION_PROFILE,
        particles=True,
        text_bank=None,
        world_text_prob=0.3,
        fg_prob=0.85,
    ):
        dev = stills.device
        n, S = stills.shape[0], stills.shape[1]
        rnd = lambda *shape: torch.rand(*shape, device=dev, generator=g)  # noqa: E731
        tex = stills.permute(0, 3, 1, 2).float() / 255.0
        self.n, self.S, self.device, self.out_px = n, S, dev, out_px
        self.bg = tex
        perm = torch.roll(torch.arange(n, device=dev), 1)
        flat = (rnd(n) < 0.25)[:, None, None, None]
        flat_col = rnd(n, 3, 1, 1) * (0.85 + 0.3 * rnd(n, 1, S, S)).clamp(0, 1)
        self.fg = torch.where(flat, flat_col.expand(-1, -1, S, S), tex[perm])
        centre_tex = rnd(n, 2) * 1.0 - 0.5
        self.fg_alpha = (
            foreground_alpha(n, S, g, dev, centre_tex)
            * (rnd(n) < fg_prob).float()[:, None, None, None]
        )
        if (
            text_bank is not None and world_text_prob > 0
        ):  # world-space text moves with the background
            tiles = text_bank.to(dev).float() / 255.0
            canvas, a = self.bg.clone(), torch.zeros(n, 1, S, S, device=dev)
            idx = torch.where(
                rnd(n) < world_text_prob,
                torch.randint(0, len(tiles), (n,), device=dev, generator=g),
                torch.full((n,), -1, device=dev),
            )
            xy = torch.stack(
                [
                    torch.randint(
                        0, max(1, S - tiles.shape[-1]), (n,), device=dev, generator=g
                    ),
                    torch.randint(
                        0, max(1, S - tiles.shape[-2]), (n,), device=dev, generator=g
                    ),
                ],
                1,
            )
            stamp_tiles(canvas, a, tiles, idx, xy)
            self.bg = canvas
        self.cam = random_layer(n, g, dev, out_px, S, duration, profile)
        target_screen = rnd(n, 2) * 1.2 - 0.6
        self.obj = random_layer(
            n,
            g,
            dev,
            out_px,
            S,
            duration,
            profile,
            centre_screen=(centre_tex, target_screen),
            rot_scale=2.0,
        )
        self.d_bg = 0.6 + 0.4 * rnd(n)
        self.d_fg = 0.15 + 0.35 * rnd(n)
        self.exposure = torch.exp2((rnd(n) * 2 - 1) * 0.8)
        k = 6
        self.p_on = (rnd(n, k) < (0.5 if particles else 0.0)).float()
        self.p_pos = rnd(n, k, 2) * 2 - 1
        self.p_vel = (rnd(n, k, 2) * 2 - 1) * 0.06
        self.p_rad = (2 + 10 * rnd(n, k)) * (2.0 / out_px)
        self.p_col = 0.6 + 0.4 * rnd(n, k, 3)
        self.p_amp = 0.2 + 0.5 * rnd(n, k)

    def matrices(self, t):
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32).expand(self.n)
        return layer_matrix(self.cam, t), layer_matrix(self.obj, t)

    def render(self, t, pts):
        """Sample the scene at normalised screen positions pts (N, h, w, 2) at time t.
        Returns colour (N, 3, h, w) in display sRGB, fg coverage (N, 1, h, w), depth, reactive mask.
        """
        m_bg, m_fg = self.matrices(t)
        bg = F.grid_sample(
            self.bg,
            apply_affine(m_bg, pts),
            mode="bilinear",
            padding_mode="reflection",
            align_corners=False,
        )
        g_fg = apply_affine(m_fg, pts)
        fg = F.grid_sample(
            self.fg, g_fg, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        a = F.grid_sample(
            self.fg_alpha,
            g_fg,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        col = a * fg + (1 - a) * bg
        tex_bg = apply_affine(m_bg, pts)
        d_bg = self.d_bg[:, None, None, None] * (
            1 + 0.3 * tex_bg[..., 1:2].permute(0, 3, 1, 2)
        )
        depth = torch.where(a > 0.5, self.d_fg[:, None, None, None].expand_as(a), d_bg)
        tt = (
            torch.as_tensor(t, device=self.device, dtype=torch.float32)
            .expand(self.n)
            .view(self.n, 1, 1)
        )
        pos = self.p_pos + self.p_vel * tt  # (N, K, 2)
        d2 = ((pts[:, None] - pos[:, :, None, None, :]) ** 2).sum(-1)  # (N, K, h, w)
        gk = (
            self.p_on[..., None, None]
            * self.p_amp[..., None, None]
            * torch.exp(-d2 / (2 * self.p_rad[..., None, None] ** 2))
        )
        reactive = 1 - torch.prod(1 - gk, dim=1, keepdim=True)
        for k in range(gk.shape[1]):
            col = (
                col * (1 - gk[:, k : k + 1])
                + self.p_col[:, k, :, None, None] * gk[:, k : k + 1]
            )
        lin = srgb_to_linear(col.clamp(0, 1)) * self.exposure[:, None, None, None]
        return linear_to_srgb(lin), a, depth, reactive

    def frame(self, t, h, w, jitter=None, ssaa=False):
        """Render a full frame of h x w pixels. jitter: (N, 2) sub-pixel offsets (pixels) or None."""
        if not ssaa:
            return self.render(t, pixel_grid(self.n, h, w, self.device, jitter))
        acc = None
        for ox, oy in SSAA_OFFSETS:
            off = torch.tensor([[ox, oy]], device=self.device).expand(self.n, 2)
            col, _, _, _ = self.render(t, pixel_grid(self.n, h, w, self.device, off))
            acc = col if acc is None else acc + col
        _, a, depth, reactive = self.render(t, pixel_grid(self.n, h, w, self.device))
        return acc / len(SSAA_OFFSETS), a, depth, reactive

    def flow(self, t_from, t_to, h, w, jitter=None):
        """For every pixel of the frame at t_from: displacement (pixels of an h x w grid) to the same surface point
        at t_to, of the visible layer, and validity (1 = visible at t_to, inside the view, not occluded).
        """
        pts = pixel_grid(self.n, h, w, self.device, jitter)
        bg_f, fg_f = self.matrices(t_from)
        bg_t, fg_t = self.matrices(t_to)
        a_from = F.grid_sample(
            self.fg_alpha,
            apply_affine(fg_f, pts),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        to_bg = apply_affine(invert_affine(bg_t), apply_affine(bg_f, pts))
        to_fg = apply_affine(invert_affine(fg_t), apply_affine(fg_f, pts))
        on_fg = (a_from > 0.5).permute(0, 2, 3, 1)
        dest = torch.where(on_fg, to_fg, to_bg)
        a_to = F.grid_sample(
            self.fg_alpha,
            apply_affine(fg_t, to_bg),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        inside = (dest.abs() <= 1.0).all(-1, keepdim=True).permute(0, 3, 1, 2)
        visible = torch.where(
            on_fg.permute(0, 3, 1, 2),
            torch.ones_like(a_to, dtype=torch.bool),
            a_to < 0.5,
        )
        disp = (dest - pts) * torch.tensor([w / 2.0, h / 2.0], device=self.device)
        return disp.permute(0, 3, 1, 2), (inside & visible).float()


# ---------------------------------------------------------------------------------------------------------------
# Batches for the two tasks
# ---------------------------------------------------------------------------------------------------------------
def sr_batch(
    stills,
    scale,
    T,
    P,
    g,
    jitter_phases=16,
    profile=MOTION_PROFILE,
    text_bank=None,
    room_frames=None,
):
    """Temporal super-resolution sequence batch. Output-resolution P x P targets with 4-sample anti-aliasing,
    jittered single-sample low-resolution inputs of (P / scale)^2, exact motion vectors at low resolution
    (current -> previous frame, low-resolution pixels), validity, depth, reactive masks and exposure.
    room_frames: number of frames the camera path must stay inside the texture (default T); a smaller value keeps
    fast motion in long sequences, after which the view continues onto the mirrored texture (still exact).
    """
    n, dev = stills.shape[0], stills.device
    h = P // scale
    scene = Scene(stills, P, room_frames or T, g, profile=profile, text_bank=text_bank)
    phase = torch.randint(0, jitter_phases, (n,), device=dev, generator=g)
    seq = jitter_sequence(T + jitter_phases, jitter_phases).to(dev)
    out = {
        k: []
        for k in (
            "lr",
            "hr",
            "mv_lr",
            "valid_lr",
            "mv_hr",
            "valid_hr",
            "depth_lr",
            "reactive_lr",
            "jitter",
        )
    }
    for t in range(T):
        jit = seq[(phase + t) % jitter_phases]
        lr, _, depth, reactive = scene.frame(float(t), h, h, jitter=jit)
        hr, _, _, _ = scene.frame(float(t), P, P, ssaa=True)
        if t == 0:
            mv_lr, va_lr = torch.zeros(n, 2, h, h, device=dev), torch.zeros(
                n, 1, h, h, device=dev
            )
            mv_hr, va_hr = torch.zeros(n, 2, P, P, device=dev), torch.zeros(
                n, 1, P, P, device=dev
            )
        else:
            mv_lr, va_lr = scene.flow(float(t), float(t - 1), h, h, jitter=jit)
            mv_hr, va_hr = scene.flow(float(t), float(t - 1), P, P)
        for k, v in zip(
            out, (lr, hr, mv_lr, va_lr, mv_hr, va_hr, depth, reactive, jit)
        ):
            out[k].append(v)
    batch = {k: torch.stack(v, 1) for k, v in out.items()}
    batch["depth_lr"] = 1.0 / (
        1.0
        + batch["depth_lr"]
        / batch["depth_lr"].flatten(1).median(1).values.view(n, 1, 1, 1, 1)
    )
    batch["exposure"] = torch.log2(scene.exposure)[:, None].expand(n, T)
    batch["has_depth"] = torch.ones(n, device=dev)
    batch["mv_quality"] = torch.ones(n, device=dev)
    batch["speed_class"] = scene.cam["speed_class"]
    return batch


def ui_layer(n, H, W, tiles, g, device, n_slots=4, menu=None):
    """Screen-space HUD for n samples in two states (before and after a subtitle / HUD change at t = 1).
    Returns premultiplied colour [(N, 3, H, W)] * 2 and alpha [(N, 1, H, W)] * 2; composite as
    frame * (1 - alpha) + colour. tiles: float (K, 4, th, tw). menu: bool (N,) adds a full-screen panel.
    """
    rgb = [torch.zeros(n, 3, H, W, device=device) for _ in range(2)]
    alp = [torch.zeros(n, 1, H, W, device=device) for _ in range(2)]
    if menu is not None:
        m = menu[:, None, None, None].float()
        for s in range(2):
            rgb[s] = rgb[s] + 0.6 * 0.05 * m
            alp[s] = alp[s] + 0.6 * m
    K, _, th, tw = tiles.shape
    for slot in range(n_slots):
        on = torch.rand(n, device=device, generator=g) < 0.7
        idx = torch.where(
            on,
            torch.randint(0, K, (n,), device=device, generator=g),
            torch.full((n,), -1, device=device),
        )
        xy = torch.stack(
            [
                torch.randint(0, max(1, W - tw // 2), (n,), device=device, generator=g),
                torch.randint(0, max(1, H - th // 2), (n,), device=device, generator=g),
            ],
            1,
        )
        change = (torch.rand(n, device=device, generator=g) < 0.3) & on
        idx2 = torch.where(
            change, torch.randint(0, K, (n,), device=device, generator=g), idx
        )
        stamp_tiles(rgb[0], alp[0], tiles, idx, xy)
        stamp_tiles(rgb[1], alp[1], tiles, idx2, xy)
    return rgb, alp


def fg_batch(
    stills,
    P,
    g,
    tiles=None,
    p_extrap=0.5,
    p_ui=0.5,
    p_cut=0.03,
    p_menu=0.02,
    p_flip=0.1,
    profile=MOTION_PROFILE,
):
    """Frame-generation batch at P x P. Real frames I0 (t=0) and I1 (t=1); target at tau = 0.5 (interpolation)
    or tau = 1.5 (extrapolation). Returns frames, the engine motion vector of I1 (I1 -> I0), depth of I1, the
    camera-only flow from the target to I1 (known from the latest input, available to extrapolation), exact
    target -> I0 / I1 flows, UI composites and masks, and flags for cuts and menus."""
    n, dev = stills.shape[0], stills.device
    scene = Scene(stills, P, 2.0, g, profile=profile)
    flip = (
        torch.rand(n, device=dev, generator=g) < p_flip
    )  # rapid input change: camera reverses after t = 1
    scene.cam["flip"], scene.cam["t_flip"] = flip.float(), torch.ones(n, device=dev)
    extrap = torch.rand(n, device=dev, generator=g) < p_extrap
    tau = torch.where(
        extrap, torch.full((n,), 1.5, device=dev), torch.full((n,), 0.5, device=dev)
    )
    i0, _, _, _ = scene.frame(0.0, P, P, ssaa=True)
    i1, _, d1, _ = scene.frame(1.0, P, P, ssaa=True)
    tgt, _, _, _ = scene.frame(tau, P, P, ssaa=True)
    mv1, mv1_valid = scene.flow(1.0, 0.0, P, P)
    f_t0, v_t0 = scene.flow(tau, 0.0, P, P)
    f_t1, v_t1 = scene.flow(tau, 1.0, P, P)
    # camera-only prior: map every target pixel through the background (camera) layer to t = 1
    pts = pixel_grid(n, P, P, dev)
    m_bg, bg1 = layer_matrix(scene.cam, tau), layer_matrix(
        scene.cam, torch.ones(n, device=dev)
    )
    to1 = apply_affine(invert_affine(bg1), apply_affine(m_bg, pts))
    cam_prior = ((to1 - pts) * (P / 2.0)).permute(0, 3, 1, 2)
    batch = {
        "i0": i0,
        "i1": i1,
        "target": tgt,
        "tau": tau,
        "extrap": extrap.float(),
        "mv1": mv1,
        "mv1_valid": mv1_valid,
        "depth1": 1.0 / (1.0 + d1 / d1.flatten(1).median(1).values.view(n, 1, 1, 1)),
        "cam_prior": cam_prior,
        "flow_t0": f_t0,
        "flow_t1": f_t1,
        "valid_t0": v_t0,
        "valid_t1": v_t1,
        "speed_class": scene.cam["speed_class"],
        "flip": flip.float(),
    }
    # scene cuts: I1 and the target come from a different scene; the correct output is I1 (repeat)
    cut = torch.rand(n, device=dev, generator=g) < p_cut
    if cut.any():
        perm = torch.roll(torch.arange(n, device=dev), 1)
        for k in ("i1", "mv1", "depth1"):
            batch[k] = torch.where(cut[:, None, None, None], batch[k][perm], batch[k])
        batch["target"] = torch.where(
            cut[:, None, None, None], batch["i1"], batch["target"]
        )
        for k in ("mv1_valid", "valid_t0", "valid_t1"):
            batch[k] = torch.where(
                cut[:, None, None, None], torch.zeros_like(batch[k]), batch[k]
            )
    batch["cut"] = cut.float()
    # HUD / subtitles / menus composited in screen space after rendering (static, may change at t = 1)
    batch["ui_mask"] = torch.zeros(n, 1, P, P, device=dev)
    batch["menu"] = torch.zeros(n, device=dev)
    if tiles is not None:
        use_ui = torch.rand(n, device=dev, generator=g) < p_ui
        menu = (torch.rand(n, device=dev, generator=g) < p_menu) & use_ui
        rgb, alp = ui_layer(n, P, P, tiles, g, dev, menu=menu)
        u = use_ui[:, None, None, None].float()
        a0, a1, c0, c1 = alp[0] * u, alp[1] * u, rgb[0] * u, rgb[1] * u
        batch["hudless_i0"], batch["hudless_i1"], batch["hudless_target"] = (
            batch["i0"],
            batch["i1"],
            batch["target"],
        )
        batch["ui_rgb"], batch["ui_alpha"] = c1, a1
        batch["i0"] = batch["i0"] * (1 - a0) + c0
        batch["i1"] = batch["i1"] * (1 - a1) + c1
        # the UI shown in a generated frame is the UI of the newest real frame (I1): never interpolated
        batch["target"] = batch["target"] * (1 - a1) + c1
        batch["ui_mask"] = torch.maximum(a0, a1)
        batch["menu"] = menu.float()
    return batch


# ---------------------------------------------------------------------------------------------------------------
# Batches from cached real sequences (TartanAir, Sintel, video), built on the GPU
# ---------------------------------------------------------------------------------------------------------------
def _flip_fields(x, fx, fy, vec=False):
    """Flip (N, T, C, H, W) or (N, C, H, W) tensors per sample; vector fields also negate their components."""
    shape = (-1,) + (1,) * (x.dim() - 1)
    x = torch.where(fx.view(shape), x.flip(-1), x)
    x = torch.where(fy.view(shape), x.flip(-2), x)
    if vec:
        cdim = x.dim() - 3
        sx = torch.where(fx, -1.0, 1.0).view(shape)
        sy = torch.where(fy, -1.0, 1.0).view(shape)
        x = torch.cat([x.narrow(cdim, 0, 1) * sx, x.narrow(cdim, 1, 1) * sy], cdim)
    return x


def sr_batch_from_frames(
    rgb,
    mv_half,
    valid_half,
    depth_half,
    has_depth,
    mv_quality,
    scale,
    g,
    jitter_phases=16,
):
    """Temporal SR batch from cached real frames: rgb uint8 (N, T, P, P, 3); mv_half (N, T, P/2, P/2, 2) in P-grid
    pixels (current -> previous); valid_half (N, T, P/2, P/2); depth_half (N, T, P/2, P/2). The low-resolution
    input is a jittered single-sample read of the full-resolution frame (aliased, like a low-resolution render).
    """
    n, T, P = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    dev, h = rgb.device, P // scale
    hr = rgb.permute(0, 1, 4, 2, 3).float() / 255.0
    mv = mv_half.permute(0, 1, 4, 2, 3).float()
    valid = valid_half[:, :, None].float()
    depth = depth_half[:, :, None].float()
    fx = torch.rand(n, device=dev, generator=g) < 0.5
    fy = torch.rand(n, device=dev, generator=g) < 0.5
    hr, valid, depth = (
        _flip_fields(hr, fx, fy),
        _flip_fields(valid, fx, fy),
        _flip_fields(depth, fx, fy),
    )
    mv = _flip_fields(mv, fx, fy, vec=True)
    gain = torch.exp2((torch.rand(n, device=dev, generator=g) * 2 - 1) * 0.8)
    hr = linear_to_srgb(srgb_to_linear(hr) * gain.view(n, 1, 1, 1, 1))
    phase = torch.randint(0, jitter_phases, (n,), device=dev, generator=g)
    seq = jitter_sequence(T + jitter_phases, jitter_phases).to(dev)
    keys = (
        "lr",
        "hr",
        "mv_lr",
        "valid_lr",
        "mv_hr",
        "valid_hr",
        "depth_lr",
        "reactive_lr",
        "jitter",
    )
    out = {k: [] for k in keys}
    for t in range(T):
        jit = seq[(phase + t) % jitter_phases]
        pts = pixel_grid(n, h, h, dev, jit)
        lr = F.grid_sample(
            hr[:, t], pts, mode="bilinear", padding_mode="border", align_corners=False
        )
        mv_lr = (
            F.grid_sample(
                mv[:, t],
                pts,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            / scale
        )
        va_lr = F.grid_sample(
            valid[:, t], pts, mode="nearest", padding_mode="border", align_corners=False
        )
        d_lr = F.grid_sample(
            depth[:, t],
            pts,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        mv_hr = F.interpolate(
            mv[:, t], scale_factor=2, mode="bilinear", align_corners=False
        )
        va_hr = F.interpolate(valid[:, t], scale_factor=2, mode="nearest")
        for k, v in zip(
            keys,
            (
                lr,
                hr[:, t],
                mv_lr,
                va_lr,
                mv_hr,
                va_hr,
                d_lr,
                torch.zeros_like(va_lr),
                jit,
            ),
        ):
            out[k].append(v)
    batch = {k: torch.stack(v, 1) for k, v in out.items()}
    batch["exposure"] = torch.log2(gain)[:, None].expand(n, T)
    batch["has_depth"], batch["mv_quality"] = has_depth.float(), mv_quality.float()
    batch["speed_class"] = torch.full((n,), -1, device=dev, dtype=torch.long)
    return batch


def fg_batch_from_frames(
    rgb, flows_half, flow_ok, g, tiles=None, p_extrap=0.5, p_ui=0.5
):
    """Frame-generation batch from cached real quads: rgb uint8 (N, 4, P, P, 3) = frames k..k+3; flows_half
    (N, 4, P/2, P/2, 2) in P-grid pixels = [2->0, 1->0, 1->2, 3->2] (teacher optical flow); flow_ok (N,).
    Interpolation: I0 = k, I1 = k+2, target k+1. Extrapolation: I0 = k, I1 = k+2, target k+3 (tau = 1.5).
    """
    n, P, dev = rgb.shape[0], rgb.shape[2], rgb.device
    fr = rgb.permute(0, 1, 4, 2, 3).float() / 255.0
    fl = flows_half.permute(0, 1, 4, 2, 3).float()
    fx = torch.rand(n, device=dev, generator=g) < 0.5
    fy = torch.rand(n, device=dev, generator=g) < 0.5
    fr, fl = _flip_fields(fr, fx, fy), _flip_fields(fl, fx, fy, vec=True)
    up = lambda x: F.interpolate(
        x, scale_factor=2, mode="bilinear", align_corners=False
    )  # noqa: E731
    extrap = torch.rand(n, device=dev, generator=g) < p_extrap
    e4 = extrap.view(n, 1, 1, 1)
    ok = flow_ok.float().view(n, 1, 1, 1).expand(n, 1, P, P)
    batch = {
        "i0": fr[:, 0],
        "i1": fr[:, 2],
        "target": torch.where(e4, fr[:, 3], fr[:, 1]),
        "tau": torch.where(extrap, 1.5, 0.5).float(),
        "extrap": extrap.float(),
        "mv1": up(fl[:, 0]),
        "mv1_valid": ok,
        "depth1": torch.zeros(n, 1, P, P, device=dev),
        "cam_prior": torch.zeros(n, 2, P, P, device=dev),
        "flow_t0": up(fl[:, 1]),
        "valid_t0": ok * (~e4).float(),
        "flow_t1": torch.where(e4, up(fl[:, 3]), up(fl[:, 2])),
        "valid_t1": ok,
        "speed_class": torch.full((n,), -1, device=dev, dtype=torch.long),
        "flip": torch.zeros(n, device=dev),
        "cut": torch.zeros(n, device=dev),
        "menu": torch.zeros(n, device=dev),
        "ui_mask": torch.zeros(n, 1, P, P, device=dev),
    }
    if tiles is not None:
        use_ui = torch.rand(n, device=dev, generator=g) < p_ui
        rgb_ui, alp = ui_layer(n, P, P, tiles, g, dev)
        u = use_ui[:, None, None, None].float()
        a0, a1, c0, c1 = alp[0] * u, alp[1] * u, rgb_ui[0] * u, rgb_ui[1] * u
        batch["hudless_i0"], batch["hudless_i1"], batch["hudless_target"] = (
            batch["i0"],
            batch["i1"],
            batch["target"],
        )
        batch["ui_rgb"], batch["ui_alpha"] = c1, a1
        batch["i0"] = batch["i0"] * (1 - a0) + c0
        batch["i1"] = batch["i1"] * (1 - a1) + c1
        batch["target"] = batch["target"] * (1 - a1) + c1
        batch["ui_mask"] = torch.maximum(a0, a1)
    return batch
