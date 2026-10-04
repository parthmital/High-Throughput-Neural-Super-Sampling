"""Rep-TNSR super-resolution library: model, reparameterisation, losses, metrics and degradation."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

SOBEL_X = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
SOBEL_Y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
LAPLACIAN = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]

TIERS = {
    "performance": {"channels": 16, "blocks": 4},
    "quality": {"channels": 20, "blocks": 4},
    "ultra": {"channels": 24, "blocks": 6},
}


class RepConvBlock(nn.Module):
    """Training-time multi-branch block that collapses into one 3x3 convolution plus PReLU.

    Branches: 3x3 conv, 1x1 conv, identity (when in == out), and fixed Sobel-x, Sobel-y and
    Laplacian depthwise filters followed by a learnable 1x1 projection. Filtering before the
    projection keeps the fusion exact, including at zero-padded borders.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        kernels = torch.tensor([SOBEL_X, SOBEL_Y, LAPLACIAN]).unsqueeze(1)
        self.register_buffer(
            "edge_dw", kernels.repeat(in_ch, 1, 1, 1), persistent=False
        )
        self.edge_pw = nn.Conv2d(3 * in_ch, out_ch, 1)
        self.act = nn.PReLU(out_ch)
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return self.act(self.fused(x))
        out = self.conv3(x) + self.conv1(x)
        edges = F.conv2d(x, self.edge_dw.to(x.dtype), padding=1, groups=self.in_ch)
        out = out + self.edge_pw(edges)
        if self.in_ch == self.out_ch:
            out = out + x
        return self.act(out)

    @torch.no_grad()
    def fused_conv(self):
        weight = self.conv3.weight.detach().clone()
        bias = self.conv3.bias.detach().clone()
        weight += F.pad(self.conv1.weight, [1, 1, 1, 1])
        bias += self.conv1.bias
        pointwise = self.edge_pw.weight[:, :, 0, 0].reshape(self.out_ch, self.in_ch, 3)
        weight += torch.einsum(
            "oik,khw->oihw", pointwise, self.edge_dw[:3, 0].to(pointwise.dtype)
        )
        bias += self.edge_pw.bias
        if self.in_ch == self.out_ch:
            idx = torch.arange(self.out_ch, device=weight.device)
            weight[idx, idx, 1, 1] += 1.0
        conv = nn.Conv2d(self.in_ch, self.out_ch, 3, padding=1).to(
            weight.device, weight.dtype
        )
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
        return conv

    def convert_to_fused(self):
        conv = self.fused_conv()
        del self.conv3, self.conv1, self.edge_pw, self.edge_dw
        self.fused = conv


class RepTNSR(nn.Module):
    """Rep-TNSR v2 trunk: N RepConv blocks in LR space, 3x3 projection to 3*s^2, pixel shuffle.

    The input colour is repeated s^2 times and added before the pixel shuffle (ECBSR-style
    nearest-neighbour residual), so the network only learns the high-frequency correction.
    """

    def __init__(self, in_ch=3, channels=24, blocks=6, scale=3):
        super().__init__()
        self.scale = scale
        self.blocks = nn.ModuleList(
            [
                RepConvBlock(in_ch if i == 0 else channels, channels)
                for i in range(blocks)
            ]
        )
        self.conv_out = nn.Conv2d(channels, 3 * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        feat = x
        for block in self.blocks:
            feat = block(feat)
        sub = self.scale * self.scale
        residual = torch.cat(
            [x[:, c : c + 1].expand(-1, sub, -1, -1) for c in range(3)], dim=1
        )
        return self.shuffle(self.conv_out(feat) + residual)

    def fused_copy(self):
        fused = copy.deepcopy(self).eval()
        for block in fused.blocks:
            block.convert_to_fused()
        return fused


def build_model(tier, scale, in_ch=3):
    spec = TIERS[tier]
    return RepTNSR(
        in_ch=in_ch, channels=spec["channels"], blocks=spec["blocks"], scale=scale
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def fused_macs_per_lr_pixel(in_ch, channels, blocks, scale):
    """Multiply-accumulates per LR pixel of the fused network (convolutions only)."""
    return (
        in_ch * channels * 9
        + (blocks - 1) * channels * channels * 9
        + channels * 3 * scale * scale * 9
    )


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


def edge_loss(pred, target):
    """L1 distance between central-difference gradients in x and y."""

    def grads(img):
        return img[..., :, 2:] - img[..., :, :-2], img[..., 2:, :] - img[..., :-2, :]

    (pred_x, pred_y), (gt_x, gt_y) = grads(pred), grads(target)
    return (pred_x - gt_x).abs().mean() + (pred_y - gt_y).abs().mean()


def bicubic_down(hr, scale):
    """Anti-aliased bicubic downsampling; HR height and width must be divisible by scale."""
    h, w = hr.shape[-2:]
    lr = F.interpolate(
        hr,
        size=(h // scale, w // scale),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    )
    return lr.clamp_(0.0, 1.0)


def bicubic_up(lr, scale):
    h, w = lr.shape[-2:]
    return F.interpolate(
        lr, size=(h * scale, w * scale), mode="bicubic", align_corners=False
    ).clamp_(0.0, 1.0)


def random_dihedral(x):
    """Independent random flip and transpose per sample (the 8 dihedral variants of a square patch)."""
    flags = torch.rand(x.shape[0], 3, device=x.device) < 0.5
    x = torch.where(flags[:, 0, None, None, None], x.flip(-1), x)
    x = torch.where(flags[:, 1, None, None, None], x.flip(-2), x)
    return torch.where(flags[:, 2, None, None, None], x.transpose(-1, -2), x)


def quantize(x):
    return (x.clamp(0.0, 1.0) * 255.0).round() / 255.0


def rgb_to_y(x):
    """ITU-R BT.601 luma used by SR benchmarks, returned on a [0, 1] scale."""
    return (
        16.0 + 65.481 * x[:, 0:1] + 128.553 * x[:, 1:2] + 24.966 * x[:, 2:3]
    ) / 255.0


def shave(x, border):
    return x[..., border:-border, border:-border] if border > 0 else x


def psnr(a, b, border=0):
    """Per-image PSNR in dB for tensors in [0, 1]."""
    mse = ((shave(a, border).float() - shave(b, border).float()) ** 2).mean(
        dim=(1, 2, 3)
    )
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(channels, size=11, sigma=1.5, device=None):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(a, b, border=0):
    """Per-image SSIM (11x11 Gaussian window, sigma 1.5, valid region only)."""
    a, b = shave(a, border).float(), shave(b, border).float()
    channels = a.shape[1]
    window = _gaussian_window(channels, device=a.device)
    mu_a = F.conv2d(a, window, groups=channels)
    mu_b = F.conv2d(b, window, groups=channels)
    var_a = F.conv2d(a * a, window, groups=channels) - mu_a**2
    var_b = F.conv2d(b * b, window, groups=channels) - mu_b**2
    cov = F.conv2d(a * b, window, groups=channels) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return score.mean(dim=(1, 2, 3))
