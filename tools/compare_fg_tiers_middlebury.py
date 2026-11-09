"""Compare frame-generation tiers on the Middlebury 'other' interpolation set (frame10 + frame11 -> frame10i11), colour only
(no engine data), on CPU.

Methods, named as in the FG results: average (frame averaging); rife_4_26 (RIFE 4.26 unchanged, flow at full resolution),
rife_4_26_flow_half and rife_4_26_flow_quarter (the same network with flow at half and quarter resolution, blend output);
high, mid, low and igpu (the trained FG tiers of a Kaggle run, EMA weights; high is the fg_teacher run, which has the
high tier's architecture). GMAC is per generated 1080p frame.

Writes the PSNR of every sequence, the mean PSNR per method (sequence 'mean', with the GMAC per method) to
neural-frame-generation/local-evaluation/middlebury_psnr.csv, which tools/build_paper_results.py reads.

Inputs (defaults): .agent-local/downloads/middlebury/extracted (other-color-allframes and other-gt-interp from
vision.middlebury.edu/flow/data/comp/zip/), the RIFE 4.26 flownet.pkl in .agent-local/downloads/rife426/extracted, and the
checkpoints fg_{teacher,mid,low,igpu}_best.pt of the FG run in neural-frame-generation/outputs/weights.
Run from the repository root: .venv/Scripts/python -I tools/compare_fg_tiers_middlebury.py [--middlebury DIR] [--rife FILE]
    [--weights DIR] [--out CSV]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "neuralss"))
import nss_models as NM  # noqa: E402

DEFAULT_MIDDLEBURY = ROOT / ".agent-local/downloads/middlebury/extracted"
DEFAULT_RIFE = (
    ROOT / ".agent-local/downloads/rife426/extracted/RIFEv4.26_0921/flownet.pkl"
)
DEFAULT_WEIGHTS = ROOT / "neural-frame-generation/outputs/weights"
DEFAULT_OUT = ROOT / "neural-frame-generation/local-evaluation/middlebury_psnr.csv"
# RIFE 4.26 flow scale -> flow_lines for the frames of this comparison (380 to 480 lines, flow halved per scale step)
RIFE_FLOW_LINES = {1.0: 4096, 0.5: 360, 0.25: 180}
# RIFE 4.26 flow scale -> flow_lines for the GMAC count of a 1080p frame
RIFE_GMAC_FLOW_LINES = {1.0: 1152, 0.5: 576, 0.25: 288}
RIFE_METHODS = {
    "rife_4_26": 1.0,
    "rife_4_26_flow_half": 0.5,
    "rife_4_26_flow_quarter": 0.25,
}
# method name -> (checkpoint of the FG run, tier whose network is built)
FG_METHODS = {
    "high": ("fg_teacher", "teacher"),
    "mid": ("fg_mid", "mid"),
    "low": ("fg_low", "low"),
    "igpu": ("fg_igpu", "igpu"),
}
torch.set_grad_enabled(False)


def load(path):
    return (
        torch.from_numpy(np.array(Image.open(path).convert("RGB")))
        .permute(2, 0, 1)[None]
        .float()
        / 255
    )


def psnr(a, b):
    return float(10 * torch.log10(1 / ((a.clamp(0, 1) - b) ** 2).mean()))


def inputs(i0, i1):
    n, _, h, w = i0.shape
    z = torch.zeros(n)
    return {
        "i0": i0,
        "i1": i1,
        "mv1": torch.zeros(n, 2, h, w),
        "depth1": torch.zeros(n, 1, h, w),
        "cam_prior": torch.zeros(n, 2, h, w),
        "tau": torch.full((n,), 0.5),
        "extrap": z,
        "mv_avail": z,
        "cam_avail": z,
        "depth_avail": z,
    }


def rife_at(state, scale):
    # flow_lines just below the frame height halves it log2(1 / scale) times on the 380 to 480-line frames
    net = NM.FGNet(NM.RIFE_WIDTHS, 8, 32, flow_lines=RIFE_FLOW_LINES[scale])
    NM.load_ifnet_weights(net.ifnet, state)
    return net.eval()


def gmac_1080p(net):
    net.flow_lines = net.flow_lines * 272 / 1080  # the 1080p flow scale at 272 lines
    f = NM.fg_inputs(1, 272, 480)
    return NM.count_macs(net, lambda m: m(f)) * (1080 * 1920) / (272 * 480) / 1e9


def gmac_by_method():
    gmac = {}
    for name, scale in RIFE_METHODS.items():
        gmac[name] = gmac_1080p(
            NM.FGNet(
                NM.RIFE_WIDTHS, 8, 32, flow_lines=RIFE_GMAC_FLOW_LINES[scale]
            ).eval()
        )
    for name, (_, tier) in FG_METHODS.items():
        gmac[name] = gmac_1080p(NM.build_fg(tier).eval())
    return gmac


def parse_args():
    parser = argparse.ArgumentParser(
        description="Middlebury PSNR and GMAC of the FG tiers and RIFE 4.26 (CPU)."
    )
    parser.add_argument(
        "--middlebury",
        type=Path,
        default=DEFAULT_MIDDLEBURY,
        help="extracted other-color-allframes and other-gt-interp",
    )
    parser.add_argument(
        "--rife", type=Path, default=DEFAULT_RIFE, help="RIFE 4.26 flownet.pkl"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="directory with the fg_*_best.pt checkpoints",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="CSV to write")
    return parser.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.rife, map_location="cpu", weights_only=True)
    nets = {
        name: (rife_at(state, scale), "merged") for name, scale in RIFE_METHODS.items()
    }
    for name, (run, tier) in FG_METHODS.items():
        path = args.weights / f"{run}_best.pt"
        if not path.exists():
            print("checkpoint missing, method skipped:", path)
            continue
        net = NM.build_fg(tier)
        net.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)["ema"]
        )
        nets[name] = (net.eval(), "out")
    gmac = gmac_by_method()
    for name, value in gmac.items():
        print(f"{name}: {value:.2f} GMAC per 1080p frame")
    methods = ["average", *nets]
    rows = []
    per_method = {name: [] for name in methods}
    gt_dir = args.middlebury / "other-gt-interp"
    for seq in sorted(p.name for p in gt_dir.iterdir()):
        i0 = load(args.middlebury / "other-data" / seq / "frame10.png")
        i1 = load(args.middlebury / "other-data" / seq / "frame11.png")
        gt = load(gt_dir / seq / "frame10i11.png")
        values = {"average": psnr(0.5 * (i0 + i1), gt)}
        for name, (net, key) in nets.items():
            out, aux = net(inputs(i0, i1))
            values[name] = psnr(out if key == "out" else aux["merged"], gt)
        size = f"{gt.shape[-2]}x{gt.shape[-1]}"
        for name in methods:
            per_method[name].append(values[name])
            rows.append(
                {
                    "sequence": seq,
                    "size": size,
                    "method": name,
                    "psnr": f"{values[name]:.4f}",
                    "gmac_1080p": "",
                }
            )
        print(seq, size, {k: round(v, 2) for k, v in values.items()})
    means = {name: float(np.mean(per_method[name])) for name in methods}
    print("mean PSNR (dB):", {k: round(v, 2) for k, v in means.items()})
    for name in methods:
        rows.append(
            {
                "sequence": "mean",
                "size": "",
                "method": name,
                "psnr": f"{means[name]:.4f}",
                "gmac_1080p": f"{gmac[name]:.2f}" if name in gmac else "",
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sequence", "size", "method", "psnr", "gmac_1080p"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
