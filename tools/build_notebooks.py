"""Build the Kaggle notebooks from their percent-format sources and the shared library.

Sources: notebooks/<name>.py (cells separated by "# %%" and "# %% [markdown]"). A code cell whose whole body is
"# WRITEFILE <name>.py" becomes a %%writefile cell that embeds src/neuralss/<name>.py into /kaggle/working/code,
so every notebook is self-contained on Kaggle while the library has one source of truth in this repository.

Usage: python tools/build_notebooks.py [--check]   (--check: fail if any built notebook differs from the repo copy)
Requires: nbformat.
"""

import argparse
import copy
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "src" / "neuralss"
NOTEBOOKS = ["neural-data-pipeline", "neural-supersampling", "neural-frame-generation"]
WRITEFILE = re.compile(r"^# WRITEFILE (\S+\.py)$")
KAGGLE_META = {
    "accelerator": "nvidiaTeslaT4",
    "isInternetEnabled": True,
    "isGpuEnabled": True,
    "language": "python",
    "sourceType": "notebook",
    "dataSources": [],
}


def build(name):
    text = (ROOT / "notebooks" / f"{name}.py").read_text(encoding="utf-8")
    chunks = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for header, body in zip(chunks[1::2], chunks[2::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            lines = [
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in body.splitlines()
            ]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip()))
            continue
        m = WRITEFILE.match(body.strip())
        if m:
            src = (LIB / m.group(1)).read_text(encoding="utf-8").rstrip("\n")
            cells.append(
                nbformat.v4.new_code_cell(
                    f"%%writefile /kaggle/working/code/{m.group(1)}\n{src}"
                )
            )
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    for k, cell in enumerate(cells):  # stable ids keep diffs small between builds
        cell["id"] = f"{name[:12]}-{k:03d}"
    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    meta = copy.deepcopy(KAGGLE_META)
    out = ROOT / name / f"{name}.ipynb"
    if out.exists():
        old = nbformat.read(out, as_version=4).metadata.get("kaggle", {})
        if old.get("dataSources"):
            meta["dataSources"] = copy.deepcopy(old["dataSources"])
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "kaggle": meta,
    }
    nbformat.validate(nb)
    return out, nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for name in NOTEBOOKS:
        out, nb = build(name)
        text = nbformat.writes(nb) + "\n"
        if args.check:
            if not out.exists() or out.read_text(encoding="utf-8") != text:
                stale.append(str(out))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
        n_code = sum(c.cell_type == "code" for c in nb.cells)
        print(f"wrote {out.relative_to(ROOT)}: {len(nb.cells)} cells ({n_code} code)")
    if stale:
        print("stale notebooks (run tools/build_notebooks.py):", *stale, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
