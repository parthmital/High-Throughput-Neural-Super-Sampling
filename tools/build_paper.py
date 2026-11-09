"""Build the paper: regenerate paper/results-values.tex, then compile the six PDFs with Tectonic in paper/.

Documents are built in the order that xr-hyper needs (each document reads the auxiliary file of the other one of its pair):
  main, supplementary, main                                     the article and its supplement
  main-anonymised, supplementary-anonymised, main-anonymised    the submission copy and its supplement
  main-title-page, main-no-references                           the title page and the similarity-check copy
Tectonic runs with --keep-intermediates --keep-logs --reruns 3. The page count of each PDF is printed (pypdf). A failed
generator or Tectonic run stops the build with exit code 1.

Options:
  --supplement-only  regenerate the values, then compile supplementary.tex once
  --no-generate      skip tools/build_paper_results.py
Tectonic: paper/tectonic.exe if it exists, else tectonic on PATH.
Run from the repository root: .venv/Scripts/python -I tools/build_paper.py [--supplement-only] [--no-generate]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
GENERATOR = ROOT / "tools" / "build_paper_results.py"
TECTONIC_OPTIONS = [
    "--keep-intermediates",
    "--keep-logs",
    "--reruns",
    "3",
    "--color",
    "never",
]
MAIN_PAIR = ["main", "supplementary", "main"]
ANONYMISED_PAIR = ["main-anonymised", "supplementary-anonymised", "main-anonymised"]
OTHER_DOCUMENTS = ["main-title-page", "main-no-references"]
PDF_NAMES = [
    "main",
    "supplementary",
    "main-anonymised",
    "supplementary-anonymised",
    "main-title-page",
    "main-no-references",
]


def find_tectonic():
    local = PAPER / "tectonic.exe"
    if local.exists():
        return str(local)
    return shutil.which("tectonic")


def compile_document(tectonic, name):
    """Run Tectonic on paper/<name>.tex. Returns False, with the end of its output, when it fails."""
    result = subprocess.run(
        [tectonic, *TECTONIC_OPTIONS, f"{name}.tex"],
        cwd=PAPER,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"Tectonic failed on {name}.tex (exit {result.returncode})")
        print((result.stdout + result.stderr)[-4000:])
        return False
    print(f"built paper/{name}.pdf")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the paper values and build the six PDFs."
    )
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="compile supplementary.tex once after the generator",
    )
    parser.add_argument(
        "--no-generate", action="store_true", help="skip tools/build_paper_results.py"
    )
    args = parser.parse_args()

    if not args.no_generate:
        generated = subprocess.run([sys.executable, "-I", str(GENERATOR)], cwd=ROOT)
        if generated.returncode != 0:
            print("tools/build_paper_results.py failed: the PDFs are not built")
            return 1

    tectonic = find_tectonic()
    if tectonic is None:
        print("Tectonic not found: neither paper/tectonic.exe nor tectonic on PATH")
        return 1

    sequence = (
        ["supplementary"]
        if args.supplement_only
        else MAIN_PAIR + ANONYMISED_PAIR + OTHER_DOCUMENTS
    )
    for name in sequence:
        if not compile_document(tectonic, name):
            return 1

    for name in PDF_NAMES:
        if name in sequence:
            pages = len(PdfReader(str(PAPER / f"{name}.pdf")).pages)
            print(f"paper/{name}.pdf: {pages} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
