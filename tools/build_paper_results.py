r"""Generate paper/results-values.tex, the numbers the paper quotes, from the notebooks' saved metric files, and recreate
the paper figures that are copies of notebook outputs.

Sources (read only): neural-data-pipeline (prefix data.), neural-supersampling (tsr.) and neural-frame-generation (fg.):
outputs/metrics/*.csv and *.json, outputs/logs/notebook.log and the notebook (run dates and hours); and
neural-frame-generation/local-evaluation/middlebury_psnr.csv (prefix mb., written by tools/compare_fg_tiers_middlebury.py).

Output: paper/results-values.tex, a generated header comment and then one SetResult line per key, sorted by key:
\SetResult{key}{value}. A key is lowercase [a-z0-9_.-]: <prefix><file stem>.<identifying values>.<metric column>, where
the identifying values come from TABLES (or GROUPS). Values are formatted from the column name: dB to 2 decimals, SSIM 3
(4 at 0.99 and above), LPIPS and FLIP 3, rates and probabilities 3, temporal errors 4, ms and GMAC 1, parameter counts as
138K or 1.41M with a raw key (.raw), integers with thousands commas, anything else to 4 significant figures.
Summaries: per-frame and per-sample CSVs by group (n, mean, min, max); history CSVs by last, min and max; the FG
test_per_sample.csv as in .agent-local/scripts/readme_tables.py (not-gated means, sample counts .n, .gated and .kept).
*_history.json are skipped because they repeat the history CSVs. Run dates and hours: run.<notebook>.start, .end and .hours.

Derived keys: every ResDiff{a}{b} (signed a - b), ResAbsDiff{a}{b} (|a - b|) and ResRatio{a}{b} (a / b, 2 decimals) in
paper/*.tex creates diff:a:b, absdiff:a:b and ratio:a:b. Every Res{key} and derived call must name a known key. Otherwise
the error is printed with file:line and the script exits 1, after writing the file.

Change report: before the file is overwritten, the added, removed and changed keys are printed, and for each removed or
changed key the file:line of every paper use, so that the sentences can be checked again.

Figures: sync_figures() copies the paper's figures from their notebook outputs when the bytes differ: fig-tsr-*.png and
fig-fg-ui-{target,average,rife,student,student-ui-layer}.png are byte copies of the saved predictions (LaTeX scales them).
fig-fg-ui-mask.png is a pixel crop of the predicted UI mask panel of plots/visual_ui.png (FIGURE_CROPS).

Camera prior: fg.camera_prior.<suite>.<mode>.<inputs>.<prior_on|prior_off>.<method>.<metric>.{mean,min,max} for the
samples of each camera prior setting (the reversal suite). Configuration values (*.config.*) are written without padded
zeros (0.5 not 0.500).

Run from the repository root: .venv/Scripts/python -I tools/build_paper_results.py
"""

import json
import math
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUTPUT = PAPER / "results-values.tex"
MIDDLEBURY_CSV = (
    ROOT / "neural-frame-generation" / "local-evaluation" / "middlebury_psnr.csv"
)
NOTEBOOK_DIRS = {
    "data": ROOT / "neural-data-pipeline",
    "tsr": ROOT / "neural-supersampling",
    "fg": ROOT / "neural-frame-generation",
}
FIGURE_SOURCES = {
    "fig-tsr-static.png": "neural-supersampling/outputs/plots/static_convergence.png",
    "fig-tsr-vid4-bicubic.png": "neural-supersampling/outputs/predictions/visual_vid4_plain_bicubic.png",
    "fig-tsr-vid4-ground-truth.png": "neural-supersampling/outputs/predictions/visual_vid4_plain_ground_truth.png",
    "fig-tsr-vid4-student.png": "neural-supersampling/outputs/predictions/visual_vid4_plain_student_s3.png",
    "fig-tsr-vid4-teacher.png": "neural-supersampling/outputs/predictions/visual_vid4_plain_teacher_s3.png",
    "fig-fg-ui-target.png": "neural-frame-generation/outputs/predictions/visual_ui_target.png",
    "fig-fg-ui-average.png": "neural-frame-generation/outputs/predictions/visual_ui_average.png",
    "fig-fg-ui-rife.png": "neural-frame-generation/outputs/predictions/visual_ui_rife_4.26.png",
    "fig-fg-ui-student.png": "neural-frame-generation/outputs/predictions/visual_ui_student.png",
    "fig-fg-ui-student-ui-layer.png": "neural-frame-generation/outputs/predictions/visual_ui_student_ui_layer.png",
}
# Figures cut from a multi-panel sheet: (source, (left, top, right, bottom) in pixels). The predicted UI mask panel is
# the ninth of the eleven panels of plots/visual_ui.png; its image area starts at row 89 (as the target panel does).
FIGURE_CROPS = {
    "fig-fg-ui-mask.png": (
        "neural-frame-generation/outputs/plots/visual_ui.png",
        (2278, 89, 2515, 325),
    ),
}
# Identifying columns of each metrics CSV (empty: only the row count is kept)
TABLES = {
    "data": {
        "catalogue_summary": ["source", "kind", "role"],
        "dataset_card": ["source"],
        "discovery_kaggle": ["source"],
        "leakage_audit": ["split", "reference", "metric"],
        "leakage_violations": [],
        "split_by_source": ["source"],
        "stage_times": ["stage"],
    },
    "tsr": {
        "test_results": ["suite", "mode", "scale", "method"],
        "ablations": ["suite", "mode", "method"],
        "best_validation": ["run"],
        "batch_probe": ["run", "batch"],
        "latency": ["model", "scale", "output"],
        "model_tiers": ["tier", "scale"],
        "static_convergence": ["method"],
        "stage_times": ["stage"],
    },
    "fg": {
        "test_results": ["suite", "mode", "inputs", "method"],
        "ablations": ["suite", "mode", "inputs", "method"],
        "ui_results": ["mode", "inputs", "method"],
        "cut_gate_rates": ["suite", "mode", "inputs"],
        "best_validation": ["run"],
        "batch_probe": ["run", "batch"],
        "latency": ["model", "output"],
        "model_tiers": ["tier"],
        "pacing_model": ["mode", "render_ms"],
        "stage_times": ["stage"],
        "temporal_consistency": ["suite"],
    },
}
# Summaries by group: (prefix, file stem) -> identifying columns
GROUPS = {
    ("data", "calibration_scores"): ["label", "kind"],
    ("data", "duplicate_pairs"): ["rule"],
    ("data", "keyframe_stats"): ["source"],
    ("tsr", "test_per_frame"): ["suite", "mode", "scale", "method"],
    ("fg", "temporal_consistency_per_clip"): ["suite", "method"],
}
FG_PER_SAMPLE_METRICS = [
    "psnr",
    "ssim",
    "ie",
    "lpips",
    "psnr_ui",
    "ui_ghost_rate",
    "psnr_occluded",
    "mv_px",
]
# Columns that flag subsets of rows, or are row indices: neither a metric nor an identifier
FLAG_AND_INDEX_COLUMNS = {
    "extrap",
    "cut",
    "menu",
    "flip",
    "sample",
    "frame",
    "speed_class",
    "item",
    "kf_a",
    "kf_b",
    "item_a",
    "item_b",
}
HISTORY_EXCLUDED_COLUMNS = FLAG_AND_INDEX_COLUMNS | {"round", "step", "lr"}
MINUS = r"\ensuremath{-}"
TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
# Files of paper/ that define macros or hold the results: not scanned for uses
NOT_SCANNED = {"paper-preamble.tex", "results-values.tex"}
RES_SINGLE = re.compile(r"\\Res\{([^{}]+)\}")
RES_PAIR = re.compile(r"\\(ResDiff|ResAbsDiff|ResRatio)\{([^{}]+)\}\{([^{}]+)\}")
DERIVED_PREFIX = {"ResDiff": "diff", "ResAbsDiff": "absdiff", "ResRatio": "ratio"}


class Value(NamedTuple):
    text: str
    number: float | None
    decimals: int | None


class Use(NamedTuple):
    location: str
    macro: str
    produces: str
    depends: tuple


def key_part(text):
    """A key segment: lowercase, with each run of characters other than a-z, 0-9, '-' and '_' replaced by '_'."""
    return re.sub(r"[^a-z0-9_-]+", "_", str(text).lower()).strip("_")


def escape_text(text):
    return "".join(TEX_SPECIAL.get(char, char) for char in text)


def fixed(number, decimals):
    """Fixed-point text with thousands commas; a negative value gets the LaTeX minus sign."""
    rounded = round(float(number), decimals)
    text = f"{abs(rounded):,.{decimals}f}"
    return MINUS + text if rounded < 0 else text


def integer_text(number):
    number = int(number)
    return MINUS + f"{-number:,}" if number < 0 else f"{number:,}"


def significant(number):
    """Four significant figures with trailing zeros dropped; returns the text and its decimal places."""
    if number == 0:
        return "0", 0
    decimals = max(0, 3 - math.floor(math.log10(abs(number))))
    text = fixed(number, decimals)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text, decimals


def decimals_of(column, value):
    """Decimal places of a float metric from its column name; None means four significant figures."""
    name = column.lower()
    words = set(re.findall(r"[a-z0-9]+", name))
    if "gmac" in name or "ms" in words:
        return 1
    if "psnr" in name:
        return 2
    if "ssim" in name:
        return 4 if value >= 0.99 else 3
    if words & {
        "lpips",
        "flip",
        "rate",
        "share",
        "recall",
        "precision",
        "fraction",
        "prob",
    }:
        return 3
    if "temporal" in name or "tce" in name or "warp" in name:
        return 4
    return None


def format_metric(column, value):
    decimals = decimals_of(column, value)
    if decimals is None:
        return significant(value)
    return fixed(value, decimals), decimals


def params_text(number):
    """Parameter count as 138K or 1.41M, with its decimal places."""
    if abs(number) >= 1e6:
        return fixed(number / 1e6, 2) + "M", 2
    if abs(number) >= 1e3:
        return fixed(round(number / 1e3), 0) + "K", 0
    return integer_text(number), 0


def put(values, key, text, number, decimals):
    if key in values:
        raise ValueError(f"duplicate key {key}")
    values[key] = Value(text, number, decimals)


def put_metric(values, key, column, value):
    """Add one metric value under key, formatted from its column name (params also get a raw key)."""
    if "params" in re.findall(r"[a-z0-9]+", column.lower()):
        text, decimals = params_text(float(value))
        put(values, key, text, float(value), decimals)
        put(values, key + ".raw", integer_text(round(float(value))), float(value), 0)
    elif isinstance(value, (int, np.integer)):
        put(values, key, integer_text(value), float(value), 0)
    else:
        text, decimals = format_metric(column, float(value))
        put(values, key, text, float(value), decimals)


def metric_columns(df, excluded):
    """Numeric, non-boolean columns that are not excluded."""
    return [
        column
        for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def add_table(values, prefix, stem, df, id_columns):
    base = f"{prefix}.{stem}"
    put(values, f"{base}.rows", integer_text(len(df)), len(df), 0)
    if not id_columns:
        return
    metrics = metric_columns(df, set(id_columns) | FLAG_AND_INDEX_COLUMNS)
    for record in df.to_dict("records"):
        parts = [key_part(record[column]) for column in id_columns]
        row_base = ".".join([base] + [part for part in parts if part])
        for column in metrics:
            value = record[column]
            if value is None or pd.isna(value):
                continue
            put_metric(values, f"{row_base}.{key_part(column)}", column, value)


def add_groups(values, prefix, stem, df, group_columns):
    base = f"{prefix}.{stem}"
    metrics = metric_columns(df, set(group_columns) | FLAG_AND_INDEX_COLUMNS)
    for group_key, group in df.groupby(group_columns, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_base = ".".join([base] + [key_part(part) for part in group_key])
        put(values, f"{group_base}.n", integer_text(len(group)), len(group), 0)
        for column in metrics:
            series = group[column].dropna()
            if series.empty:
                continue
            for stat, stat_value in (
                ("mean", series.mean()),
                ("min", series.min()),
                ("max", series.max()),
            ):
                put_metric(
                    values,
                    f"{group_base}.{key_part(column)}.{stat}",
                    column,
                    stat_value,
                )


def add_history(values, prefix, stem, df):
    base = f"{prefix}.{stem}"
    put(values, f"{base}.rounds", integer_text(len(df)), len(df), 0)
    for column in metric_columns(df, HISTORY_EXCLUDED_COLUMNS):
        series = df[column].dropna()
        if series.empty:
            continue
        name = key_part(column)
        put_metric(values, f"{base}.{name}.last", column, series.iloc[-1])
        put_metric(values, f"{base}.{name}.min", column, series.min())
        put_metric(values, f"{base}.{name}.max", column, series.max())


def add_fg_test_per_sample(values, df):
    """Not-gated means and gate counts of the FG per-sample results, computed as .agent-local/scripts/readme_tables.py does.

    A sample is gated when its RIFE 4.26 output equals frame repeat exactly (the cut gate replaced it). Means are taken over
    the samples that were not gated, for the suites that are not synthetic-*. Only samples with cut == 0 and menu == 0 count.
    """
    base = "fg.test_per_sample"
    s = df[(df["cut"] == 0) & (df["menu"] == 0)].copy()
    s["sid"] = s.groupby(["suite", "mode", "inputs", "method"]).cumcount()
    repeat = s[s["method"] == "repeat"][
        ["suite", "mode", "inputs", "sid", "psnr"]
    ].rename(columns={"psnr": "repeat_psnr"})
    rife = s[s["method"] == "rife_4.26"].merge(
        repeat, on=["suite", "mode", "inputs", "sid"]
    )
    rife["gated"] = rife["psnr"] == rife["repeat_psnr"]
    counts = (
        rife.groupby(["suite", "mode", "inputs"])
        .agg(samples=("gated", "size"), gated=("gated", "sum"))
        .reset_index()
    )
    for record in counts.to_dict("records"):
        group_base = f"{base}.{key_part(record['suite'])}.{key_part(record['mode'])}.{key_part(record['inputs'])}"
        samples, gated = int(record["samples"]), int(record["gated"])
        put(values, f"{group_base}.n", integer_text(samples), samples, 0)
        put(values, f"{group_base}.gated", integer_text(gated), gated, 0)
        put(
            values,
            f"{group_base}.kept",
            integer_text(samples - gated),
            samples - gated,
            0,
        )
    kept = rife.loc[~rife["gated"], ["suite", "mode", "inputs", "sid"]]
    not_gated = s.merge(kept, on=["suite", "mode", "inputs", "sid"])
    not_gated = not_gated[~not_gated["suite"].str.startswith("synthetic-")]
    means = (
        not_gated.groupby(["suite", "mode", "inputs", "method"])[FG_PER_SAMPLE_METRICS]
        .mean()
        .reset_index()
    )
    for record in means.to_dict("records"):
        group_base = f"{base}.{key_part(record['suite'])}.{key_part(record['mode'])}.{key_part(record['inputs'])}"
        for metric in FG_PER_SAMPLE_METRICS:
            value = record[metric]
            if pd.isna(value):
                continue
            put_metric(
                values,
                f"{group_base}.{key_part(record['method'])}.{key_part(metric)}",
                metric,
                value,
            )


def add_json(values, prefix, stem, data, keep_text):
    """Flatten a JSON document to numeric leaves (booleans are skipped). Text leaves are kept only when keep_text is set,
    and never for paths. Lists are indexed; keys containing '/' are split into path segments.
    """
    base = f"{prefix}.{stem}"

    def walk(node, path):
        if isinstance(node, dict):
            for name, child in node.items():
                walk(child, path + str(name).split("/"))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path + [str(index)])
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            parts = [key_part(part) for part in path if key_part(part)]
            key = ".".join([base] + parts)
            if stem == "config" and isinstance(node, float):
                text, decimals = significant(node)
                put(values, key, text, float(node), decimals)
            else:
                put_metric(values, key, path[-1], node)
        elif (
            isinstance(node, str) and keep_text and "/" not in node and "\\" not in node
        ):
            parts = [key_part(part) for part in path if key_part(part)]
            put(values, ".".join([base] + parts), escape_text(node), None, None)

    walk(data, [])


def add_middlebury(values, errors):
    if not MIDDLEBURY_CSV.exists():
        errors.append(
            f"{MIDDLEBURY_CSV.relative_to(ROOT).as_posix()} is missing: run tools/compare_fg_tiers_middlebury.py"
        )
        return
    df = pd.read_csv(MIDDLEBURY_CSV)
    add_table(values, "mb", "middlebury_psnr", df, ["sequence", "method"])
    sequences = int(df.loc[df["sequence"] != "mean", "sequence"].nunique())
    put(
        values,
        "mb.middlebury_psnr.sequences",
        integer_text(sequences),
        sequences,
        0,
    )


def add_camera_prior_groups(values, df):
    """Samples split by the camera prior setting (test_per_sample.csv: the reversal suite; other suites have no flag)."""
    prior = df.dropna(subset=["camera_prior"]).copy()
    prior["camera_prior"] = (
        prior["camera_prior"]
        .astype(str)
        .str.lower()
        .map({"true": "prior_on", "false": "prior_off"})
    )
    prior = prior.dropna(subset=["camera_prior"])
    add_groups(
        values,
        "fg",
        "camera_prior",
        prior,
        ["suite", "mode", "inputs", "camera_prior", "method"],
    )


def add_metrics(values, errors):
    """Every metrics file of the three notebooks, through its rule. A file without a rule is an error, so that no
    number is silently left out."""
    for prefix, notebook in NOTEBOOK_DIRS.items():
        metrics_dir = notebook / "outputs" / "metrics"
        for path in sorted(metrics_dir.glob("*.csv")):
            stem = path.stem
            df = pd.read_csv(path)
            if df.columns[0].startswith("Unnamed"):
                df = df.rename(columns={df.columns[0]: "stage"})
            if stem.startswith("history_"):
                add_history(values, prefix, stem, df)
            elif (prefix, stem) in GROUPS:
                add_groups(values, prefix, stem, df, GROUPS[(prefix, stem)])
            elif (prefix, stem) == ("fg", "test_per_sample"):
                add_fg_test_per_sample(values, df)
                add_camera_prior_groups(values, df)
            elif stem in TABLES.get(prefix, {}):
                add_table(values, prefix, stem, df, TABLES[prefix][stem])
            else:
                errors.append(
                    f"{path.relative_to(ROOT).as_posix()}: no rule for this metrics file"
                )
        for path in sorted(metrics_dir.glob("*.json")):
            if path.stem.endswith("_history"):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            add_json(values, prefix, path.stem, data, keep_text=path.stem == "hardware")


def add_runs(values, errors):
    """Start and end date and total hours of each notebook run, from its log (clock times) and its notebook (dates)."""
    for prefix, notebook in NOTEBOOK_DIRS.items():
        name = notebook.name
        log = notebook / "outputs" / "logs" / "notebook.log"
        executed = notebook / "outputs" / f"{name}.executed.ipynb"
        source = executed if executed.exists() else notebook / f"{name}.ipynb"
        log_text = log.read_text(encoding="utf-8")
        clocks = re.findall(r"^(\d{2}:\d{2}:\d{2}) \|", log_text, flags=re.M)
        hours = re.findall(r"run total ([0-9.]+) h\)", log_text)
        stamps = [
            datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
            for stamp in re.findall(
                r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^"]*)"',
                source.read_text(encoding="utf-8"),
            )
        ]
        if not clocks or not hours or not stamps:
            errors.append(
                f"{name}: run start, end or hours not found in the log and notebook"
            )
            continue
        start = datetime.combine(
            min(stamps).date(), datetime.strptime(clocks[0], "%H:%M:%S").time()
        )
        end = datetime.combine(
            max(stamps).date(), datetime.strptime(clocks[-1], "%H:%M:%S").time()
        )
        if end < start:
            end += timedelta(days=1)
        put(values, f"run.{name}.start", f"{start.day} {start:%B %Y}", None, None)
        put(values, f"run.{name}.end", f"{end.day} {end:%B %Y}", None, None)
        put_metric(values, f"run.{name}.hours", "hours", float(hours[-1]))


def sync_figures(errors):
    """Recreate each figure that is a copy or a crop of a notebook output, whenever it differs from the source."""
    for figure, source in FIGURE_SOURCES.items():
        source_path = ROOT / source
        target = PAPER / figure
        if not source_path.exists():
            errors.append(f"figure source missing: {source}")
            continue
        if target.exists() and target.read_bytes() == source_path.read_bytes():
            continue
        shutil.copyfile(source_path, target)
        print(f"figure updated: paper/{figure} <- {source}")
    for figure, (source, box) in FIGURE_CROPS.items():
        source_path = ROOT / source
        target = PAPER / figure
        if not source_path.exists():
            errors.append(f"figure source missing: {source}")
            continue
        crop = Image.open(source_path).convert("RGB").crop(box)
        if target.exists() and np.array_equal(
            np.asarray(Image.open(target).convert("RGB")), np.asarray(crop)
        ):
            continue
        crop.save(target)
        print(f"figure updated: paper/{figure} <- {source} crop {box}")


def scan_paper_uses():
    uses = []
    for path in sorted(PAPER.glob("*.tex")):
        if path.name in NOT_SCANNED:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            code = re.sub(r"(?<!\\)%.*$", "", line)
            location = f"{path.relative_to(ROOT).as_posix()}:{number}"
            for match in RES_PAIR.finditer(code):
                macro, first, second = match.groups()
                uses.append(
                    Use(
                        location,
                        macro,
                        f"{DERIVED_PREFIX[macro]}:{first}:{second}",
                        (first, second),
                    )
                )
            for match in RES_SINGLE.finditer(code):
                uses.append(Use(location, "Res", match.group(1), (match.group(1),)))
    return uses


def add_derived(values, uses, errors):
    for use in uses:
        if use.macro == "Res" or use.produces in values:
            continue
        first, second = use.depends
        unknown = [
            key
            for key in use.depends
            if key not in values or values[key].number is None
        ]
        if unknown:
            errors.append(
                f"{use.location}: \\{use.macro} uses unknown key {unknown[0]}"
            )
            continue
        a, b = values[first], values[second]
        if use.macro == "ResDiff":
            difference = a.number - b.number
            text = (
                "+" + fixed(difference, a.decimals)
                if round(difference, a.decimals) > 0
                else fixed(difference, a.decimals)
            )
            values[use.produces] = Value(text, difference, a.decimals)
        elif use.macro == "ResAbsDiff":
            difference = abs(a.number - b.number)
            values[use.produces] = Value(
                fixed(difference, a.decimals), difference, a.decimals
            )
        else:
            if b.number == 0:
                errors.append(f"{use.location}: \\ResRatio divides by zero ({second})")
                continue
            ratio = a.number / b.number
            values[use.produces] = Value(fixed(ratio, 2), ratio, 2)


def check_res_uses(values, uses, errors):
    for use in uses:
        if use.macro == "Res" and use.produces not in values:
            errors.append(f"{use.location}: \\Res{{{use.produces}}} is not a known key")


def read_previous():
    if not OUTPUT.exists():
        return None
    pattern = re.compile(r"^\\SetResult\{([^{}]+)\}\{(.*)\}$")
    previous = {}
    for line in OUTPUT.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            previous[match.group(1)] = match.group(2)
    return previous


def report_changes(previous, values, uses):
    if previous is None:
        print("results-values.tex does not exist yet: every key is new")
    old = previous or {}
    added = sorted(set(values) - set(old))
    removed = sorted(set(old) - set(values))
    changed = sorted(
        key for key in set(values) & set(old) if old[key] != values[key].text
    )
    print(f"keys: added {len(added)}, removed {len(removed)}, changed {len(changed)}")
    for key in added:
        print(f"  added    {key} = {values[key].text}")
    for key in removed:
        print(f"  removed  {key} (was {old[key]})")
        print_uses(key, uses)
    for key in changed:
        print(f"  changed  {key}: {old[key]} -> {values[key].text}")
        print_uses(key, uses)


def print_uses(key, uses):
    for use in uses:
        if use.produces == key or key in use.depends:
            print(f"           {use.location} (\\{use.macro})")


def write_output(values):
    lines = [
        "%% Generated by tools/build_paper_results.py. Do not edit: run the script to regenerate this file.",
        "%% Sources: neural-data-pipeline, neural-supersampling and neural-frame-generation outputs/metrics (csv and json),",
        "%% their notebook logs and executed notebooks, and neural-frame-generation/local-evaluation/middlebury_psnr.csv.",
        "%% Each line sets one value, read by \\Res{key} (see paper-preamble.tex).",
    ]
    lines += [
        "\\SetResult{" + key + "}{" + values[key].text + "}" for key in sorted(values)
    ]
    OUTPUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def print_key_counts(values):
    counts = {}
    for key in values:
        prefix = key.split(":")[0] if ":" in key else key.split(".")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    print(
        "keys per prefix:",
        ", ".join(f"{prefix} {count}" for prefix, count in sorted(counts.items())),
    )


def main():
    errors = []
    sync_figures(errors)
    values = {}
    add_metrics(values, errors)
    add_middlebury(values, errors)
    add_runs(values, errors)
    uses = scan_paper_uses()
    previous = read_previous()
    add_derived(values, uses, errors)
    check_res_uses(values, uses, errors)
    report_changes(previous, values, uses)
    write_output(values)
    print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()} with {len(values)} keys")
    print_key_counts(values)
    for error in errors:
        print("ERROR", error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
