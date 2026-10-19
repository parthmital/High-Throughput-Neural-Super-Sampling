"""Shared notebook runtime: output folders, logging, stage timing, figure saving, hardware report, pip helper."""

import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import psutil
import torch

WORK = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
SUBDIRS = ["code", "weights", "plots", "metrics", "logs", "predictions"]


class Run:
    """Holds the run clock, output folders, logger and per-stage timings of one notebook run."""

    def __init__(self, name, work=WORK):
        self.start = time.time()
        self.work = Path(work)
        self.dirs = {d: self.work / d for d in SUBDIRS}
        for folder in self.dirs.values():
            folder.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        self.log.propagate = False
        for handler in (
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(self.dirs["logs"] / "notebook.log"),
        ):
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
            )
            self.log.addHandler(handler)
        self.stage_times = {}

    def hours(self):
        return (time.time() - self.start) / 3600

    @contextmanager
    def stage(self, name):
        t0 = time.time()
        self.log.info("start: %s", name)
        try:
            yield
        finally:
            self.stage_times[name] = time.time() - t0
            self.log.info(
                "done: %s in %.1f min (run total %.2f h)",
                name,
                self.stage_times[name] / 60,
                self.hours(),
            )

    def show(self, fig, name):
        from IPython.display import display

        fig.savefig(self.dirs["plots"] / f"{name}.png", dpi=120, bbox_inches="tight")
        display(fig)
        plt.close(fig)

    def save_json(self, obj, name):
        with open(self.dirs["metrics"] / name, "w") as f:
            json.dump(obj, f, indent=2, default=str)

    def save_csv(self, df, name, **kwargs):
        df.to_csv(self.dirs["metrics"] / name, **kwargs)


def hardware_report(work=WORK):
    """GPU, CPU, RAM and disk detected at run time (nothing hard-coded)."""
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "gpu": i,
                "name": p.name,
                "vram_gb": round(p.total_memory / 2**30, 2),
                "sm": p.multi_processor_count,
                "cc": f"{p.major}.{p.minor}",
            }
        )
    vm = psutil.virtual_memory()
    report = {
        "gpus": len(gpus),
        "gpu_list": gpus,
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
        "disk_free_gb_working": round(shutil.disk_usage(work).free / 2**30, 1),
        "disk_free_gb_tmp": (
            round(shutil.disk_usage("/tmp").free / 2**30, 1)
            if Path("/tmp").exists()
            else None
        ),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    return report


def pip_install(packages, log=None):
    """Install missing packages quietly; returns {package: importable}. Needs Kaggle Internet on."""
    status = {}
    for pip_name, module in packages.items():
        try:
            importlib.import_module(module)
            status[pip_name] = True
            continue
        except ImportError:
            pass
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pip_name],
            capture_output=True,
            text=True,
        )
        try:
            importlib.invalidate_caches()
            importlib.import_module(module)
            status[pip_name] = True
        except ImportError:
            status[pip_name] = False
            if log:
                log.warning(
                    "could not install %s: %s", pip_name, result.stderr.strip()[-300:]
                )
    return status


def gpu_utilisation():
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return [
            [float(v) for v in line.split(",")] for line in out.strip().splitlines()
        ]
    except Exception:
        return []


def stream_process(cmd, env, on_line, log_path):
    """Run a subprocess, call on_line for every stdout line, mirror everything into log_path; returns exit code."""
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            on_line(line.rstrip("\n"))
        return proc.wait()


def launch_training(
    script, cfg, cfg_path, log_path, gpus, desc, port=29500, on_round=None
):
    """Run a training script on the given GPUs (torch.distributed.run for several GPUs, a plain process for one),
    write its config, mirror its output to log_path, drive a tqdm bar from PROGRESS lines and collect ROUND rows.
    Returns (rounds, done). Raises with the log tail if the process fails."""
    from tqdm.auto import tqdm

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    env = os.environ.copy()
    env.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus),
        }
    )
    if len(gpus) > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={len(gpus)}",
            f"--master_port={port}",
            str(script),
            "--config",
            str(cfg_path),
        ]
    else:
        env["CUDA_DEVICE"] = "0"
        cmd = [sys.executable, str(script), "--config", str(cfg_path)]
    rounds, done, tail = [], {}, []
    bar = tqdm(
        total=100,
        desc=desc,
        unit="%",
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}% [{elapsed}<{remaining}] {postfix}",
    )

    def on_line(line):
        tail.append(line)
        del tail[:-40]
        kind, _, payload = line.partition(" ")
        if kind in ("PROGRESS", "ROUND", "DONE", "LOG"):
            try:
                data = json.loads(payload)
            except ValueError:
                return
            if kind == "PROGRESS":
                bar.n = round(100 * data["progress"], 1)
                bar.set_postfix(
                    loss=f"{data['loss']:.4f}",
                    sps=f"{data['samples_per_s']:.0f}",
                    lr=f"{data['lr']:.1e}",
                    mem=data["mem_gb"],
                )
                bar.refresh()
            elif kind == "ROUND":
                rounds.append(data)
                if on_round:
                    on_round(rounds)
            elif kind == "DONE":
                done.update(data)
            else:
                print(f"[{desc}] {data.get('msg')}", flush=True)
        elif "Error" in line or "Traceback" in line:
            print(f"[{desc}] {line}", flush=True)

    code = stream_process(cmd, env, on_line, log_path)
    bar.n = 100
    bar.close()
    if code != 0:
        raise RuntimeError(
            f"{desc} failed with exit code {code}; last lines:\n"
            + "\n".join(tail[-15:])
        )
    return rounds, done
