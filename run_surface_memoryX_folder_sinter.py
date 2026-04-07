#!/usr/bin/env python3
"""
run_surface_memoryX_folder_sinter.py

Runs sinter on rotated surface code memory_X circuits stored as .stim files under:
  ./tests/testdata/surface   (or --circuits-dir)

Filters to:
  circuit=rotated_surface_code_memory_X

Decoders (matching your plot):
  mem-bp (XYZ=False/True)
  msl-bp (XYZ=False/True)
  relay-bp (XYZ=False/True)
  pymatching (XYZ=False only)

XZ vs XYZ matches RotatedSurfaceCodeAnalysis.ipynb:
  - XYZ=False: filter_detectors_by_basis(circuit, "X") + DEM(decompose_errors=True)
  - XYZ=True : no filtering + DEM(decompose_errors=False)

Early-stop:
  --max-errors N stops each task after N logical failures (or max_shots reached).

Output:
  results_surface_x/surfaceX_seed<seed>.csv
"""

from __future__ import annotations

import argparse
import inspect
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional

import stim
import sinter

import relay_bp.stim as relay_stim
from relay_bp.stim.sinter.utils import write_stats


# ---------------------------
# CPU helpers (Slurm-friendly)
# ---------------------------

def allocated_cpu_count() -> int:
    s = os.environ.get("SLURM_CPUS_PER_TASK")
    if s:
        try:
            n = int(s)
            if n > 0:
                return n
        except ValueError:
            pass
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        return multiprocessing.cpu_count()

def set_thread_env_to_1() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------
# Import filter_detectors_by_basis from repo tests/testdata
# ---------------------------

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "tests" / "testdata").exists():
            return p
    raise FileNotFoundError("Could not find repo root with tests/testdata. Run from repo root or pass --repo-root.")

def import_filter_detectors_by_basis(repo_root: Path):
    tests_dir = repo_root / "tests"
    sys.path.insert(0, str(tests_dir.resolve()))
    from testdata import filter_detectors_by_basis  # type: ignore
    return filter_detectors_by_basis


# ---------------------------
# sinter.collect seed compatibility
# ---------------------------

def sinter_collect_optional_seed(*, seed: Optional[int], **kwargs):
    sig = inspect.signature(sinter.collect)
    if seed is not None and "seed" in sig.parameters:
        return sinter.collect(seed=seed, **kwargs)
    return sinter.collect(**kwargs)


# ---------------------------
# Parse the filename key=value,... naming scheme
# ---------------------------

@dataclass(frozen=True)
class Meta:
    path: Path
    circuit: str
    distance: int
    rounds: int
    p: float
    noise_model: str

def parse_kv_filename(path: Path) -> Optional[Meta]:
    name = path.name
    if not name.endswith(".stim"):
        return None
    stem = name[:-5]  # strip .stim
    parts = stem.split(",")
    kv = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()

    required = ("circuit", "distance", "rounds", "error_rate")
    if any(k not in kv for k in required):
        return None

    try:
        return Meta(
            path=path,
            circuit=kv["circuit"],
            distance=int(kv["distance"]),
            rounds=int(kv["rounds"]),
            p=float(kv["error_rate"]),
            noise_model=kv.get("noise_model", ""),
        )
    except Exception:
        return None


# ---------------------------
# Shot schedule (adjustable)
# ---------------------------

def default_max_shots(p: float) -> int:
    if p >= 0.010: return 2_000
    if p >= 0.008: return 3_000
    if p >= 0.006: return 5_000
    if p >= 0.004: return 10_000
    if p >= 0.003: return 50_000
    if p >= 0.002: return 200_000
    if p >= 0.001: return 1_000_000
    return 5_000_000


# ---------------------------
# Decoder bundle (surface optimized params from the notebook)
# ---------------------------

def build_custom_decoders_surface_optimized() -> Dict[str, object]:
    params = dict(
        alpha=1.0,
        gamma0=0.65,
        gamma_dist_interval=[-0.25453802248, 0.98518104204],
        num_sets=300,
        pre_iter=80,
        set_max_iter=60,
        stop_nconv=5,
        max_iter=10000,
    )
    sig = inspect.signature(relay_stim.sinter_decoders)
    filtered = {k: v for k, v in params.items() if k in sig.parameters}
    return relay_stim.sinter_decoders(**filtered)


# ---------------------------
# Build tasks from folder
# ---------------------------

def iter_tasks(
    circuits_dir: Path,
    *,
    filter_detectors_by_basis,
    include_xyz: bool,
    include_pymatching: bool,
    shot_scale: float,
    max_shots_cap: Optional[int],
) -> Iterator[sinter.Task]:

    # Walk all .stim files under the directory
    stim_files = sorted(circuits_dir.rglob("*.stim"))
    if not stim_files:
        raise RuntimeError(f"No .stim files found under {circuits_dir}")

    for f in stim_files:
        meta = parse_kv_filename(f)
        if meta is None:
            continue
        if meta.circuit != "rotated_surface_code_memory_X":
            continue  # ONLY memory_X

        text = f.read_text(encoding="utf-8", errors="replace")
        circ0 = stim.Circuit(text)

        xyz_flags = [False] + ([True] if include_xyz else [])
        for xyz in xyz_flags:
            if not xyz:
                circ = filter_detectors_by_basis(circ0, "X")
                dem = circ.detector_error_model(decompose_errors=True)
            else:
                circ = circ0
                dem = circ.detector_error_model(decompose_errors=False)

            decoders = ["mem-bp", "msl-bp", "relay-bp"]
            if include_pymatching and not xyz:
                decoders.append("pymatching")

            ms = int(default_max_shots(meta.p) * shot_scale)
            if max_shots_cap is not None:
                ms = min(ms, max_shots_cap)
            ms = max(1, ms)

            for dec in decoders:
                yield sinter.Task(
                    circuit=circ,
                    detector_error_model=dem,
                    decoder=dec,
                    collection_options=sinter.CollectionOptions(max_shots=ms),
                    json_metadata={
                        "family": "rotated_surface_code",
                        "file": str(f),
                        "circuit": meta.circuit,
                        "d": meta.distance,
                        "r": meta.rounds,
                        "p": meta.p,
                        "noise_model": meta.noise_model,
                        "xyz": bool(xyz),
                    },
                )


# ---------------------------
# Main
# ---------------------------

def main():
    set_thread_env_to_1()

    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits-dir", type=str, default="tests/testdata/surface",
                    help="Directory containing .stim circuits (default: tests/testdata/surface).")
    ap.add_argument("--repo-root", type=str, default=None, help="Relay repo root (for importing tests/testdata).")
    ap.add_argument("--outdir", type=str, default="results_surface_x")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-errors", type=int, default=200,
                    help="STOP each task when it hits this many logical errors.")
    ap.add_argument("--include-xyz", action="store_true")
    ap.add_argument("--include-pymatching", action="store_true")
    ap.add_argument("--shot-scale", type=float, default=1.0)
    ap.add_argument("--max-shots-cap", type=int, default=None)
    args = ap.parse_args()

    circuits_dir = Path(args.circuits_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"surfaceX_seed{args.seed}.csv"

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    filter_detectors_by_basis = import_filter_detectors_by_basis(repo_root)

    num_workers = args.num_workers if args.num_workers is not None else allocated_cpu_count()

    print("=== run_surface_memoryX_folder_sinter ===")
    print("circuits_dir:", circuits_dir)
    print("out_csv:", out_csv)
    print("seed:", args.seed)
    print("num_workers:", num_workers)
    print("max_errors:", args.max_errors)
    print("include_xyz:", args.include_xyz)
    print("include_pymatching:", args.include_pymatching)
    print("shot_scale:", args.shot_scale)
    print("max_shots_cap:", args.max_shots_cap)
    print()

    custom_decoders = build_custom_decoders_surface_optimized()

    tasks = list(iter_tasks(
        circuits_dir,
        filter_detectors_by_basis=filter_detectors_by_basis,
        include_xyz=args.include_xyz,
        include_pymatching=args.include_pymatching,
        shot_scale=args.shot_scale,
        max_shots_cap=args.max_shots_cap,
    ))
    print("tasks:", len(tasks))

    samples = sinter_collect_optional_seed(
        seed=args.seed,
        tasks=tasks,
        num_workers=num_workers,
        custom_decoders=custom_decoders,
        max_errors=args.max_errors,
        print_progress=True,
        save_resume_filepath=str(out_csv),
        existing_data_filepaths=[str(out_csv)] if out_csv.exists() else [],
    )

    write_stats(samples, str(out_csv))
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
