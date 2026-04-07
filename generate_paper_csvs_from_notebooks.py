#!/usr/bin/env python3
"""
generate_paper_csvs_from_notebooks.py

Goal
----
Generate the same CSV datasets produced by:
  - examples/BivariateBicycleCodeAnalysis.ipynb
  - examples/RotatedSurfaceCodeAnalysis.ipynb

Specifically, this reproduces the data for the notebooks' section:
  "## Performing a detailed error rate characterization"
which is the dataset used to plot Logical Error Rate vs Physical Error Rate.

It also optionally generates the gamma0 sweep dataset from:
  "## Tuning Relay-BP memory factor"

This script is designed for sbatch:
  - uses sinter multiprocessing (num_workers ~ allocated CPUs)
  - supports Slurm array sharding so each node writes a separate CSV

Outputs
-------
Writes consolidated sinter CSVs to ./data/, matching the notebooks' naming:
  data/{notebook_str}_{experiment_name}.csv

Where:
  - notebook_str for BB: bicycle_bivariate_{n}_{k}_{d}
  - notebook_str for surface: rotated_surface_code

Important notebook-matching details
-----------------------------------
BivariateBicycle notebook:
  - circuits: bicycle_bivariate_{n}_{k}_{d}_memory_X and _Z
  - distance: d=12 (when test=False)
  - rounds = d
  - error_rates = [0.001, 0.002, 0.003, 0.004, 0.005]
  - relay defaults: gamma0=0.15, gamma_dist=[-0.226..., 0.6216...], num_sets=300, pre_iter=80, set_max_iter=60, stop_nconv=5
  - includes "bp-osd" via stimbposd.SinterDecoder_BPOSD (osd_order=10)
  - XZ tasks: filter_detectors_by_basis(circ, basis); dem=circ.detector_error_model()
  - XYZ tasks: no filtering; dem=circ.detector_error_model()

RotatedSurface notebook:
  - circuits: rotated_surface_code_memory_X and _Z (via testdata.get_test_circuit)
  - distance=11 (when test=False), rounds=distance
  - error_rates = [0.001, 0.002, 0.003, 0.004, 0.005]
  - relay defaults: gamma0=0.65, gamma_dist=[-0.2545..., 0.9851...], num_sets=300, pre_iter=80, set_max_iter=60, stop_nconv=5
  - decoders list includes "pymatching"
  - XZ: filter_detectors_by_basis + dem=detector_error_model(decompose_errors=True)
  - XYZ: no filtering + dem=detector_error_model(decompose_errors=False)
"""

from __future__ import annotations

import argparse
import inspect
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import sinter

import relay_bp.stim as r
from relay_bp.stim.sinter.utils import write_stats


# -------------------------
# CPU / Slurm helpers
# -------------------------

def allocated_cpu_count() -> int:
    """Prefer Slurm allocation; fall back to affinity; then cpu_count()."""
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
    """Avoid BLAS/OpenMP oversubscription inside each worker process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# -------------------------
# Repo testdata import
# -------------------------

def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "tests" / "testdata").exists():
            return p
    raise FileNotFoundError(
        "Could not find relay repo root containing tests/testdata.\n"
        "Run from the repo root or pass --repo-root."
    )


def import_testdata(repo_root: Path):
    tests_dir = repo_root / "tests"
    sys.path.insert(0, str(tests_dir.resolve()))
    # Matches the notebooks' approach (they import testdata helpers from tests/)
    from testdata import get_test_circuit, filter_detectors_by_basis  # type: ignore
    return get_test_circuit, filter_detectors_by_basis


# -------------------------
# Sharding
# -------------------------

def shard_filter(i: int, shard_index: int, num_shards: int) -> bool:
    return (i % num_shards) == shard_index


def shard_tasks(tasks: Iterable[sinter.Task], shard_index: int, num_shards: int) -> Iterator[sinter.Task]:
    for i, t in enumerate(tasks):
        if shard_filter(i, shard_index, num_shards):
            yield t


# -------------------------
# Sinter collect with optional seed kwarg
# (sinter versions differ; this keeps it robust)
# -------------------------

def sinter_collect_optional_seed(*, seed: Optional[int], **kwargs):
    sig = inspect.signature(sinter.collect)
    if seed is not None and "seed" in sig.parameters:
        return sinter.collect(seed=seed, **kwargs)
    return sinter.collect(**kwargs)


# =========================================================
# Plot Group: Bivariate Bicycle Code — PER vs LER curves
# Corresponds to the plot produced in BivariateBicycleCodeAnalysis.ipynb
# cell with sinter.plot_error_rate(...) over fine_ler_sweep_samples.
# =========================================================

def bb_defaults(test: bool):
    # Matches notebook cell "Some good defaults from previous optimization run"
    gamma0 = 0.15000000000000002
    alpha = 1.0
    relay_gamma_dist_interval = [-0.22628432386414646, 0.6216020925981884]

    if test:
        relay_num_sets = 60
        relay_stop_nconv = 1
        osd_order = 0
        msl_max_iter = 200
    else:
        relay_num_sets = 300
        relay_stop_nconv = 5
        osd_order = 10
        msl_max_iter = 10000

    relay_params = dict(
        alpha=alpha,
        gamma0=gamma0,
        gamma_dist_interval=relay_gamma_dist_interval,
        num_sets=relay_num_sets,
        pre_iter=80,
        set_max_iter=60,
        stop_nconv=relay_stop_nconv,
    )
    mem_params = dict(alpha=alpha, max_iter=msl_max_iter, gamma0=gamma0)
    msl_params = dict(alpha=alpha, max_iter=msl_max_iter)

    # The notebook uses BP+OSD via stimbposd with these params:
    osd_params = dict(
        max_bp_iters=80,
        bp_method="minimum_sum",
        osd_order=osd_order,
        osd_method="osd_cs",
    )
    return relay_params, mem_params, msl_params, osd_params


def bb_select_num_shots_notebook(p: float, test: bool) -> int:
    # Matches notebook: fine_select_num_shots when test=False, rough when test=True
    if test:
        # rough_select_num_shots
        if p >= 0.004:
            return 500
        if p < 0.004 and p >= 0.003:
            return 1000
        if p < 0.003 and p >= 0.002:
            return 1000
        return 1000
    else:
        # fine_select_num_shots
        if p >= 0.005:
            return 5_000
        if p < 0.005 and p >= 0.004:
            return 10_000
        if p < 0.004 and p >= 0.003:
            return 100_000
        if p < 0.003 and p >= 0.002:
            return 1_000_000
        return 10_000_000


def bb_bp_select_num_shots_notebook(p: float) -> int:
    # Matches notebook cell "bp_select_num_shots" (used for mem-bp and msl-bp)
    if p >= 0.005:
        return 500
    if p < 0.005 and p >= 0.004:
        return 1_000
    if p < 0.004 and p >= 0.003:
        return 10_000
    if p < 0.003 and p >= 0.002:
        return 50_000
    return 100_000


def generate_bb_fine_ler_tasks(
    *,
    get_test_circuit,
    filter_detectors_by_basis,
    n: int,
    k: int,
    d: int,
    error_rates: List[float],
    relay_params: dict,
    mem_params: dict,
    msl_params: dict,
    osd_params: dict,
    include_xyz: bool,
    test: bool,
):
    circuits = [
        f"bicycle_bivariate_{n}_{k}_{d}_memory_X",
        f"bicycle_bivariate_{n}_{k}_{d}_memory_Z",
    ]
    rounds = d

    # The notebook compares: relay-bp, mem-bp, msl-bp, bp-osd (and whatever sinter_decoders provides)
    # It constructs all_decoders = list(sinter_decoders().keys()) + ["bp-osd"]
    all_decoders = list(r.sinter_decoders().keys()) + ["bp-osd"]

    # We add BP+OSD custom decoder explicitly, as the notebook does.
    from stimbposd import SinterDecoder_BPOSD  # type: ignore
    custom_decoders: Dict[str, object] = r.sinter_decoders(**relay_params)
    custom_decoders["bp-osd"] = SinterDecoder_BPOSD(**osd_params)

    def tasks_for_xyz_flag(xyz: bool):
        for p in error_rates:
            for circuit_name in circuits:
                circ = get_test_circuit(circuit=circuit_name, distance=d, rounds=rounds, error_rate=p)
                if not xyz:
                    # XZ data (used for XZ curves in plot)
                    circ = filter_detectors_by_basis(circ, circuit_name[-1])
                dem = circ.detector_error_model()

                for decoder in all_decoders:
                    md = {
                        "circuit": circuit_name,
                        "p": p,
                        "n": n,
                        "k": k,
                        "d": d,
                        "r": rounds,
                        "xyz": xyz,
                        "decoder_params": {},
                    }

                    num_shots = bb_select_num_shots_notebook(p, test=test)

                    if decoder == "relay-bp":
                        md["decoder_params"] = relay_params
                    elif decoder == "mem-bp":
                        md["decoder_params"] = mem_params
                        num_shots = bb_bp_select_num_shots_notebook(p)
                    elif decoder == "msl-bp":
                        md["decoder_params"] = msl_params
                        num_shots = bb_bp_select_num_shots_notebook(p)
                    elif decoder == "bp-osd":
                        md["decoder_params"] = osd_params

                    yield sinter.Task(
                        circuit=circ,
                        detector_error_model=dem,
                        decoder=decoder,
                        json_metadata=md,
                        collection_options=sinter.CollectionOptions(max_shots=num_shots),
                    )

    # Yield XZ always; XYZ optionally
    yield from tasks_for_xyz_flag(False)
    if include_xyz:
        yield from tasks_for_xyz_flag(True)

    return custom_decoders


def generate_bb_gamma0_sweep_tasks(
    *,
    get_test_circuit,
    filter_detectors_by_basis,
    n: int,
    k: int,
    d: int,
    opt_error_rate: float,
    test: bool,
):
    # Matches notebook generate_gamma0_tasks / membp_gamma0s setup (mem-bp only)
    circuits = [
        f"bicycle_bivariate_{n}_{k}_{d}_memory_X",
        f"bicycle_bivariate_{n}_{k}_{d}_memory_Z",
    ]
    membp_gamma0s = np.linspace(0, 1, 11 if test else 21).tolist()
    membp_sweep_xyz = False  # notebook sets False for BB

    for gamma0 in membp_gamma0s:
        decoder_params = dict(alpha=1.0, max_iter=80, gamma0=gamma0)
        for circuit_name in circuits:
            circ = get_test_circuit(circuit=circuit_name, distance=d, rounds=d, error_rate=opt_error_rate)
            if not membp_sweep_xyz:
                circ = filter_detectors_by_basis(circ, circuit_name[-1])
            dem = circ.detector_error_model()
            yield sinter.Task(
                circuit=circ,
                detector_error_model=dem,
                decoder="mem-bp",
                json_metadata={
                    "circuit": circuit_name,
                    "p": opt_error_rate,
                    "n": n,
                    "k": k,
                    "d": d,
                    "r": d,
                    "decoder_params": decoder_params,
                    "xyz": membp_sweep_xyz,
                },
                collection_options=sinter.CollectionOptions(max_shots=1000),
            )


# =========================================================
# Plot Group: Rotated Surface Code — PER vs LER curves
# Corresponds to the plot produced in RotatedSurfaceCodeAnalysis.ipynb
# cell with sinter.plot_error_rate(...) over fine_ler_sweep_samples.
# =========================================================

def surface_defaults(test: bool):
    # Matches notebook cell "Some good defaults"
    gamma0 = 0.65
    alpha = 1.0
    relay_gamma_dist_interval = [-0.25453802248, 0.98518104204]

    if test:
        relay_num_sets = 60
        relay_stop_nconv = 1
        msl_max_iter = 200
    else:
        relay_num_sets = 300
        relay_stop_nconv = 5
        msl_max_iter = 10000

    relay_params = dict(
        alpha=alpha,
        gamma0=gamma0,
        gamma_dist_interval=relay_gamma_dist_interval,
        num_sets=relay_num_sets,
        pre_iter=80,
        set_max_iter=60,
        stop_nconv=relay_stop_nconv,
    )
    mem_params = dict(alpha=alpha, max_iter=msl_max_iter, gamma0=gamma0)
    msl_params = dict(alpha=alpha, max_iter=msl_max_iter)
    return relay_params, mem_params, msl_params


def surface_select_num_shots_notebook(p: float, test: bool) -> int:
    # Matches notebook: select_num_shots = fine_select_num_shots when test=False
    if test:
        # rough_select_num_shots
        if p >= 0.005:
            return 500
        if p < 0.005 and p >= 0.004:
            return 500
        if p < 0.004 and p >= 0.003:
            return 1_000
        if p < 0.003 and p >= 0.002:
            return 5_000
        return 5_000
    else:
        # fine_select_num_shots
        if p >= 0.005:
            return 5_000
        if p < 0.005 and p >= 0.004:
            return 10_000
        if p < 0.004 and p >= 0.003:
            return 100_000
        if p < 0.003 and p >= 0.002:
            return 1_000_000
        return 10_000_000


def generate_surface_fine_ler_tasks(
    *,
    get_test_circuit,
    filter_detectors_by_basis,
    distance: int,
    error_rates: List[float],
    relay_params: dict,
    mem_params: dict,
    msl_params: dict,
    include_xyz: bool,
    test: bool,
):
    circuits = ["rotated_surface_code_memory_X", "rotated_surface_code_memory_Z"]
    rounds = distance

    # Notebook: all_decoders = list(sinter_decoders().keys()) + ["pymatching"]
    all_decoders = list(r.sinter_decoders().keys()) + ["pymatching"]

    custom_decoders: Dict[str, object] = r.sinter_decoders(**relay_params)

    def tasks_for_xyz_flag(xyz: bool):
        for p in error_rates:
            for circuit_name in circuits:
                circ = get_test_circuit(circuit=circuit_name, distance=distance, rounds=rounds, error_rate=p)

                if not xyz:
                    # XZ mode: basis filter + decompose_errors=True
                    circ = filter_detectors_by_basis(circ, circuit_name[-1])
                    dem = circ.detector_error_model(decompose_errors=True)
                else:
                    # XYZ mode: no filter + decompose_errors=False
                    dem = circ.detector_error_model(decompose_errors=False)

                for decoder in all_decoders:
                    md = {
                        "circuit": circuit_name,
                        "p": p,
                        "d": distance,
                        "xyz": xyz,
                        "decoder_params": {},
                    }

                    num_shots = surface_select_num_shots_notebook(p, test=test)

                    if decoder == "relay-bp":
                        md["decoder_params"] = relay_params
                    elif decoder == "mem-bp":
                        md["decoder_params"] = mem_params
                        num_shots = 5000 if test else 5000  # notebook uses rough_select for mem/msl here
                    elif decoder == "msl-bp":
                        md["decoder_params"] = msl_params
                        num_shots = 5000 if test else 5000

                    yield sinter.Task(
                        circuit=circ,
                        detector_error_model=dem,
                        decoder=decoder,
                        json_metadata=md,
                        collection_options=sinter.CollectionOptions(max_shots=num_shots),
                    )

    yield from tasks_for_xyz_flag(False)
    if include_xyz:
        yield from tasks_for_xyz_flag(True)

    return custom_decoders


def generate_surface_gamma0_sweep_tasks(
    *,
    get_test_circuit,
    filter_detectors_by_basis,
    distance: int,
    opt_error_rate: float,
    test: bool,
):
    # Matches notebook setup: membp_sweep_xyz = True for surface when test=False
    circuits = ["rotated_surface_code_memory_X", "rotated_surface_code_memory_Z"]
    membp_gamma0s = np.linspace(0, 1, 11 if test else 21).tolist()
    membp_sweep_xyz = True if not test else False

    for gamma0 in membp_gamma0s:
        decoder_params = dict(alpha=1.0, max_iter=80, gamma0=gamma0)
        for circuit_name in circuits:
            circ = get_test_circuit(circuit=circuit_name, distance=distance, rounds=distance, error_rate=opt_error_rate)
            if not membp_sweep_xyz:
                circ = filter_detectors_by_basis(circ, circuit_name[-1])
            dem = circ.detector_error_model(decompose_errors=not membp_sweep_xyz)
            yield sinter.Task(
                circuit=circ,
                detector_error_model=dem,
                decoder="mem-bp",
                json_metadata={
                    "circuit": circuit_name,
                    "p": opt_error_rate,
                    "d": distance,
                    "r": distance,
                    "decoder_params": decoder_params,
                    "xyz": membp_sweep_xyz,
                },
                collection_options=sinter.CollectionOptions(max_shots=1000),
            )


# -------------------------
# Main
# -------------------------

def main():
    set_thread_env_to_1()

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=None, help="Path to relay repo root; otherwise inferred.")
    ap.add_argument("--num-workers", type=int, default=None, help="Override worker count; default uses allocation.")
    ap.add_argument("--seed", type=int, default=None, help="Sinter seed if supported; also recorded in filenames/metadata.")
    ap.add_argument("--max-errors", type=int, default=200, help="Stop each task after this many errors.")
    ap.add_argument("--start-batch-size", type=int, default=10_000, help="Matches notebooks' sinter_collect default.")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SINTER_NUM_SHARDS", "1")))
    ap.add_argument("--which", choices=["bb", "surface", "all"], default="all")
    ap.add_argument("--test", action="store_true", help="Notebook test mode (small/fast).")
    ap.add_argument("--include-xyz", action="store_true", help="Also generate XYZ tasks (no basis filtering).")
    ap.add_argument("--do-gamma0-sweep", action="store_true", help="Also generate the gamma0 sweep dataset.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    get_test_circuit, filter_detectors_by_basis = import_testdata(repo_root)

    num_workers = args.num_workers if args.num_workers is not None else allocated_cpu_count()

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=== notebook-matched data generation ===")
    print("repo_root:", repo_root)
    print("num_workers:", num_workers)
    print("seed:", args.seed)
    print("shard:", f"{args.shard_index}/{args.num_shards}")
    print("include_xyz:", args.include_xyz)
    print("do_gamma0_sweep:", args.do_gamma0_sweep)
    print()

    # Notebook error_rates (both notebooks when test=False)
    if args.test:
        error_rates = [0.001, 0.002, 0.003]
    else:
        error_rates = [0.001, 0.002, 0.003, 0.004, 0.005]

    # BB parameters (notebook defaults when test=False)
    if args.test:
        n, k, d = 18, 4, 3
        opt_error_rate = 0.001
    else:
        n, k, d = 144, 12, 12
        opt_error_rate = 0.003

    # Surface parameters (notebook defaults when test=False)
    distance = 5 if args.test else 11
    surface_opt_error_rate = 0.003

    # Collect tasks, shard them, run sinter.collect, then consolidate with write_stats.
    # This mimics the notebooks' sinter_collect() helper behavior.

    def run_experiment(tasks: Iterable[sinter.Task], custom_decoders: Dict[str, object], out_csv: Path):
        sharded = list(shard_tasks(tasks, args.shard_index, args.num_shards))
        print(f"Running: {out_csv.name}")
        print(f"  tasks total/shard: (unknown)/{len(sharded)}")
        samples = sinter_collect_optional_seed(
            seed=args.seed,
            tasks=sharded,
            num_workers=num_workers,
            custom_decoders=custom_decoders,
            start_batch_size=args.start_batch_size,
            save_resume_filepath=str(out_csv),
            existing_data_filepaths=[str(out_csv)] if out_csv.exists() else [],
            print_progress=True,
            max_errors=args.max_errors,
        )
        write_stats(samples, str(out_csv))
        print(f"  wrote: {out_csv}")
        print()

    # -------------------------
    # Plot Group (BB): PER vs LER curves
    # -------------------------
    if args.which in ("bb", "all"):
        notebook_str = f"bicycle_bivariate_{n}_{k}_{d}"

        relay_params, mem_params, msl_params, osd_params = bb_defaults(args.test)

        # Build BB fine sweep tasks (THIS DATA POWERS THE NOTEBOOK'S MAIN PER→LER PLOT)
        bb_tasks = []
        gen = generate_bb_fine_ler_tasks(
            get_test_circuit=get_test_circuit,
            filter_detectors_by_basis=filter_detectors_by_basis,
            n=n, k=k, d=d,
            error_rates=error_rates,
            relay_params=relay_params,
            mem_params=mem_params,
            msl_params=msl_params,
            osd_params=osd_params,
            include_xyz=args.include_xyz,
            test=args.test,
        )
        # generate_bb_fine_ler_tasks returns a generator; custom_decoders is returned at end, so we rebuild cleanly:
        # (keep logic simple: build custom_decoders explicitly here)
        from stimbposd import SinterDecoder_BPOSD  # type: ignore
        bb_custom_decoders: Dict[str, object] = r.sinter_decoders(**relay_params)
        bb_custom_decoders["bp-osd"] = SinterDecoder_BPOSD(**osd_params)

        # Re-create tasks without consuming generator-return trick:
        def bb_task_iter():
            yield from generate_bb_fine_ler_tasks(
                get_test_circuit=get_test_circuit,
                filter_detectors_by_basis=filter_detectors_by_basis,
                n=n, k=k, d=d,
                error_rates=error_rates,
                relay_params=relay_params,
                mem_params=mem_params,
                msl_params=msl_params,
                osd_params=osd_params,
                include_xyz=args.include_xyz,
                test=args.test,
            )

        out_csv = data_dir / f"{notebook_str}_fine_ler_sweep_shard{args.shard_index}-of-{args.num_shards}_seed{args.seed}.csv"
        run_experiment(bb_task_iter(), bb_custom_decoders, out_csv)

        # Optional: gamma0 sweep dataset (Tuning Relay-BP memory factor)
        if args.do_gamma0_sweep:
            out_csv2 = data_dir / f"{notebook_str}_gamma0_sweep_shard{args.shard_index}-of-{args.num_shards}_seed{args.seed}.csv"
            bb_gamma_tasks = generate_bb_gamma0_sweep_tasks(
                get_test_circuit=get_test_circuit,
                filter_detectors_by_basis=filter_detectors_by_basis,
                n=n, k=k, d=d,
                opt_error_rate=opt_error_rate,
                test=args.test,
            )
            # mem-bp is already in r.sinter_decoders()
            run_experiment(bb_gamma_tasks, r.sinter_decoders(**relay_params), out_csv2)

    # -------------------------
    # Plot Group (Surface): PER vs LER curves
    # -------------------------
    if args.which in ("surface", "all"):
        notebook_str = "rotated_surface_code"
        relay_params, mem_params, msl_params = surface_defaults(args.test)
        surface_custom_decoders: Dict[str, object] = r.sinter_decoders(**relay_params)

        def surface_task_iter():
            yield from generate_surface_fine_ler_tasks(
                get_test_circuit=get_test_circuit,
                filter_detectors_by_basis=filter_detectors_by_basis,
                distance=distance,
                error_rates=error_rates,
                relay_params=relay_params,
                mem_params=mem_params,
                msl_params=msl_params,
                include_xyz=args.include_xyz,
                test=args.test,
            )

        out_csv = data_dir / f"{notebook_str}_fine_ler_sweep_shard{args.shard_index}-of-{args.num_shards}_seed{args.seed}.csv"
        run_experiment(surface_task_iter(), surface_custom_decoders, out_csv)

        # Optional: gamma0 sweep dataset (Tuning Relay-BP memory factor)
        if args.do_gamma0_sweep:
            out_csv2 = data_dir / f"{notebook_str}_gamma0_sweep_shard{args.shard_index}-of-{args.num_shards}_seed{args.seed}.csv"
            surface_gamma_tasks = generate_surface_gamma0_sweep_tasks(
                get_test_circuit=get_test_circuit,
                filter_detectors_by_basis=filter_detectors_by_basis,
                distance=distance,
                opt_error_rate=surface_opt_error_rate,
                test=args.test,
            )
            run_experiment(surface_gamma_tasks, r.sinter_decoders(**relay_params), out_csv2)

    print("DONE.")


if __name__ == "__main__":
    main()
