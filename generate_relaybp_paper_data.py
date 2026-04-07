#!/usr/bin/env python3
"""
generate_relaybp_paper_data.py

Purpose
-------
Generate sinter CSV data for the Relay-BP paper-style experiments using:
  - Relay-BP custom decoders (relay-bp, mem-bp, msl-bp) via relay_bp.stim.sinter_decoders
  - Stim circuits (rotated surface code) and repo-provided test circuits (bivariate bicycle)

This is designed to be run in a Slurm sbatch job (one job == one node), using sinter's
multiprocessing (num_workers ~ allocated CPUs).

Output
------
Writes ONE CSV per run (per seed/shard), which you can later merge.

Notes
-----
- This script mirrors the repo's GettingStarted notebook sinter workflow:
  tasks with json_metadata, save_resume_filepath, write_stats, etc.
- Exact p grids / distances used in the paper may be set in the repo notebooks:
    examples/BivariateBicycleCodeAnalysis.ipynb
    examples/RotatedSurfaceCodeAnalysis.ipynb
  This script provides reasonable defaults + CLI overrides.
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import stim
import sinter

import relay_bp.stim as relay_stim
from relay_bp.stim.sinter.utils import write_stats


# ---------------------------
# Slurm / CPU helpers
# ---------------------------

def _allocated_cpu_count() -> int:
    """
    Best-effort "how many CPUs should I use?"
    - Prefer SLURM_CPUS_PER_TASK when present.
    - Else prefer Linux CPU affinity (sched_getaffinity).
    - Else fallback to multiprocessing.cpu_count().
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            n = int(slurm_cpus)
            if n > 0:
                return n
        except ValueError:
            pass

    # Linux: respect cpuset/cgroup affinity when configured.
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        pass

    return multiprocessing.cpu_count()


def _maybe_set_blas_threads_to_one() -> None:
    """
    When using multiprocessing, avoid oversubscribing cores due to BLAS/OpenMP threads.
    Safe defaults for HPC: 1 thread per process.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ---------------------------
# Repo testdata import
# ---------------------------

def _find_repo_root(start: Path) -> Path:
    """
    Finds a repo root by walking upward until tests/testdata exists.
    """
    for p in [start, *start.parents]:
        if (p / "tests" / "testdata").exists():
            return p
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata.\n"
        "Run from the relay repo root, or pass --repo-root."
    )


def _import_testdata(repo_root: Path):
    """
    Imports testdata helpers from the repo's tests/ directory.
    This matches the repo's GettingStarted example approach.
    """
    tests_dir = repo_root / "tests"
    sys.path.insert(0, str(tests_dir.resolve()))
    from testdata import filter_detectors_by_basis, get_test_circuit  # type: ignore
    return get_test_circuit, filter_detectors_by_basis


# ---------------------------
# Task config / sharding
# ---------------------------

@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    num_shards: int

    def keep(self, i: int) -> bool:
        return (i % self.num_shards) == self.shard_index


def _shard_tasks(tasks: Iterable[sinter.Task], shard: ShardSpec) -> Iterator[sinter.Task]:
    for i, t in enumerate(tasks):
        if shard.keep(i):
            yield t


def _sinter_collect_with_optional_seed(*, seed: Optional[int], **kwargs):
    """
    Sinter API has varied slightly over versions; this keeps the script resilient.
    If sinter.collect supports seed=..., we pass it; otherwise we don't.
    """
    sig = inspect.signature(sinter.collect)
    if seed is not None and "seed" in sig.parameters:
        return sinter.collect(seed=seed, **kwargs)
    return sinter.collect(**kwargs)


# ---------------------------
# Experiment definitions
# ---------------------------

def _default_bb_decoder_params() -> dict:
    """
    Relay parameters tuned for the Gross BB code in the repo's GettingStarted example.
    (Paper performance is sensitive to these parameters; tune if needed.)
    """
    return dict(
        gamma0=0.1,
        pre_iter=80,
        num_sets=300,
        set_max_iter=60,
        gamma_dist_interval=[-0.24, 0.66],
        stop_nconv=5,
    )


def _select_max_shots(p: float, decoder_key: str, *, family: str) -> int:
    """
    Shot scheduling heuristic.
    Mirrors the GettingStarted notebook idea: spend more shots at lower p,
    and typically spend more shots on relay-bp points.
    """
    # Non-relay baselines: keep cheap by default.
    if decoder_key != "relay-bp":
        return 500

    # Relay-BP: more shots deeper in the low-error regime
    # (same structure as the GettingStarted example, adjustable if desired).
    if p >= 0.005:
        return 500
    if 0.004 <= p < 0.005:
        return 1_000
    if 0.003 <= p < 0.004:
        return 5_000
    if 0.002 <= p < 0.003:
        return 20_000
    return 100_000


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


# ---------------------------
# MAIN
# ---------------------------

def main() -> None:
    _maybe_set_blas_threads_to_one()

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=None,
                    help="Path to relay repo root (containing tests/testdata). "
                         "If omitted, script searches upward from CWD.")
    ap.add_argument("--outdir", type=str, default="paper_data_out",
                    help="Directory where per-run CSVs are written.")
    ap.add_argument("--tag", type=str, default="relaybp_paper",
                    help="Prefix used in output filename.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Base seed for sinter (if supported) + recorded in metadata.")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Override worker process count. Default: allocated CPUs.")
    ap.add_argument("--max-errors", type=int, default=200,
                    help="Stop each task after this many logical errors are observed (global cap).")
    ap.add_argument("--p-list", type=str, default=None,
                    help="Comma-separated physical error rates. Overrides p-min/p-max/p-num.")
    ap.add_argument("--p-min", type=float, default=0.002,
                    help="Default min p if --p-list not given.")
    ap.add_argument("--p-max", type=float, default=0.006,
                    help="Default max p if --p-list not given.")
    ap.add_argument("--p-num", type=int, default=9,
                    help="Default number of p points if --p-list not given.")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
                    help="Which shard to run (for Slurm array jobs).")
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("SINTER_NUM_SHARDS", "1")),
                    help="Total shards (for Slurm array jobs).")
    ap.add_argument("--which", type=str, default="all",
                    choices=["all", "bb", "surface"],
                    help="Which experiment group(s) to run.")
    ap.add_argument("--bb-bases", type=str, default="Z",
                    choices=["X", "Z", "both"],
                    help="Which BB basis circuits to run (for CSS split decoding workflows).")
    ap.add_argument("--bb-xyz", action="store_true",
                    help="If set, do NOT filter detectors by basis (XYZ-style decoding).")
    ap.add_argument("--surface-basis", type=str, default="x",
                    choices=["x", "z", "both"],
                    help="Which rotated surface memory basis to run.")
    ap.add_argument("--surface-distances", type=str, default="3,5,7,9,11",
                    help="Comma-separated surface code distances.")
    ap.add_argument("--include-pymatching", action="store_true",
                    help="If set, include 'pymatching' decoder (useful for surface-code curves).")
    args = ap.parse_args()

    shard = ShardSpec(shard_index=args.shard_index, num_shards=args.num_shards)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _find_repo_root(Path.cwd())
    get_test_circuit, filter_detectors_by_basis = _import_testdata(repo_root)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    num_workers = args.num_workers if args.num_workers is not None else _allocated_cpu_count()

    # p grid
    if args.p_list:
        p_values = _parse_float_list(args.p_list)
    else:
        # BB testdata circuits only exist at specific physical error rates.
        # If BB is involved, default to the notebook/testdata grid.
        if args.which in ("all", "bb"):
            p_values = [0.001, 0.002, 0.003, 0.004, 0.005]
        else:
            p_values = [float(x) for x in np.linspace(args.p_min, args.p_max, args.p_num)]

    # Build custom decoders (relay-bp + variants) for sinter
    # -------------------------------
    # DATASET: All paper plots that use Relay-BP / Mem-BP / MSL-BP
    # (These are the same decoder keys you smoke-tested.)
    # -------------------------------
    decoder_params = _default_bb_decoder_params()
    custom_decoders: Dict[str, object] = relay_stim.sinter_decoders(**decoder_params)

    base_decoder_keys: List[str] = list(custom_decoders.keys())
    bb_decoder_keys: List[str] = list(base_decoder_keys)
    surface_decoder_keys: List[str] = list(base_decoder_keys)
    if args.include_pymatching:
        surface_decoder_keys.append("pymatching")

    # Make run output file name
    seed_str = f"seed{args.seed}" if args.seed is not None else "seedNone"
    out_csv = outdir / f"{args.tag}_{seed_str}_shard{shard.shard_index}-of-{shard.num_shards}.csv"

    print("=== Relay-BP paper-style data generation ===")
    print(f"repo_root         : {repo_root}")
    print(f"out_csv           : {out_csv}")
    print(f"num_workers       : {num_workers}")
    print(f"seed              : {args.seed}")
    print(f"shard             : {shard.shard_index}/{shard.num_shards}")
    print(f"p_values          : {p_values}")
    print(f"bb_decoder_keys   : {bb_decoder_keys}")
    print(f"surface_decoder_keys: {surface_decoder_keys}")
    print(f"decoder_params    : {json.dumps(decoder_params)}")
    print()

    all_tasks: List[sinter.Task] = []

    # ============================================================
    # PLOT GROUP A (paper): Bivariate bicycle code logical error-rate curves
    #   - Includes the Gross / Two-Gross type BB code results in the paper.
    #   - Uses repo-provided test circuits via get_test_circuit + optional basis filtering.
    # ============================================================
    if args.which in ("all", "bb"):
        bases: List[str]
        if args.bb_bases == "both":
            bases = ["X", "Z"]
        else:
            bases = [args.bb_bases]

        # Default code list:
        #   The repo's GettingStarted uses [[144,12,12]] with rounds=d.
        #   If you want to match the paper exactly, you can extend this list (or
        #   point it at your own circuit names) to include the "Two-Gross" BB code.
        bb_codes: List[Tuple[int, int, int]] = [
            (144, 12, 12),  # Gross code (as in repo example)
            # Add the "Two-Gross" (n,k,d) triple from the paper/notebook here if desired.
        ]

        for (n, k, d) in bb_codes:
            rounds = d
            for basis in bases:
                circuit_str = f"bicycle_bivariate_{n}_{k}_{d}_memory_{basis}"
                for p in p_values:
                    circuit = get_test_circuit(
                        circuit=circuit_str,
                        distance=d,
                        rounds=rounds,
                        error_rate=p,
                    )

                    # If bb_xyz is False, mimic CSS-style split decoding:
                    # keep only detectors from the chosen basis.
                    if not args.bb_xyz:
                        circuit = filter_detectors_by_basis(circuit, basis)

                    dem = circuit.detector_error_model()

                    for dec in bb_decoder_keys:
                        max_shots = _select_max_shots(p, dec, family="bb")
                        all_tasks.append(
                            sinter.Task(
                                circuit=circuit,
                                detector_error_model=dem,
                                decoder=dec,
                                json_metadata={
                                    # Useful for grouping/plotting later:
                                    "plot_group": "bb_code_curves",  # (paper BB performance plots)
                                    "code_family": "bivariate_bicycle",
                                    "circuit": circuit_str,
                                    "n": n,
                                    "k": k,
                                    "d": d,
                                    "rounds": rounds,
                                    "basis": basis,
                                    "xyz": bool(args.bb_xyz),
                                    "p": float(p),
                                    "decoder_params": decoder_params,
                                    "seed": args.seed,
                                    "shard_index": shard.shard_index,
                                    "num_shards": shard.num_shards,
                                },
                                collection_options=sinter.CollectionOptions(
                                    max_shots=max_shots
                                ),
                            )
                        )

    # ============================================================
    # PLOT GROUP B (paper): Rotated surface code memory logical error curves
    #   - This matches the style of results in the paper comparing Relay-BP vs MWPM.
    #   - Stim-generated circuits.
    # ============================================================
    if args.which in ("all", "surface"):
        distances = [int(x) for x in args.surface_distances.split(",") if x.strip()]
        if args.surface_basis == "both":
            sc_bases = ["x", "z"]
        else:
            sc_bases = [args.surface_basis]

        for b in sc_bases:
            gen_name = f"surface_code:rotated_memory_{b}"
            for d in distances:
                rounds = d
                for p in p_values:
                    circuit = stim.Circuit.generated(
                        gen_name,
                        distance=d,
                        rounds=rounds,
                        after_clifford_depolarization=p,
                    )
                    dem = circuit.detector_error_model()
                    for dec in surface_decoder_keys:
                        max_shots = _select_max_shots(p, dec, family="surface")
                        all_tasks.append(
                            sinter.Task(
                                circuit=circuit,
                                detector_error_model=dem,
                                decoder=dec,
                                json_metadata={
                                    "plot_group": "surface_code_curves",  # (paper surface-code plots)
                                    "code_family": "rotated_surface_code",
                                    "stim_gen": gen_name,
                                    "d": d,
                                    "rounds": rounds,
                                    "basis": b,
                                    "p": float(p),
                                    "seed": args.seed,
                                    "shard_index": shard.shard_index,
                                    "num_shards": shard.num_shards,
                                },
                                collection_options=sinter.CollectionOptions(
                                    max_shots=max_shots
                                ),
                            )
                        )

    # Apply sharding (for Slurm array runs)
    sharded = list(_shard_tasks(all_tasks, shard))
    print(f"Total tasks generated: {len(all_tasks)}")
    print(f"Tasks in this shard  : {len(sharded)}")
    print()

    # Run sinter
    # - save_resume_filepath lets you resume this exact shard file if preempted.
    samples = _sinter_collect_with_optional_seed(
        seed=args.seed,
        tasks=sharded,
        num_workers=num_workers,
        custom_decoders=custom_decoders,
        max_errors=args.max_errors,
        print_progress=True,
        save_resume_filepath=str(out_csv),
    )

    # Overwrite with consolidated CSV (repo utility)
    write_stats(samples, str(out_csv))

    print("\nDONE.")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
