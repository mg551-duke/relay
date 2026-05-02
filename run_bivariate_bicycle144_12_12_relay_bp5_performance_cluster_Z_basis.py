#!/usr/bin/env python3
"""Sinter-based Relay-BP-5 performance sweep with mean iteration tracking.

This is a circuit-level performance runner for the
`bicycle_bivariate_144_12_12_memory_Z` family using Z-basis detector filtering.
It is modeled on `run_bivariate_bicycle144_12_12_native_bpgd_cluster_Z_basis.py`
but focuses on three Relay-BP-5 variants:

1. Relay-BP-5 without damping
2. Relay-BP-5 with random damping in [0.85, 0.95]
3. Relay-BP-5 with trained two-point discrete memory

Shared decoder settings:
- p sweep: 0.001, 0.002, 0.003, 0.004, 0.005
- max logical errors per p: 50
- max shots per p: 1e8
- gamma0 = 0.125
- relay weights sampled from [-0.24, 0.66], except for trained discrete memory
- R = 601 total legs (600 relay continuation legs)
- stop after 5 converged relay legs

The runner still uses `sinter.collect`, but uses a custom sinter sampler so it
can aggregate iteration totals into `custom_counts` without persisting per-shot
detail records.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter
import stim

from relay_bp.stim import CheckMatrices, SinterDecoder_RelayBP
from relay_bp.stim.sinter.runner import _filter_detectors_by_basis
from relay_bp.stim.sinter.utils import write_stats
from sinter._decoding._sampler import CompiledSampler, Sampler

SELECTED_P_VALUES = [0.001, 0.002, 0.003, 0.004, 0.005]
ERRORS_BY_P = {
    0.001: 50,
    0.002: 50,
    0.003: 50,
    0.004: 50,
    0.005: 50,
}
MAX_SHOTS_SAFETY_CAP = 100_000_000

BASIS_FILTER = "Z"

RELAY_ALPHA = 1.0
RELAY_GAMMA0 = 0.125
RELAY_PRE_ITER = 80
RELAY_NUM_SETS = 600
RELAY_SET_MAX_ITER = 60
RELAY_GAMMA_DIST_INTERVAL = (-0.24, 0.66)
RELAY_STOP_NCONV = 5
RELAY_POSTERIORS = True
RELAY_DAMPING_INTERVAL = (0.85, 0.95)
# Native wrapper order: (negative, positive, p_positive).
RELAY_BERNOULLI_GAMMA = (
    -0.0687506131581227,
    0.3557972204473916,
    0.5451025293701163,
)
RELAY_BASE_SEED = 0

DEFAULT_DECODER_SEQUENCE = [
    "relay-bp5-r601-no-damping",
    "relay-bp5-r601-damp085-095",
    "relay-bp5-r601-discrete-memory",
]
DEFAULT_DECODER_INDICES: list[int] | None = None
DEFAULT_MAX_BATCH_SIZE = 64


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tests" / "testdata" / "bicycle_bivariate").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata/bicycle_bivariate "
        f"starting from {start}."
    )


def resolve_num_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, multiprocessing.cpu_count())


def parse_name_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"stim_path": str(path)}
    for part in path.stem.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key] = value
    if "error_rate" in metadata:
        metadata["p"] = float(metadata["error_rate"])
    return metadata


def parse_float_csv(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def find_circuits(circuits_dir: Path, selected_p_values: list[float]) -> list[Path]:
    wanted = {str(p) for p in selected_p_values}
    matched = []
    for path in sorted(
        circuits_dir.glob("circuit=bicycle_bivariate_144_12_12_memory_Z,*.stim")
    ):
        metadata = parse_name_metadata(path)
        if metadata.get("error_rate") in wanted:
            matched.append(path)

    matched = sorted(matched, key=lambda path: parse_name_metadata(path)["p"])
    found = {parse_name_metadata(path)["error_rate"] for path in matched}
    missing = sorted(float(p) for p in (wanted - found))
    if missing:
        raise FileNotFoundError(
            f"Missing 144_12_12 memory_Z circuits for p values: {missing} in {circuits_dir}."
        )
    return matched


def select_num_errors(p: float, errors_by_p: dict[float, int]) -> int:
    key = round(float(p), 3)
    if key not in errors_by_p:
        raise KeyError(f"No error budget configured for p={p}.")
    return int(errors_by_p[key])


def build_tasks(circuit_paths: list[Path]) -> list[sinter.Task]:
    tasks: list[sinter.Task] = []
    for circuit_path in circuit_paths:
        full_circuit = stim.Circuit.from_file(circuit_path)
        filtered_circuit = _filter_detectors_by_basis(full_circuit, BASIS_FILTER)
        dem = filtered_circuit.detector_error_model()
        metadata = parse_name_metadata(circuit_path)
        metadata["basis_filter_applied"] = BASIS_FILTER
        tasks.append(
            sinter.Task(
                circuit=filtered_circuit,
                detector_error_model=dem,
                json_metadata=metadata,
                collection_options=sinter.CollectionOptions(
                    max_errors=select_num_errors(metadata["p"], ERRORS_BY_P),
                    max_shots=MAX_SHOTS_SAFETY_CAP,
                ),
            )
        )
    return tasks


class RelayBPIterationSampler(Sampler):
    """Custom sinter sampler that aggregates iteration totals per batch."""

    def __init__(self, *, decoder: SinterDecoder_RelayBP):
        self.decoder = decoder

    def compiled_sampler_for_task(self, task: sinter.Task) -> CompiledSampler:
        return CompiledRelayBPIterationSampler(decoder=self.decoder, task=task)


class CompiledRelayBPIterationSampler(CompiledSampler):
    def __init__(self, *, decoder: SinterDecoder_RelayBP, task: sinter.Task):
        self.task = task
        self.decoder = decoder
        self.check_matrices = CheckMatrices.from_dem(
            task.detector_error_model,
            decomposed_hyperedges=decoder.decomposed_hyperedges,
            prune_decided_errors=decoder.prune_decided_errors,
            threshold=decoder.threshold,
        )
        self.observable_decoder = decoder.build_observable_decoder(self.check_matrices)
        self.stim_sampler = task.circuit.compile_detector_sampler()
        self.num_detectors = task.circuit.num_detectors
        self.num_observables = task.circuit.num_observables
        self.postselection_mask = task.postselection_mask
        self.postselected_observables_mask = None
        if task.postselected_observables_mask is not None:
            self.postselected_observables_mask = np.unpackbits(
                task.postselected_observables_mask,
                bitorder="little",
            )[: self.num_observables].astype(np.uint8)

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets_packed, actual_obs_packed = self.stim_sampler.sample(
            shots=suggested_shots,
            bit_packed=True,
            separate_observables=True,
        )
        num_sampled_shots = int(dets_packed.shape[0])

        if self.postselection_mask is not None:
            discarded_flags = np.any(dets_packed & self.postselection_mask, axis=1)
            num_discards_1 = int(np.count_nonzero(discarded_flags))
            if num_discards_1:
                kept_flags = ~discarded_flags
                dets_packed = dets_packed[kept_flags, :]
                actual_obs_packed = actual_obs_packed[kept_flags, :]
        else:
            num_discards_1 = 0

        num_kept_after_det = int(dets_packed.shape[0])
        if num_kept_after_det == 0:
            return sinter.AnonTaskStats(
                shots=num_sampled_shots,
                errors=0,
                discards=num_discards_1,
                seconds=time.monotonic() - t0,
                custom_counts=collections.Counter(),
            )

        syndromes = np.unpackbits(dets_packed, axis=1, bitorder="little")[
            :, : self.num_detectors
        ].astype(np.uint8)
        actual_obs = np.unpackbits(actual_obs_packed, axis=1, bitorder="little")[
            :, : self.num_observables
        ].astype(np.uint8)

        if self.check_matrices.syndrome_bias is not None:
            syndromes = (syndromes + self.check_matrices.syndrome_bias) % 2

        detailed_results = self.observable_decoder.decode_observables_detailed_batch(
            syndromes,
            parallel=self.decoder.parallel,
            progress_bar=self.decoder.show_progress,
            leave_progress_bar_on_finish=self.decoder.leave_progress_bar_on_finish,
        )

        predicted_obs = np.asarray(
            [
                np.asarray(detail.observables, dtype=np.uint8)
                for detail in detailed_results
            ],
            dtype=np.uint8,
        )

        if self.check_matrices.observables_bias is not None:
            predicted_obs = (predicted_obs + self.check_matrices.observables_bias) % 2

        kept_for_error = np.ones(num_kept_after_det, dtype=bool)
        if self.postselected_observables_mask is not None:
            mismatches = actual_obs ^ predicted_obs
            discard_obs = np.any(
                mismatches & self.postselected_observables_mask, axis=1
            )
            kept_for_error &= ~discard_obs
            num_discards_2 = int(np.count_nonzero(discard_obs))
        else:
            num_discards_2 = 0

        if np.any(kept_for_error):
            logical_errors = np.any(
                actual_obs[kept_for_error] != predicted_obs[kept_for_error],
                axis=1,
            )
            num_errors = int(np.count_nonzero(logical_errors))
            iteration_sum = int(
                sum(
                    int(detailed_results[idx].iterations)
                    for idx, keep in enumerate(kept_for_error)
                    if keep
                )
            )
            decoded_shots = int(np.count_nonzero(kept_for_error))
        else:
            num_errors = 0
            iteration_sum = 0
            decoded_shots = 0

        return sinter.AnonTaskStats(
            shots=num_sampled_shots,
            errors=num_errors,
            discards=num_discards_1 + num_discards_2,
            seconds=time.monotonic() - t0,
            custom_counts=collections.Counter(
                {
                    "decoded_shots": decoded_shots,
                    "iteration_sum": iteration_sum,
                }
            ),
        )


def build_custom_decoders() -> dict[str, Sampler]:
    common = dict(
        alpha=RELAY_ALPHA,
        gamma0=RELAY_GAMMA0,
        pre_iter=RELAY_PRE_ITER,
        num_sets=RELAY_NUM_SETS,
        set_max_iter=RELAY_SET_MAX_ITER,
        gamma_dist_interval=RELAY_GAMMA_DIST_INTERVAL,
        relay_posteriors=RELAY_POSTERIORS,
        stop_nconv=RELAY_STOP_NCONV,
        stopping_criterion="nconv",
        logging=False,
        show_progress=False,
        leave_progress_bar_on_finish=False,
    )
    return {
        "relay-bp5-r601-no-damping": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                seed=RELAY_BASE_SEED,
                decoder_label="relay-bp5-r601-no-damping",
            )
        ),
        "relay-bp5-r601-damp085-095": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                c_damp_dist_interval=RELAY_DAMPING_INTERVAL,
                seed=RELAY_BASE_SEED + 1,
                decoder_label="relay-bp5-r601-damp085-095",
            )
        ),
        "relay-bp5-r601-discrete-memory": RelayBPIterationSampler(
            decoder=SinterDecoder_RelayBP(
                **common,
                gamma_bernoulli=RELAY_BERNOULLI_GAMMA,
                seed=RELAY_BASE_SEED + 2,
                decoder_label="relay-bp5-r601-discrete-memory",
            )
        ),
    }


def samples_to_df(samples: list[sinter.TaskStats]) -> pd.DataFrame:
    rows = []
    for stat in samples:
        circuit_name = stat.json_metadata["circuit"]
        detail_shots = int(stat.custom_counts.get("decoded_shots", 0))
        detail_iteration_sum = int(stat.custom_counts.get("iteration_sum", 0))
        rows.append(
            {
                "decoder": stat.decoder,
                "circuit": circuit_name,
                "memory": circuit_name.rsplit("_", 1)[-1],
                "p": float(stat.json_metadata["p"]),
                "shots": int(stat.shots),
                "errors": int(stat.errors),
                "seconds": float(stat.seconds),
                "basis_filter_applied": stat.json_metadata.get(
                    "basis_filter_applied", "none"
                ),
                "detail_shots": detail_shots,
                "detail_iteration_sum": detail_iteration_sum,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "decoder",
                "circuit",
                "memory",
                "p",
                "shots",
                "errors",
                "seconds",
                "basis_filter_applied",
                "detail_shots",
                "detail_iteration_sum",
                "ler",
                "stderr",
                "mean_iterations",
                "hit_target",
                "shot_cap_reached",
            ]
        )

    df = pd.DataFrame(rows)
    summary_df = (
        df.groupby(
            ["decoder", "circuit", "memory", "p", "basis_filter_applied"],
            as_index=False,
        )
        .agg(
            shots=("shots", "sum"),
            errors=("errors", "sum"),
            seconds=("seconds", "sum"),
            detail_shots=("detail_shots", "sum"),
            detail_iteration_sum=("detail_iteration_sum", "sum"),
        )
        .sort_values(["decoder", "p"])
        .reset_index(drop=True)
    )
    summary_df["ler"] = summary_df["errors"] / summary_df["shots"]
    summary_df["stderr"] = (
        (summary_df["ler"] * (1.0 - summary_df["ler"]) / summary_df["shots"]).clip(
            lower=0.0
        )
    ) ** 0.5
    summary_df["mean_iterations"] = np.where(
        summary_df["detail_shots"] > 0,
        summary_df["detail_iteration_sum"] / summary_df["detail_shots"],
        np.nan,
    )
    summary_df["hit_target"] = summary_df["errors"] >= summary_df["p"].map(ERRORS_BY_P)
    summary_df["shot_cap_reached"] = summary_df["shots"] >= MAX_SHOTS_SAFETY_CAP
    return summary_df


def plot_logical_error_rate(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    available = [
        name for name in decoder_sequence if name in set(summary_df["decoder"])
    ]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax.plot(group["p"], group["ler"], marker="o", linewidth=1.8, label=decoder_name)

    ax.set_title("Relay-BP-5 logical error performance (Z-basis filtered)")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_mean_iterations(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    available = [
        name for name in decoder_sequence if name in set(summary_df["decoder"])
    ]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax.plot(
            group["p"],
            group["mean_iterations"],
            marker="o",
            linewidth=1.8,
            label=decoder_name,
        )

    ax.set_title("Relay-BP-5 mean iterations per decoded shot")
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("mean iterations")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_combined_summary(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, (ax_ler, ax_iter) = plt.subplots(1, 2, figsize=(12, 5))
    available = [
        name for name in decoder_sequence if name in set(summary_df["decoder"])
    ]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax_ler.plot(
            group["p"], group["ler"], marker="o", linewidth=1.8, label=decoder_name
        )
        ax_iter.plot(
            group["p"],
            group["mean_iterations"],
            marker="o",
            linewidth=1.8,
            label=decoder_name,
        )

    ax_ler.set_title("Logical error rate")
    ax_ler.set_xlabel("physical error rate p")
    ax_ler.set_ylabel("logical error rate")
    ax_ler.set_yscale("log")
    ax_ler.grid(True, which="both", alpha=0.3)

    ax_iter.set_title("Mean iterations")
    ax_iter.set_xlabel("physical error rate p")
    ax_iter.set_ylabel("mean iterations")
    ax_iter.set_yscale("log")
    ax_iter.grid(True, which="both", alpha=0.3)
    ax_iter.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def build_setup_dict(
    *,
    repo_root: Path,
    output_dir: Path,
    plots_dir: Path,
    matched_circuits: list[Path],
    selected_p_values: list[float],
    decoder_sequence: list[str],
    num_workers: int,
    max_batch_size: int,
) -> dict[str, Any]:
    return {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "plots_dir": str(plots_dir),
        "matched_circuits": [str(path) for path in matched_circuits],
        "selected_p_values": [float(p) for p in selected_p_values],
        "errors_by_p": {str(k): int(v) for k, v in ERRORS_BY_P.items()},
        "max_shots_safety_cap": int(MAX_SHOTS_SAFETY_CAP),
        "basis_filter": BASIS_FILTER,
        "relay_alpha": float(RELAY_ALPHA),
        "relay_gamma0": float(RELAY_GAMMA0),
        "relay_pre_iter": int(RELAY_PRE_ITER),
        "relay_num_sets": int(RELAY_NUM_SETS),
        "relay_total_legs": int(RELAY_NUM_SETS + 1),
        "relay_set_max_iter": int(RELAY_SET_MAX_ITER),
        "relay_gamma_dist_interval": list(RELAY_GAMMA_DIST_INTERVAL),
        "relay_stop_nconv": int(RELAY_STOP_NCONV),
        "relay_posteriors": bool(RELAY_POSTERIORS),
        "relay_damping_interval": list(RELAY_DAMPING_INTERVAL),
        "relay_bernoulli_gamma": list(RELAY_BERNOULLI_GAMMA),
        "decoder_sequence": list(decoder_sequence),
        "num_workers": int(num_workers),
        "max_batch_size": int(max_batch_size),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
    }


def validate_resume_setup(setup_json_path: Path, current_setup: dict[str, Any]) -> None:
    if not setup_json_path.exists():
        return
    existing_setup = json.loads(setup_json_path.read_text(encoding="utf-8"))
    fields_to_compare = [
        "selected_p_values",
        "errors_by_p",
        "max_shots_safety_cap",
        "basis_filter",
        "relay_alpha",
        "relay_gamma0",
        "relay_pre_iter",
        "relay_num_sets",
        "relay_set_max_iter",
        "relay_gamma_dist_interval",
        "relay_stop_nconv",
        "relay_posteriors",
        "relay_damping_interval",
        "max_batch_size",
    ]
    mismatches = [
        field
        for field in fields_to_compare
        if existing_setup.get(field) != current_setup.get(field)
    ]
    existing_decoder_sequence = list(existing_setup.get("decoder_sequence", []))
    current_decoder_sequence = list(current_setup.get("decoder_sequence", []))
    decoder_sequence_compatible = (
        existing_decoder_sequence == current_decoder_sequence
        or current_decoder_sequence[: len(existing_decoder_sequence)]
        == existing_decoder_sequence
    )
    if not decoder_sequence_compatible:
        mismatches.append("decoder_sequence")

    if "relay_bernoulli_gamma" in existing_setup and existing_setup.get(
        "relay_bernoulli_gamma"
    ) != current_setup.get("relay_bernoulli_gamma"):
        mismatches.append("relay_bernoulli_gamma")

    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Existing performance outputs were created with different settings: "
            f"{mismatch_text}. Use --reset-data or choose a different output directory."
        )


def resolve_selected_decoder_steps(
    decoder_sequence: list[str],
    decoder_indices: list[int] | None,
) -> list[tuple[int, str]]:
    if decoder_indices is None:
        return [(index + 1, name) for index, name in enumerate(decoder_sequence)]

    total = len(decoder_sequence)
    invalid = [index for index in decoder_indices if index < 1 or index > total]
    if invalid:
        raise ValueError(
            f"Decoder indices out of range: {invalid}. Valid one-based indices are 1..{total}."
        )

    selected_steps: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index in decoder_indices:
        if index in seen:
            continue
        seen.add(index)
        selected_steps.append((index, decoder_sequence[index - 1]))
    return selected_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root containing tests/testdata/bicycle_bivariate. Auto-detected by default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for csv outputs and plots/. Defaults under examples/notebook_data/.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Delete prior csv/json outputs before running.",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        default=False,
        help="Enable sinter progress output.",
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        default=DEFAULT_DECODER_SEQUENCE,
        help="Subset or reordered decoder names to run.",
    )
    parser.add_argument(
        "--decoder-indices",
        nargs="+",
        type=int,
        default=DEFAULT_DECODER_INDICES,
        help=(
            "Optional one-based positions into the decoder list to actually run. "
            "Example: --decoder-indices 2 runs only the damping variant."
        ),
    )
    parser.add_argument(
        "--p-values",
        type=str,
        default=None,
        help="Optional comma-separated p values. Defaults to 0.001,0.002,0.003,0.004,0.005.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help="Maximum sinter batch size per worker. Lower this if detailed decoding uses too much memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = (
        args.repo_root.resolve()
        if args.repo_root
        else find_repo_root(Path.cwd().resolve())
    )
    circuits_dir = repo_root / "tests" / "testdata" / "bicycle_bivariate"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root
        / "examples"
        / "notebook_data"
        / "bicycle_bivariate_144_12_12_relay_bp5_performance_cluster_Z_basis"
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_stem = "bicycle_bivariate_144_12_12_relay_bp5_performance_cluster_Z_basis"
    resume_csv = output_dir / f"{save_stem}_resume.csv"
    summary_csv = output_dir / f"{save_stem}_summary.csv"
    flat_summary_csv = output_dir / f"{save_stem}_flat_summary.csv"
    setup_json = output_dir / f"{save_stem}_setup.json"

    if args.reset_data:
        for path in [resume_csv, summary_csv, flat_summary_csv, setup_json]:
            if path.exists():
                path.unlink()

    custom_decoders = build_custom_decoders()
    decoder_sequence = args.decoders
    unknown = [name for name in decoder_sequence if name not in custom_decoders]
    if unknown:
        raise KeyError(f"Unknown decoders requested: {unknown}")
    selected_decoder_steps = resolve_selected_decoder_steps(
        decoder_sequence,
        args.decoder_indices,
    )
    selected_p_values = parse_float_csv(args.p_values) or list(SELECTED_P_VALUES)
    max_batch_size = max(1, int(args.max_batch_size))

    matched_circuits = find_circuits(circuits_dir, selected_p_values)
    tasks = build_tasks(matched_circuits)
    matched_metadata_df = pd.DataFrame(
        [parse_name_metadata(path) for path in matched_circuits]
    )
    matched_metadata_df["basis_filter_applied"] = BASIS_FILTER
    matched_metadata_df.to_csv(output_dir / "matched_circuits.csv", index=False)

    num_workers = resolve_num_workers()
    setup = build_setup_dict(
        repo_root=repo_root,
        output_dir=output_dir,
        plots_dir=plots_dir,
        matched_circuits=matched_circuits,
        selected_p_values=selected_p_values,
        decoder_sequence=decoder_sequence,
        num_workers=num_workers,
        max_batch_size=max_batch_size,
    )
    validate_resume_setup(setup_json, setup)
    setup_json.write_text(json.dumps(setup, indent=2, sort_keys=True), encoding="utf-8")

    print(f"repo_root   = {repo_root}")
    print(f"circuits_dir= {circuits_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"plots_dir   = {plots_dir}")
    print(f"num_workers = {num_workers}")
    print(f"max_batch_size = {max_batch_size}")
    print("selected_p_values =", selected_p_values)
    print("errors_by_p =", ERRORS_BY_P)
    print("max_shots safety cap =", MAX_SHOTS_SAFETY_CAP)
    print("basis_filter =", BASIS_FILTER)
    print("relay_bernoulli_gamma =", RELAY_BERNOULLI_GAMMA)
    print("decoders =", decoder_sequence)
    if args.decoder_indices is None:
        print("selected_decoder_indices = all")
    else:
        print("selected_decoder_indices =", args.decoder_indices)

    samples: list[sinter.TaskStats] = []
    summary_df = pd.DataFrame()

    for step_index, decoder_name in selected_decoder_steps:
        print(
            f"\n=== Running decoder {step_index}/{len(decoder_sequence)}: {decoder_name} ==="
        )
        samples = sinter.collect(
            tasks=tasks,
            decoders=[decoder_name],
            custom_decoders={decoder_name: custom_decoders[decoder_name]},
            num_workers=num_workers,
            print_progress=args.print_progress,
            save_resume_filepath=resume_csv,
            max_batch_size=max_batch_size,
        )

        write_stats(samples, summary_csv)
        summary_df = samples_to_df(samples)
        summary_df.to_csv(flat_summary_csv, index=False)

        plot_logical_error_rate(
            summary_df,
            decoder_sequence=decoder_sequence,
            save_path=plots_dir / f"step_{step_index:02d}_logical_error_rate.png",
        )
        plot_mean_iterations(
            summary_df,
            decoder_sequence=decoder_sequence,
            save_path=plots_dir / f"step_{step_index:02d}_mean_iterations.png",
        )
        plot_combined_summary(
            summary_df,
            decoder_sequence=decoder_sequence,
            save_path=plots_dir / f"step_{step_index:02d}_combined.png",
        )

    plot_logical_error_rate(
        summary_df,
        decoder_sequence=decoder_sequence,
        save_path=plots_dir / "logical_error_rate.png",
    )
    plot_mean_iterations(
        summary_df,
        decoder_sequence=decoder_sequence,
        save_path=plots_dir / "mean_iterations.png",
    )
    plot_combined_summary(
        summary_df,
        decoder_sequence=decoder_sequence,
        save_path=plots_dir / "combined_summary.png",
    )

    print("\nDone.")
    print(f"resume_csv       = {resume_csv}")
    print(f"summary_csv      = {summary_csv}")
    print(f"flat_summary_csv = {flat_summary_csv}")
    print(f"plots_dir        = {plots_dir}")


if __name__ == "__main__":
    main()
