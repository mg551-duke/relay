#!/usr/bin/env python3
"""Headless cluster runner for BivariateBicycle144_12_12NativeBPGD.ipynb.

Runs the five notebook decoders on bicycle_bivariate_144_12_12_memory_X circuits
using max_errors targets instead of max_shots-only targets, saves cumulative plots
after each decoder into a new plots/ folder, and writes a final combined plot.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sinter
import stim

from relay_bp.stim import SinterDecoder_BPGD, SinterDecoder_DampedBP, SinterDecoder_MSLBP
from relay_bp.stim.sinter.utils import write_stats


SELECTED_P_VALUES = [0.001, 0.002, 0.003, 0.004, 0.005]
ERRORS_BY_P = {
    0.001: 50,
    0.002: 50,
    0.003: 50,
    0.004: 50,
    0.005: 50,
}
MAX_SHOTS_SAFETY_CAP = 1_000_000

ALPHA = 1.0
BASELINE_MAX_ITER = 100
C_DAMP = 0.5

PRE_ITER = 50
R_LOW = 5
T_LOW = 10
T_HIGH = 10
R_HIGH_MAX = 1_000_000  # Sentinel; native BPGD caps this internally.

DEFAULT_DECODER_SEQUENCE = [
    "msl-bp",
    "msl-bp-damp0p5",
    "bpgd-low-only",
    "bpgd-high-init0p98",
    "bpgd-high-init0p9",
]


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tests" / "testdata" / "bicycle_bivariate").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find repo root containing tests/testdata/bicycle_bivariate "
        f"starting from {start}."
    )


def parse_name_metadata(path: Path) -> dict:
    metadata = {"stim_path": str(path)}
    for part in path.stem.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key] = value
    if "error_rate" in metadata:
        metadata["p"] = float(metadata["error_rate"])
    return metadata


def find_circuits(circuits_dir: Path, selected_p_values: list[float]) -> list[Path]:
    wanted = {str(p) for p in selected_p_values}
    matched = []
    for path in sorted(circuits_dir.glob("circuit=bicycle_bivariate_144_12_12_memory_X,*.stim")):
        metadata = parse_name_metadata(path)
        if metadata.get("error_rate") in wanted:
            matched.append(path)

    matched = sorted(matched, key=lambda path: parse_name_metadata(path)["p"])
    found = {parse_name_metadata(path)["error_rate"] for path in matched}
    missing = sorted(float(p) for p in (wanted - found))
    if missing:
        raise FileNotFoundError(
            f"Missing 144_12_12 memory_X circuits for p values: {missing} in {circuits_dir}."
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
        circuit = stim.Circuit.from_file(circuit_path)
        dem = circuit.detector_error_model()
        metadata = parse_name_metadata(circuit_path)
        tasks.append(
            sinter.Task(
                circuit=circuit,
                detector_error_model=dem,
                json_metadata=metadata,
                collection_options=sinter.CollectionOptions(
                    max_errors=select_num_errors(metadata["p"], ERRORS_BY_P),
                    max_shots=MAX_SHOTS_SAFETY_CAP,
                ),
            )
        )
    return tasks


def samples_to_df(samples: list[sinter.TaskStats]) -> pd.DataFrame:
    rows = []
    for stat in samples:
        circuit_name = stat.json_metadata["circuit"]
        rows.append(
            {
                "decoder": stat.decoder,
                "circuit": circuit_name,
                "memory": circuit_name.rsplit("_", 1)[-1],
                "p": float(stat.json_metadata["p"]),
                "shots": stat.shots,
                "errors": stat.errors,
                "seconds": stat.seconds,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["decoder", "circuit", "memory", "p", "shots", "errors", "seconds", "ler", "stderr"]
        )

    df = pd.DataFrame(rows)
    summary_df = (
        df.groupby(["decoder", "circuit", "memory", "p"], as_index=False)
        .agg(shots=("shots", "sum"), errors=("errors", "sum"), seconds=("seconds", "sum"))
        .sort_values(["decoder", "p"])
        .reset_index(drop=True)
    )
    summary_df["ler"] = summary_df["errors"] / summary_df["shots"]
    summary_df["stderr"] = (
        (summary_df["ler"] * (1.0 - summary_df["ler"]) / summary_df["shots"]).clip(lower=0.0)
    ) ** 0.5
    return summary_df


def plot_summary(
    summary_df: pd.DataFrame,
    *,
    decoder_sequence: list[str],
    title: str,
    save_path: Path,
) -> None:
    if summary_df.empty:
        print(f"[warn] No samples available to plot for {save_path.name}.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    available = [name for name in decoder_sequence if name in set(summary_df["decoder"])]
    for decoder_name in available:
        group = summary_df[summary_df["decoder"] == decoder_name].sort_values("p")
        ax.plot(group["p"], group["ler"], marker="o", linewidth=1.8, label=decoder_name)

    ax.set_title(title)
    ax.set_xlabel("physical error rate p")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def build_custom_decoders() -> dict[str, object]:
    return {
        "msl-bp": SinterDecoder_MSLBP(
            max_iter=BASELINE_MAX_ITER,
            alpha=ALPHA,
            decoder_label="msl-bp",
        ),
        "msl-bp-damp0p5": SinterDecoder_DampedBP(
            max_iter=BASELINE_MAX_ITER,
            alpha=ALPHA,
            c_damp=C_DAMP,
            decoder_label="msl-bp-damp0p5",
        ),
        "bpgd-low-only": SinterDecoder_BPGD(
            pre_iter=PRE_ITER,
            initial_decimation_percentage=0.0,
            r_low=R_LOW,
            t_low=T_LOW,
            r_high=0,
            t_high=T_HIGH,
            alpha=ALPHA,
            c_damp=C_DAMP,
            fallback_kind="none",
            decoder_label="bpgd-low-only",
        ),
        "bpgd-high-init0p98": SinterDecoder_BPGD(
            pre_iter=PRE_ITER,
            initial_decimation_percentage=0.98,
            r_low=R_LOW,
            t_low=T_LOW,
            r_high=R_HIGH_MAX,
            t_high=T_HIGH,
            alpha=ALPHA,
            c_damp=C_DAMP,
            fallback_kind="none",
            decoder_label="bpgd-high-init0p98",
        ),
        "bpgd-high-init0p9": SinterDecoder_BPGD(
            pre_iter=PRE_ITER,
            initial_decimation_percentage=0.9,
            r_low=R_LOW,
            t_low=T_LOW,
            r_high=R_HIGH_MAX,
            t_high=T_HIGH,
            alpha=ALPHA,
            c_damp=C_DAMP,
            fallback_kind="none",
            decoder_label="bpgd-high-init0p9",
        ),
    }


def resolve_num_workers() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return max(1, multiprocessing.cpu_count())


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
        help="Directory for csv outputs and plots/. Defaults to examples/notebook_data/bicycle_bivariate_144_12_12_native_bpgd_cluster under the repo root.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Delete prior csv outputs before running.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd().resolve())
    circuits_dir = repo_root / "tests" / "testdata" / "bicycle_bivariate"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "examples" / "notebook_data" / "bicycle_bivariate_144_12_12_native_bpgd_cluster"
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_stem = "bicycle_bivariate_144_12_12_native_bpgd_cluster"
    resume_csv = output_dir / f"{save_stem}_resume.csv"
    summary_csv = output_dir / f"{save_stem}_summary.csv"
    flat_summary_csv = output_dir / f"{save_stem}_flat_summary.csv"

    if args.reset_data:
        for path in [resume_csv, summary_csv, flat_summary_csv]:
            if path.exists():
                path.unlink()

    custom_decoders = build_custom_decoders()
    decoder_sequence = args.decoders
    unknown = [name for name in decoder_sequence if name not in custom_decoders]
    if unknown:
        raise KeyError(f"Unknown decoders requested: {unknown}")

    matched_circuits = find_circuits(circuits_dir, SELECTED_P_VALUES)
    tasks = build_tasks(matched_circuits)
    matched_metadata_df = pd.DataFrame([parse_name_metadata(path) for path in matched_circuits])
    matched_metadata_df.to_csv(output_dir / "matched_circuits.csv", index=False)

    num_workers = resolve_num_workers()

    print(f"repo_root   = {repo_root}")
    print(f"circuits_dir= {circuits_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"plots_dir   = {plots_dir}")
    print(f"num_workers = {num_workers}")
    print("selected_p_values =", SELECTED_P_VALUES)
    print("errors_by_p =", ERRORS_BY_P)
    print("max_shots safety cap =", MAX_SHOTS_SAFETY_CAP)
    print("decoders =", decoder_sequence)

    samples: list[sinter.TaskStats] = []
    summary_df = pd.DataFrame()

    for step_index, decoder_name in enumerate(decoder_sequence, start=1):
        print(f"\n=== Running decoder {step_index}/{len(decoder_sequence)}: {decoder_name} ===")
        samples = sinter.collect(
            tasks=tasks,
            decoders=[decoder_name],
            custom_decoders={decoder_name: custom_decoders[decoder_name]},
            num_workers=num_workers,
            print_progress=args.print_progress,
            save_resume_filepath=resume_csv,
        )

        write_stats(samples, summary_csv)
        summary_df = samples_to_df(samples)
        summary_df.to_csv(flat_summary_csv, index=False)

        step_plot_path = plots_dir / f"step_{step_index:02d}_after_{decoder_name}.png"
        plot_summary(
            summary_df,
            decoder_sequence=decoder_sequence,
            title=f"Cumulative logical error rate after {decoder_name}",
            save_path=step_plot_path,
        )

    final_plot_path = plots_dir / "combined_all_decoders.png"
    plot_summary(
        summary_df,
        decoder_sequence=decoder_sequence,
        title="Logical error rate: all decoders",
        save_path=final_plot_path,
    )

    print("\nDone.")
    print(f"resume_csv       = {resume_csv}")
    print(f"summary_csv      = {summary_csv}")
    print(f"flat_summary_csv = {flat_summary_csv}")
    print(f"plots_dir        = {plots_dir}")


if __name__ == "__main__":
    main()
