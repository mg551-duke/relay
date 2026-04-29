#!/usr/bin/env python3
"""Circuit-level Relay-BP-1 edge-weight heatmaps at fixed gamma values.

The heatmap grid controls explicit per-edge message weights. Damping is not
used. The run is short-relay Relay-BP-1:

- R = 31 total legs (`num_sets = 30`)
- `stop_nconv = 1`
- fixed gamma values run back to back
- default gamma list: `0, 0.125, 0.250, 0.375, 0.500`

The `gamma=0` run is the no-memory-effect baseline. Nonzero gamma runs use a
fixed uniform gamma on the first leg and all relay legs, not sampled memory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sinter
from matplotlib.colors import LogNorm

from relay_bp.stim import SinterDecoder_RelayBP
from relay_bp.stim.sinter.utils import write_stats
from run_bivariate_bicycle144_12_12_paper_heatmap_cluster_Z_basis import (
    best_points_table,
    build_task,
    find_circuit_path,
    find_repo_root,
    format_float_token,
    grid_to_artifact,
    heatmap_edges,
    load_filtered_problem,
    load_samples_from_resume,
    parse_float_csv,
    resolve_grid_values,
    resolve_num_workers,
    samples_to_df,
)


DEFAULT_SELECTED_P = 0.003
DEFAULT_MEMORY_BASIS = "Z"
DEFAULT_BASIS_FILTER = "Z"

DEFAULT_ALPHA = 1.0
DEFAULT_PRE_ITER = 80
DEFAULT_NUM_SETS = 30
DEFAULT_SET_MAX_ITER = 60
DEFAULT_STOP_NCONV = 1
DEFAULT_GAMMA_VALUES = "0,0.125,0.25,0.375,0.5"

DEFAULT_CENTER_MIN = 0.0
DEFAULT_CENTER_MAX = 1.0
DEFAULT_CENTER_COUNT = 11
DEFAULT_WIDTH_MIN = 0.0
DEFAULT_WIDTH_MAX = 2.0
DEFAULT_WIDTH_COUNT = 11

DEFAULT_TARGET_LOGICAL_ERRORS = 10
DEFAULT_MAX_SHOTS_PER_POINT = 500_000
DEFAULT_BASE_RELAY_SEED = 0


def gamma_token(gamma: float) -> str:
    return f"g{format_float_token(float(gamma))}"


def interval_from_center_width(center: float, width: float) -> tuple[float, float]:
    if float(width) < 0.0:
        raise ValueError("width must be non-negative")
    half_width = float(width) / 2.0
    return float(center) - half_width, float(center) + half_width


def make_decoder_name(*, gamma: float, center: float, width: float) -> str:
    return (
        "relay-bp1-r31-fixed-gamma-edge-weight-"
        f"{gamma_token(gamma)}-"
        f"c{format_float_token(center)}-"
        f"w{format_float_token(width)}"
    )


def parse_gamma_values(text: str) -> np.ndarray:
    values = parse_float_csv(text)
    if values is None or values.size == 0:
        raise ValueError("--gamma-values must contain at least one comma-separated value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("--gamma-values must be finite.")
    return values.astype(np.float64)


def build_heatmap_points(
    *,
    gamma: float,
    center_values: np.ndarray,
    width_values: np.ndarray,
    base_relay_seed: int,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for width_idx, width in enumerate(width_values):
        for center_idx, center in enumerate(center_values):
            point_id = width_idx * len(center_values) + center_idx
            interval_low, interval_high = interval_from_center_width(float(center), float(width))
            points.append(
                {
                    "point_id": int(point_id),
                    "width_idx": int(width_idx),
                    "center_idx": int(center_idx),
                    "center": float(center),
                    "width": float(width),
                    "interval_low": float(interval_low),
                    "interval_high": float(interval_high),
                    "gamma": float(gamma),
                    "decoder": make_decoder_name(
                        gamma=float(gamma),
                        center=float(center),
                        width=float(width),
                    ),
                    "relay_seed": int(base_relay_seed + 1_000_003 * point_id),
                }
            )
    return points


def build_edge_message_weights(
    *,
    point: dict[str, Any],
    num_edges: int,
) -> np.ndarray:
    low = float(point["interval_low"])
    high = float(point["interval_high"])
    if low > high:
        raise ValueError(f"edge-weight interval low must be <= high; got [{low}, {high}]")
    if math.isclose(low, high, abs_tol=1e-15):
        return np.full(int(num_edges), low, dtype=np.float64)
    rng = np.random.default_rng(int(point["relay_seed"]))
    return rng.uniform(low, high, size=int(num_edges)).astype(np.float64)


def build_point_decoder(
    *,
    point: dict[str, Any],
    gamma: float,
    num_sets: int,
    num_fault_mechanisms: int,
    num_edges: int,
    alpha: float,
    pre_iter: int,
    set_max_iter: int,
    relay_posteriors: bool,
    stop_nconv: int,
) -> SinterDecoder_RelayBP:
    return SinterDecoder_RelayBP(
        alpha=float(alpha),
        gamma0=float(gamma),
        pre_iter=int(pre_iter),
        num_sets=int(num_sets),
        set_max_iter=int(set_max_iter),
        gamma_dist_interval=(float(gamma), float(gamma) + 1e-12),
        explicit_gammas=np.full(
            (int(num_sets), int(num_fault_mechanisms)),
            float(gamma),
            dtype=np.float64,
        ),
        explicit_edge_message_weights=build_edge_message_weights(
            point=point,
            num_edges=num_edges,
        ),
        relay_posteriors=bool(relay_posteriors),
        stop_nconv=int(stop_nconv),
        stopping_criterion="nconv",
        logging=False,
        seed=int(point["relay_seed"]),
        decoder_label=str(point["decoder"]),
    )


def samples_to_df_with_point_metadata(
    *,
    samples: list[sinter.TaskStats],
    points: list[dict[str, Any]],
    points_by_decoder: dict[str, dict[str, Any]],
    target_logical_errors: int,
    max_shots_per_point: int,
) -> pd.DataFrame:
    summary_df = samples_to_df(
        samples=samples,
        points_by_decoder=points_by_decoder,
        target_logical_errors=target_logical_errors,
        max_shots_per_point=max_shots_per_point,
    )
    metadata_df = pd.DataFrame(
        [
            {
                "decoder": point["decoder"],
                "gamma": point["gamma"],
            }
            for point in points
        ]
    )
    if summary_df.empty:
        for column in metadata_df.columns:
            if column not in summary_df.columns:
                summary_df[column] = pd.Series(dtype=metadata_df[column].dtype)
        return summary_df
    return summary_df.merge(metadata_df, on="decoder", how="left")


def logical_error_lognorm(
    logical_error_rate: np.ndarray,
    *,
    max_shots_per_point: int,
) -> tuple[np.ndarray, LogNorm]:
    display_values = np.asarray(logical_error_rate, dtype=np.float64).copy()
    finite_mask = np.isfinite(display_values)
    positive_mask = finite_mask & (display_values > 0.0)
    floor = 0.5 / max_shots_per_point
    display_values[finite_mask & (display_values <= 0.0)] = floor
    if np.any(positive_mask):
        vmin = float(np.nanmin(display_values[positive_mask]))
        vmax = float(np.nanmax(display_values[finite_mask]))
    else:
        vmin = floor
        vmax = floor * 10.0
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 1.01
    return display_values, LogNorm(vmin=vmin, vmax=vmax)


def positive_lognorm(values: np.ndarray, *, floor: float = 1.0) -> tuple[np.ndarray, LogNorm]:
    display_values = np.asarray(values, dtype=np.float64).copy()
    finite_mask = np.isfinite(display_values)
    positive_mask = finite_mask & (display_values > 0.0)
    display_values[finite_mask & (display_values <= 0.0)] = floor
    if np.any(positive_mask):
        vmin = float(np.nanmin(display_values[positive_mask]))
        vmax = float(np.nanmax(display_values[finite_mask]))
    else:
        vmin = floor
        vmax = floor * 10.0
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 1.01
    return display_values, LogNorm(vmin=vmin, vmax=vmax)


def plot_edge_weight_heatmap(
    *,
    artifact: dict[str, np.ndarray],
    center_values: np.ndarray,
    width_values: np.ndarray,
    metadata: dict[str, Any],
    selected_p: float,
    basis_filter: str,
    output_path: Path,
    max_shots_per_point: int,
    gamma: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    relay_posteriors: bool,
) -> None:
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    display_values, norm = logical_error_lognorm(
        artifact["logical_error_rate"],
        max_shots_per_point=max_shots_per_point,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(
        center_edges,
        width_edges,
        display_values,
        shading="auto",
        cmap="viridis_r",
        norm=norm,
    )
    ax.set_xlabel("edge-weight interval center")
    ax.set_ylabel("edge-weight interval width")
    ax.set_title(
        "Relay-BP-1 edge-weight logical error rate | "
        f"{metadata.get('circuit', 'unknown')} | p = {selected_p:.3g} | basis filter = {basis_filter}"
    )
    subtitle = (
        f"gamma = {gamma:.3f}, pre_iter = {pre_iter}, R = {num_sets + 1}, "
        f"set_max_iter = {set_max_iter}, stop_nconv = {stop_nconv}, "
        f"relay_posteriors = {relay_posteriors}; damping disabled"
    )
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("logical error rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_collection_heatmaps(
    *,
    artifact: dict[str, np.ndarray],
    center_values: np.ndarray,
    width_values: np.ndarray,
    metadata: dict[str, Any],
    selected_p: float,
    basis_filter: str,
    output_path: Path,
    max_shots_per_point: int,
) -> None:
    center_edges = heatmap_edges(center_values)
    width_edges = heatmap_edges(width_values)
    metrics = [
        ("logical_error_rate", "Logical error rate", "log"),
        ("shots", "Shots", "positive_log"),
        ("logical_error_count", "Logical errors observed", "linear"),
        ("seconds", "Wall-clock seconds", "positive_log"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (key, title, scale) in zip(axes.flat, metrics):
        values = np.asarray(artifact[key], dtype=np.float64)
        norm = None
        display_values = values
        if scale == "log":
            display_values, norm = logical_error_lognorm(
                values,
                max_shots_per_point=max_shots_per_point,
            )
        elif scale == "positive_log":
            display_values, norm = positive_lognorm(values)
        mesh = ax.pcolormesh(
            center_edges,
            width_edges,
            display_values,
            shading="auto",
            cmap="viridis_r" if key == "logical_error_rate" else "viridis",
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel("edge-weight interval center")
        ax.set_ylabel("edge-weight interval width")
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Relay-BP-1 edge-weight collection summary heatmaps | "
        f"{metadata.get('circuit', 'unknown')} | p = {selected_p:.3g} | basis filter = {basis_filter}",
        fontsize=13,
        y=1.01,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    *,
    summary_df: pd.DataFrame,
    center_values: np.ndarray,
    width_values: np.ndarray,
    metadata: dict[str, Any],
    selected_p: float,
    basis_filter: str,
    output_dir: Path,
    save_stem: str,
    gamma: float,
    pre_iter: int,
    num_sets: int,
    set_max_iter: int,
    stop_nconv: int,
    max_shots_per_point: int,
    relay_posteriors: bool,
) -> None:
    grid_csv = output_dir / f"{save_stem}_grid.csv"
    best_csv = output_dir / f"{save_stem}_best_points.csv"
    artifact_npz = output_dir / f"{save_stem}_artifact.npz"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    main_plot_png = plots_dir / "edge_weight_logical_error_heatmap.png"
    collection_plot_png = plots_dir / "collection_heatmaps.png"

    summary_df.sort_values(["width_idx", "center_idx"]).to_csv(grid_csv, index=False)
    best_points_table(summary_df).to_csv(best_csv, index=False)
    if summary_df.empty:
        return

    artifact = grid_to_artifact(
        summary_df,
        center_values=center_values,
        width_values=width_values,
    )
    np.savez_compressed(
        artifact_npz,
        center_values=center_values,
        width_values=width_values,
        **artifact,
    )
    plot_edge_weight_heatmap(
        artifact=artifact,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=selected_p,
        basis_filter=basis_filter,
        output_path=main_plot_png,
        max_shots_per_point=max_shots_per_point,
        gamma=gamma,
        pre_iter=pre_iter,
        num_sets=num_sets,
        set_max_iter=set_max_iter,
        stop_nconv=stop_nconv,
        relay_posteriors=relay_posteriors,
    )
    plot_collection_heatmaps(
        artifact=artifact,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=selected_p,
        basis_filter=basis_filter,
        output_path=collection_plot_png,
        max_shots_per_point=max_shots_per_point,
    )


def validate_resume_setup(setup_json_path: Path, current_setup: dict[str, Any]) -> None:
    if not setup_json_path.exists():
        return
    existing_setup = json.loads(setup_json_path.read_text(encoding="utf-8"))
    fields_to_compare = [
        "selected_p",
        "memory_basis",
        "basis_filter",
        "alpha",
        "gamma",
        "pre_iter",
        "num_sets",
        "set_max_iter",
        "stop_nconv",
        "relay_posteriors",
        "center_values",
        "width_values",
        "target_logical_errors",
        "max_shots_per_point",
        "base_relay_seed",
        "edge_weight_mode",
    ]
    mismatches = [
        field
        for field in fields_to_compare
        if existing_setup.get(field) != current_setup.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Existing fixed-gamma edge-weight heatmap outputs were created with different settings: "
            f"{', '.join(mismatches)}. Use --reset-data or choose a different output directory."
        )


def run_one_gamma(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    circuits_dir: Path,
    root_output_dir: Path,
    gamma: float,
    center_values: np.ndarray,
    width_values: np.ndarray,
) -> None:
    save_stem = (
        "bicycle_bivariate_144_12_12_relay_bp1_r31_"
        f"fixed_{gamma_token(gamma)}_edge_weight_heatmap_cluster_Z_basis"
    )
    output_dir = root_output_dir / gamma_token(gamma)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    resume_csv = output_dir / f"{save_stem}_resume.csv"
    summary_csv = output_dir / f"{save_stem}_summary.csv"
    setup_json = output_dir / f"{save_stem}_setup.json"
    selected_circuit_csv = output_dir / "selected_circuit.csv"
    point_manifest_csv = output_dir / f"{save_stem}_point_manifest.csv"

    if args.reset_data:
        for path in output_dir.glob(f"{save_stem}_*"):
            if path.is_file():
                path.unlink()
        for path in plots_dir.glob("*.png"):
            path.unlink()
        if selected_circuit_csv.exists():
            selected_circuit_csv.unlink()

    stim_path = find_circuit_path(circuits_dir, args.p, args.memory_basis)
    problem = load_filtered_problem(stim_path=stim_path, basis_filter=args.basis_filter)
    metadata = dict(problem["metadata"])
    metadata["stim_path"] = str(stim_path)
    metadata["fixed_gamma"] = float(gamma)
    metadata["relay_bp_variant"] = "Relay-BP-1"
    metadata["edge_weight_mode"] = "sampled_per_edge_message_weight"
    metadata["damping_disabled"] = True
    metadata["total_legs"] = int(args.num_sets + 1)
    pd.DataFrame([metadata]).to_csv(selected_circuit_csv, index=False)

    num_workers = resolve_num_workers()
    relay_posteriors = not args.no_relay_posteriors
    tasks = build_task(
        filtered_circuit=problem["filtered_circuit"],
        dem=problem["dem"],
        metadata=metadata,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
    )

    gamma_seed_offset = int(round((float(gamma) + 10.0) * 1_000_000))
    points = build_heatmap_points(
        gamma=gamma,
        center_values=center_values,
        width_values=width_values,
        base_relay_seed=int(args.base_relay_seed + gamma_seed_offset),
    )
    points_by_decoder = {str(point["decoder"]): point for point in points}
    pd.DataFrame(points).to_csv(point_manifest_csv, index=False)

    num_fault_mechanisms = int(problem["check_matrices"].check_matrix.shape[1])
    num_edges = int(problem["check_matrices"].check_matrix.nnz)
    current_setup = {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "selected_p": float(args.p),
        "memory_basis": str(args.memory_basis),
        "basis_filter": str(args.basis_filter),
        "alpha": float(args.alpha),
        "gamma": float(gamma),
        "pre_iter": int(args.pre_iter),
        "num_sets": int(args.num_sets),
        "total_legs": int(args.num_sets + 1),
        "set_max_iter": int(args.set_max_iter),
        "stop_nconv": int(args.stop_nconv),
        "relay_posteriors": bool(relay_posteriors),
        "center_values": center_values.tolist(),
        "width_values": width_values.tolist(),
        "target_logical_errors": int(args.target_logical_errors),
        "max_shots_per_point": int(args.max_shots_per_point),
        "base_relay_seed": int(args.base_relay_seed),
        "edge_weight_mode": "sampled_per_edge_message_weight",
        "damping_disabled": True,
        "num_detectors": int(problem["dem"].num_detectors),
        "num_observables": int(problem["dem"].num_observables),
        "num_fault_mechanisms": int(num_fault_mechanisms),
        "num_edges": int(num_edges),
        "num_workers": int(num_workers),
    }
    validate_resume_setup(setup_json, current_setup)
    setup_json.write_text(json.dumps(current_setup, indent=2, sort_keys=True), encoding="utf-8")

    initial_samples = load_samples_from_resume(resume_csv)
    initial_summary_df = samples_to_df_with_point_metadata(
        samples=initial_samples,
        points=points,
        points_by_decoder=points_by_decoder,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
    )
    if initial_samples:
        write_stats(initial_samples, summary_csv)
    save_outputs(
        summary_df=initial_summary_df,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=args.p,
        basis_filter=args.basis_filter,
        output_dir=output_dir,
        save_stem=save_stem,
        gamma=gamma,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        max_shots_per_point=args.max_shots_per_point,
        relay_posteriors=relay_posteriors,
    )

    completed_decoders = set(
        initial_summary_df.loc[
            initial_summary_df["hit_target"] | initial_summary_df["shot_cap_reached"],
            "decoder",
        ].tolist()
    )

    print("\n" + "=" * 96)
    print(f"Running fixed-gamma edge-weight heatmap: gamma={gamma:.6g}")
    print("=" * 96)
    print(f"stim_path   = {stim_path}")
    print(f"output_dir  = {output_dir}")
    print(f"num_workers = {num_workers}")
    print(f"selected_p  = {args.p}")
    print(f"basis_filter= {args.basis_filter}")
    print(f"num_edges   = {num_edges}")
    print(f"num_sets    = {args.num_sets} (R = {args.num_sets + 1})")
    print(f"stop_nconv  = {args.stop_nconv}")
    print(f"center_values = {center_values.tolist()}")
    print(f"width_values  = {width_values.tolist()}")
    print(f"completed_points = {len(completed_decoders)}/{len(points)}")

    for step_index, point in enumerate(points, start=1):
        decoder_name = str(point["decoder"])
        if decoder_name in completed_decoders:
            print(
                f"\n=== Skipping fixed-gamma edge-weight point {step_index}/{len(points)}: "
                f"{decoder_name} (already complete) ==="
            )
            continue

        print(
            f"\n=== Running fixed-gamma edge-weight point {step_index}/{len(points)}: {decoder_name} "
            f"| gamma={gamma:.3f} center={point['center']:.3f} width={point['width']:.3f} "
            f"| interval=[{point['interval_low']:.3f}, {point['interval_high']:.3f}] ==="
        )
        decoder = build_point_decoder(
            point=point,
            gamma=gamma,
            num_sets=args.num_sets,
            num_fault_mechanisms=num_fault_mechanisms,
            num_edges=num_edges,
            alpha=args.alpha,
            pre_iter=args.pre_iter,
            set_max_iter=args.set_max_iter,
            relay_posteriors=relay_posteriors,
            stop_nconv=args.stop_nconv,
        )
        sinter.collect(
            tasks=tasks,
            decoders=[decoder_name],
            custom_decoders={decoder_name: decoder},
            num_workers=num_workers,
            print_progress=args.print_progress,
            save_resume_filepath=resume_csv,
        )

        samples = load_samples_from_resume(resume_csv)
        write_stats(samples, summary_csv)
        summary_df = samples_to_df_with_point_metadata(
            samples=samples,
            points=points,
            points_by_decoder=points_by_decoder,
            target_logical_errors=args.target_logical_errors,
            max_shots_per_point=args.max_shots_per_point,
        )
        save_outputs(
            summary_df=summary_df,
            center_values=center_values,
            width_values=width_values,
            metadata=metadata,
            selected_p=args.p,
            basis_filter=args.basis_filter,
            output_dir=output_dir,
            save_stem=save_stem,
            gamma=gamma,
            pre_iter=args.pre_iter,
            num_sets=args.num_sets,
            set_max_iter=args.set_max_iter,
            stop_nconv=args.stop_nconv,
            max_shots_per_point=args.max_shots_per_point,
            relay_posteriors=relay_posteriors,
        )
        completed_decoders.add(decoder_name)

    final_samples = load_samples_from_resume(resume_csv)
    final_summary_df = samples_to_df_with_point_metadata(
        samples=final_samples,
        points=points,
        points_by_decoder=points_by_decoder,
        target_logical_errors=args.target_logical_errors,
        max_shots_per_point=args.max_shots_per_point,
    )
    if final_samples:
        write_stats(final_samples, summary_csv)
    save_outputs(
        summary_df=final_summary_df,
        center_values=center_values,
        width_values=width_values,
        metadata=metadata,
        selected_p=args.p,
        basis_filter=args.basis_filter,
        output_dir=output_dir,
        save_stem=save_stem,
        gamma=gamma,
        pre_iter=args.pre_iter,
        num_sets=args.num_sets,
        set_max_iter=args.set_max_iter,
        stop_nconv=args.stop_nconv,
        max_shots_per_point=args.max_shots_per_point,
        relay_posteriors=relay_posteriors,
    )

    print(f"Done gamma={gamma:.6g}.")
    print(f"resume_csv = {resume_csv}")
    print(f"summary_csv = {summary_csv}")
    print(f"grid_csv = {output_dir / f'{save_stem}_grid.csv'}")
    print(f"best_csv = {output_dir / f'{save_stem}_best_points.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reset-data", action="store_true")
    parser.add_argument("--print-progress", action="store_true", default=False)
    parser.add_argument("--p", type=float, default=DEFAULT_SELECTED_P)
    parser.add_argument("--memory-basis", choices=["X", "Z"], default=DEFAULT_MEMORY_BASIS)
    parser.add_argument("--basis-filter", choices=["X", "Z"], default=DEFAULT_BASIS_FILTER)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--pre-iter", type=int, default=DEFAULT_PRE_ITER)
    parser.add_argument("--num-sets", type=int, default=DEFAULT_NUM_SETS)
    parser.add_argument("--set-max-iter", type=int, default=DEFAULT_SET_MAX_ITER)
    parser.add_argument("--stop-nconv", type=int, default=DEFAULT_STOP_NCONV)
    parser.add_argument("--gamma-values", type=str, default=DEFAULT_GAMMA_VALUES)
    parser.add_argument("--center-values", type=str, default=None)
    parser.add_argument("--width-values", type=str, default=None)
    parser.add_argument("--center-min", type=float, default=DEFAULT_CENTER_MIN)
    parser.add_argument("--center-max", type=float, default=DEFAULT_CENTER_MAX)
    parser.add_argument("--center-count", type=int, default=DEFAULT_CENTER_COUNT)
    parser.add_argument("--width-min", type=float, default=DEFAULT_WIDTH_MIN)
    parser.add_argument("--width-max", type=float, default=DEFAULT_WIDTH_MAX)
    parser.add_argument("--width-count", type=int, default=DEFAULT_WIDTH_COUNT)
    parser.add_argument("--target-logical-errors", type=int, default=DEFAULT_TARGET_LOGICAL_ERRORS)
    parser.add_argument("--max-shots-per-point", type=int, default=DEFAULT_MAX_SHOTS_PER_POINT)
    parser.add_argument("--base-relay-seed", type=int, default=DEFAULT_BASE_RELAY_SEED)
    parser.add_argument("--no-relay-posteriors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_sets != 30:
        print(f"WARNING: --num-sets={args.num_sets}; total R will be {args.num_sets + 1}, not 31.")
    if args.stop_nconv != 1:
        print(f"WARNING: --stop-nconv={args.stop_nconv}; this is not Relay-BP-1.")
    if args.target_logical_errors <= 0:
        raise ValueError("--target-logical-errors must be positive.")
    if args.max_shots_per_point <= 0:
        raise ValueError("--max-shots-per-point must be positive.")

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd().resolve())
    circuits_dir = repo_root / "tests" / "testdata" / "bicycle_bivariate"
    save_stem = "bicycle_bivariate_144_12_12_relay_bp1_r31_fixed_gamma_edge_weight_heatmap_cluster_Z_basis"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "examples" / "notebook_data" / save_stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    center_values = resolve_grid_values(
        explicit_values=parse_float_csv(args.center_values),
        lower=args.center_min,
        upper=args.center_max,
        count=args.center_count,
        name="center_values",
    )
    width_values = resolve_grid_values(
        explicit_values=parse_float_csv(args.width_values),
        lower=args.width_min,
        upper=args.width_max,
        count=args.width_count,
        name="width_values",
    )
    if np.any(width_values < 0.0):
        raise ValueError("All width values must be non-negative.")

    gamma_values = parse_gamma_values(args.gamma_values)
    manifest = pd.DataFrame(
        [
            {
                "gamma": float(gamma),
                "gamma_token": gamma_token(float(gamma)),
                "output_dir": str(output_dir / gamma_token(float(gamma))),
            }
            for gamma in gamma_values
        ]
    )
    manifest.to_csv(output_dir / f"{save_stem}_gamma_manifest.csv", index=False)

    print(f"repo_root = {repo_root}")
    print(f"output_dir = {output_dir}")
    print(f"gamma_values = {gamma_values.tolist()}")
    print(f"R = {args.num_sets + 1}")
    print(f"stop_nconv = {args.stop_nconv}")
    print(f"target_logical_errors = {args.target_logical_errors}")
    print(f"max_shots_per_point = {args.max_shots_per_point}")
    print(f"center_values = {center_values.tolist()}")
    print(f"width_values = {width_values.tolist()}")

    for gamma in gamma_values:
        run_one_gamma(
            args=args,
            repo_root=repo_root,
            circuits_dir=circuits_dir,
            root_output_dir=output_dir,
            gamma=float(gamma),
            center_values=center_values,
            width_values=width_values,
        )

    print("\nAll fixed-gamma edge-weight heatmaps done.")
    print(f"gamma_manifest = {output_dir / f'{save_stem}_gamma_manifest.csv'}")


if __name__ == "__main__":
    main()
