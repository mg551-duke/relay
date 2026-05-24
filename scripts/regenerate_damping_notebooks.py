from __future__ import annotations

import json
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _lines(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.split("\n")]


def _markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def _code(text: str, *, tags: list[str] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _lines(text),
    }


def _notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(_notebook(cells), indent=2) + "\n", encoding="utf-8")


def build_heatmap_notebook() -> list[dict[str, object]]:
    return [
        _markdown("""
            # Code-Capacity Damping Heat Map

            This mirrors the random-error memory-coefficient heat-map notebook, but the disordered parameter now lives on message edges. The main sweep uses interval-sampled damping coefficients and scores logical success first, with exact recovery kept as a secondary diagnostic.
            """),
        _code("""
            import importlib
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from tqdm.auto import tqdm
            """),
        _code("""
            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = find_repo_root(Path.cwd())
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from relay_bp.analysis.code_capacity_common import default_export_code_paths

            CODE_PATHS = default_export_code_paths(REPO_ROOT)

            selected_code = "gross"  # change here
            npz_path = CODE_PATHS[selected_code]

            p_random_error = 0.06
            prior_error_rate = p_random_error
            max_iter = 40
            alpha = 1.0

            center_values = np.linspace(0.50, 1.00, 11)
            width_values = np.linspace(0.00, 0.50, 11)
            random_draws_per_point = 2
            n_error_samples = 300
            base_seed = 0

            show_progress = True
            show_inner_case_progress = False

            sample_limit = None  # set to an integer for a quick smoke run

            assert np.all((0.0 <= center_values) & (center_values <= 1.0)), "center_values must stay inside [0, 1]."
            assert np.all(width_values >= 0.0), "All width values must be non-negative."
            assert np.all(np.diff(center_values) >= 0), "center_values must be sorted."
            assert np.all(np.diff(width_values) >= 0), "width_values must be sorted."
            assert n_error_samples > 0, "n_error_samples must be positive."
            """),
        _code(
            """
            import importlib
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd


            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = globals().get("REPO_ROOT", find_repo_root(Path.cwd()))
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            import relay_bp.analysis.code_capacity_common as code_capacity_common_module
            import relay_bp.analysis.code_capacity_damping as code_capacity_damping_module

            importlib.reload(code_capacity_common_module)
            importlib.reload(code_capacity_damping_module)

            from relay_bp.analysis.code_capacity_common import (
                default_export_code_paths,
                load_code_capacity_problem,
                sample_random_error_cases,
            )
            from relay_bp.analysis.code_capacity_damping import (
                evaluate_interval_heatmap,
                heatmap_rows,
            )

            CODE_PATHS = globals().get("CODE_PATHS", default_export_code_paths(REPO_ROOT))
            selected_code = globals().get("selected_code", "gross")
            npz_path = globals().get("npz_path", CODE_PATHS[selected_code])
            p_random_error = float(globals().get("p_random_error", 0.06))
            prior_error_rate = float(globals().get("prior_error_rate", p_random_error))
            max_iter = int(globals().get("max_iter", 40))
            alpha = float(globals().get("alpha", 1.0))
            center_values = np.asarray(globals().get("center_values", np.linspace(0.50, 1.00, 11)), dtype=np.float64)
            width_values = np.asarray(globals().get("width_values", np.linspace(0.00, 0.50, 11)), dtype=np.float64)
            random_draws_per_point = int(globals().get("random_draws_per_point", 2))
            n_error_samples = int(globals().get("n_error_samples", 300))
            base_seed = int(globals().get("base_seed", 0))
            show_progress = bool(globals().get("show_progress", False))
            show_inner_case_progress = bool(globals().get("show_inner_case_progress", False))
            sample_limit = globals().get("sample_limit", None)


            def load_css_code(path: Path) -> dict[str, object]:
                problem = load_code_capacity_problem(path, default_prior_error_rate=prior_error_rate)
                return {
                    "problem": problem,
                    "hx": problem.hx,
                    "hz": problem.hz,
                    "lz": problem.lz,
                    "metadata": problem.metadata,
                    "error_priors": problem.error_priors,
                }


            def sample_statistics(cases: list[dict[str, object]]) -> pd.DataFrame:
                error_weights = np.asarray([case["error_weight"] for case in cases], dtype=int)
                syndrome_weights = np.asarray([case["syndrome_weight"] for case in cases], dtype=int)
                return pd.DataFrame(
                    [
                        {
                            "code": selected_code,
                            "n_samples": int(len(cases)),
                            "p_random_error": float(p_random_error),
                            "mean_error_weight": float(error_weights.mean()) if len(error_weights) else np.nan,
                            "mean_syndrome_weight": float(syndrome_weights.mean()) if len(syndrome_weights) else np.nan,
                            "zero_error_rate": float((error_weights == 0).mean()) if len(error_weights) else np.nan,
                            "zero_syndrome_rate": float((syndrome_weights == 0).mean()) if len(syndrome_weights) else np.nan,
                            "max_error_weight": int(error_weights.max()) if len(error_weights) else -1,
                            "max_syndrome_weight": int(syndrome_weights.max()) if len(syndrome_weights) else -1,
                        }
                    ]
                )


            def sample_manifest(cases: list[dict[str, object]]) -> pd.DataFrame:
                rows = []
                for case in cases:
                    rows.append(
                        {
                            "sample_idx": int(case["sample_idx"]),
                            "error_weight": int(case["error_weight"]),
                            "syndrome_weight": int(case["syndrome_weight"]),
                            "error_support": np.flatnonzero(case["error"]).astype(int).tolist(),
                        }
                    )
                return pd.DataFrame(rows)


            def evaluate_parameter_heatmaps(cases: list[dict[str, object]], check_matrix, lz_matrix) -> dict[str, object]:
                del check_matrix, lz_matrix
                return evaluate_interval_heatmap(
                    problem=code_data["problem"],
                    cases=cases,
                    max_iter=max_iter,
                    alpha=alpha,
                    centers=center_values.tolist(),
                    widths=width_values.tolist(),
                    base_seed=base_seed,
                    random_draws_per_point=random_draws_per_point,
                    support_only=False,
                    show_progress=show_progress,
                    show_inner_case_progress=show_inner_case_progress,
                )


            def grid_table(artifact: dict[str, object]) -> pd.DataFrame:
                return pd.DataFrame(heatmap_rows(artifact)).sort_values(
                    ["logical_success_rate", "convergence_rate", "exact_recovery_rate", "mean_iterations"],
                    ascending=[False, False, False, True],
                )


            def best_points_table(artifact: dict[str, object], top_k: int = 8) -> pd.DataFrame:
                return grid_table(artifact).head(int(top_k)).reset_index(drop=True)


            def plot_parameter_heatmaps(artifact: dict[str, object]):
                metrics = [
                    ("logical_success_rate", "Logical success rate"),
                    ("convergence_rate", "Convergence rate"),
                    ("exact_recovery_rate", "Exact recovery rate"),
                    ("mean_iterations", "Mean iterations"),
                ]
                centers = np.asarray(artifact["centers"], dtype=np.float64)
                widths = np.asarray(artifact["widths"], dtype=np.float64)
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
                for axis, (metric_key, title) in zip(axes.flat, metrics):
                    image = axis.imshow(
                        np.asarray(artifact[metric_key], dtype=np.float64),
                        origin="lower",
                        aspect="auto",
                        extent=[float(centers.min()), float(centers.max()), float(widths.min()), float(widths.max())],
                        cmap="viridis",
                    )
                    axis.set_title(title)
                    axis.set_xlabel("interval center")
                    axis.set_ylabel("interval width")
                    fig.colorbar(image, ax=axis, shrink=0.85)
                fig.suptitle(f"{selected_code}: disordered damping heat maps", fontsize=14)
                return fig


            code_data = load_css_code(npz_path)
            hz_check_matrix = code_data["hz"]
            lz_matrix = code_data["lz"]
            random_error_cases = sample_random_error_cases(
                code_data["problem"],
                num_samples=n_error_samples,
                error_rate=p_random_error,
                base_seed=base_seed,
            )
            if sample_limit is not None:
                random_error_cases = random_error_cases[: int(sample_limit)]
            sample_stats_df = sample_statistics(random_error_cases)
            sample_manifest_df = sample_manifest(random_error_cases)


            _smoke_problem = load_code_capacity_problem(CODE_PATHS["gross"])
            _smoke_cases = sample_random_error_cases(_smoke_problem, num_samples=2, error_rate=0.03, base_seed=11)
            _smoke_artifact = evaluate_interval_heatmap(
                problem=_smoke_problem,
                cases=_smoke_cases,
                max_iter=8,
                alpha=1.0,
                centers=[0.75],
                widths=[0.10],
                base_seed=5,
                random_draws_per_point=1,
                support_only=False,
            )
            assert _smoke_artifact["logical_success_rate"].shape == (1, 1)
            assert int(_smoke_artifact["trial_count"][0, 0]) == len(_smoke_cases)
            """,
            tags=["smoke-test"],
        ),
        _markdown("""
            ## Random Error Sample Summary
            """),
        _code("""
            display(sample_stats_df)
            display(sample_manifest_df.head(20))
            """),
        _markdown("""
            ## Coefficient Heat Maps
            """),
        _code("""
            artifact = evaluate_parameter_heatmaps(random_error_cases, hz_check_matrix, lz_matrix)
            heatmap_df = grid_table(artifact)
            best_points_df = best_points_table(artifact)

            display(best_points_df)
            display(heatmap_df.head(16))
            fig = plot_parameter_heatmaps(artifact)
            fig
            """),
        _markdown("""
            ## Optional Extension

            One useful follow-up, kept at the end so the main flow stays familiar: compare the best disordered interval against the scalar-damping slice at the same center. That separates "interval disorder helps" from "the best point just moved the average damping."
            """),
        _code("""
            best_point = best_points_df.iloc[0]
            scalar_artifact = evaluate_interval_heatmap(
                problem=code_data["problem"],
                cases=random_error_cases,
                max_iter=max_iter,
                alpha=alpha,
                centers=[float(best_point["center"])],
                widths=[0.0],
                base_seed=base_seed,
                random_draws_per_point=random_draws_per_point,
                support_only=False,
                show_progress=show_progress,
                show_inner_case_progress=show_inner_case_progress,
            )
            scalar_row = pd.DataFrame(heatmap_rows(scalar_artifact)).iloc[0]
            comparison_df = pd.DataFrame(
                [
                    {
                        "setting": "best disordered interval",
                        "center": float(best_point["center"]),
                        "width": float(best_point["width"]),
                        "logical_success_rate": float(best_point["logical_success_rate"]),
                        "convergence_rate": float(best_point["convergence_rate"]),
                        "exact_recovery_rate": float(best_point["exact_recovery_rate"]),
                        "mean_iterations": float(best_point["mean_iterations"]),
                    },
                    {
                        "setting": "same-center scalar damping",
                        "center": float(scalar_row["center"]),
                        "width": float(scalar_row["width"]),
                        "logical_success_rate": float(scalar_row["logical_success_rate"]),
                        "convergence_rate": float(scalar_row["convergence_rate"]),
                        "exact_recovery_rate": float(scalar_row["exact_recovery_rate"]),
                        "mean_iterations": float(scalar_row["mean_iterations"]),
                    },
                ]
            )
            display(comparison_df)
            """),
    ]


def build_trace_notebook() -> list[dict[str, object]]:
    return [
        _markdown("""
            # Code-Capacity Trace Of Disordered Damping Coefficients

            This mirrors the signed-memory trace notebook cell-for-cell as closely as possible, but the perturbation now lives on check-to-bit damping messages. The two random families separate global edge disorder from stabilizer-support-local disorder.
            """),
        _code("""
            import importlib
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            """),
        _code("""
            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = find_repo_root(Path.cwd())
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from relay_bp.analysis.code_capacity_common import default_export_code_paths

            CODE_PATHS = default_export_code_paths(REPO_ROOT)

            selected_code = "gross"  # change here
            npz_path = CODE_PATHS[selected_code]

            prior_error_rate = 0.10
            max_iter = 40
            alpha = 1.0
            same_damping_coefficient = 0.75
            random_damping_interval = (0.50, 1.00)
            random_coefficient_seed_offsets = [0, 1, 2]
            core_seed_zero = 0
            core_seed_grid = list(range(1, 11))

            DECODER_SPECS = {
                "bp": {
                    "label": "Vanilla BP",
                    "coeff_seed_offsets": [0],
                },
                "damp_same": {
                    "label": "Damp-BP / same coefficient",
                    "coeff_seed_offsets": [0],
                },
                "damp_random_all_edges": {
                    "label": "Damp-BP / random interval on all edges",
                    "coeff_seed_offsets": random_coefficient_seed_offsets,
                },
                "damp_random_support_only": {
                    "label": "Damp-BP / random interval on stabilizer support",
                    "coeff_seed_offsets": random_coefficient_seed_offsets,
                },
            }
            """),
        _code(
            """
            import importlib
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd


            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = globals().get("REPO_ROOT", find_repo_root(Path.cwd()))
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            import relay_bp.analysis.code_capacity_common as code_capacity_common_module

            importlib.reload(code_capacity_common_module)

            from relay_bp.analysis.code_capacity_common import (
                build_edge_damping_messages,
                build_half_stabilizer_case_from_seed,
                default_export_code_paths,
                evaluate_decode_result,
                load_code_capacity_problem,
                make_min_sum_tracer,
            )

            CODE_PATHS = globals().get("CODE_PATHS", default_export_code_paths(REPO_ROOT))
            selected_code = globals().get("selected_code", "gross")
            npz_path = globals().get("npz_path", CODE_PATHS[selected_code])
            prior_error_rate = float(globals().get("prior_error_rate", 0.10))
            max_iter = int(globals().get("max_iter", 40))
            alpha = float(globals().get("alpha", 1.0))
            same_damping_coefficient = float(globals().get("same_damping_coefficient", 0.75))
            random_damping_interval = tuple(float(x) for x in globals().get("random_damping_interval", (0.50, 1.00)))
            random_coefficient_seed_offsets = list(globals().get("random_coefficient_seed_offsets", [0, 1, 2]))
            core_seed_zero = int(globals().get("core_seed_zero", 0))
            core_seed_grid = list(globals().get("core_seed_grid", list(range(1, 11))))
            DECODER_SPECS = globals().get(
                "DECODER_SPECS",
                {
                    "bp": {"label": "Vanilla BP", "coeff_seed_offsets": [0]},
                    "damp_same": {"label": "Damp-BP / same coefficient", "coeff_seed_offsets": [0]},
                    "damp_random_all_edges": {
                        "label": "Damp-BP / random interval on all edges",
                        "coeff_seed_offsets": random_coefficient_seed_offsets,
                    },
                    "damp_random_support_only": {
                        "label": "Damp-BP / random interval on stabilizer support",
                        "coeff_seed_offsets": random_coefficient_seed_offsets,
                    },
                },
            )


            def load_css_code(path: Path) -> dict[str, object]:
                problem = load_code_capacity_problem(path, default_prior_error_rate=prior_error_rate)
                return {
                    "problem": problem,
                    "hx": problem.hx,
                    "hz": problem.hz,
                    "lz": problem.lz,
                    "metadata": problem.metadata,
                }


            def build_half_stabilizer_x_error(problem_seed: int, code_data: dict[str, object]) -> dict[str, object]:
                case = build_half_stabilizer_case_from_seed(code_data["problem"], problem_seed)
                return {
                    "core_seed": int(problem_seed),
                    "hx_row": int(case["row_idx"]),
                    "support_bits": list(case["support_bits"]),
                    "flipped_bits": list(case["flipped_bits"]),
                    "error": case["error"],
                    "syndrome": case["syndrome"],
                    "row_weight": int(case["row_weight"]),
                    "syndrome_weight": int(case["syndrome_weight"]),
                }


            def build_damping_messages(decoder_kind: str, check_matrix, support_bits: list[int], coeff_seed: int):
                if decoder_kind == "bp":
                    return None
                if decoder_kind == "damp_same":
                    return build_edge_damping_messages(
                        check_matrix,
                        interval=(same_damping_coefficient, same_damping_coefficient),
                        coeff_seed=0,
                        support_bits=None,
                    )
                if decoder_kind == "damp_random_all_edges":
                    return build_edge_damping_messages(
                        check_matrix,
                        interval=random_damping_interval,
                        coeff_seed=coeff_seed,
                        support_bits=None,
                    )
                if decoder_kind == "damp_random_support_only":
                    return build_edge_damping_messages(
                        check_matrix,
                        interval=random_damping_interval,
                        coeff_seed=coeff_seed,
                        support_bits=tuple(int(bit) for bit in support_bits),
                    )
                raise ValueError(f"Unsupported decoder kind: {decoder_kind}")


            def summarize_messages(messages: np.ndarray | None) -> dict[str, object]:
                if messages is None:
                    return {
                        "min": np.nan,
                        "max": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                        "nontrivial_edges": 0,
                    }
                messages = np.asarray(messages, dtype=np.float64)
                return {
                    "min": float(messages.min()),
                    "max": float(messages.max()),
                    "mean": float(messages.mean()),
                    "std": float(messages.std()),
                    "nontrivial_edges": int(np.sum(np.abs(messages - 1.0) > 1e-12)),
                }


            def run_single_trace(problem: dict[str, object], decoder_kind: str, coeff_seed_offset: int, code_data: dict[str, object], check_matrix) -> dict[str, object]:
                coeff_seed = int(problem["core_seed"]) + (10_000 * int(coeff_seed_offset))
                tracer = make_min_sum_tracer(code_data["problem"], max_iter=max_iter, alpha=alpha, gamma0=None)
                tracer.reset()
                tracer.set_memory_strengths(np.zeros(code_data["problem"].n_bits, dtype=np.float64))
                tracer.clear_explicit_c_damp_messages()
                messages = build_damping_messages(decoder_kind, check_matrix, problem["support_bits"], coeff_seed)
                if messages is not None:
                    tracer.set_explicit_c_damp_messages(np.asarray(messages, dtype=np.float64))
                snapshots = [tracer.snapshot(np.asarray(problem["syndrome"], dtype=np.uint8))]
                while tracer.current_iteration < snapshots[-1].max_iter and not snapshots[-1].success:
                    snapshots.append(tracer.run_iteration(np.asarray(problem["syndrome"], dtype=np.uint8)))
                final_result = snapshots[-1]
                evaluation = evaluate_decode_result(
                    result=final_result,
                    error=np.asarray(problem["error"], dtype=np.uint8),
                    lz=code_data["problem"].lz,
                )
                posterior_trace = np.vstack(
                    [np.asarray(snapshot.posterior_ratios, dtype=np.float64) for snapshot in snapshots]
                )
                summary = summarize_messages(messages)
                summary.update(
                    {
                        "decoder_kind": decoder_kind,
                        "label": DECODER_SPECS[decoder_kind]["label"],
                        "coeff_seed_offset": int(coeff_seed_offset),
                        "coeff_seed": int(coeff_seed),
                        "iterations": int(final_result.iterations),
                        "converged": bool(final_result.success),
                        "logical_success": bool(evaluation.logical_success),
                        "exact_recovery": bool(evaluation.exact_recovery),
                        "logical_weight": int(evaluation.logical_weight),
                        "damping_mode": str(final_result.damping_mode),
                        "damping_min": final_result.damping_min,
                        "damping_max": final_result.damping_max,
                    }
                )
                return {
                    "summary": summary,
                    "posterior_trace": posterior_trace,
                    "iterations": np.asarray([int(snapshot.iterations) for snapshot in snapshots], dtype=int),
                    "traced_bits": tuple(int(bit) for bit in problem["flipped_bits"]),
                    "messages": None if messages is None else np.asarray(messages, dtype=np.float64),
                }


            def plot_trace_runs(problem: dict[str, object], decoder_kind: str, runs: list[dict[str, object]]):
                fig, axes = plt.subplots(len(runs), 1, figsize=(10, 3.0 * len(runs)), sharex=True, constrained_layout=True)
                axes = np.atleast_1d(axes)
                traced_bits = tuple(int(bit) for bit in problem["flipped_bits"])
                for axis, run in zip(axes, runs):
                    posterior_trace = np.asarray(run["posterior_trace"], dtype=np.float64)
                    iterations = np.asarray(run["iterations"], dtype=int)
                    for bit in traced_bits:
                        axis.plot(iterations, posterior_trace[:, bit], marker="o", linewidth=1.8, label=f"bit {bit}")
                    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
                    axis.set_ylabel("posterior ratio")
                    axis.set_title(
                        f"{DECODER_SPECS[decoder_kind]['label']} | offset={run['summary']['coeff_seed_offset']} | logical={run['summary']['logical_success']} | exact={run['summary']['exact_recovery']}"
                    )
                    axis.grid(alpha=0.25)
                axes[-1].set_xlabel("iteration")
                if traced_bits:
                    axes[0].legend(loc="best", ncol=min(4, len(traced_bits)))
                fig.suptitle(
                    f"core_seed={problem['core_seed']} | row={problem['hx_row']} | support={problem['support_bits']} | flipped={problem['flipped_bits']}",
                    fontsize=12,
                )
                return fig


            def run_decoder_experiment(core_seed: int, decoder_kind: str, code_data: dict[str, object], check_matrix, coeff_seed_offsets: list[int] | None = None):
                problem = build_half_stabilizer_x_error(core_seed, code_data)
                offsets = list(DECODER_SPECS[decoder_kind]["coeff_seed_offsets"] if coeff_seed_offsets is None else coeff_seed_offsets)
                runs = [run_single_trace(problem, decoder_kind, offset, code_data, check_matrix) for offset in offsets]
                summary_df = pd.DataFrame([run["summary"] for run in runs])
                coefficient_columns = ["coeff_seed_offset", "coeff_seed", "min", "max", "mean", "std", "nontrivial_edges", "damping_mode", "damping_min", "damping_max"]
                coefficient_df = summary_df[coefficient_columns].copy()
                fig = plot_trace_runs(problem, decoder_kind, runs)
                return {
                    "problem": problem,
                    "runs": runs,
                    "summary": summary_df,
                    "coefficients": coefficient_df,
                    "figure": fig,
                }


            def plot_seed_grid(decoder_kind: str, core_seeds: list[int], code_data: dict[str, object], check_matrix):
                experiments = [run_decoder_experiment(seed, decoder_kind, code_data, check_matrix) for seed in core_seeds]
                coeff_rows = []
                fig, axes = plt.subplots(len(experiments), 1, figsize=(10, 2.6 * len(experiments)), sharex=False, constrained_layout=True)
                axes = np.atleast_1d(axes)
                for axis, experiment in zip(axes, experiments):
                    run = experiment["runs"][0]
                    coeff_rows.extend(experiment["summary"].to_dict(orient="records"))
                    posterior_trace = np.asarray(run["posterior_trace"], dtype=np.float64)
                    iterations = np.asarray(run["iterations"], dtype=int)
                    traced_bits = tuple(int(bit) for bit in experiment["problem"]["flipped_bits"])
                    for bit in traced_bits:
                        axis.plot(iterations, posterior_trace[:, bit], marker="o", linewidth=1.5, label=f"bit {bit}")
                    axis.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
                    axis.set_title(
                        f"seed={experiment['problem']['core_seed']} | row={experiment['problem']['hx_row']} | logical={run['summary']['logical_success']} | exact={run['summary']['exact_recovery']}"
                    )
                    axis.grid(alpha=0.25)
                axes[-1].set_xlabel("iteration")
                axes[0].set_ylabel("posterior ratio")
                fig.suptitle(DECODER_SPECS[decoder_kind]["label"], fontsize=12)
                coeffs_df = pd.DataFrame(coeff_rows)
                return experiments, coeffs_df, fig


            code_data = load_css_code(npz_path)
            hz_check_matrix = code_data["hz"]
            core_seed_case = build_half_stabilizer_x_error(core_seed_zero, code_data)


            _smoke_code_data = load_css_code(CODE_PATHS["gross"])
            _smoke_experiment = run_decoder_experiment(
                0,
                "damp_random_support_only",
                _smoke_code_data,
                _smoke_code_data["hz"],
                coeff_seed_offsets=[0],
            )
            assert len(_smoke_experiment["runs"]) == 1
            assert "logical_success" in _smoke_experiment["summary"].columns
            assert _smoke_experiment["summary"]["coeff_seed_offset"].tolist() == [0]
            """,
            tags=["smoke-test"],
        ),
        _markdown("""
            ## Core Seed = 0
            """),
        _code("""
            experiment_bp_seed0 = run_decoder_experiment(core_seed_zero, "bp", code_data, hz_check_matrix)
            display(experiment_bp_seed0["summary"])
            display(experiment_bp_seed0["coefficients"])
            experiment_bp_seed0["figure"]
            """),
        _code("""
            experiment_damp_same_seed0 = run_decoder_experiment(core_seed_zero, "damp_same", code_data, hz_check_matrix)
            display(experiment_damp_same_seed0["summary"])
            display(experiment_damp_same_seed0["coefficients"])
            experiment_damp_same_seed0["figure"]
            """),
        _code("""
            experiment_damp_random_all_edges_seed0 = run_decoder_experiment(core_seed_zero, "damp_random_all_edges", code_data, hz_check_matrix)
            display(experiment_damp_random_all_edges_seed0["summary"])
            display(experiment_damp_random_all_edges_seed0["coefficients"])
            experiment_damp_random_all_edges_seed0["figure"]
            """),
        _code("""
            experiment_damp_random_support_only_seed0 = run_decoder_experiment(core_seed_zero, "damp_random_support_only", code_data, hz_check_matrix)
            display(experiment_damp_random_support_only_seed0["summary"])
            display(experiment_damp_random_support_only_seed0["coefficients"])
            experiment_damp_random_support_only_seed0["figure"]
            """),
        _markdown("""
            ## Core Seeds 1 To 10
            """),
        _code("""
            experiments_bp_grid, coeffs_bp_grid, fig_bp_grid = plot_seed_grid("bp", core_seed_grid, code_data, hz_check_matrix)
            display(coeffs_bp_grid.head(20))
            fig_bp_grid
            """),
        _code("""
            experiments_damp_same_grid, coeffs_damp_same_grid, fig_damp_same_grid = plot_seed_grid("damp_same", core_seed_grid, code_data, hz_check_matrix)
            display(coeffs_damp_same_grid.head(20))
            fig_damp_same_grid
            """),
        _code("""
            experiments_damp_random_all_edges_grid, coeffs_damp_random_all_edges_grid, fig_damp_random_all_edges_grid = plot_seed_grid("damp_random_all_edges", core_seed_grid, code_data, hz_check_matrix)
            display(coeffs_damp_random_all_edges_grid.head(20))
            fig_damp_random_all_edges_grid
            """),
        _code("""
            experiments_damp_random_support_only_grid, coeffs_damp_random_support_only_grid, fig_damp_random_support_only_grid = plot_seed_grid("damp_random_support_only", core_seed_grid, code_data, hz_check_matrix)
            display(coeffs_damp_random_support_only_grid.head(20))
            fig_damp_random_support_only_grid
            """),
        _markdown("""
            ## Optional Extension

            The extra check I like here is a direct same-seed comparison between global edge disorder and support-local disorder with the same coefficient offsets. That isolates whether the local placement of the disorder matters more than the interval itself.
            """),
        _code("""
            extension_rows = []
            for all_row, local_row in zip(
                experiment_damp_random_all_edges_seed0["summary"].to_dict(orient="records"),
                experiment_damp_random_support_only_seed0["summary"].to_dict(orient="records"),
            ):
                extension_rows.append(
                    {
                        "coeff_seed_offset": int(all_row["coeff_seed_offset"]),
                        "all_edges_logical_success": bool(all_row["logical_success"]),
                        "support_only_logical_success": bool(local_row["logical_success"]),
                        "all_edges_exact_recovery": bool(all_row["exact_recovery"]),
                        "support_only_exact_recovery": bool(local_row["exact_recovery"]),
                        "all_edges_nontrivial_edges": int(all_row["nontrivial_edges"]),
                        "support_only_nontrivial_edges": int(local_row["nontrivial_edges"]),
                    }
                )
            display(pd.DataFrame(extension_rows))
            """),
        _code(""),
    ]


def build_half_stabilizer_performance_notebook() -> list[dict[str, object]]:
    return [
        _markdown("""
            # Code-Capacity Performance On Half-Stabilizer Errors With Disordered Damping

            This mirrors the half-stabilizer memory-performance notebook, but uses edge-wise damping instead of per-bit memory strengths. The two random families separate all-edge disorder from support-local disorder on the chosen half-stabilizer.
            """),
        _code("""
            import importlib
            from math import comb
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from tqdm.auto import tqdm
            """),
        _code("""
            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = find_repo_root(Path.cwd())
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from relay_bp.analysis.code_capacity_common import default_export_code_paths

            CODE_PATHS = default_export_code_paths(REPO_ROOT)

            selected_code = "surface13"  # change here to switch code
            npz_path = CODE_PATHS[selected_code]

            prior_error_rate = 0.10
            max_iter = 40
            alpha = 1.0
            same_damping_coefficient = 0.75
            random_damping_interval = (0.50, 1.00)
            base_disorder_seed = 0
            random_coefficient_seed_offsets = [0, 1, 2]

            DECODER_SPECS = {
                "bp": {
                    "label": "Vanilla BP",
                    "coeff_seed_offsets": [0],
                },
                "damp_same": {
                    "label": "Damp-BP / same coefficient",
                    "coeff_seed_offsets": [0],
                },
                "damp_random_all_edges": {
                    "label": "Damp-BP / random interval on all edges",
                    "coeff_seed_offsets": random_coefficient_seed_offsets,
                },
                "damp_random_support_only": {
                    "label": "Damp-BP / random interval on stabilizer support",
                    "coeff_seed_offsets": random_coefficient_seed_offsets,
                },
            }
            """),
        _code(
            """
            import importlib
            from math import comb
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd


            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = globals().get("REPO_ROOT", find_repo_root(Path.cwd()))
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            import relay_bp.analysis.code_capacity_common as code_capacity_common_module

            importlib.reload(code_capacity_common_module)

            from relay_bp.analysis.code_capacity_common import (
                build_edge_damping_messages,
                default_export_code_paths,
                decode_with_edge_damping,
                decode_with_plain_bp,
                enumerate_half_stabilizer_cases,
                evaluate_decode_result,
                load_code_capacity_problem,
                make_min_sum_tracer,
            )

            CODE_PATHS = globals().get("CODE_PATHS", default_export_code_paths(REPO_ROOT))
            selected_code = globals().get("selected_code", "surface13")
            npz_path = globals().get("npz_path", CODE_PATHS[selected_code])
            prior_error_rate = float(globals().get("prior_error_rate", 0.10))
            max_iter = int(globals().get("max_iter", 40))
            alpha = float(globals().get("alpha", 1.0))
            same_damping_coefficient = float(globals().get("same_damping_coefficient", 0.75))
            random_damping_interval = tuple(float(x) for x in globals().get("random_damping_interval", (0.50, 1.00)))
            base_disorder_seed = int(globals().get("base_disorder_seed", 0))
            random_coefficient_seed_offsets = list(globals().get("random_coefficient_seed_offsets", [0, 1, 2]))
            DECODER_SPECS = globals().get(
                "DECODER_SPECS",
                {
                    "bp": {"label": "Vanilla BP", "coeff_seed_offsets": [0]},
                    "damp_same": {"label": "Damp-BP / same coefficient", "coeff_seed_offsets": [0]},
                    "damp_random_all_edges": {
                        "label": "Damp-BP / random interval on all edges",
                        "coeff_seed_offsets": random_coefficient_seed_offsets,
                    },
                    "damp_random_support_only": {
                        "label": "Damp-BP / random interval on stabilizer support",
                        "coeff_seed_offsets": random_coefficient_seed_offsets,
                    },
                },
            )


            def load_css_code(path: Path) -> dict[str, object]:
                problem = load_code_capacity_problem(path, default_prior_error_rate=prior_error_rate)
                return {
                    "problem": problem,
                    "hx": problem.hx,
                    "hz": problem.hz,
                    "lz": problem.lz,
                    "metadata": problem.metadata,
                }


            def stabilizer_statistics(hx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
                weights = hx.sum(axis=1).astype(int)
                unique_weights, counts = np.unique(weights, return_counts=True)
                rows = []
                total_raw_half_subsets = 0
                total_even_weight_stabilizers = 0
                for weight, count in zip(unique_weights, counts):
                    weight = int(weight)
                    count = int(count)
                    is_even = weight > 0 and weight % 2 == 0
                    raw_half_subsets_per_row = comb(weight, weight // 2) if is_even else 0
                    total_raw = count * raw_half_subsets_per_row
                    if is_even:
                        total_raw_half_subsets += total_raw
                        total_even_weight_stabilizers += count
                    rows.append(
                        {
                            "row_weight": weight,
                            "n_x_stabilizers": count,
                            "is_even_weight": is_even,
                            "raw_half_subsets_per_row": raw_half_subsets_per_row,
                            "total_raw_half_subsets": total_raw,
                        }
                    )
                summary = pd.DataFrame(
                    [
                        {
                            "code": selected_code,
                            "Hx_rows": int(hx.shape[0]),
                            "n_qubits": int(hx.shape[1]),
                            "even_weight_x_stabilizers": total_even_weight_stabilizers,
                            "total_half_flip_patterns": total_raw_half_subsets,
                        }
                    ]
                )
                return pd.DataFrame(rows), summary


            def case_manifest(cases: list[dict[str, object]], head: int | None = 20) -> pd.DataFrame:
                manifest = pd.DataFrame(
                    [
                        {
                            "case_id": int(case["case_id"]),
                            "row_idx": int(case["row_idx"]),
                            "row_weight": int(case["row_weight"]),
                            "half_subset_idx": int(case["half_subset_idx"]),
                            "syndrome_weight": int(case["syndrome_weight"]),
                            "support_bits": list(case["support_bits"]),
                            "flipped_bits": list(case["flipped_bits"]),
                        }
                        for case in cases
                    ]
                )
                return manifest.head(head) if head is not None else manifest


            def build_damping_messages(decoder_kind: str, check_matrix, case: dict[str, object], coeff_seed: int):
                if decoder_kind == "bp":
                    return None
                if decoder_kind == "damp_same":
                    return build_edge_damping_messages(
                        check_matrix,
                        interval=(same_damping_coefficient, same_damping_coefficient),
                        coeff_seed=0,
                        support_bits=None,
                    )
                if decoder_kind == "damp_random_all_edges":
                    return build_edge_damping_messages(
                        check_matrix,
                        interval=random_damping_interval,
                        coeff_seed=coeff_seed,
                        support_bits=None,
                    )
                if decoder_kind == "damp_random_support_only":
                    return build_edge_damping_messages(
                        check_matrix,
                        interval=random_damping_interval,
                        coeff_seed=coeff_seed,
                        support_bits=tuple(int(bit) for bit in case["support_bits"]),
                    )
                raise ValueError(f"Unsupported decoder kind: {decoder_kind}")


            def evaluate_decoder(decoder_kind: str, cases: list[dict[str, object]], check_matrix, lz_matrix):
                del lz_matrix
                rows = []
                plain_tracer = make_min_sum_tracer(code_data["problem"], max_iter=max_iter, alpha=alpha, gamma0=None)
                damping_tracer = make_min_sum_tracer(code_data["problem"], max_iter=max_iter, alpha=alpha, gamma0=None)
                for case in cases:
                    for coeff_seed_offset in DECODER_SPECS[decoder_kind]["coeff_seed_offsets"]:
                        coeff_seed = int(base_disorder_seed) + (10_000 * int(coeff_seed_offset)) + int(case["case_id"])
                        if decoder_kind == "bp":
                            result = decode_with_plain_bp(plain_tracer, case["syndrome"], n_bits=code_data["problem"].n_bits)
                            messages = None
                        else:
                            messages = build_damping_messages(decoder_kind, check_matrix, case, coeff_seed)
                            result = decode_with_edge_damping(
                                damping_tracer,
                                case["syndrome"],
                                messages,
                                n_bits=code_data["problem"].n_bits,
                            )
                        evaluation = evaluate_decode_result(result=result, error=case["error"], lz=code_data["problem"].lz)
                        rows.append(
                            {
                                "decoder_kind": decoder_kind,
                                "label": DECODER_SPECS[decoder_kind]["label"],
                                "case_id": int(case["case_id"]),
                                "row_idx": int(case["row_idx"]),
                                "row_weight": int(case["row_weight"]),
                                "half_subset_idx": int(case["half_subset_idx"]),
                                "syndrome_weight": int(case["syndrome_weight"]),
                                "coeff_seed_offset": int(coeff_seed_offset),
                                "coeff_seed": int(coeff_seed),
                                "converged": bool(result.success),
                                "logical_success": bool(evaluation.logical_success),
                                "exact_recovery": bool(evaluation.exact_recovery),
                                "logical_weight": int(evaluation.logical_weight),
                                "iterations": int(result.iterations),
                                "damping_min": np.nan if messages is None else float(np.min(messages)),
                                "damping_max": np.nan if messages is None else float(np.max(messages)),
                                "nontrivial_edges": 0 if messages is None else int(np.sum(np.abs(np.asarray(messages) - 1.0) > 1e-12)),
                            }
                        )
                rows_df = pd.DataFrame(rows)
                summary_df = pd.DataFrame(
                    [
                        {
                            "decoder_kind": decoder_kind,
                            "label": DECODER_SPECS[decoder_kind]["label"],
                            "n_trials": int(len(rows_df)),
                            "logical_success_rate": float(rows_df["logical_success"].mean()),
                            "exact_recovery_rate": float(rows_df["exact_recovery"].mean()),
                            "convergence_rate": float(rows_df["converged"].mean()),
                            "mean_iterations": float(rows_df["iterations"].mean()),
                            "mean_logical_weight": float(rows_df["logical_weight"].mean()),
                        }
                    ]
                )
                by_row_weight = (
                    rows_df.groupby("row_weight", as_index=False)
                    .agg(
                        n_trials=("case_id", "size"),
                        logical_success_rate=("logical_success", "mean"),
                        exact_recovery_rate=("exact_recovery", "mean"),
                        convergence_rate=("converged", "mean"),
                        mean_iterations=("iterations", "mean"),
                    )
                    .sort_values("row_weight")
                )
                by_syndrome_weight = (
                    rows_df.groupby("syndrome_weight", as_index=False)
                    .agg(
                        n_trials=("case_id", "size"),
                        logical_success_rate=("logical_success", "mean"),
                        exact_recovery_rate=("exact_recovery", "mean"),
                        convergence_rate=("converged", "mean"),
                    )
                    .sort_values("syndrome_weight")
                )
                fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
                axes[0].bar(
                    ["logical", "exact", "converged"],
                    [
                        float(summary_df.iloc[0]["logical_success_rate"]),
                        float(summary_df.iloc[0]["exact_recovery_rate"]),
                        float(summary_df.iloc[0]["convergence_rate"]),
                    ],
                    color=["tab:blue", "tab:orange", "tab:green"],
                )
                axes[0].set_ylim(0.0, 1.0)
                axes[0].set_title(DECODER_SPECS[decoder_kind]["label"])
                axes[1].plot(by_row_weight["row_weight"], by_row_weight["logical_success_rate"], marker="o", label="logical")
                axes[1].plot(by_row_weight["row_weight"], by_row_weight["exact_recovery_rate"], marker="o", label="exact")
                axes[1].set_xlabel("row weight")
                axes[1].set_ylabel("rate")
                axes[1].set_ylim(0.0, 1.0)
                axes[1].legend(loc="best")
                axes[2].plot(by_syndrome_weight["syndrome_weight"], by_syndrome_weight["logical_success_rate"], marker="o", label="logical")
                axes[2].plot(by_syndrome_weight["syndrome_weight"], by_syndrome_weight["exact_recovery_rate"], marker="o", label="exact")
                axes[2].set_xlabel("syndrome weight")
                axes[2].set_ylim(0.0, 1.0)
                axes[2].legend(loc="best")
                return {
                    "rows": rows_df,
                    "summary": summary_df,
                    "by_row_weight": by_row_weight,
                    "by_syndrome_weight": by_syndrome_weight,
                    "figure": fig,
                }


            def exact_vs_logical_ranking(results_map: dict[str, dict[str, object]]) -> pd.DataFrame:
                rows = []
                for name, payload in results_map.items():
                    summary = payload["summary"].iloc[0]
                    rows.append(
                        {
                            "decoder_kind": name,
                            "label": str(summary["label"]),
                            "logical_success_rate": float(summary["logical_success_rate"]),
                            "exact_recovery_rate": float(summary["exact_recovery_rate"]),
                            "convergence_rate": float(summary["convergence_rate"]),
                            "logical_rank": 0,
                            "exact_rank": 0,
                        }
                    )
                ranking = pd.DataFrame(rows)
                ranking["logical_rank"] = ranking["logical_success_rate"].rank(method="dense", ascending=False).astype(int)
                ranking["exact_rank"] = ranking["exact_recovery_rate"].rank(method="dense", ascending=False).astype(int)
                return ranking.sort_values(["logical_rank", "exact_rank", "decoder_kind"]).reset_index(drop=True)


            code_data = load_css_code(npz_path)
            hz_check_matrix = code_data["hz"]
            lz_matrix = code_data["lz"]
            stabilizer_rows, stabilizer_summary = stabilizer_statistics(code_data["hx"])
            all_cases = enumerate_half_stabilizer_cases(code_data["problem"])
            case_manifest_df = case_manifest(all_cases)


            _smoke_cases = all_cases[:4]
            _smoke_results = evaluate_decoder("damp_random_support_only", _smoke_cases, hz_check_matrix, lz_matrix)
            assert int(_smoke_results["summary"].iloc[0]["n_trials"]) == len(_smoke_cases) * len(random_coefficient_seed_offsets)
            assert "logical_success_rate" in _smoke_results["summary"].columns
            """,
            tags=["smoke-test"],
        ),
        _markdown("""
            ## Half-Stabilizer Error Set Statistics
            """),
        _code("""
            display(stabilizer_summary)
            display(stabilizer_rows)
            display(case_manifest_df)
            """),
        _markdown("""
            ## Decoder Cells
            """),
        _code("""
            results_bp = evaluate_decoder("bp", all_cases, hz_check_matrix, lz_matrix)
            display(results_bp["summary"])
            display(results_bp["by_row_weight"])
            display(results_bp["by_syndrome_weight"])
            results_bp["figure"]
            """),
        _code("""
            results_damp_same = evaluate_decoder("damp_same", all_cases, hz_check_matrix, lz_matrix)
            display(results_damp_same["summary"])
            display(results_damp_same["by_row_weight"])
            display(results_damp_same["by_syndrome_weight"])
            results_damp_same["figure"]
            """),
        _code("""
            results_damp_random_all_edges = evaluate_decoder("damp_random_all_edges", all_cases, hz_check_matrix, lz_matrix)
            display(results_damp_random_all_edges["summary"])
            display(results_damp_random_all_edges["by_row_weight"])
            display(results_damp_random_all_edges["by_syndrome_weight"])
            results_damp_random_all_edges["figure"]
            """),
        _code("""
            results_damp_random_support_only = evaluate_decoder("damp_random_support_only", all_cases, hz_check_matrix, lz_matrix)
            display(results_damp_random_support_only["summary"])
            display(results_damp_random_support_only["by_row_weight"])
            display(results_damp_random_support_only["by_syndrome_weight"])
            results_damp_random_support_only["figure"]
            """),
        _markdown("""
            ## Optional Extension

            A useful end-of-notebook cross-check is to rank decoder families once by logical success and once by exact recovery. For half-stabilizer cases, that quickly shows whether a family is genuinely better or is only choosing a different representative of the same logical class.
            """),
        _code("""
            ranking_df = exact_vs_logical_ranking(
                {
                    "bp": results_bp,
                    "damp_same": results_damp_same,
                    "damp_random_all_edges": results_damp_random_all_edges,
                    "damp_random_support_only": results_damp_random_support_only,
                }
            )
            display(ranking_df)
            """),
    ]


def build_half_stabilizer_heatmap_notebook() -> list[dict[str, object]]:
    return [
        _markdown("""
            # Code-Capacity Damping Heat Maps For Half-Stabilizer Errors

            This follows the half-stabilizer memory heat-map notebook closely, but sweeps message-edge damping intervals instead of per-bit memory strengths. The application scope switch keeps the same notebook structure while toggling between all-edge disorder and stabilizer-support-local disorder.
            """),
        _code("""
            import importlib
            from math import comb
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from tqdm.auto import tqdm
            """),
        _code("""
            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = find_repo_root(Path.cwd())
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from relay_bp.analysis.code_capacity_common import default_export_code_paths

            CODE_PATHS = default_export_code_paths(REPO_ROOT)

            selected_code = "gross"  # change here
            npz_path = CODE_PATHS[selected_code]

            prior_error_rate = 0.10
            max_iter = 40
            alpha = 1.0
            damping_application_scope = "stabilizer_support"  # options: "stabilizer_support", "all_edges"

            center_values = np.linspace(0.50, 1.00, 11)
            width_values = np.linspace(0.00, 0.50, 11)
            random_draws_per_point = 2
            base_seed = 0

            show_progress = True
            show_inner_case_progress = False

            selected_row_weights = None  # e.g. {4, 6}
            case_limit = None  # set to an integer for a quick smoke run

            assert np.all((0.0 <= center_values) & (center_values <= 1.0)), "center_values must stay inside [0, 1]."
            assert np.all(width_values >= 0), "All width values must be non-negative."
            assert np.all(np.diff(center_values) >= 0), "center_values must be sorted."
            assert np.all(np.diff(width_values) >= 0), "width_values must be sorted."
            """),
        _code(
            """
            import importlib
            from math import comb
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd


            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root from the current working directory.")


            REPO_ROOT = globals().get("REPO_ROOT", find_repo_root(Path.cwd()))
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            import relay_bp.analysis.code_capacity_common as code_capacity_common_module
            import relay_bp.analysis.code_capacity_damping as code_capacity_damping_module

            importlib.reload(code_capacity_common_module)
            importlib.reload(code_capacity_damping_module)

            from relay_bp.analysis.code_capacity_common import (
                default_export_code_paths,
                enumerate_half_stabilizer_cases,
                load_code_capacity_problem,
            )
            from relay_bp.analysis.code_capacity_damping import (
                evaluate_interval_heatmap,
                heatmap_rows,
            )

            CODE_PATHS = globals().get("CODE_PATHS", default_export_code_paths(REPO_ROOT))
            selected_code = globals().get("selected_code", "gross")
            npz_path = globals().get("npz_path", CODE_PATHS[selected_code])
            prior_error_rate = float(globals().get("prior_error_rate", 0.10))
            max_iter = int(globals().get("max_iter", 40))
            alpha = float(globals().get("alpha", 1.0))
            damping_application_scope = globals().get("damping_application_scope", "stabilizer_support")
            center_values = np.asarray(globals().get("center_values", np.linspace(0.50, 1.00, 11)), dtype=np.float64)
            width_values = np.asarray(globals().get("width_values", np.linspace(0.00, 0.50, 11)), dtype=np.float64)
            random_draws_per_point = int(globals().get("random_draws_per_point", 2))
            base_seed = int(globals().get("base_seed", 0))
            show_progress = bool(globals().get("show_progress", False))
            show_inner_case_progress = bool(globals().get("show_inner_case_progress", False))
            selected_row_weights = globals().get("selected_row_weights", None)
            case_limit = globals().get("case_limit", None)


            def load_css_code(path: Path) -> dict[str, object]:
                problem = load_code_capacity_problem(path, default_prior_error_rate=prior_error_rate)
                return {
                    "problem": problem,
                    "hx": problem.hx,
                    "hz": problem.hz,
                    "lz": problem.lz,
                    "metadata": problem.metadata,
                }


            def stabilizer_statistics(hx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
                weights = hx.sum(axis=1).astype(int)
                unique_weights, counts = np.unique(weights, return_counts=True)
                rows = []
                total_even_weight_stabilizers = 0
                total_half_flip_patterns = 0
                for weight, count in zip(unique_weights, counts):
                    weight = int(weight)
                    count = int(count)
                    is_even = weight > 0 and weight % 2 == 0
                    half_subsets_per_row = comb(weight, weight // 2) if is_even else 0
                    total_half = count * half_subsets_per_row
                    if is_even:
                        total_even_weight_stabilizers += count
                        total_half_flip_patterns += total_half
                    rows.append(
                        {
                            "row_weight": weight,
                            "n_x_stabilizers": count,
                            "is_even_weight": is_even,
                            "half_flip_patterns_per_row": half_subsets_per_row,
                            "total_half_flip_patterns": total_half,
                        }
                    )
                summary = pd.DataFrame(
                    [
                        {
                            "code": selected_code,
                            "Hx_rows": int(hx.shape[0]),
                            "n_qubits": int(hx.shape[1]),
                            "even_weight_x_stabilizers": total_even_weight_stabilizers,
                            "total_half_flip_patterns": total_half_flip_patterns,
                        }
                    ]
                )
                return pd.DataFrame(rows), summary


            def case_manifest(cases: list[dict[str, object]], head: int | None = 20) -> pd.DataFrame:
                manifest = pd.DataFrame(
                    [
                        {
                            "case_id": int(case["case_id"]),
                            "row_idx": int(case["row_idx"]),
                            "row_weight": int(case["row_weight"]),
                            "half_subset_idx": int(case["half_subset_idx"]),
                            "syndrome_weight": int(case["syndrome_weight"]),
                            "support_bits": list(case["support_bits"]),
                            "flipped_bits": list(case["flipped_bits"]),
                        }
                        for case in cases
                    ]
                )
                return manifest.head(head) if head is not None else manifest


            def evaluate_parameter_heatmaps(cases: list[dict[str, object]], check_matrix, lz_matrix) -> dict[str, object]:
                del check_matrix, lz_matrix
                return evaluate_interval_heatmap(
                    problem=code_data["problem"],
                    cases=cases,
                    max_iter=max_iter,
                    alpha=alpha,
                    centers=center_values.tolist(),
                    widths=width_values.tolist(),
                    base_seed=base_seed,
                    random_draws_per_point=random_draws_per_point,
                    support_only=damping_application_scope == "stabilizer_support",
                    show_progress=show_progress,
                    show_inner_case_progress=show_inner_case_progress,
                )


            def grid_table(artifact: dict[str, object]) -> pd.DataFrame:
                return pd.DataFrame(heatmap_rows(artifact)).sort_values(
                    ["logical_success_rate", "convergence_rate", "exact_recovery_rate", "mean_iterations"],
                    ascending=[False, False, False, True],
                )


            def best_points_table(artifact: dict[str, object], top_k: int = 8) -> pd.DataFrame:
                return grid_table(artifact).head(int(top_k)).reset_index(drop=True)


            def plot_parameter_heatmaps(artifact: dict[str, object]):
                metrics = [
                    ("logical_success_rate", "Logical success rate"),
                    ("convergence_rate", "Convergence rate"),
                    ("exact_recovery_rate", "Exact recovery rate"),
                    ("mean_iterations", "Mean iterations"),
                ]
                centers = np.asarray(artifact["centers"], dtype=np.float64)
                widths = np.asarray(artifact["widths"], dtype=np.float64)
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
                for axis, (metric_key, title) in zip(axes.flat, metrics):
                    image = axis.imshow(
                        np.asarray(artifact[metric_key], dtype=np.float64),
                        origin="lower",
                        aspect="auto",
                        extent=[float(centers.min()), float(centers.max()), float(widths.min()), float(widths.max())],
                        cmap="viridis",
                    )
                    axis.set_title(title)
                    axis.set_xlabel("interval center")
                    axis.set_ylabel("interval width")
                    fig.colorbar(image, ax=axis, shrink=0.85)
                fig.suptitle(
                    f"{selected_code}: half-stabilizer damping heat maps ({damping_application_scope})",
                    fontsize=14,
                )
                return fig


            code_data = load_css_code(npz_path)
            hz_check_matrix = code_data["hz"]
            lz_matrix = code_data["lz"]
            stabilizer_rows, stabilizer_summary = stabilizer_statistics(code_data["hx"])
            all_cases = enumerate_half_stabilizer_cases(
                code_data["problem"],
                selected_row_weights=None if selected_row_weights is None else set(int(weight) for weight in selected_row_weights),
                max_cases=case_limit,
            )
            case_manifest_df = case_manifest(all_cases)


            _smoke_code_data = load_css_code(CODE_PATHS["surface13"])
            _smoke_cases = enumerate_half_stabilizer_cases(_smoke_code_data["problem"], max_cases=4)
            _smoke_artifact = evaluate_interval_heatmap(
                problem=_smoke_code_data["problem"],
                cases=_smoke_cases,
                max_iter=8,
                alpha=1.0,
                centers=[0.75],
                widths=[0.10],
                base_seed=3,
                random_draws_per_point=1,
                support_only=True,
            )
            assert _smoke_artifact["logical_success_rate"].shape == (1, 1)
            assert int(_smoke_artifact["trial_count"][0, 0]) == len(_smoke_cases)
            """,
            tags=["smoke-test"],
        ),
        _markdown("""
            ## Half-Stabilizer Catalogue
            """),
        _code("""
            display(stabilizer_summary)
            display(stabilizer_rows)
            display(case_manifest_df)
            """),
        _markdown("""
            ## Coefficient Heat Maps
            """),
        _code("""
            artifact = evaluate_parameter_heatmaps(all_cases, hz_check_matrix, lz_matrix)
            heatmap_df = grid_table(artifact)
            best_points_df = best_points_table(artifact)

            display(best_points_df)
            display(heatmap_df.head(16))
            fig = plot_parameter_heatmaps(artifact)
            fig
            """),
        _markdown("""
            ## Optional Extension

            A useful end-of-notebook comparison is to take the best point from the chosen application scope and rerun it under the other scope. That tells you whether the gain came from the interval values themselves or from keeping the disorder local to the half-stabilizer support.
            """),
        _code("""
            best_point = best_points_df.iloc[0]
            alternate_scope_artifact = evaluate_interval_heatmap(
                problem=code_data["problem"],
                cases=all_cases,
                max_iter=max_iter,
                alpha=alpha,
                centers=[float(best_point["center"])],
                widths=[float(best_point["width"])],
                base_seed=base_seed,
                random_draws_per_point=random_draws_per_point,
                support_only=not (damping_application_scope == "stabilizer_support"),
                show_progress=show_progress,
                show_inner_case_progress=show_inner_case_progress,
            )
            alternate_row = pd.DataFrame(heatmap_rows(alternate_scope_artifact)).iloc[0]
            comparison_df = pd.DataFrame(
                [
                    {
                        "scope": damping_application_scope,
                        "center": float(best_point["center"]),
                        "width": float(best_point["width"]),
                        "logical_success_rate": float(best_point["logical_success_rate"]),
                        "exact_recovery_rate": float(best_point["exact_recovery_rate"]),
                        "mean_iterations": float(best_point["mean_iterations"]),
                    },
                    {
                        "scope": "all_edges" if damping_application_scope == "stabilizer_support" else "stabilizer_support",
                        "center": float(alternate_row["center"]),
                        "width": float(alternate_row["width"]),
                        "logical_success_rate": float(alternate_row["logical_success_rate"]),
                        "exact_recovery_rate": float(alternate_row["exact_recovery_rate"]),
                        "mean_iterations": float(alternate_row["mean_iterations"]),
                    },
                ]
            )
            display(comparison_df)
            """),
    ]


def main() -> None:
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityDampingPerformanceHeatMap.ipynb",
        build_heatmap_notebook(),
    )
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityDampingTrace.ipynb", build_trace_notebook()
    )
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityHalfStabilizerDampingPerformance.ipynb",
        build_half_stabilizer_performance_notebook(),
    )
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityHalfStabilizerDampingHeatmaps.ipynb",
        build_half_stabilizer_heatmap_notebook(),
    )


if __name__ == "__main__":
    main()
