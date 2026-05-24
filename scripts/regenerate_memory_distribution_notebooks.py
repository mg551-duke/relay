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
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


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
            # Code-Capacity Gaussian Memory Heatmaps

            This notebook compares uniform interval sampling against truncated Gaussian sampling on the same support interval. The scripts in `scripts/` are the source of truth; this notebook is a lightweight companion for interactive inspection.
            """),
        _code("""
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            """),
        _code("""
            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root.")


            REPO_ROOT = find_repo_root(Path.cwd())
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from relay_bp.analysis import (
                best_heatmap_rows,
                default_export_code_paths,
                evaluate_matched_support_heatmaps,
                load_code_capacity_problem,
                matched_support_heatmap_rows,
                sample_random_error_cases,
                with_uniform_error_rate,
            )

            CODE_PATHS = default_export_code_paths(REPO_ROOT)
            selected_code = "surface13"
            p_random_error = 0.09
            centers = np.linspace(-0.1, 0.3, 5)
            widths = np.linspace(0.0, 0.6, 5)
            heatmap_draws = 2
            n_error_samples = 64
            max_iter = 20
            alpha = 1.0
            """),
        _code("""
            base_problem = load_code_capacity_problem(CODE_PATHS[selected_code])
            problem = with_uniform_error_rate(base_problem, p_random_error)
            cases = sample_random_error_cases(
                problem,
                num_samples=n_error_samples,
                error_rate=p_random_error,
                base_seed=0,
            )
            artifact = evaluate_matched_support_heatmaps(
                problem=problem,
                cases=cases,
                max_iter=max_iter,
                alpha=alpha,
                centers=centers.tolist(),
                widths=widths.tolist(),
                base_seed=0,
                random_draws_per_point=heatmap_draws,
                application_scope="all_bits",
                selection_mode="single_draw",
                logical_weight_penalty=4.0,
                convergence_penalty=1.0,
                iteration_penalty=0.05,
            )
            heatmap_table = pd.DataFrame(matched_support_heatmap_rows(artifact))
            best_points = pd.DataFrame(best_heatmap_rows(heatmap_table.to_dict("records"), topk=8))
            display(best_points)
            """),
        _code("""
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            for ax, family_name in zip(axes, ["uniform_interval", "truncated_gaussian"]):
                image = ax.imshow(
                    artifact["families"][family_name]["logical_success_rate"],
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                )
                ax.set_title(family_name)
                ax.set_xlabel("center index")
                ax.set_ylabel("width index")
                fig.colorbar(image, ax=ax, shrink=0.8)
            plt.show()
            """),
        _code(
            """
            from pathlib import Path
            import sys


            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root.")


            _repo_root = find_repo_root(Path.cwd())
            _src_root = _repo_root / "src"
            if str(_src_root) not in sys.path:
                sys.path.insert(0, str(_src_root))

            from relay_bp.analysis import (
                default_export_code_paths,
                evaluate_matched_support_heatmaps,
                load_code_capacity_problem,
                matched_support_heatmap_rows,
                sample_random_error_cases,
                with_uniform_error_rate,
            )

            _code_paths = default_export_code_paths(_repo_root)
            smoke_problem = with_uniform_error_rate(load_code_capacity_problem(_code_paths["surface13"]), 0.09)
            smoke_cases = sample_random_error_cases(smoke_problem, num_samples=8, error_rate=0.09, base_seed=11)
            smoke_artifact = evaluate_matched_support_heatmaps(
                problem=smoke_problem,
                cases=smoke_cases,
                max_iter=10,
                alpha=1.0,
                centers=[0.0, 0.2],
                widths=[0.0, 0.4],
                base_seed=5,
                random_draws_per_point=1,
                application_scope="all_bits",
                selection_mode="single_draw",
                logical_weight_penalty=4.0,
                convergence_penalty=1.0,
                iteration_penalty=0.05,
            )
            smoke_rows = matched_support_heatmap_rows(smoke_artifact)
            assert len(smoke_rows) == 8
            assert int(smoke_artifact["families"]["uniform_interval"]["trial_count"][0, 0]) == len(smoke_cases)
            """,
            tags=["smoke-test"],
        ),
    ]


def build_half_study_notebook() -> list[dict[str, object]]:
    return [
        _markdown("""
            # Code-Capacity Half-Stabilizer Gaussian Study

            This notebook focuses on half-stabilizer behaviour, including Gaussian refinement, practical K-draw aggregation, and equivalence reporting between deterministic and rerolled strategies.
            """),
        _code("""
            from pathlib import Path
            import sys

            import numpy as np
            import pandas as pd
            """),
        _code("""
            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root.")


            REPO_ROOT = find_repo_root(Path.cwd())
            SRC_ROOT = REPO_ROOT / "src"
            if str(SRC_ROOT) not in sys.path:
                sys.path.insert(0, str(SRC_ROOT))

            from relay_bp.analysis import (
                EquivalenceThresholds,
                MemorySamplingSpec,
                bootstrap_confidence_interval,
                default_export_code_paths,
                enumerate_half_stabilizer_cases,
                evaluate_equivalence,
                evaluate_gaussian_refinement_heatmap,
                evaluate_sampling_strategy,
                gaussian_refinement_heatmap_rows,
                load_code_capacity_problem,
                with_uniform_error_rate,
            )

            CODE_PATHS = default_export_code_paths(REPO_ROOT)
            selected_code = "surface13"
            p_random_error = 0.09
            """),
        _code("""
            base_problem = load_code_capacity_problem(CODE_PATHS[selected_code])
            problem = with_uniform_error_rate(base_problem, p_random_error)
            half_cases = enumerate_half_stabilizer_cases(problem, max_cases=64, shuffle_seed=0)
            refinement = evaluate_gaussian_refinement_heatmap(
                problem=problem,
                cases=half_cases,
                max_iter=20,
                alpha=1.0,
                means=np.linspace(-0.1, 0.3, 5).tolist(),
                sigmas=np.linspace(0.01, 0.15, 5).tolist(),
                low=-0.1,
                high=0.3,
                base_seed=17,
                random_draws_per_point=2,
                application_scope="stabilizer_support",
                selection_mode="single_draw",
                logical_weight_penalty=4.0,
                convergence_penalty=1.0,
                iteration_penalty=0.05,
            )
            refinement_rows = pd.DataFrame(gaussian_refinement_heatmap_rows(refinement))
            display(refinement_rows.sort_values(["loss_mean", "mean_iterations"]).head(8))
            """),
        _code("""
            practical_spec = MemorySamplingSpec(
                sampling_family="truncated_gaussian",
                application_scope="stabilizer_support",
                selection_mode="k_draw_practical",
                draw_count=4,
                low=-0.1,
                high=0.3,
                mean=0.1,
                sigma=0.05,
            )
            practical_metrics = evaluate_sampling_strategy(
                problem=problem,
                cases=half_cases,
                sampling_spec=practical_spec,
                max_iter=20,
                alpha=1.0,
                logical_weight_penalty=4.0,
                convergence_penalty=1.0,
                iteration_penalty=0.05,
                base_seed=5,
            )
            pd.DataFrame([practical_metrics])
            """),
        _code("""
            equivalence = evaluate_equivalence(
                candidate_rows=[
                    {"logical_success_rate": 0.80, "convergence_rate": 0.70, "mean_iterations": 8.8},
                    {"logical_success_rate": 0.82, "convergence_rate": 0.73, "mean_iterations": 8.5},
                ],
                baseline_rows=[
                    {"logical_success_rate": 0.81, "convergence_rate": 0.72, "mean_iterations": 8.7},
                    {"logical_success_rate": 0.82, "convergence_rate": 0.73, "mean_iterations": 8.6},
                ],
                thresholds=EquivalenceThresholds(bootstrap_samples=512, seed=3),
            )
            pd.DataFrame([equivalence["logical_success_difference"], equivalence["convergence_difference"]])
            """),
        _code(
            """
            from pathlib import Path
            import sys


            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "pyproject.toml").exists() and (candidate / "src" / "relay_bp").exists():
                        return candidate
                raise FileNotFoundError("Could not locate the relay repo root.")


            _repo_root = find_repo_root(Path.cwd())
            _src_root = _repo_root / "src"
            if str(_src_root) not in sys.path:
                sys.path.insert(0, str(_src_root))

            from relay_bp.analysis import (
                MemorySamplingSpec,
                bootstrap_confidence_interval,
                default_export_code_paths,
                enumerate_half_stabilizer_cases,
                evaluate_sampling_strategy,
                load_code_capacity_problem,
                with_uniform_error_rate,
            )

            _code_paths = default_export_code_paths(_repo_root)
            smoke_problem = with_uniform_error_rate(load_code_capacity_problem(_code_paths["surface13"]), 0.09)
            smoke_cases = enumerate_half_stabilizer_cases(smoke_problem, max_cases=8, shuffle_seed=2)
            smoke_spec = MemorySamplingSpec(
                sampling_family="uniform_interval",
                application_scope="stabilizer_support",
                selection_mode="k_draw_practical",
                draw_count=2,
                low=0.0,
                high=0.2,
            )
            smoke_metrics = evaluate_sampling_strategy(
                problem=smoke_problem,
                cases=smoke_cases,
                sampling_spec=smoke_spec,
                max_iter=10,
                alpha=1.0,
                logical_weight_penalty=4.0,
                convergence_penalty=1.0,
                iteration_penalty=0.05,
                base_seed=13,
            )
            assert smoke_metrics["num_cases"] == len(smoke_cases)
            assert "mean" in bootstrap_confidence_interval([0.0, 1.0, 1.0], bootstrap_samples=64, seed=1)
            """,
            tags=["smoke-test"],
        ),
    ]


def main() -> None:
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityMemoryDistributionHeatmaps.ipynb",
        build_heatmap_notebook(),
    )
    _write_notebook(
        EXAMPLES_DIR / "CodeCapacityHalfStabilizerMemoryDistributionStudy.ipynb",
        build_half_study_notebook(),
    )


if __name__ == "__main__":
    main()
