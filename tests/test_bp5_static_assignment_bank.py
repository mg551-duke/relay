import json
from pathlib import Path

import numpy as np
import pytest

import run_bivariate_bicycle144_12_12_relay_bp5_performance_cluster_Z_basis as bp5


def _write_candidate(
    base: Path,
    seed: int,
    *,
    gamma_value: float,
    validation_metrics: dict[str, float] | None = None,
    params_metrics: dict[str, float] | None = None,
) -> Path:
    seed_dir = base / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    np.save(
        seed_dir / "discrete_best_static_gammas.npy",
        np.full((1, 4), gamma_value, dtype=np.float64),
    )
    if validation_metrics is not None:
        (seed_dir / "validation_metrics.json").write_text(
            json.dumps({"discrete": validation_metrics}),
            encoding="utf-8",
        )
    if params_metrics is not None:
        (seed_dir / "discrete_best_params.json").write_text(
            json.dumps({"metrics": params_metrics}),
            encoding="utf-8",
        )
    return seed_dir


def _metrics(
    logical_failure_rate: float,
    *,
    mean_iterations: float = 10.0,
    convergence_rate: float = 0.5,
) -> dict[str, float]:
    return {
        "logical_failure_rate": logical_failure_rate,
        "mean_iterations": mean_iterations,
        "convergence_rate": convergence_rate,
    }


def test_static_discrete_assignment_bank_selects_top_10_with_validation_priority(
    tmp_path: Path,
) -> None:
    base = tmp_path / "training"
    _write_candidate(
        base,
        0,
        gamma_value=0.0,
        validation_metrics=_metrics(1.0),
        params_metrics=_metrics(0.0),
    )
    for seed in range(1, 10):
        _write_candidate(
            base,
            seed,
            gamma_value=float(seed),
            validation_metrics=_metrics(0.1 + seed * 0.01),
        )
    _write_candidate(
        base,
        10,
        gamma_value=10.0,
        params_metrics=_metrics(0.005),
    )

    bank, metadata = bp5.load_static_discrete_assignment_bank(base, top_k=10)

    assert bank is not None
    assert metadata is not None
    assert bank.shape == (10, 4)
    assert bank[:, 0].tolist() == [
        9.0,
        10.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ]
    selected_seed_dirs = [
        Path(item["seed_dir"]).name for item in metadata["selected_assignments"]
    ]
    assert selected_seed_dirs == [
        "seed_10",
        "seed_1",
        "seed_2",
        "seed_3",
        "seed_4",
        "seed_5",
        "seed_6",
        "seed_7",
        "seed_8",
        "seed_9",
    ]
    assert metadata["selected_assignments"][0]["metrics_source"] == (
        "discrete_best_params.json:metrics"
    )
    assert metadata["selected_assignments"][1]["metrics_source"] == (
        "validation_metrics.json:discrete"
    )


def test_static_discrete_assignment_bank_requires_enough_candidates(
    tmp_path: Path,
) -> None:
    base = tmp_path / "training"
    for seed in range(9):
        _write_candidate(
            base,
            seed,
            gamma_value=float(seed),
            validation_metrics=_metrics(0.1 + seed * 0.01),
        )

    with pytest.raises(
        ValueError, match="Need at least 10 valid static discrete assignments"
    ):
        bp5.load_static_discrete_assignment_bank(base, top_k=10)


def test_static_discrete_assignment_bank_decoder_uses_fixed_damping() -> None:
    bank = np.zeros((10, 4), dtype=np.float64)

    decoders = bp5.build_custom_decoders(static_discrete_assignment_bank=bank)

    sampler = decoders[bp5.STATIC_DISCRETE_TOPK_DAMP_DECODER_NAME]
    decoder = sampler.decoder
    assert decoder.explicit_gammas is bank
    assert decoder.explicit_gammas.shape == (10, 4)
    assert decoder.c_damp == pytest.approx(0.9)
