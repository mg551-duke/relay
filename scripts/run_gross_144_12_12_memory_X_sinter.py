#!/usr/bin/env python3
"""Run the native Gross [[144,12,12]] memory-X Sinter benchmark."""

from __future__ import annotations

from pathlib import Path

from run_stim_sinter_benchmarks import build_parser, run_from_args


def main() -> None:
    parser = build_parser(
        __doc__,
        default_circuits_dir=Path("tests/testdata/bicycle_bivariate"),
        default_decoder_config=Path("configs/decoder_specs_gross_144_12_12_memory_X_sinter.json"),
        default_outdir=Path("results_gross_144_12_12_memory_X_sinter"),
        default_filename_contains="circuit=bicycle_bivariate_144_12_12_memory_X",
        default_max_shots=10_000,
        default_save_name="gross_144_12_12_memory_X",
    )
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
