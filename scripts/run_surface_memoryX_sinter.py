#!/usr/bin/env python3
"""Run the native rotated surface-code memory-X Sinter benchmark."""

from __future__ import annotations

from pathlib import Path

from run_stim_sinter_benchmarks import build_parser, run_from_args


def main() -> None:
    parser = build_parser(
        __doc__,
        default_circuits_dir=Path("tests/testdata/surface"),
        default_decoder_config=Path("configs/decoder_specs_surface.json"),
        default_outdir=Path("results_surface_x_sinter"),
        default_filename_contains="rotated_surface_code_memory_X",
        default_max_shots=2_000,
        default_save_name="surface_memoryX",
    )
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
