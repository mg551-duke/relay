# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = [
    REPO_ROOT / "examples" / "CodeCapacityEdgeWeightHalfStabilizerHeatmaps.ipynb",
    REPO_ROOT / "examples" / "CodeCapacityEdgeWeightPerformanceHeatmaps.ipynb",
    REPO_ROOT / "examples" / "CodeCapacityEdgeWeightTrace.ipynb",
]


def test_edge_weight_notebook_smoke_cells_execute():
    for notebook_path in NOTEBOOKS:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        smoke_cell_count = 0
        executed_cells: set[int] = set()
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            tags = cell.get("metadata", {}).get("tags", [])
            if "smoke-test" not in tags:
                continue
            smoke_cell_count += 1
            for prerequisite_index, prerequisite_cell in enumerate(notebook["cells"][: cell_index + 1]):
                if prerequisite_index in executed_cells or prerequisite_cell.get("cell_type") != "code":
                    continue
                exec(
                    compile("".join(prerequisite_cell["source"]), f"{notebook_path}:{prerequisite_index}", "exec"),
                    namespace,
                )
                executed_cells.add(prerequisite_index)
        assert smoke_cell_count >= 1, f"{notebook_path.name} is missing a smoke-test cell."
