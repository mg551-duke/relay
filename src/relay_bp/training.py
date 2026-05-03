# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Native training helpers exposed from the Rust extension."""

from __future__ import annotations

__all__ = [
    "train_relayed_bpgd_bernoulli_memory",
    "train_relayed_bpgd_static_continuous_memory",
    "train_relayed_bpgd_static_discrete_memory",
]

from ._relay_bp import _training  # pylint: disable=E0611

train_relayed_bpgd_bernoulli_memory = (
    _training.train_relayed_bpgd_bernoulli_memory_py
)
train_relayed_bpgd_static_discrete_memory = (
    _training.train_relayed_bpgd_static_discrete_memory_py
)
train_relayed_bpgd_static_continuous_memory = (
    _training.train_relayed_bpgd_static_continuous_memory_py
)
