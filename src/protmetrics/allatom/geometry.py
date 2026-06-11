"""Sidechain covalent geometry RMSZ — lowest priority.

STATUS: stub. Extends backbone bond/angle RMSZ to sidechain bonds/angles.
Inherits the open faithfulness gap (fixed Engh & Huber sigma vs CCTBX's
conformation-dependent / all-atom restraints — see the backbone bond/angle
issue), and generators that emit ideal covalent geometry pass it trivially.
Implement only if sidechain covalent geometry is a suspected failure mode.
"""

from torch import Tensor


def sidechain_geometry_metrics(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
) -> dict[str, Tensor]:
    raise NotImplementedError("Lower priority; see module docstring.")
