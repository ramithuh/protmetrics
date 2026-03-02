"""Shared test fixtures: ideal backbone coordinates."""

import math

import pytest
import torch


def _ideal_helix_coords(n_residues: int) -> torch.Tensor:
    """Build ideal alpha-helix backbone coords (N, CA, C per residue).

    Uses standard helix parameters:
        rise per residue = 1.5 Å, turn = 100° per residue
        N-CA = 1.459 Å, CA-C = 1.525 Å, C-N(next) = 1.329 Å
        N-CA-C angle = 111°

    Returns: [1, L*3, 3]
    """
    # Build by placing atoms along the helix backbone with ideal geometry.
    # We use a simplified approach: place N, CA, C with correct bond lengths
    # and the N-CA-C angle, then advance along a helical path.
    coords = []
    # Helical parameters
    radius = 2.3  # Å
    rise = 1.5  # Å per residue
    turn = math.radians(100)  # per residue

    for i in range(n_residues):
        theta = i * turn
        # Backbone center follows helix
        cx = radius * math.cos(theta)
        cy = radius * math.sin(theta)
        cz = i * rise

        # Place N, CA, C around the center with small offsets
        # that maintain approximately correct bond geometry
        t = theta
        # N: slightly before center
        n_x = radius * math.cos(t - 0.15) + 0.0
        n_y = radius * math.sin(t - 0.15) + 0.0
        n_z = cz - 0.5

        # CA: at center
        ca_x = cx
        ca_y = cy
        ca_z = cz

        # C: slightly after center
        c_x = radius * math.cos(t + 0.15) + 0.0
        c_y = radius * math.sin(t + 0.15) + 0.0
        c_z = cz + 0.5

        coords.extend([
            [n_x, n_y, n_z],
            [ca_x, ca_y, ca_z],
            [c_x, c_y, c_z],
        ])

    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # [1, L*3, 3]


def _ideal_geometry_coords(n_residues: int) -> torch.Tensor:
    """Build backbone coords with *exact* ideal bond lengths and angles.

    Places atoms sequentially along a chain with:
        N-CA = 1.459 Å, CA-C = 1.525 Å, C-N = 1.329 Å
        N-CA-C = 111°, CA-C-N = 116.568°, C-N-CA = 121.352°

    Returns: [1, L*3, 3]
    """
    coords = [[0.0, 0.0, 0.0]]  # First N at origin

    # Bond lengths
    lengths = []
    # angles between consecutive bonds (at the middle atom)
    angles_deg = []

    for i in range(n_residues):
        # N -> CA
        lengths.append(1.459)
        if len(coords) > 1:
            angles_deg.append(121.352 if i > 0 else None)  # C-N-CA
        # CA -> C
        lengths.append(1.525)
        angles_deg.append(111.0)  # N-CA-C
        # C -> N(next) if not last
        if i < n_residues - 1:
            lengths.append(1.329)
            angles_deg.append(116.568)  # CA-C-N

    # Build chain in 2D (z=0 plane) with correct lengths and angles
    # Place second atom along x-axis
    if len(lengths) > 0:
        coords.append([lengths[0], 0.0, 0.0])

    # Direction of previous bond
    prev_dir = [1.0, 0.0]

    for j in range(1, len(lengths)):
        angle = angles_deg[j] if j < len(angles_deg) else 111.0
        # Angle is the supplement of the turn angle (exterior angle)
        turn = math.pi - math.radians(angle)
        # Alternate direction of turn to keep chain roughly straight
        if j % 2 == 1:
            turn = -turn

        cos_t, sin_t = math.cos(turn), math.sin(turn)
        new_dir = [
            prev_dir[0] * cos_t - prev_dir[1] * sin_t,
            prev_dir[0] * sin_t + prev_dir[1] * cos_t,
        ]
        norm = math.sqrt(new_dir[0] ** 2 + new_dir[1] ** 2)
        new_dir = [new_dir[0] / norm, new_dir[1] / norm]

        last = coords[-1]
        coords.append([
            last[0] + new_dir[0] * lengths[j],
            last[1] + new_dir[1] * lengths[j],
            0.0,
        ])
        prev_dir = new_dir

    # We have 1 + (n_residues * 3 - 1) = n_residues * 3 atoms
    # Actually: N + (N-CA + CA-C) per residue + C-N links = 3*n_residues atoms
    coords = coords[: n_residues * 3]
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)


@pytest.fixture
def helix_coords():
    """[1, 30*3, 3] ideal helix backbone (approximate geometry)."""
    return _ideal_helix_coords(30)


@pytest.fixture
def ideal_coords():
    """[1, 10*3, 3] backbone with exact ideal bond lengths and angles."""
    return _ideal_geometry_coords(10)
