"""Tests for backbone clash detection."""

import math

import torch
from protmetrics.backbone.clashes import backbone_clash_score, reconstruct_O
from protmetrics.backbone.constants import (
    CLASH_OVERLAP_THRESHOLD,
    IDEAL_C_O,
    IDEAL_CA_C,
    IDEAL_CA_C_O,
    IDEAL_C_N,
    IDEAL_N_CA,
    VDW_RADIUS_C,
    VDW_RADIUS_N,
)


def test_no_clashes_ideal_geometry(ideal_coords):
    """Well-separated ideal chain should have zero clashes."""
    m = backbone_clash_score(ideal_coords)
    assert m["clash/count"].item() == 0.0
    assert m["clash/score"].item() == 0.0


def test_obvious_clash():
    """Two residues with overlapping atoms from non-sequential residues."""
    # Build 3 residues: residue 0 and 2 are placed on top of each other
    # so their N atoms clash (they are not sequential neighbors).
    coords = torch.zeros(1, 9, 3)
    # Residue 0: near origin
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])          # N
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])    # CA
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C
    # Residue 1: extended along x
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 3] = torch.tensor([x1, 0.0, 0.0])
    coords[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    # Residue 2: folded back to overlap with residue 0
    coords[0, 6] = torch.tensor([0.1, 0.0, 0.0])           # N near residue 0 N
    coords[0, 7] = torch.tensor([0.1 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 8] = torch.tensor([0.1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    m = backbone_clash_score(coords)
    assert m["clash/count"].item() > 0


def test_sequential_exclusion():
    """Peptide bond C_i-N_{i+1} should NOT be counted as a clash even at short distance."""
    # Two residues with extremely short C-N peptide bond (0.5 A)
    coords = torch.zeros(1, 6, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])          # N
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])    # CA
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C
    # Next residue: N only 0.5 A from C (would be huge overlap if not excluded)
    x = IDEAL_N_CA + IDEAL_CA_C + 0.5
    coords[0, 3] = torch.tensor([x, 0.0, 0.0])            # N
    coords[0, 4] = torch.tensor([x + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    m = backbone_clash_score(coords)
    # All atom pairs between sequential residues up to 1-4 are excluded.
    # Remaining pairs (N_0-CA_1, N_0-C_1, CA_0-C_1) are 1-5 or 1-6
    # connected and at reasonable distances in a linear chain.
    assert m["clash/count"].item() == 0.0


def test_masking_excludes_padded(ideal_coords):
    """Masked (padded) residues should not contribute clashes."""
    B, L3, _ = ideal_coords.shape
    L = L3 // 3
    mask = torch.ones(B, L)
    mask[:, -3:] = 0  # mask last 3 residues

    # Corrupt masked positions to create overlaps
    corrupted = ideal_coords.clone()
    corrupted[:, -9:, :] = 0.0  # pile all masked atoms at origin

    m = backbone_clash_score(corrupted, backbone_mask=mask)
    assert m["clash/count"].item() == 0.0


def test_batch_dimension():
    """B=4 batch should work."""
    coords = torch.zeros(1, 9, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 3] = torch.tensor([x1, 0.0, 0.0])
    coords[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    x2 = x1 + IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 6] = torch.tensor([x2, 0.0, 0.0])
    coords[0, 7] = torch.tensor([x2 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 8] = torch.tensor([x2 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    batch = coords.expand(4, -1, -1).clone()
    m = backbone_clash_score(batch)
    assert m["clash/score"].item() == 0.0
    assert m["clash/total_atoms"].item() == 9.0


def test_single_residue():
    """L=1: no inter-residue pairs, so zero clashes."""
    coords = torch.zeros(1, 3, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    m = backbone_clash_score(coords)
    assert m["clash/count"].item() == 0.0
    assert m["clash/total_atoms"].item() == 3.0


def test_reconstruct_O_geometry():
    """Reconstructed O should have correct bond length and angle."""
    # Build 2-residue linear chain along x
    coords = torch.zeros(1, 6, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])          # N0
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])    # CA0
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C0
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 3] = torch.tensor([x1, 0.0, 0.0])            # N1
    coords[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    result = reconstruct_O(coords)
    assert result.shape == (1, 2, 4, 3)

    # Check C-O bond length for residue 0
    C0 = result[0, 0, 2]  # C atom
    O0 = result[0, 0, 3]  # O atom
    d_CO = (C0 - O0).norm().item()
    assert abs(d_CO - IDEAL_C_O) < 0.01, f"C-O distance {d_CO:.4f} != {IDEAL_C_O}"

    # Check CA-C-O angle for residue 0
    CA0 = result[0, 0, 1]
    v1 = CA0 - C0
    v2 = O0 - C0
    cos_angle = torch.dot(v1, v2) / (v1.norm() * v2.norm())
    angle_deg = math.degrees(torch.acos(cos_angle.clamp(-1, 1)).item())
    assert abs(angle_deg - IDEAL_CA_C_O) < 1.0, f"CA-C-O angle {angle_deg:.1f} != {IDEAL_CA_C_O}"


def test_include_O_flag():
    """include_O=True should add O atoms and detect clashes with them."""
    # With O, there are 4 atoms per residue instead of 3
    coords = torch.zeros(1, 6, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 3] = torch.tensor([x1, 0.0, 0.0])
    coords[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    m = backbone_clash_score(coords, include_O=True)
    assert m["clash/total_atoms"].item() == 8.0  # 2 residues * 4 atoms


def test_4d_input():
    """Passing [B, L, 4, 3] directly should work."""
    coords_3 = torch.zeros(1, 6, 3)
    coords_3[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords_3[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords_3[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords_3[0, 3] = torch.tensor([x1, 0.0, 0.0])
    coords_3[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])
    coords_3[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    coords_4d = reconstruct_O(coords_3)  # [1, 2, 4, 3]
    m = backbone_clash_score(coords_4d)
    assert m["clash/total_atoms"].item() == 8.0
    assert m["clash/count"].item() == 0.0


def test_clashscore_units():
    """Verify clash/score = count / total_atoms * 1000."""
    # 3 residues, force residue 0 and 2 to overlap
    coords = torch.zeros(1, 9, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 3] = torch.tensor([x1, 0.0, 0.0])
    coords[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    # Residue 2 on top of residue 0
    coords[0, 6] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 7] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 8] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    m = backbone_clash_score(coords)
    count = m["clash/count"].item()
    total = m["clash/total_atoms"].item()
    score = m["clash/score"].item()
    assert count > 0
    assert abs(score - count / total * 1000) < 0.01


def test_helix_no_clashes(helix_coords):
    """An ideal helix should have no clashes."""
    m = backbone_clash_score(helix_coords)
    assert m["clash/count"].item() == 0.0


def test_backward_compat_keys(ideal_coords):
    """All expected keys must be present."""
    m = backbone_clash_score(ideal_coords)
    for key in ["clash/score", "clash/count", "clash/total_atoms"]:
        assert key in m, f"Missing key: {key}"
