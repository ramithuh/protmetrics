"""Tests for bond length metrics."""

import torch
from protmetrics.bonds import bond_length_metrics
from protmetrics.constants import IDEAL_CA_C, IDEAL_C_N, IDEAL_N_CA


def test_ideal_geometry_bond_lengths(ideal_coords):
    """Exact ideal geometry should have zero violations and correct means."""
    m = bond_length_metrics(ideal_coords)

    assert abs(m["bond/N_CA_mean"].item() - IDEAL_N_CA) < 0.001
    assert abs(m["bond/CA_C_mean"].item() - IDEAL_CA_C) < 0.001
    assert abs(m["bond/C_N_mean"].item() - IDEAL_C_N) < 0.001
    assert m["bond/violation_frac"].item() == 0.0


def test_perturbed_increases_violations(ideal_coords):
    """Adding noise should produce nonzero violation fraction."""
    noisy = ideal_coords + torch.randn_like(ideal_coords) * 0.5
    m = bond_length_metrics(noisy)
    assert m["bond/violation_frac"].item() > 0.0


def test_masking_excludes_padded(ideal_coords):
    """Masked residues should not affect statistics."""
    B, L3, _ = ideal_coords.shape
    L = L3 // 3
    mask = torch.ones(B, L)
    # Mask out last 5 residues
    mask[:, -5:] = 0

    # Corrupt the masked positions
    corrupted = ideal_coords.clone()
    corrupted[:, -15:, :] = 999.0  # L=10, last 5 residues = last 15 atoms

    m = bond_length_metrics(corrupted, backbone_mask=mask)
    # Stats should only reflect the unmasked (ideal) residues
    assert abs(m["bond/N_CA_mean"].item() - IDEAL_N_CA) < 0.001
    assert m["bond/violation_frac"].item() == 0.0


def test_batch_dimension():
    """Batch of B=4 identical structures should work."""
    # Simple 3-residue chain with ideal N-CA bond length
    coords_single = torch.zeros(1, 9, 3)
    # Residue 0: N at origin, CA at (1.459, 0, 0), C at (2.984, 0, 0)
    coords_single[0, 0] = torch.tensor([0.0, 0.0, 0.0])  # N
    coords_single[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])  # CA
    coords_single[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C
    # Residue 1: continue the chain
    x = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords_single[0, 3] = torch.tensor([x, 0.0, 0.0])  # N
    coords_single[0, 4] = torch.tensor([x + IDEAL_N_CA, 0.0, 0.0])  # CA
    coords_single[0, 5] = torch.tensor([x + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C
    # Residue 2
    x2 = x + IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords_single[0, 6] = torch.tensor([x2, 0.0, 0.0])
    coords_single[0, 7] = torch.tensor([x2 + IDEAL_N_CA, 0.0, 0.0])
    coords_single[0, 8] = torch.tensor([x2 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    batch = coords_single.expand(4, -1, -1).clone()
    m = bond_length_metrics(batch)

    assert abs(m["bond/N_CA_mean"].item() - IDEAL_N_CA) < 0.001
    assert abs(m["bond/CA_C_mean"].item() - IDEAL_CA_C) < 0.001
    assert abs(m["bond/C_N_mean"].item() - IDEAL_C_N) < 0.001


def test_single_residue():
    """L=1: intra-residue bonds exist, no peptide bond."""
    coords = torch.zeros(1, 3, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    m = bond_length_metrics(coords)
    assert abs(m["bond/N_CA_mean"].item() - IDEAL_N_CA) < 0.001
    assert abs(m["bond/CA_C_mean"].item() - IDEAL_CA_C) < 0.001
    # C-N peptide bond: no inter-residue data with L=1 → should still have a key
    # but the violation_frac should only include intra-residue bonds
    assert "bond/violation_frac" in m
