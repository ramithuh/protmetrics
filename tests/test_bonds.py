"""Tests for bond length metrics."""

import torch
from protmetrics.backbone.bonds import bond_length_metrics
from protmetrics.backbone.constants import (
    AA_PRO,
    IDEAL_CA_C,
    IDEAL_C_N,
    IDEAL_C_N_PRO,
    IDEAL_N_CA,
    STDDEV_C_N,
    STDDEV_C_N_PRO,
)


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


def test_rmsz_ideal_geometry(ideal_coords):
    """Ideal geometry should have RMSZ close to 0 and no 4-sigma outliers."""
    m = bond_length_metrics(ideal_coords)
    assert m["bond/rmsz"].item() < 0.1
    assert m["bond/outlier_frac_4sigma"].item() == 0.0


def test_proline_cn_zscore():
    """C-N bond to Pro should be scored against Pro ideal, not general ideal."""
    # Build 3-residue chain: ALA-PRO-ALA
    # Residue 0 = ALA (idx 0), Residue 1 = PRO (idx 14), Residue 2 = ALA (idx 0)
    coords = torch.zeros(1, 9, 3)
    # Residue 0
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])          # N
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])    # CA
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C
    # Residue 1 (Pro): C-N bond uses Pro ideal (1.341)
    x1 = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N_PRO
    coords[0, 3] = torch.tensor([x1, 0.0, 0.0])            # N
    coords[0, 4] = torch.tensor([x1 + IDEAL_N_CA, 0.0, 0.0])  # CA
    coords[0, 5] = torch.tensor([x1 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])  # C
    # Residue 2 (Ala): C-N bond uses general ideal (1.329)
    x2 = x1 + IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    coords[0, 6] = torch.tensor([x2, 0.0, 0.0])
    coords[0, 7] = torch.tensor([x2 + IDEAL_N_CA, 0.0, 0.0])
    coords[0, 8] = torch.tensor([x2 + IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])

    aa_seq = torch.tensor([[0, AA_PRO, 0]])  # ALA, PRO, ALA

    m = bond_length_metrics(coords, aa_seq=aa_seq)
    # With correct Pro ideal, RMSZ should be ~0
    assert m["bond/rmsz"].item() < 0.1
    # Without aa_seq, the Pro C-N bond would be scored against 1.329 instead of 1.341
    m_no_aa = bond_length_metrics(coords)
    # The C-N to Pro has length 1.341 vs general ideal 1.329, Z = (1.341-1.329)/0.014 ≈ 0.86
    # So RMSZ should be noticeably higher
    assert m_no_aa["bond/rmsz"].item() > m["bond/rmsz"].item()


def test_bond_4sigma_outlier_detection():
    """Bonds with |Z| > 4 should be counted as outliers."""
    coords = torch.zeros(1, 6, 3)
    # Residue 0: ideal geometry
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([IDEAL_N_CA, 0.0, 0.0])
    coords[0, 2] = torch.tensor([IDEAL_N_CA + IDEAL_CA_C, 0.0, 0.0])
    # Residue 1: drastically wrong N-CA (should trigger 4-sigma outlier)
    x = IDEAL_N_CA + IDEAL_CA_C + IDEAL_C_N
    bad_n_ca = IDEAL_N_CA + 0.5  # 0.5 Å off → Z = 0.5/0.020 = 25
    coords[0, 3] = torch.tensor([x, 0.0, 0.0])
    coords[0, 4] = torch.tensor([x + bad_n_ca, 0.0, 0.0])
    coords[0, 5] = torch.tensor([x + bad_n_ca + IDEAL_CA_C, 0.0, 0.0])

    m = bond_length_metrics(coords)
    assert m["bond/outlier_frac_4sigma"].item() > 0.0


def test_backward_compat_keys(ideal_coords):
    """All original metric keys must still be present."""
    m = bond_length_metrics(ideal_coords)
    for key in [
        "bond/N_CA_mean", "bond/N_CA_std", "bond/N_CA_median", "bond/N_CA_dev_mean",
        "bond/CA_C_mean", "bond/CA_C_std", "bond/CA_C_median", "bond/CA_C_dev_mean",
        "bond/C_N_mean", "bond/C_N_std", "bond/C_N_median", "bond/C_N_dev_mean",
        "bond/violation_frac",
        "bond/rmsz", "bond/outlier_frac_4sigma",
    ]:
        assert key in m, f"Missing key: {key}"
