"""Tests for bond angle metrics."""

import math

import torch
from protmetrics.backbone.angles import bond_angle_metrics
from protmetrics.backbone.constants import IDEAL_N_CA_C


def test_ideal_geometry_angles(ideal_coords):
    """Ideal geometry fixture should have N-CA-C angle close to ideal.

    The 2D chain fixture has slight angle drift from alternating turn
    directions, so we use a 2° tolerance rather than exact match.
    The exact-angle test (test_known_angle) verifies the math precisely.
    """
    m = bond_angle_metrics(ideal_coords)
    assert abs(m["angle/N_CA_C_mean"].item() - IDEAL_N_CA_C) < 2.0


def test_known_angle():
    """Three atoms at a known angle should recover that angle exactly."""
    # Place 3 atoms (one residue) with N-CA-C angle = 90 degrees
    # N at (1, 0, 0), CA at origin, C at (0, 1, 0) → angle at CA = 90°
    coords = torch.zeros(1, 3, 3)
    coords[0, 0] = torch.tensor([1.0, 0.0, 0.0])  # N
    coords[0, 1] = torch.tensor([0.0, 0.0, 0.0])  # CA
    coords[0, 2] = torch.tensor([0.0, 1.0, 0.0])  # C
    m = bond_angle_metrics(coords)
    assert abs(m["angle/N_CA_C_mean"].item() - 90.0) < 0.01


def test_masking_excludes_padded(ideal_coords):
    """Masked positions should be excluded from statistics."""
    B, L3, _ = ideal_coords.shape
    L = L3 // 3
    mask = torch.ones(B, L)
    mask[:, -3:] = 0

    corrupted = ideal_coords.clone()
    corrupted[:, -9:, :] = 999.0

    m_clean = bond_angle_metrics(ideal_coords, backbone_mask=torch.ones(B, L))
    m_masked = bond_angle_metrics(corrupted, backbone_mask=mask)

    # The masked version should still show valid angles for the unmasked part
    assert abs(m_masked["angle/N_CA_C_mean"].item() - m_clean["angle/N_CA_C_mean"].item()) < 1.0


def test_violation_counting():
    """Deliberately bad angles should produce violation fraction ≈ 1.0."""
    # Make a 2-residue chain with near-zero angles (atoms collinear)
    coords = torch.zeros(1, 6, 3)
    # Residue 0: all on x-axis → angle ≈ 180° (deviation > 10° from ideal 111°)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])  # N
    coords[0, 1] = torch.tensor([1.0, 0.0, 0.0])  # CA
    coords[0, 2] = torch.tensor([2.0, 0.0, 0.0])  # C
    # Residue 1: also collinear
    coords[0, 3] = torch.tensor([3.0, 0.0, 0.0])  # N
    coords[0, 4] = torch.tensor([4.0, 0.0, 0.0])  # CA
    coords[0, 5] = torch.tensor([5.0, 0.0, 0.0])  # C

    m = bond_angle_metrics(coords)
    # 180° deviates from all ideals (111°, 116.6°, 121.4°) by >> 10°
    assert m["angle/violation_frac"].item() > 0.9


def test_rmsz_ideal_geometry():
    """Single residue with exact ideal N-CA-C angle → RMSZ ≈ 0."""
    angle_rad = math.radians(IDEAL_N_CA_C)
    coords = torch.zeros(1, 3, 3)
    coords[0, 0] = torch.tensor([1.0, 0.0, 0.0])  # N
    coords[0, 1] = torch.tensor([0.0, 0.0, 0.0])  # CA (vertex)
    # C at ideal angle from N-CA direction
    coords[0, 2] = torch.tensor([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    m = bond_angle_metrics(coords)
    # Only intra-residue angle, should be exactly ideal → RMSZ ≈ 0
    assert m["angle/rmsz"].item() < 0.1
    assert m["angle/outlier_frac_4sigma"].item() == 0.0


def test_angle_4sigma_outlier_detection():
    """Collinear atoms (180°) should produce 4-sigma outliers.

    N-CA-C ideal=111°, stddev=2.8° → Z = (180-111)/2.8 ≈ 24.6 → outlier.
    """
    coords = torch.zeros(1, 6, 3)
    coords[0, 0] = torch.tensor([0.0, 0.0, 0.0])
    coords[0, 1] = torch.tensor([1.0, 0.0, 0.0])
    coords[0, 2] = torch.tensor([2.0, 0.0, 0.0])
    coords[0, 3] = torch.tensor([3.0, 0.0, 0.0])
    coords[0, 4] = torch.tensor([4.0, 0.0, 0.0])
    coords[0, 5] = torch.tensor([5.0, 0.0, 0.0])

    m = bond_angle_metrics(coords)
    assert m["angle/outlier_frac_4sigma"].item() > 0.9


def test_backward_compat_keys(ideal_coords):
    """All original metric keys must still be present."""
    m = bond_angle_metrics(ideal_coords)
    for key in [
        "angle/N_CA_C_mean", "angle/N_CA_C_std", "angle/N_CA_C_dev_mean",
        "angle/CA_C_N_mean", "angle/CA_C_N_std", "angle/CA_C_N_dev_mean",
        "angle/C_N_CA_mean", "angle/C_N_CA_std", "angle/C_N_CA_dev_mean",
        "angle/violation_frac",
        "angle/rmsz", "angle/outlier_frac_4sigma",
    ]:
        assert key in m, f"Missing key: {key}"
