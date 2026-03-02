"""Tests for dihedral angle computation."""

import math

import torch
from protmetrics.backbone.dihedrals import _dihedral, compute_dihedrals


class TestDihedralPrimitive:
    """Test the _dihedral helper on known 4-atom arrangements."""

    def test_180_degrees(self):
        """Planar trans arrangement → ±180°."""
        p0 = torch.tensor([1.0, 0.0, 0.0])
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 1.0, 0.0])
        p3 = torch.tensor([-1.0, 1.0, 0.0])
        d = _dihedral(p0, p1, p2, p3).item()
        assert abs(abs(d) - 180.0) < 0.1

    def test_minus_90_degrees(self):
        """Known -90° dihedral (IUPAC convention)."""
        p0 = torch.tensor([1.0, 0.0, 0.0])
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 1.0, 0.0])
        p3 = torch.tensor([0.0, 1.0, 1.0])
        d = _dihedral(p0, p1, p2, p3).item()
        assert abs(d - (-90.0)) < 0.1

    def test_90_degrees(self):
        """Known +90° dihedral (IUPAC convention)."""
        p0 = torch.tensor([1.0, 0.0, 0.0])
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 1.0, 0.0])
        p3 = torch.tensor([0.0, 1.0, -1.0])
        d = _dihedral(p0, p1, p2, p3).item()
        assert abs(d - 90.0) < 0.1

    def test_0_degrees(self):
        """Cis arrangement → 0°."""
        p0 = torch.tensor([1.0, 0.0, 0.0])
        p1 = torch.tensor([0.0, 0.0, 0.0])
        p2 = torch.tensor([0.0, 1.0, 0.0])
        p3 = torch.tensor([1.0, 1.0, 0.0])
        d = _dihedral(p0, p1, p2, p3).item()
        assert abs(d) < 0.1


class TestComputeDihedrals:
    """Test compute_dihedrals on backbone coordinates."""

    def test_nan_boundaries(self):
        """First phi and last psi must be NaN."""
        # 3-residue chain (9 atoms)
        coords = torch.randn(1, 9, 3)
        phi, psi, omega = compute_dihedrals(coords)
        assert torch.isnan(phi[0, 0]), "phi[0] should be NaN"
        assert torch.isnan(psi[0, -1]), "psi[-1] should be NaN"
        # Middle values should NOT be NaN
        assert not torch.isnan(phi[0, 1]), "phi[1] should be defined"
        assert not torch.isnan(psi[0, 0]), "psi[0] should be defined"

    def test_batch_consistency(self):
        """Same structure repeated → identical dihedrals."""
        coords = torch.randn(1, 15, 3)  # 5 residues
        batch = coords.expand(4, -1, -1).clone()
        phi, psi, omega = compute_dihedrals(batch)
        for i in range(1, 4):
            assert torch.allclose(phi[0], phi[i], equal_nan=True)
            assert torch.allclose(psi[0], psi[i], equal_nan=True)

    def test_mask_invalidates_neighbors(self):
        """Masking residue i should NaN phi[i], phi[i+1], psi[i], psi[i-1]."""
        coords = torch.randn(1, 15, 3)  # 5 residues
        mask = torch.ones(1, 5)
        mask[0, 2] = 0  # mask middle residue

        phi, psi, omega = compute_dihedrals(coords, backbone_mask=mask)
        # phi[2] needs residues 1&2 → NaN because 2 is masked
        assert torch.isnan(phi[0, 2])
        # phi[3] needs residues 2&3 → NaN because 2 is masked
        assert torch.isnan(phi[0, 3])
        # psi[1] needs residues 1&2 → NaN because 2 is masked
        assert torch.isnan(psi[0, 1])
        # psi[2] needs residues 2&3 → NaN because 2 is masked
        assert torch.isnan(psi[0, 2])

    def test_single_residue_all_nan(self):
        """L=1: both phi and psi should be NaN."""
        coords = torch.randn(1, 3, 3)
        phi, psi, omega = compute_dihedrals(coords)
        assert torch.isnan(phi[0, 0])
        assert torch.isnan(psi[0, 0])

    def test_two_residues(self):
        """L=2: phi[0]=NaN, phi[1]=defined, psi[0]=defined, psi[1]=NaN."""
        coords = torch.randn(1, 6, 3)
        phi, psi, omega = compute_dihedrals(coords)
        assert torch.isnan(phi[0, 0])
        assert not torch.isnan(phi[0, 1])
        assert not torch.isnan(psi[0, 0])
        assert torch.isnan(psi[0, 1])
