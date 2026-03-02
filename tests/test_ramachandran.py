"""Tests for Ramachandran classification."""

import torch
from protmetrics.backbone.ramachandran import (
    _angle_to_bin,
    _angle_to_fractional_bin,
    _bilinear_lookup,
    _load_tables,
    ramachandran_metrics,
)


class TestTableLoading:
    """Verify the extracted lookup tables are sane."""

    def test_all_tables_present(self):
        tables = _load_tables()
        expected = {"general", "glycine", "cis_proline", "trans_proline", "pre_proline", "ile_val"}
        assert set(tables.keys()) == expected

    def test_table_shapes(self):
        tables = _load_tables()
        for name, t in tables.items():
            assert t.shape == (180, 180), f"{name} has wrong shape: {t.shape}"

    def test_tables_non_negative(self):
        tables = _load_tables()
        for name, t in tables.items():
            assert (t >= 0).all(), f"{name} has negative values"

    def test_general_table_has_content(self):
        """General table should have many nonzero bins."""
        tables = _load_tables()
        nonzero = (tables["general"] > 0).sum().item()
        assert nonzero > 10000, f"Expected >10k nonzero bins, got {nonzero}"


class TestRamachandranMetrics:
    """Test classification of known phi/psi regions."""

    def test_alpha_helix_favored(self):
        """Alpha helix (phi=-57, psi=-47) should be favored."""
        B, L = 1, 20
        phi = torch.full((B, L), -57.0)
        psi = torch.full((B, L), -47.0)
        # First phi and last psi undefined
        phi[:, 0] = float("nan")
        psi[:, -1] = float("nan")

        m = ramachandran_metrics(phi, psi)
        assert m["rama/favored_frac"].item() > 0.95

    def test_beta_sheet_favored(self):
        """Beta sheet (phi=-120, psi=120) should be in favored region."""
        B, L = 1, 20
        phi = torch.full((B, L), -120.0)
        psi = torch.full((B, L), 120.0)
        phi[:, 0] = float("nan")
        psi[:, -1] = float("nan")

        m = ramachandran_metrics(phi, psi)
        # Beta sheet is well within favored for General table
        assert m["rama/favored_frac"].item() > 0.9

    def test_outlier_region(self):
        """phi=0, psi=0 is a known outlier for the General table."""
        B, L = 1, 20
        phi = torch.full((B, L), 0.0)
        psi = torch.full((B, L), 0.0)
        phi[:, 0] = float("nan")
        psi[:, -1] = float("nan")

        m = ramachandran_metrics(phi, psi)
        assert m["rama/outlier_frac"].item() > 0.9

    def test_all_nan_returns_nan(self):
        """All NaN input → NaN output."""
        phi = torch.full((1, 5), float("nan"))
        psi = torch.full((1, 5), float("nan"))
        m = ramachandran_metrics(phi, psi)
        assert torch.isnan(m["rama/favored_frac"])

    def test_fractions_sum_to_one(self):
        """favored + allowed + outlier should sum to 1.0."""
        phi = torch.full((1, 50), -57.0)
        psi = torch.full((1, 50), -47.0)
        phi[:, 0] = float("nan")
        psi[:, -1] = float("nan")

        m = ramachandran_metrics(phi, psi)
        total = m["rama/favored_frac"] + m["rama/allowed_frac"] + m["rama/outlier_frac"]
        assert abs(total.item() - 1.0) < 1e-5


class TestBilinearInterpolation:
    """Test bilinear lookup and fractional bin conversion."""

    def test_bin_center_matches_table(self):
        """At exact bin center, bilinear lookup should equal direct table value."""
        tables = _load_tables()
        table = tables["general"]
        # Bin 60 center → angle = 60*2 - 179 = -59° (bin center)
        # At exact integer bin, fractional part is 0, so bilinear = table[i, j]
        phi_angle = torch.tensor([-59.0])
        psi_angle = torch.tensor([-47.0])
        phi_frac = _angle_to_fractional_bin(phi_angle)
        psi_frac = _angle_to_fractional_bin(psi_angle)
        # Fractional bin should be integer at bin center
        phi_bin = _angle_to_bin(phi_angle)
        psi_bin = _angle_to_bin(psi_angle)
        direct = table[phi_bin[0], psi_bin[0]].item()
        interp = _bilinear_lookup(table, phi_frac, psi_frac)[0].item()
        assert abs(interp - direct) < 1e-5

    def test_wrapping_at_boundary(self):
        """Angles near ±180° should wrap correctly."""
        tables = _load_tables()
        table = tables["general"]
        # phi = 179° and phi = -179° should give similar results (both near bin 179)
        phi_pos = torch.tensor([179.0])
        phi_neg = torch.tensor([-179.0])
        psi = torch.tensor([0.0])
        frac_pos = _angle_to_fractional_bin(phi_pos)
        frac_neg = _angle_to_fractional_bin(phi_neg)
        psi_frac = _angle_to_fractional_bin(psi)
        val_pos = _bilinear_lookup(table, frac_pos, psi_frac)[0].item()
        val_neg = _bilinear_lookup(table, frac_neg, psi_frac)[0].item()
        # Both should be valid (not NaN or negative)
        assert val_pos >= 0.0
        assert val_neg >= 0.0

    def test_midpoint_interpolation(self):
        """Value at midpoint between two bins should be average of neighbors."""
        tables = _load_tables()
        table = tables["general"]
        # Midpoint between bin 60 and bin 61 for phi, exact bin 80 for psi
        # Bin 60 center angle = 60*2 - 179 = -59, bin 61 center = -57
        # Midpoint = -58
        phi_mid = torch.tensor([-58.0])
        psi_exact = torch.tensor([-19.0])  # bin 80 center = 80*2-179 = -19
        phi_frac = _angle_to_fractional_bin(phi_mid)
        psi_frac = _angle_to_fractional_bin(psi_exact)
        interp = _bilinear_lookup(table, phi_frac, psi_frac)[0].item()
        # Should be average of table[60, 80] and table[61, 80]
        expected = 0.5 * (table[60, 80].item() + table[61, 80].item())
        assert abs(interp - expected) < 1e-5

    def test_fractions_still_sum_to_one(self):
        """Bilinear interpolation should preserve favored+allowed+outlier=1."""
        # Use non-bin-center angles to exercise interpolation
        phi = torch.full((1, 50), -58.3)  # not a bin center
        psi = torch.full((1, 50), -46.7)  # not a bin center
        phi[:, 0] = float("nan")
        psi[:, -1] = float("nan")

        m = ramachandran_metrics(phi, psi)
        total = m["rama/favored_frac"] + m["rama/allowed_frac"] + m["rama/outlier_frac"]
        assert abs(total.item() - 1.0) < 1e-5
