"""Tests for Ramachandran classification."""

import torch
from protmetrics.ramachandran import _load_tables, ramachandran_metrics


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
