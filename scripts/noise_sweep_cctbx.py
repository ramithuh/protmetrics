#!/usr/bin/env python3
"""Noise sweep with CCTBX cross-validation.

Takes a real PDB, adds increasing noise, runs both CCTBX ramalyze and
protmetrics on each noisy version. Compares rama classification side by side.

Requires: cctbx-base, protmetrics, torch

Usage:
    python scripts/noise_sweep_cctbx.py [PDB_ID]
"""

import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

from iotbx import pdb as iotbx_pdb
from mmtbx.validation import ramalyze

from protmetrics import compute_structural_metrics
from protmetrics.dihedrals import compute_dihedrals
from protmetrics.ramachandran import ramachandran_metrics

_AA3_TO_IDX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


def download_pdb(pdb_id: str) -> str:
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    tmp = Path(tempfile.mkdtemp()) / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(url, tmp)
    return str(tmp)


def extract_backbone(pdb_path: str):
    """Extract backbone coords, aa_seq, and the raw hierarchy for rewriting."""
    pdb_inp = iotbx_pdb.input(file_name=pdb_path)
    hierarchy = pdb_inp.construct_hierarchy()

    model = hierarchy.models()[0]
    chain = model.chains()[0]

    coords_list = []
    aa_indices = []
    # Track atom objects so we can write back noisy coords
    backbone_atoms = []  # list of (N, CA, C) atom objects per residue

    for rg in chain.residue_groups():
        ag = rg.atom_groups()[0]
        atoms = {a.name.strip(): a for a in ag.atoms()}
        resname = ag.resname.strip()

        if "N" in atoms and "CA" in atoms and "C" in atoms and resname in _AA3_TO_IDX:
            coords_list.extend([list(atoms["N"].xyz), list(atoms["CA"].xyz), list(atoms["C"].xyz)])
            aa_indices.append(_AA3_TO_IDX[resname])
            backbone_atoms.append((atoms["N"], atoms["CA"], atoms["C"]))

    coords = torch.tensor(coords_list, dtype=torch.float32).unsqueeze(0)
    aa_seq = torch.tensor(aa_indices, dtype=torch.long).unsqueeze(0)
    return coords, aa_seq, hierarchy, backbone_atoms


def apply_noise_to_hierarchy(hierarchy, backbone_atoms, noisy_coords):
    """Write noisy coords back into the CCTBX hierarchy for ramalyze."""
    import copy
    h = hierarchy.deep_copy()

    # Get the same chain/atoms from the copy
    model = h.models()[0]
    chain = model.chains()[0]

    i = 0
    for rg in chain.residue_groups():
        ag = rg.atom_groups()[0]
        atoms = {a.name.strip(): a for a in ag.atoms()}
        resname = ag.resname.strip()

        if "N" in atoms and "CA" in atoms and "C" in atoms and resname in _AA3_TO_IDX:
            for name_idx, name in enumerate(["N", "CA", "C"]):
                atom_idx = i * 3 + name_idx
                x, y, z = noisy_coords[0, atom_idx].tolist()
                atoms[name].set_xyz((x, y, z))
            i += 1

    return h


def run_cctbx_rama(hierarchy):
    """Run ramalyze on a hierarchy, return favored/allowed/outlier fracs."""
    rama = ramalyze.ramalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    n = rama.n_total
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    return rama.n_favored / n, rama.n_allowed / n, rama.n_outliers / n


def compute_cctbx_dihedrals(hierarchy):
    """Compute phi/psi from CCTBX hierarchy, return per-residue values."""
    rama = ramalyze.ramalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    phis = []
    psis = []
    for r in rama.results:
        phis.append(r.phi)
        psis.append(r.psi)
    return phis, psis


def main():
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1ubq"
    print(f"Downloading {pdb_id.upper()}...")
    pdb_path = download_pdb(pdb_id)
    coords, aa_seq, hierarchy, backbone_atoms = extract_backbone(pdb_path)
    n_res = aa_seq.shape[1]
    print(f"Extracted {n_res} residues\n")

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    # ── Ramachandran comparison ──
    print("═" * 110)
    print("  RAMACHANDRAN")
    print("═" * 110)
    print(f"{'σ (Å)':>8s}  │ {'CCTBX fav':>10s} {'PM fav':>10s} {'Δfav':>8s}  │ "
          f"{'CCTBX out':>10s} {'PM out':>10s} {'Δout':>8s}  │ "
          f"{'CCTBX alw':>10s} {'PM alw':>10s} {'Δalw':>8s}")
    print("─" * 110)

    torch.manual_seed(42)
    all_noisy = {}
    for sigma in noise_levels:
        if sigma == 0.0:
            noisy = coords.clone()
        else:
            noisy = coords + torch.randn_like(coords) * sigma
        all_noisy[sigma] = noisy

        # protmetrics
        phi, psi, omega = compute_dihedrals(noisy)
        pm = ramachandran_metrics(phi, psi, aa_seq=aa_seq, omega=omega)
        pm_fav = pm["rama/favored_frac"].item()
        pm_alw = pm["rama/allowed_frac"].item()
        pm_out = pm["rama/outlier_frac"].item()

        # CCTBX
        noisy_h = apply_noise_to_hierarchy(hierarchy, backbone_atoms, noisy)
        c_fav, c_alw, c_out = run_cctbx_rama(noisy_h)

        print(
            f"{sigma:8.2f}  │ "
            f"{c_fav:10.4f} {pm_fav:10.4f} {pm_fav - c_fav:+8.4f}  │ "
            f"{c_out:10.4f} {pm_out:10.4f} {pm_out - c_out:+8.4f}  │ "
            f"{c_alw:10.4f} {pm_alw:10.4f} {pm_alw - c_alw:+8.4f}"
        )

    # ── Phi/Psi comparison ──
    print(f"\n{'═' * 80}")
    print("  PHI/PSI DIHEDRALS (mean absolute difference vs CCTBX)")
    print("═" * 80)
    print(f"{'σ (Å)':>8s}  │ {'mean |Δphi|':>12s}  {'max |Δphi|':>12s}  │ "
          f"{'mean |Δpsi|':>12s}  {'max |Δpsi|':>12s}  │ {'n_compared':>10s}")
    print("─" * 80)

    for sigma in noise_levels:
        noisy = all_noisy[sigma]
        noisy_h = apply_noise_to_hierarchy(hierarchy, backbone_atoms, noisy)

        phi, psi, omega = compute_dihedrals(noisy)
        phi_vals = phi[0]  # [L]
        psi_vals = psi[0]

        c_phis, c_psis = compute_cctbx_dihedrals(noisy_h)
        if not c_phis:
            print(f"{sigma:8.2f}  │ {'N/A':>12s}  {'N/A':>12s}  │ {'N/A':>12s}  {'N/A':>12s}  │ {'0':>10s}")
            continue

        phi_diffs = []
        psi_diffs = []
        for i in range(len(c_phis)):
            pm_idx = i + 1  # CCTBX starts at residue 2
            if pm_idx >= len(phi_vals):
                break
            if not torch.isnan(phi_vals[pm_idx]):
                phi_diffs.append(abs(phi_vals[pm_idx].item() - c_phis[i]))
            if not torch.isnan(psi_vals[pm_idx]):
                psi_diffs.append(abs(psi_vals[pm_idx].item() - c_psis[i]))

        if phi_diffs:
            print(
                f"{sigma:8.2f}  │ "
                f"{sum(phi_diffs)/len(phi_diffs):12.6f}  {max(phi_diffs):12.6f}  │ "
                f"{sum(psi_diffs)/len(psi_diffs):12.6f}  {max(psi_diffs):12.6f}  │ "
                f"{len(phi_diffs):>10d}"
            )
        else:
            print(f"{sigma:8.2f}  │ {'N/A':>12s}  {'N/A':>12s}  │ {'N/A':>12s}  {'N/A':>12s}  │ {'0':>10s}")

    # ── Bond/Angle metrics (protmetrics only — deterministic geometry) ──
    print(f"\n{'═' * 95}")
    print("  BOND LENGTHS & ANGLES (protmetrics — deterministic from coordinates)")
    print("═" * 95)
    print(f"{'σ (Å)':>8s}  │ {'N-CA':>8s} {'CA-C':>8s} {'C-N':>8s} {'bond_viol':>10s}  │ "
          f"{'N-CA-C':>8s} {'CA-C-N':>8s} {'C-N-CA':>8s} {'ang_viol':>10s}")
    print("─" * 95)

    for sigma in noise_levels:
        noisy = all_noisy[sigma]
        m = compute_structural_metrics(noisy, aa_seq=aa_seq)
        print(
            f"{sigma:8.2f}  │ "
            f"{m['bond/N_CA_mean'].item():8.3f} "
            f"{m['bond/CA_C_mean'].item():8.3f} "
            f"{m['bond/C_N_mean'].item():8.3f} "
            f"{m['bond/violation_frac'].item():10.4f}  │ "
            f"{m['angle/N_CA_C_mean'].item():8.2f} "
            f"{m['angle/CA_C_N_mean'].item():8.2f} "
            f"{m['angle/C_N_CA_mean'].item():8.2f} "
            f"{m['angle/violation_frac'].item():10.4f}"
        )


if __name__ == "__main__":
    main()
