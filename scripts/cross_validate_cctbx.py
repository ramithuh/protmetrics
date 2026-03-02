#!/usr/bin/env python3
"""Cross-validate protmetrics against CCTBX on real PDB structures.

Requires: cctbx-base (conda install -c conda-forge cctbx-base)
          protmetrics (pip install -e .)
          torch

Usage:
    python scripts/cross_validate_cctbx.py [PDB_ID ...]
    python scripts/cross_validate_cctbx.py 1ubq 2igd  # specific structures
    python scripts/cross_validate_cctbx.py              # defaults to 1ubq
"""

import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# CCTBX imports
# ---------------------------------------------------------------------------
from iotbx import pdb as iotbx_pdb
from mmtbx.validation import ramalyze

# ---------------------------------------------------------------------------
# protmetrics imports
# ---------------------------------------------------------------------------
from protmetrics import compute_structural_metrics
from protmetrics.backbone.dihedrals import compute_dihedrals
from protmetrics.backbone.ramachandran import ramachandran_metrics

# Standard alphabetical AA ordering (0-indexed, matches RF2AA / OpenFold)
_AA3_TO_IDX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


def download_pdb(pdb_id: str) -> str:
    """Download PDB file, return path."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    tmp = Path(tempfile.mkdtemp()) / f"{pdb_id}.pdb"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    return str(tmp)


def extract_backbone(pdb_path: str) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
    """Extract N, CA, C coords and residue types from first chain.

    Returns:
        coords: [1, L*3, 3] tensor
        aa_seq: [1, L] integer amino acid indices (0-indexed)
        resnames: list of 3-letter residue names
        n_residues: number of residues
    """
    pdb_inp = iotbx_pdb.input(file_name=pdb_path)
    hierarchy = pdb_inp.construct_hierarchy()

    model = hierarchy.models()[0]
    chain = model.chains()[0]

    coords_list = []
    resnames = []
    aa_indices = []

    for rg in chain.residue_groups():
        ag = rg.atom_groups()[0]
        atoms = {a.name.strip(): a for a in ag.atoms()}
        resname = ag.resname.strip()

        if "N" in atoms and "CA" in atoms and "C" in atoms and resname in _AA3_TO_IDX:
            coords_list.append(list(atoms["N"].xyz))
            coords_list.append(list(atoms["CA"].xyz))
            coords_list.append(list(atoms["C"].xyz))
            resnames.append(resname)
            aa_indices.append(_AA3_TO_IDX[resname])

    coords = torch.tensor(coords_list, dtype=torch.float32).unsqueeze(0)
    aa_seq = torch.tensor(aa_indices, dtype=torch.long).unsqueeze(0)
    return coords, aa_seq, resnames, len(resnames)


def _first_model_hierarchy(hierarchy):
    """Truncate a PDB hierarchy to its first model (fixes NMR multi-model files)."""
    if len(hierarchy.models()) > 1:
        sel = hierarchy.atom_selection_cache().selection(f"model_id {hierarchy.models()[0].id}")
        hierarchy = hierarchy.select(sel)
    return hierarchy


def run_cctbx_ramalyze(pdb_path: str) -> dict:
    """Run CCTBX ramalyze and return summary stats."""
    pdb_inp = iotbx_pdb.input(file_name=pdb_path)
    hierarchy = _first_model_hierarchy(pdb_inp.construct_hierarchy())

    rama = ramalyze.ramalyze(pdb_hierarchy=hierarchy, outliers_only=False)

    per_residue = []
    for result in rama.results:
        per_residue.append({
            "resname": result.resname.strip(),
            "resseq": result.resseq.strip(),
            "phi": result.phi,
            "psi": result.psi,
            "score": result.score,
            "rama_type": result.rama_type,  # 2=FAVORED, 1=ALLOWED, 0=OUTLIER
        })

    return {
        "n_total": rama.n_total,
        "n_favored": rama.n_favored,
        "n_allowed": rama.n_allowed,
        "n_outliers": rama.n_outliers,
        "favored_frac": rama.n_favored / rama.n_total if rama.n_total > 0 else 0,
        "allowed_frac": rama.n_allowed / rama.n_total if rama.n_total > 0 else 0,
        "outlier_frac": rama.n_outliers / rama.n_total if rama.n_total > 0 else 0,
        "per_residue": per_residue,
    }


def compare(pdb_id: str):
    """Compare protmetrics vs CCTBX on a single structure."""
    print(f"\n{'='*60}")
    print(f"  {pdb_id.upper()}")
    print(f"{'='*60}")

    pdb_path = download_pdb(pdb_id)
    coords, aa_seq, resnames, n_res = extract_backbone(pdb_path)
    print(f"Extracted {n_res} residues from chain A")

    # --- CCTBX ---
    print("\nRunning CCTBX ramalyze...")
    cctbx_results = run_cctbx_ramalyze(pdb_path)

    # --- protmetrics ---
    print("Running protmetrics...")
    phi, psi, omega = compute_dihedrals(coords)
    pm_rama = ramachandran_metrics(phi, psi, aa_seq=aa_seq, omega=omega)
    pm_all = compute_structural_metrics(coords, aa_seq=aa_seq)

    # --- Compare Ramachandran ---
    print(f"\n{'Ramachandran':>25s}  {'CCTBX':>10s}  {'protmetrics':>12s}  {'diff':>8s}")
    print("-" * 60)

    for label, cctbx_key, pm_key in [
        ("Favored", "favored_frac", "rama/favored_frac"),
        ("Allowed", "allowed_frac", "rama/allowed_frac"),
        ("Outlier", "outlier_frac", "rama/outlier_frac"),
    ]:
        cv = cctbx_results[cctbx_key]
        pv = pm_rama[pm_key].item()
        diff = pv - cv
        print(f"{label:>25s}  {cv:10.4f}  {pv:12.4f}  {diff:+8.4f}")

    # --- Per-residue phi/psi comparison ---
    print(f"\n{'Per-residue phi/psi (first 10)':}")
    print(f"{'res':>6s}  {'CCTBX_phi':>10s}  {'PM_phi':>10s}  {'d_phi':>8s}  {'CCTBX_psi':>10s}  {'PM_psi':>10s}  {'d_psi':>8s}")
    print("-" * 72)

    cctbx_per_res = cctbx_results["per_residue"]
    phi_vals = phi[0]
    psi_vals = psi[0]

    n_show = min(10, len(cctbx_per_res))
    phi_diffs = []
    psi_diffs = []

    for i, cr in enumerate(cctbx_per_res[:n_show]):
        pm_idx = i + 1  # CCTBX results start at residue 2
        if pm_idx >= len(phi_vals):
            break

        c_phi, c_psi = cr["phi"], cr["psi"]
        p_phi = phi_vals[pm_idx].item()
        p_psi = psi_vals[pm_idx].item()

        d_phi = p_phi - c_phi if not torch.isnan(phi_vals[pm_idx]) else float("nan")
        d_psi = p_psi - c_psi if not torch.isnan(psi_vals[pm_idx]) else float("nan")

        if not torch.isnan(torch.tensor(d_phi)):
            phi_diffs.append(abs(d_phi))
        if not torch.isnan(torch.tensor(d_psi)):
            psi_diffs.append(abs(d_psi))

        print(
            f"{cr['resname']:>3s}{cr['resseq']:>3s}  "
            f"{c_phi:10.2f}  {p_phi:10.2f}  {d_phi:+8.2f}  "
            f"{c_psi:10.2f}  {p_psi:10.2f}  {d_psi:+8.2f}"
        )

    if phi_diffs:
        print(f"\nMean |phi diff|: {sum(phi_diffs)/len(phi_diffs):.3f}°")
    if psi_diffs:
        print(f"Mean |psi diff|: {sum(psi_diffs)/len(psi_diffs):.3f}°")

    # --- Bond/angle summary ---
    print(f"\nBond/angle metrics (protmetrics):")
    for k in sorted(pm_all.keys()):
        if k.startswith("bond/") or k.startswith("angle/"):
            print(f"  {k:30s} = {pm_all[k].item():.4f}")

    return cctbx_results, pm_rama, pm_all


def main():
    pdb_ids = sys.argv[1:] if len(sys.argv) > 1 else ["1ubq"]
    for pdb_id in pdb_ids:
        compare(pdb_id)


if __name__ == "__main__":
    main()
