#!/usr/bin/env python3
"""Extract the CA-CB slice of the Conformation-Dependent Library (CDL) into a
dense PyTorch grid, so protmetrics can reproduce CCTBX's phi/psi-dependent CA-CB
bond restraint EXACTLY (nearest-bin, no interpolation — CCTBX's default).

CDL (mmtbx.conformation_dependent_library.cdl_database) keys: <class>_<xpro>,
where class in {Gly, IleVal, Pro, NonPGIV} and xpro marks a following proline.
Each is a 36x36 phi/psi grid (10 deg bins, banker's-rounded) of restraint value
lists; CA-CB ideal is index 20, esd index 21 (verified: -1 for Gly, which has no
Cbeta). Database keys are (phi, psi) order (empirically 69/69 vs CCTBX proxies).

Gly is skipped (no Cbeta). Output: geom_data/cdl_cacb.pt with
  {groups: [...], grid: float[6, 36, 36, 2]}  indexed [group, phi_idx, psi_idx, (ideal,esd)]
  phi_idx = (round_to_ten(phi)+180)//10, same for psi.

Usage:  python scripts/extract_cdl_cacb.py   (needs the cctbx env)
"""
from pathlib import Path
import torch
from mmtbx.conformation_dependent_library import cdl_database as CDL

GROUPS = ["IleVal_nonxpro", "IleVal_xpro", "NonPGIV_nonxpro",
          "NonPGIV_xpro", "Pro_nonxpro", "Pro_xpro"]
BINS = list(range(-180, 180, 10))  # -180..170, 36 bins
CA_CB_IDEAL, CA_CB_ESD = 20, 21


def main():
    grid = torch.full((len(GROUPS), 36, 36, 2), float("nan"))
    for gi, grp in enumerate(GROUPS):
        g = CDL[grp]
        for pi, phi in enumerate(BINS):
            for si, psi in enumerate(BINS):
                entry = g.get((phi, psi))
                if entry is None:
                    continue
                ideal, esd = entry[CA_CB_IDEAL], entry[CA_CB_ESD]
                if ideal == -1:
                    continue
                grid[gi, pi, si, 0] = ideal
                grid[gi, pi, si, 1] = esd
        n_ok = int((~torch.isnan(grid[gi, :, :, 0])).sum())
        print(f"  {grp:18s} populated {n_ok}/1296 bins")

    out = Path(__file__).resolve().parent.parent / \
        "src/protmetrics/allatom/geom_data/cdl_cacb.pt"
    torch.save({"groups": GROUPS, "grid": grid}, out)
    print(f"\nSaved -> {out} ({out.stat().st_size/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
