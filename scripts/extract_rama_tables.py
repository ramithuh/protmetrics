#!/usr/bin/env python3
"""Extract rama8000 lookup tables from the CCTBX C++ header into a .pt file.

Downloads rama8000_tables.h from the CCTBX GitHub repo (BSD license) and
parses the 6 linear_table_* arrays into [180, 180] float tensors.

Grid layout (from cctbx_project/mmtbx/validation/ramachandran/convert_from_text.py):
  - 180 x 180 grid, 2-degree bins, phi-major ordering
  - Bin index: (angle + 179) // 2  (maps -179..180 -> 0..179)
  - linear_table[i] = grid[phi_bin * 180 + psi_bin]

Run once; commit the .pt file to the repo.

Source:
    https://github.com/cctbx/cctbx_project/blob/master/mmtbx/validation/ramachandran/rama8000_tables.h

Reference:
    Lovell et al. (2003) "Structure validation by Calpha geometry: phi, psi
    and Cbeta deviation." Proteins 50(3):437-50.
"""

import re
import urllib.request
from pathlib import Path

import torch

HEADER_URL = (
    "https://raw.githubusercontent.com/cctbx/cctbx_project/master/"
    "mmtbx/validation/ramachandran/rama8000_tables.h"
)

# Map C++ array name suffix -> our key name
ARRAY_NAMES = {
    "general": "general",
    "glycine": "glycine",
    "cis_pro": "cis_proline",
    "trans_pro": "trans_proline",
    "pre_pro": "pre_proline",
    "ile_val": "ile_val",
}

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "protmetrics" / "rama_data"


def parse_header(text: str) -> dict[str, torch.Tensor]:
    """Parse linear_table_* arrays from the C++ header into 180x180 tensors."""
    tables = {}

    for suffix, key in ARRAY_NAMES.items():
        # Match: const double linear_table_<suffix>[] = {<numbers>};
        pattern = rf"const\s+double\s+linear_table_{suffix}\s*\[\]\s*=\s*\{{([^}}]+)\}}"
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find linear_table_{suffix} in header")

        values_str = match.group(1)
        values = [float(x.strip()) for x in values_str.split(",") if x.strip()]

        expected = 180 * 180
        if len(values) != expected:
            raise ValueError(
                f"linear_table_{suffix}: expected {expected} values, got {len(values)}"
            )

        grid = torch.tensor(values, dtype=torch.float32).reshape(180, 180)
        tables[key] = grid
        nonzero = (grid > 0).sum().item()
        print(f"  {key}: {nonzero} non-zero bins (of {expected})")

    return tables


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {HEADER_URL}")
    with urllib.request.urlopen(HEADER_URL) as resp:
        text = resp.read().decode("utf-8")
    print(f"  Downloaded {len(text)} bytes")

    tables = parse_header(text)

    out_path = OUTPUT_DIR / "rama_tables.pt"
    torch.save(tables, out_path)
    print(f"\nSaved {len(tables)} tables to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
