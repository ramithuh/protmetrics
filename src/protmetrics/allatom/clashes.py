"""Heavy-atom clash PROXY for pocket sidechains.

STATUS: stub. This is where a clash metric is finally meaningful (sidechains
exist), but with a hard caveat: no hydrogens are present, so this is a
HEAVY-ATOM proxy that under-reports vs MolProbity clashscore (~75% of real
clashes involve H). Never expose this as `clashscore` — same discipline as the
backbone-clash caveat. Suggested key: `clash/heavy_atom_proxy`.

Design:
  - All-atom14 pairwise overlap with Bondi radii and the 0.4 A threshold.
  - Correct bonded-pair exclusions need a per-residue-type heavy-atom bond
    topology table (1-2/1-3/1-4 within a residue + the peptide linkage),
    extending backbone.clashes' sequential-exclusion approach to intra-residue
    sidechain bonds. Build that topology table next.
  - Most useful framing for pockets: pocket sidechain atoms vs
    (backbone + other pocket sidechain atoms).
"""

from torch import Tensor


def heavy_atom_clash_proxy(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
    has_sidechain: Tensor | None = None,
) -> dict[str, Tensor]:
    raise NotImplementedError(
        "Build the per-residue heavy-atom bond topology table for 1-2/1-3/1-4 "
        "exclusions, then reuse backbone.clashes' overlap logic over atom14."
    )
