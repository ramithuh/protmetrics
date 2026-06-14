# Sidechain metrics & losses — practitioner guide

`protmetrics.allatom` provides **evaluation metrics** and **training losses** for
(partial) protein sidechains, validated bit-exact against CCTBX/MolProbity on
canonically-labeled inputs. Pure PyTorch, batched, GPU-ready, single dependency
(`torch`). No CCTBX, geostd, or subprocess needed at runtime — the reference data
(rotamer grids, monomer-library ideals, CDL CA–CB grid) ships in the package.

---

## 1. Input contract

Everything takes **atom14** tensors. Convert atom37 → atom14 at the protmetrics
boundary with the standard fixed AF2/OpenFold mapping (caller's responsibility).

| arg | shape | meaning |
|---|---|---|
| `atom14_coords` | `[B, L, 14, 3]` | slots `0–3 = N, CA, C, O`; slot `4 = CB`; `5+` sidechain |
| `atom14_mask` | `[B, L, 14]` | per-slot occupancy (1 real, 0 missing/phantom) |
| `aa_seq` | `[B, L]` | residue type, **0-indexed alphabetical** (`ALA=0 … VAL=19`, `GLY=7`, `PRO=14`) |
| `has_sidechain` | `[B, L]` | 1 for pocket residues whose sidechain was generated (scoring gate) |

Two things to get right:
- **aa_seq indexing is alphabetical-3-letter, 0-based.** Wrong indexing silently
  corrupts rotamer/geometry dispatch. (`RESTYPE_TO_IDX` in `constants.py` is the source of truth.)
- **The CA–CB term and the angle metric need the full sequential backbone**
  (to read each residue's φ/ψ and its neighbors). Pass the whole chain's backbone,
  not just the isolated pocket sidechains. Atoms missing → that residue falls back
  gracefully; whole chain absent → CA–CB falls back to a static mean.

---

## 2. Evaluation metrics

One call gets everything (reference-free), plus native-reference recovery if you
pass ground truth:

```python
import protmetrics.allatom as A

metrics = A.evaluate_sidechains(
    atom14_coords, aa_seq,
    atom14_mask=atom14_mask,
    has_sidechain=has_sidechain,        # pocket-only scoring
    native_atom14_coords=native,        # optional → adds chi/recovery section
    native_atom14_mask=native_mask,     # optional
    per_restype=True,                   # optional → per-residue-type breakdown
)
# metrics is a flat {str: scalar tensor} dict → self.log_dict(metrics)
```

Run it under `torch.no_grad()` in your validation step.

### Keys produced

**Reference-free (de novo):**

| key | what it measures | CCTBX faithfulness |
|---|---|---|
| `cbeta/dev_mean`, `cbeta/dev_max` (Å) | Cβ placement vs backbone frame | exact (`cbetadev`) |
| `cbeta/outlier_frac` | fraction > 0.25 Å | exact |
| `rotamer/favored_frac`, `rotamer/allowed_frac`, `rotamer/outlier_frac` | χ in populated rotamer wells | exact (`rotalyze`) |
| `sidechain/bond_rmsz`, `sidechain/bond_outlier_frac`, `sidechain/bond_dev_mean_A` | bond-length geometry | exact |
| `sidechain/angle_rmsz`, `sidechain/angle_outlier_frac`, `sidechain/angle_dev_mean_deg` | bond-angle geometry | exact |
| `sidechain/chirality_outlier_frac` | inverted stereocenters (ILE/THR Cβ, LEU Cγ) | exact |
| `sidechain/planarity_rmsd`, `sidechain/planarity_outlier_frac` | buckled rings / amide / guanidinium / carboxylate | exact |
| `sidechain/<RES>/bond_rmsz`, `sidechain/<RES>/angle_rmsz` | per-type learnability map (`per_restype=True`) | exact |

**Native-reference (conditioned / memorization eval) — added when `native_atom14_coords` given:**

| key | what it measures |
|---|---|
| `chi/rmsd_deg`, `chi/mae_deg` | χ error vs native (symmetry-aware: ASP χ2, GLU χ3, PHE/TYR χ2) |
| `rotamer/recovery_frac` | fraction of residues within tolerance (default 40°) |

### Interpreting them

- **RMSZ ≈ 1.0 is "as good as a crystal structure"**, not 0. It's deviation in σ
  units; real proteins scatter ~1σ. Bond RMSZ ~1, angle RMSZ ~1.5–2 are normal.
- **outlier_frac** uses a 4σ cutoff (bonds/angles), 0.25 Å (Cβ), grid thresholds (rotamer).
- These are **pocket-only** when you pass `has_sidechain`. Label them as such when
  logging — e.g. `rotamer/outlier_frac` is a valid per-residue rate but not directly
  comparable to all-residue literature numbers (La-Proteina etc.).

---

## 3. Training losses

Differentiable, GPU, batched. Same arg pattern as the metrics, but return a
**scalar** and are **not** wrapped in `no_grad`. Apply to the model's predicted
**clean** coords (x0̂), never the noised input.

```python
loss_geom = A.sidechain_bond_loss(
    a14_hat, atom14_mask, aa_seq,
    residue_mask=has_sidechain,
    mode="flat_bottom", tol=4.0,     # σ-space; defaults
)
loss = loss_fm + w_geom * loss_geom  # weighting/scheduling is yours
```

Three functions: `sidechain_bond_loss`, `sidechain_angle_loss`,
`sidechain_geometry_loss` (= bond + `angle_weight`·angle).

> The bundled `sidechain_geometry_loss` takes **separate** `bond_tol` (default 4)
> and `angle_tol` (default 6), because bonds want a tight tolerance and angles a
> looser one (see the per-term notes below). For full independent control (e.g.
> different `mode` per term), call the two primitives and add them yourself.

**Modes** (penalty shape on the per-restraint deviation `dev = value − ideal`, `z = dev/σ`):
- `mse` (alias `harmonic_unweighted`) — raw `dev²` (Å² / deg²), no σ weighting. Smooth,
  dense, gentle near ideal — in practice the **most training-friendly** (empirically
  the σ-weighted modes can destabilize because `1/σ` makes stiff-bond gradients hot/uneven).
- `harmonic` — `z²`; the CCTBX restraint energy (= σ-weighted MSE / χ²). Drives RMSZ → 0,
  i.e. *over*-rigid, and σ-weighting is hot. Use only with a small weight.
- `flat_bottom` — `relu(|z| − tol)²`; σ-weighted with a dead zone inside `tol` σ. Preserves
  the natural spread but the threshold + σ-weighting give a sparse/spiky gradient.
- `berhu` — reversed Huber on **raw** `dev`: linear `|dev|` for `|dev| ≤ c`, quadratic
  beyond (escalates on gross violations). Caller passes `c` in raw units (Å for bonds,
  deg for angles). Note `c` acts like σ — a *small* `c` makes the tail hot, so size it
  generously (only genuine outliers should reach the quadratic part).

> Empirically (flow-matching pocket training): **`mse` trains cleanly; `flat_bottom`
> can degrade it** — because your coordinate/FM loss already plays CCTBX's "data term"
> role, a smooth dense restraint (mse) is well-counterbalanced, whereas the σ-weighted
> thresholded form is hot and sparse. Start with `mse`.

**Practical notes:**
- Apply on **x0̂** (predicted clean structure). Penalizing noised `xt` is meaningless.
- **Bonds are the cleanest signal** — exactly 0 on good geometry at `tol=4`. Start here.
- **Angles** carry a small real baseline at `tol=4` (real proteins have a few >4σ
  angles); use `tol=6–8` if you want it near-zero on good data.
- **Weight small** — it's a regularizer, not the objective.
- Uses **static ideals only** (no CDL/links) — irrelevant for a pull-to-ideal loss.
- Cost: ~3–7 ms GPU fwd+bwd at B=32–128 (launch-bound, ~1–5% of a training step).

---

## 3b. Compound covalent energy (one object: val metric + loss)

`sidechain_geometry_energy` is a single differentiable scalar over **all four**
covalent families — bonds + angles + chirality + planarity (no clash). It serves
both uses:

```python
# VAL METRIC — reproduces CCTBX's covalent restraint target (minus clash)
with torch.no_grad():
    E, comps = A.sidechain_geometry_energy(
        a14, mask, aa, residue_mask=pocket,
        mode="harmonic", reduction="sum", exact=True, return_components=True)
    # E ≈ CCTBX target; comps = {bond, angle, chirality, planarity}

# LOSS — same object, training-friendly mode, faster path
loss_geom = A.sidechain_geometry_energy(
    a14_hat, mask, aa, residue_mask=pocket,
    mode="mse", reduction="mean", exact=False)
```

- `mode="harmonic", reduction="sum", exact=True` **== CCTBX restraint energy** —
  validated to **<1% of CCTBX's target** on 1ubq/1crn/2igd/1rbp (angles & chirality
  exact, bonds <1%, planarity weighted-correct). Uses CDL CA–CB + proline/disulfide
  links + monomer-library chirality volumes + per-atom planarity esd (ARG CD).
- `exact=False` drops CDL + link terms (static ideals) — for the loss path where
  exactness is moot. ~10 ms GPU fwd+bwd (B=32) vs ~44 ms for `exact=True`.
- `weights={...}` rescales per family; `return_components=True` gives the per-family
  breakdown for logging.
- Note: only `mode="harmonic"` makes the four families dimensionless (z²) and additive.
  Other modes mix units (Å² vs deg² vs Å³…), so set `weights` if you combine them.

## 4. Discipline & gotchas

- **Do NOT make a rotamer-favorability loss.** It's a Goodhart magnet (pulls χ to
  the nearest dense well regardless of correctness). Keep rotamer eval-only. The
  geometry losses (bond/angle/chirality/planarity) are physical and loss-safe.
- **Symmetric-atom nomenclature.** CCTBX canonicalizes ARG NH1/NH2, ASP OD1/OD2,
  GLU OE1/OE2, PHE/TYR ring atoms before scoring. Generated atom14 is canonical by
  construction, so this never bites you — but if you feed *experimental* PDBs with
  non-standard labeling, a few symmetric-atom angles will differ.
- **Cis-peptides / chain termini** fall back to a static CA–CB mean (no φ/ψ).
  Negligible and rare; irrelevant to most pockets.
- **Clash is not implemented** (stub). The backbone-only clash elsewhere in the
  package is *not* a MolProbity clashscore — don't report it as one.

---

## 5. Minimal example

```python
import torch, protmetrics.allatom as A

B, L = 4, 128
atom14 = model_output_atom14            # [B,L,14,3], canonical slots
mask   = atom14_occupancy               # [B,L,14]
aa     = aa_seq_0indexed_alphabetical   # [B,L]
pocket = has_sidechain                  # [B,L]

# eval (validation_step, under no_grad)
with torch.no_grad():
    m = A.evaluate_sidechains(atom14, aa, atom14_mask=mask,
                              has_sidechain=pocket, per_restype=True)
    # m["sidechain/bond_rmsz"], m["rotamer/outlier_frac"], m["cbeta/dev_mean"], ...

# loss (training_step, on predicted clean coords)
lg = A.sidechain_bond_loss(atom14, mask, aa, residue_mask=pocket)
total = loss_fm + 0.5 * lg
```
