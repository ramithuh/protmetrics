# Structural Validation Metrics for Protein Geometry

A reference document covering the theory, tools, and conventions behind protein structure validation metrics — with focus on what protmetrics computes, how it relates to established tools (MolProbity, CCTBX/Phenix), and what the ML protein generation community reports.

---

## Table of Contents

1. [Overview: The Validation Ecosystem](#1-overview-the-validation-ecosystem)
2. [Ramachandran Validation](#2-ramachandran-validation)
3. [Bond Length and Bond Angle Validation (RMSZ)](#3-bond-length-and-bond-angle-validation-rmsz)
4. [Clash Score and Van der Waals Contacts](#4-clash-score-and-van-der-waals-contacts)
5. [MolProbity: The All-Atom Validation Standard](#5-molprobity-the-all-atom-validation-standard)
6. [CCTBX and Phenix](#6-cctbx-and-phenix)
7. [What ML Papers Report](#7-what-ml-papers-report)
8. [protmetrics vs CCTBX: What Aligns and What Doesn't](#8-protmetrics-vs-cctbx-what-aligns-and-what-doesnt)
9. [References](#9-references)

---

## 1. Overview: The Validation Ecosystem

Protein structure validation aims to identify errors in atomic models by checking them against known physical and statistical expectations. The key tools form a layered ecosystem:

- **MolProbity** (Richardson Lab, Duke) — the gold-standard all-atom validation service. Introduces explicit hydrogen atoms and uses probe contact analysis for clash detection. Combines clashscore, Ramachandran, and rotamer analysis into an overall MolProbity score.
- **CCTBX** (Lawrence Berkeley National Lab) — open-source computational crystallography toolbox. Its `mmtbx.validation` module re-implements MolProbity's algorithms in Python. The same code powers both the MolProbity web server and the Phenix refinement suite.
- **Phenix** — Python-based Hierarchical ENvironment for Integrated Xtallography. Built on top of CCTBX; provides automated refinement, model building, and validation tools.
- **Older tools** — PROCHECK (Laskowski et al., 1993) and WHAT_CHECK used united-atom models and rectangular Ramachandran boundaries. Largely superseded by MolProbity's data-driven, all-atom approach.

### Bird's-Eye View of the Tool Landscape

```
CCTBX  (open-source, BSD-3-Clause-LBNL, C++ with Python bindings)
│
├── Core libraries
│     ├── cctbx       (unit cells, space groups, scattering)
│     ├── iotbx       (I/O: PDB, CIF, MTZ file handling)
│     └── scitbx      (arrays, minimizers, FFT)
│
└── mmtbx  (macromolecular toolbox — where all validation lives)
      │
      ├── mmtbx.validation
      │     ├── ramalyze          Ramachandran phi/psi
      │     ├── rotalyze          Sidechain rotamer outliers
      │     ├── cbetadev          C-beta deviation
      │     ├── clashscore        Steric clashes (needs Probe + Reduce)
      │     ├── omegalyze         Cis/trans peptide bonds
      │     ├── cablam            CA-trace backbone (important for cryo-EM)
      │     ├── restraints        Bond/angle geometry vs. ideal
      │     └── molprobity/       Unified class — runs all of the above
      │
      ├── mmtbx.reduce            "reduce2" — hydrogen placement (new, in CCTBX)
      └── mmtbx.probe             "probe2" — contact analysis (new, in CCTBX)


Phenix  (free academic / commercial consortium; built ON TOP of CCTBX)
└── Bundles CCTBX + GUI + refinement + model building + phasing
    └── CLI wrappers: phenix.molprobity, phenix.ramalyze, phenix.clashscore, etc.


Richardson Lab standalone tools  (open-source, github.com/rlabduke)
├── reduce       Original C program for hydrogen placement
├── probe        Original C program for contact dot analysis
└── MolProbity   Web server (Perl/PHP) — depends on CCTBX + reduce + probe


For pip/conda users:
  conda install -c conda-forge cctbx-base
    → gets mmtbx.validation.ramalyze, rotalyze, cbetadev out of the box
    → clashscore needs reduce/probe binaries on PATH (or use reduce2/probe2)
    → bond/angle RMSZ needs a restraints manager (non-trivial setup)
```

Key points:
- **CCTBX is the foundation** — Phenix depends on CCTBX, not vice versa.
- The MolProbity validation algorithms were re-implemented in Python within `mmtbx.validation`, so CCTBX, Phenix, and the MolProbity web server all share the same core code.
- **reduce2/probe2** are newer Python/C++ reimplementations that live inside CCTBX, replacing the need for the standalone C binaries. reduce2 is slower but more accurate and handles CIF files natively.

---

## 2. Ramachandran Validation

### 2.1 Background

The Ramachandran plot (Ramachandran, Ramakrishnan & Sasisekharan, 1963) maps the backbone dihedral angles phi and psi for each residue:

- **phi**: C(i-1) — N(i) — CA(i) — C(i). Rotation around the N–CA bond.
- **psi**: N(i) — CA(i) — C(i) — N(i+1). Rotation around the CA–C bond.
- **omega**: CA(i) — C(i) — N(i+1) — CA(i+1). Usually ~180° (trans peptide bond); constrained by partial double-bond character.

Certain phi/psi combinations are disallowed because backbone atoms and the CB group would be forced closer than their Van der Waals radii (steric clash). The allowed regions correspond to recognizable secondary structures:

| Region | Approx. (phi, psi) | Description |
|--------|-------------------|-------------|
| Alpha-helix (alpha-R) | (-60°, -45°) | Right-handed alpha helix; most populated region |
| Beta-sheet | (-120°, +130°) | Extended beta-strand |
| Left-handed helix (alpha-L) | (+60°, +45°) | Rare for L-amino acids; common for glycine |
| Polyproline II | (-75°, +145°) | Found in collagen and disordered regions |

### 2.2 Residue-Type Specific Maps

Modern validation uses six separate Ramachandran distributions because different residue types have distinct conformational preferences:

1. **General** — most amino acids (excluding the categories below)
2. **Glycine** — no side chain, symmetric and expanded allowed region
3. **Trans-proline** — phi constrained near -63° by the pyrrolidine ring
4. **Cis-proline** — distinct distribution; omega near 0°
5. **Pre-proline** — residues preceding proline; unique "zeta" region at (~+130°, +80°)
6. **Ile/Val** — beta-branched at CB; slightly more restricted than general

### 2.3 Favored / Allowed / Outlier Boundaries

Boundaries are defined empirically from high-resolution structures (the **Top8000 dataset**: 7,957 protein chains, ~1 million quality-filtered residues at better than 2.0 Å, mainchain B-factor < 30 for Ramachandran contour generation):

| Classification | Contour | Meaning |
|---------------|---------|---------|
| **Favored** | Top 98% of reference data | Normal, expected conformation |
| **Allowed** | Top 99.95% (between 98% and 99.95%) | Acceptable but less common |
| **Outlier** | Outside 99.95% contour (1 in 2000 residues) | Likely error; requires investigation |

**Expected values for good structures:** >98% favored, <0.2% outliers.

### 2.4 Implementation Differences

- **MolProbity / CCTBX** — uses the Top8000-derived contour maps with six residue categories (identical codebase; `mmtbx.validation.ramalyze`). Per-residue probability looked up from 2D grid tables (`rama8000_tables.h`).
- **PROCHECK** — older rectangular region boundaries ("core", "allowed", "generously allowed"). Does not separate Ile/Val or cis/trans-Pro.
- **protmetrics** — uses bilinear interpolation on discretized Ramachandran contour maps. Closely matches CCTBX for well-folded structures (mean |diff| ~0.1% for Rama favored on generated structures). Systematic bias on ground truth (~6%) due to different boundary interpolation.

---

## 3. Bond Length and Bond Angle Validation (RMSZ)

### 3.1 Engh & Huber Parameters

The reference standard for ideal covalent geometry comes from Engh & Huber (1991, updated 2001), who derived mean bond lengths, bond angles, and their standard deviations from small-molecule crystal structures in the **Cambridge Structural Database (CSD)**. The logic: covalent geometry is a local property that should be identical whether a bond appears in a small molecule or a protein.

**Key backbone bond lengths:**

| Bond | Ideal (Å) | Sigma (Å) |
|------|-----------|-----------|
| N–CA (general) | 1.458 | 0.019 |
| N–CA (Gly) | 1.451 | 0.016 |
| N–CA (Pro) | 1.466 | 0.015 |
| CA–C (general) | 1.525 | 0.021 |
| CA–C (Gly) | 1.516 | 0.018 |
| C–N (peptide) | 1.329 | 0.014 |
| C–N (Pro peptide) | 1.341 | 0.016 |
| C=O | 1.231 | 0.020 |

### 3.2 Z-scores and RMSZ

For each bond or angle:

```
Z = (observed - ideal) / sigma
```

The aggregate quality measure is the Root Mean Square Z-score:

```
RMSZ = sqrt( mean(Z²) ) = sqrt( sum(Zi²) / N )
```

**Interpretation:**
- RMSZ = 1.0 means the deviations match the expected spread — typical for well-refined high-resolution structures
- RMSZ << 1 (e.g., 0.2–0.5) means geometry is tightly ideal — typical for heavily restrained low-resolution structures or computationally generated structures
- RMSZ > 1 indicates potential over-fitting or systematic problems

### 3.3 Violation and Outlier Thresholds

| Category | Threshold | Expected by chance (Gaussian) |
|----------|-----------|-------------------------------|
| **Violation** | \|Z\| > 2 | ~5% |
| **Outlier** | \|Z\| > 4 | ~0.006% |
| **Severe outlier** (wwPDB) | \|Z\| > 5 | ~0.00006% |

### 3.4 Circular Relationship with Refinement

Important caveat: the same Engh & Huber parameters are used as **restraints** during crystallographic refinement. The refinement target function literally minimizes Z-scores. This means:
- Heavily restrained structures will trivially have small RMSZ
- The diagnostic power comes from **individual outliers** that persist despite restraints
- ML-generated structures that use ideal bond parameters during generation will also trivially have RMSZ near 0 — this does not prove structural quality

### 3.5 Backbone-Only vs All-Atom

wwPDB validation reports compute RMSZ over all covalent bonds/angles (backbone + side chains). Backbone-only RMSZ (as protmetrics computes) covers fewer bond types per residue (~3 bonds, ~3 angles) and will generally be lower than all-atom RMSZ, since backbone geometry tends to be better-defined in both experimental and generated structures.

---

## 4. Clash Score and Van der Waals Contacts

### 4.1 Van der Waals Radii

The standard radii for protein structure validation are the **Bondi radii** (Bondi, 1964), derived from crystal packing data:

| Element | Radius (Å) |
|---------|-----------|
| H | 1.20 |
| C | 1.70 |
| N | 1.55 |
| O | 1.52 |
| S | 1.80 |

Note: Probe uses 1.0 Å for polar hydrogen (at nuclear positions), tuned to be self-consistent with its 0.4 Å clash threshold.

### 4.2 How Probe Works

**Probe** (Word et al., 1999) performs all-atom contact analysis by rolling a small sphere (0.25 Å radius) over the VdW surface of each atom. Where the probe simultaneously contacts the surface of another non-bonded atom, a contact dot is placed.

**Contact classification:**

| Category | Overlap Range | Visualization |
|----------|--------------|---------------|
| Wide contact | gap 0.25–0.5 Å | Blue dots |
| Close contact | gap 0.0–0.25 Å | Green dots |
| Small overlap | overlap 0.0–0.4 Å | Yellow dots |
| **Bad overlap (clash)** | **overlap ≥ 0.4 Å** | **Hot pink spikes** |
| H-bond | donor-acceptor overlap | Pale green dots |

A **clash** is defined as:
```
overlap = (VdW_radius_A + VdW_radius_B) - distance_AB
clash if overlap >= 0.4 Å
```

For hydrogen bond donor-acceptor pairs, higher overlap thresholds are used: Probe tolerates up to 0.6 Å overlap for regular H-bonds and 0.8 Å for charged H-bonds (vs. the standard 0.4 Å for non-donor-acceptor pairs), since H-bonded atoms naturally approach more closely.

### 4.3 Bonded Pair Exclusions

Atoms connected by few covalent bonds are excluded from clash detection:

| Connectivity | Example | Treatment |
|-------------|---------|-----------|
| **1-2** (directly bonded) | N–CA | Always excluded |
| **1-3** (one bond apart) | N···C through CA | Always excluded |
| **1-4** (two bonds apart) | N_i···N_{i+1} through CA–C–N | Excluded or reduced threshold |

Without these exclusions, every covalent bond would register as a massive "clash" since bond lengths (1.2–1.5 Å) are far shorter than VdW contact distances (3.0–3.4 Å).

### 4.4 Clashscore

```
clashscore = (number_of_clashing_pairs / total_atoms) × 1000
```

| Clashscore | Quality |
|-----------|---------|
| ~0 | Suspicious for experimental structures (possible overfitting); expected for computational models |
| < 5 | Good (well-refined experimental structures; ~2.7 is average for best high-res data) |
| < 10 | Acceptable |
| > 10 | Problematic |

### 4.5 The Role of Hydrogen Atoms

Hydrogens constitute ~50% of all atoms in a protein and participate in ~75% of all contacts. Adding explicit hydrogens (via **Reduce**) before running Probe dramatically increases clash detection sensitivity. Reduce also corrects Asn/Gln/His side-chain flips (~20% are incorrectly oriented in PDB structures).

### 4.6 Backbone-Only Clash Detection

No major published validation tool uses backbone-only clash detection. When only N, CA, C (and optionally O) are considered:

- **Side-chain clashes are invisible** — most real clashes involve side-chain atoms
- **Hydrogen-mediated clashes are missed** — H atoms are involved in ~75% of contacts
- **Sensitivity is dramatically reduced** — backbone atoms are well-separated in properly folded proteins; backbone-only clashes only occur in severely distorted structures
- Backbone-only clashscore is a **conservative lower bound** on the true clashscore

This is why protmetrics' backbone clash score diverges significantly from CCTBX's all-atom clashscore: they measure fundamentally different things.

---

## 5. MolProbity: The All-Atom Validation Standard

### 5.1 Overview

MolProbity (Richardson Lab, Duke University) is the gold-standard structure validation tool. Its key innovation is **all-atom contact analysis** — adding explicit hydrogen atoms and using Probe to detect steric clashes with far greater sensitivity than older united-atom approaches.

### 5.2 The MolProbity Pipeline

1. **Reduce** adds explicit hydrogen atoms to the structure
2. **Probe** computes all-atom dot-surface contacts
3. Ramachandran, rotamer, and C-beta analyses are run
4. Results are combined into the overall MolProbity score

### 5.3 Key Metrics

| Metric | Description | Good values |
|--------|-------------|-------------|
| **Clashscore** | Steric overlaps ≥ 0.4 Å per 1000 atoms | < 5 |
| **Rama favored** | % residues in favored phi/psi region | > 98% |
| **Rama outliers** | % residues outside allowed phi/psi region | ~ 0% |
| **Rotamer outliers** | % side chains in outlier chi conformations | < 1% |
| **C-beta deviations** | CB position vs ideal from backbone geometry | < 0.25 Å |
| **Overall MolProbity score** | Composite (see formula below) | < 2.0 |

### 5.4 Overall MolProbity Score Formula

From Chen et al. (2010):

```
MolProbity_score = 0.426 × ln(1 + clashscore)
                 + 0.33 × ln(1 + max(0, pctRotOut - 1))
                 + 0.25 × ln(1 + max(0, 100 - pctRamaFavored - 2))
                 + 0.5
```

The score is calibrated to the **crystallographic resolution scale**: a score of 2.0 means the model's quality statistics are average for 2.0 Å resolution structures. Lower is better.

### 5.5 Advantages over Older Tools

| Feature | PROCHECK / WHAT_CHECK | MolProbity |
|---------|----------------------|------------|
| Hydrogen atoms | Not used (united-atom radii) | Explicit all-atom including H |
| Clash detection | Simple center-to-center bump checks | Probe dot-surface analysis; distinguishes H-bonds from clashes |
| Asn/Gln/His flips | Not corrected | Automatically detected by Reduce |
| Reference data | Older, smaller datasets | Top8000 (~1M quality-filtered residues) |
| Composite score | G-factor (log-odds) | Resolution-calibrated MolProbity score |

---

## 6. CCTBX and Phenix

### 6.1 Architecture

- **CCTBX** (Grosse-Kunstleve et al., 2002) is the open-source core library: ISO C++ classes with Python bindings for crystallographic algorithms.
- **Phenix** (Adams et al., 2010) is built on top of CCTBX, adding automated refinement, model building, and higher-level tools.
- The **mmtbx** (macromolecular toolbox) module within CCTBX contains all validation code.

### 6.2 Validation Modules

MolProbity's validation algorithms were **re-implemented in Python within CCTBX**, creating a single shared codebase:

| Module | Function | External dependencies |
|--------|----------|----------------------|
| `mmtbx.validation.ramalyze` | Ramachandran validation | None (lookup tables built-in) |
| `mmtbx.validation.rotalyze` | Rotamer validation | None (lookup tables built-in) |
| `mmtbx.validation.cbetadev` | C-beta deviation | None |
| `mmtbx.validation.clashscore` | All-atom clash analysis | **Requires Reduce + Probe binaries** |

### 6.3 cctbx-base vs Full Phenix

- **`pip install cctbx-base`**: Core library. Ramalyze, rotalyze, cbetadev work out of the box. Clashscore needs Reduce/Probe installed separately.
- **Full Phenix suite**: Everything pre-bundled including Reduce, Probe, and all higher-level tools. Academic license required.

---

## 7. What ML Papers Report

### 7.1 The Standard Metrics Triad

The de facto standard for backbone protein generation papers is:

| Metric | Definition | Status |
|--------|-----------|--------|
| **Designability** (scTM / scRMSD) | ProteinMPNN designs sequence → AF2 predicts structure → compare to original | Universal |
| **Diversity** | Pairwise TM-score or cluster count among designable samples | Very common |
| **Novelty** | Max TM-score to nearest PDB entry | Very common |

### 7.2 Structural Quality Metrics in Major Papers

| Paper | Rama | Clash | Bond/Angle | Designability | Notes |
|-------|------|-------|------------|--------------|-------|
| **AlphaFold2** (Jumper 2021) | — | — | Violation loss during training | — | Uses AMBER relaxation post-processing |
| **RFdiffusion** (Watson 2023) | — | — | — | scRMSD < 2Å, pLDDT > 70 | Backbone-only generator |
| **FrameDiff** (Yim 2023) | — | — | — | scRMSD < 2Å | Backbone-only |
| **Chroma** (Ingraham 2023) | — | — | — | scTM > 0.5 | All-atom but focused on designability |
| **FoldFlow** (Bose 2024) | Qualitative plots | — | — | scRMSD < 2Å | Backbone-only |
| **La-Proteina** (Geffner 2025) | **Yes** | **Yes** | **Yes** | Yes | All-atom; full MolProbity suite |
| **RFdiffusion2** (2025) | — | — | — | Motif AA-RMSD < 1.5Å + Chai-1 | Focus on enzyme design |

### 7.3 Key Observations

- **Backbone-only generators** (RFdiffusion, FrameDiff, FoldFlow) validate through the designability pipeline, not MolProbity. No side chains → no meaningful all-atom clashscore.
- **All-atom generators** can and should report MolProbity metrics. **La-Proteina set the precedent** by reporting full MolProbity (clash score, Rama outliers, covalent bond outliers, rotamer analysis).
- **Bond/angle RMSZ** is rarely reported in generation papers but is standard in crystallographic validation (wwPDB reports).
- ML-generated structures with ideal bond parameters will trivially have RMSZ ≈ 0 — the meaningful backbone metric for generators is Ramachandran.

### 7.4 Typical Values

| Metric | Experimental (good) | AlphaFold2 | Generated (backbone) |
|--------|-------------------|-----------|---------------------|
| Rama favored | > 98% | ~98% | 50–80% (varies by training) |
| Rama outliers | ~0% | ~0.2% | 5–25% (varies) |
| Clashscore | < 5 | ~2 (after relaxation) | Not directly comparable |
| Bond RMSZ | 0.5–1.0 | Near ideal | Near 0 (if using ideal params) |

---

## 8. protmetrics vs CCTBX: What Aligns and What Doesn't

### Well-Aligned: Ramachandran

protmetrics Ramachandran validation closely matches CCTBX:
- **ep239/ep319 (trained models)**: mean |diff| ≈ 0.001 (~0.1%) for both favored and outlier fractions
- **Ground truth**: slightly larger diff (~0.06) due to different boundary interpolation methods (bilinear vs CCTBX's native lookup)
- **Ranking between epochs is preserved perfectly** — protmetrics correctly identifies which checkpoints are better/worse

### Not Comparable: Clash Score

protmetrics and CCTBX clash scores measure fundamentally different things:

| | protmetrics | CCTBX/MolProbity |
|---|-----------|-----------------|
| **Atoms considered** | Backbone only (N, CA, C, optionally O) | All atoms + explicit hydrogens |
| **Radii** | Bondi (N=1.55, C=1.70, O=1.52) | Bondi + Probe-specific H radii (1.0 Å polar H) |
| **Clash threshold** | ≥ 0.4 Å VdW overlap | ≥ 0.4 Å with H-bond allowance |
| **GT clashscore** | ~12 (backbone packing contacts) | ~0.2 (no real clashes) |
| **Use case** | Relative training signal | Paper-reportable MolProbity metric |

### protmetrics-Only: Bond/Angle RMSZ

Bond/angle RMSZ is backbone-only in protmetrics. CCTBX computes all-atom RMSZ. For backbone-only PDBs (the typical output of backbone generators), protmetrics RMSZ is a meaningful and practical metric. No direct CCTBX comparison is needed since both would compute the same backbone geometry — the divergence only appears when side chains are present.

### Recommendation

- **For training**: Use protmetrics Rama + bond/angle RMSZ + backbone clash as loss signals. They are internally consistent, differentiable-friendly, and track improvement.
- **For paper reporting**: Run CCTBX/MolProbity for authoritative Ramachandran and clashscore numbers. Don't call protmetrics clash score "MolProbity clashscore."

---

## 9. References

### Core Validation Methods

- Ramachandran GN, Ramakrishnan C, Sasisekharan V (1963). "Stereochemistry of polypeptide chain configurations." *J. Mol. Biol.* 7, 95–99.
- Engh RA, Huber R (1991). "Accurate bond and angle parameters for X-ray protein structure refinement." *Acta Cryst.* A47, 392–400.
- Engh RA, Huber R (2001). "Structure quality and target parameters." *International Tables for Crystallography*, Vol. F, Ch. 18.3.
- Bondi A (1964). "Van der Waals Volumes and Radii." *J. Phys. Chem.* 68, 441–451.
- Lovell SC et al. (2003). "Structure validation by Calpha geometry: phi, psi and Cbeta deviation." *Proteins* 50(3), 437–450.

### MolProbity and Probe

- Word JM et al. (1999). "Visualizing and quantifying molecular goodness-of-fit: small-probe contact dots with explicit hydrogen atoms." *J. Mol. Biol.* 285(4), 1711–1733.
- Word JM et al. (1999). "Asparagine and glutamine: using hydrogen atom contacts in the choice of side-chain amide orientation." *J. Mol. Biol.* 285(4), 1735–1747.
- Davis IW et al. (2007). "MolProbity: all-atom contacts and structure validation for proteins and nucleic acids." *Nucleic Acids Res.* 35, W375–W383.
- Chen VB et al. (2010). "MolProbity: all-atom structure validation for macromolecular crystallography." *Acta Cryst.* D66, 12–21.
- Williams CJ et al. (2018). "MolProbity: More and better reference data for improved all-atom structure validation." *Protein Sci.* 27(1), 293–315.

### CCTBX and Phenix

- Grosse-Kunstleve RW et al. (2002). "The Computational Crystallography Toolbox: crystallographic algorithms in a reusable software framework." *J. Appl. Cryst.* 35, 126–136.
- Adams PD et al. (2010). "PHENIX: a comprehensive Python-based system for macromolecular structure solution." *Acta Cryst.* D66, 213–221.

### VdW Radii

- Rowland RS, Taylor R (1996). "Intermolecular Nonbonded Contact Distances in Organic Crystal Structures." *J. Phys. Chem.* 100, 7384–7391.
- Tsai J et al. (1999). "The Packing Density in Proteins: Standard Radii and Volumes." *J. Mol. Biol.* 290, 253–266.

### Conformation-Dependent Geometry

- Tronrud DE, Berkholz DS, Karplus PA (2010). "Using a conformation-dependent stereochemical library improves crystallographic refinement of proteins." *Acta Cryst.* D66, 834–842.
- Moriarty NW et al. (2014). "Conformation-dependent backbone geometry restraints set a new standard for protein crystallographic refinement." *FEBS J.* 281, 4061–4071.

### ML Protein Generation

- Jumper J et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature* 596, 583–589.
- Watson JL et al. (2023). "De novo design of protein structure and function with RFdiffusion." *Nature* 620, 1089–1100.
- Yim J et al. (2023). "SE(3) diffusion model with application to protein backbone generation." *ICML*.
- Ingraham JB et al. (2023). "Illuminating protein space with a programmable generative model." *Nature* 623, 1070–1078.
- Bose AJ et al. (2024). "SE(3)-Stochastic Flow Matching for Protein Backbone Generation." *ICLR*.
- Geffner T et al. (2025). "La-Proteina: Generative Protein Structure-Sequence Modeling." *arXiv:2507.09466*.
