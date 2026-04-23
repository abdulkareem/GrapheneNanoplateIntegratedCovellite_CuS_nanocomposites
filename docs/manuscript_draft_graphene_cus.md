# Manuscript Draft (Journal-Ready Template)

**Working Title**  
Interfacial Charge-Transfer Engineering in Graphene Nanoplate–Covellite CuS Heterostructures for Selective Pb²⁺/Cd²⁺ Capture: A Reproducible DFT Study with ASE+GPAW (and Optional Quantum ESPRESSO Cross-Validation)

**Article Type**: Original Research Article  
**Target Journals (suggested)**: *Applied Surface Science*, *Journal of Colloid and Interface Science*, *Chemical Engineering Journal*, *ACS Applied Materials & Interfaces*

---

## Abstract
Graphene/covellite (CuS) nanocomposites have shown promising interfacial functionality in energy and environmental applications; however, atomic-level descriptors of ion-capture behavior at the graphene–CuS interface remain underexplored. Here we present a fully reproducible density-functional-theory (DFT) workflow for graphene nanoplate–CuS(001) heterostructures, implemented in ASE+GPAW with optional Quantum ESPRESSO cross-validation. The workflow includes (i) interface construction with lattice-mismatch control, (ii) geometry optimization using plane-wave PBE settings, (iii) electronic-structure analysis via total/projection-resolved density of states and band structure, (iv) adsorption thermodynamics for Pb²⁺ and Cd²⁺ probes, and (v) convergence and metadata logging for publication-quality reproducibility. We further define a high-impact novelty pathway centered on interfacial charge-transfer descriptors (charge-density-difference topology, Bader charge transfer, work-function shifts, and adsorption-energy trends under defect/termination variants). The present manuscript provides methodology, figure/table framework, and reporting standards; numerical results will be inserted after running the released pipeline.

**Keywords**: graphene nanoplatelets; covellite CuS; DFT; adsorption; Pb²⁺; Cd²⁺; GPAW; Quantum ESPRESSO; heterointerface

---

## 1. Introduction
Heavy-metal remediation and interfacial catalytic capture require materials that combine high conductivity, active surface chemistry, and favorable adsorption thermodynamics. Graphene-based composites are strong candidates due to high carrier mobility and tunable surface interactions, while covellite CuS offers chemically active Cu/S terminations and rich interfacial electronic structure.

Prior studies have reported graphene–CuS composite performance in thermoelectric or broader functional settings, indicating strong potential for interface-driven enhancement. However, mechanistic understanding for selective toxic-ion capture (notably Pb²⁺ and Cd²⁺) at graphene/CuS heterointerfaces remains incomplete at first-principles fidelity. This gap is critical for translating composite design into quantitative adsorption/selectivity rules.

In this work, we provide a reproducible DFT framework and manuscript-ready analysis blueprint emphasizing:
1. Interface-specific adsorption and charge-transfer physics (rather than only bulk/composite property trends).
2. Convergence-controlled, reproducible computational reporting.
3. Cross-engine extensibility (GPAW primary, optional QE check).

**Study hypotheses**
- H1: The graphene/CuS interface stabilizes adsorbates more strongly than isolated constituents due to interfacial charge redistribution.
- H2: Pb²⁺ and Cd²⁺ adsorption differ measurably in charge transfer and electronic-structure perturbation, enabling selectivity descriptors.
- H3: Termination/defect engineering can systematically tune adsorption energies and Fermi-level alignment.

---

## 2. Computational Methods

### 2.1 Software and Reproducibility
Calculations are performed with ASE orchestration and GPAW plane-wave DFT as the primary engine. Optional single-point cross-checks can be executed with Quantum ESPRESSO through ASE’s Espresso interface. The complete pipeline is distributed as executable scripts with metadata capture and convergence scans.

### 2.2 Model Construction
- **Graphene nanoplate**: 4×4 (or 5×5 in publication profile) supercell with vacuum ≥ 15 Å.
- **Covellite CuS slab**: hexagonal CuS bulk-derived (001) slab, typically 4–5 layers and vacuum ≥ 15 Å.
- **Composite interface**: CuS slab placed above graphene with initial spacing ~2.5 Å; in-plane mismatch constrained (target ≤ 5–8%, depending on supercell setup).

### 2.3 DFT Settings (Primary GPAW)
- Exchange-correlation: PBE-GGA.
- Basis/mode: Plane waves, cutoff typically 450 eV (quick) and 520 eV (publication profile).
- k-mesh: 3×3×1 (quick) and 5×5×1 (publication profile).
- Occupation smearing: Fermi–Dirac.
- Electronic convergence target: ~1×10⁻⁵ eV.
- Ionic relaxation: BFGS; force threshold ~0.02 eV/Å (quick) and 0.015 eV/Å (publish).

### 2.4 Optional Quantum ESPRESSO Cross-Validation
For users with `pw.x` and validated pseudopotentials, QE single-point calculations can be used to cross-check total energies and trends. Final publication should report pseudopotential library/version, cutoff selection rationale, and matching functional settings across engines.

### 2.5 Energetics Definitions
- **Binding energy**:  
  \(E_{bind}=E_{composite}-(E_{graphene}+E_{CuS})\)
- **Adsorption energy**:  
  \(E_{ads}=E_{total}-(E_{surface}+E_{adsorbate})\)

Negative values indicate exothermic stabilization.

### 2.6 Electronic-Structure and Charge Analyses
- Total DOS and projected DOS (Cu-d and other selected channels).
- Band structure along in-plane high-symmetry path.
- Charge-density difference:  
  \(\Delta\rho=\rho_{composite}-(\rho_{graphene}+\rho_{CuS})\)
- Recommended add-on for final submission: Bader charge analysis and work-function shifts.

### 2.7 Convergence and Uncertainty Reporting
A convergence matrix (k-point × cutoff) is generated for publication profile and exported as CSV+plot. Final manuscript should include uncertainty ranges from convergence sweeps and, where possible, alternative adsorption sites/orientations.

---

## 3. Results and Discussion (Insert After Running)

> Replace placeholders (`[R#]`) with your computed values.

### 3.1 Structural Relaxation and Interface Geometry
Report relaxed interlayer spacing, local bond-length changes near interface atoms, and slab distortions.

- Relaxed graphene–CuS distance: **[R1] Å**
- Maximum local Cu–S reconstruction at interface: **[R2] Å**
- Lattice mismatch applied: **[R3] %**

**Interpretation prompt**: Discuss whether interface compression/stretching correlates with charge transfer and adsorption affinity.

### 3.2 Interface Energetics
- \(E_{graphene}=\) **[R4] eV**
- \(E_{CuS}=\) **[R5] eV**
- \(E_{composite}=\) **[R6] eV**
- \(E_{bind}=\) **[R7] eV**

**Interpretation prompt**: Compare against literature-reported tendencies (if absolute values differ, focus on converged trends and model differences).

### 3.3 Electronic Structure (DOS/PDOS/Bands)
Describe:
1. Fermi-level proximity of Cu-d and S-p states.
2. Any hybridization signatures induced by graphene/CuS coupling.
3. Band dispersion changes versus isolated components.
4. Element-resolved pDOS trends (C vs Cu vs S) from `pdos_elements_C_Cu_S.png`.

**Placeholders**
- Dominant states near \(E_F\): **[R8]**
- Effective band-gap trend (if meaningful for slab model): **[R9]**
- DOS perturbation under adsorption: **[R10]**

### 3.4 Adsorption Thermodynamics and Selectivity
For Pb²⁺ and Cd²⁺ (multiple sites if possible):
- \(E_{ads}^{Pb}=\) **[R11] eV**
- \(E_{ads}^{Cd}=\) **[R12] eV**
- Selectivity metric \(\Delta E = E_{ads}^{Pb} - E_{ads}^{Cd}=\) **[R13] eV**

**Interpretation prompt**: Link selectivity to charge transfer, coordination environment, and DOS changes.

### 3.5 Charge Transfer and Interfacial Mechanism
Insert charge-density-difference maps and (if available) Bader charge table:
- Net charge transfer graphene ↔ CuS: **[R14] e**
- Net transfer adsorbate ↔ surface: **[R15] e**
- Attach 3D cube (`charge_difference.cube`) and 2D mid-plane contour (`charge_contour_2d.png`).

**Mechanism narrative template**: “Electron accumulation on [site] and depletion on [site] indicate [bonding type], consistent with [stronger/weaker] adsorption and observed DOS shifts near [energy window].”

### 3.6 Comparison with Prior Work and Novelty Positioning
Use this structure:
1. What previous graphene–CuS work already established.
2. What this study newly quantifies (interfacial adsorption/charge-transfer descriptors, convergence-backed reproducibility, defect/termination roadmap).
3. Why this advances practical design of selective heavy-metal capture materials.

---

## 4. Conclusions
This study delivers a reproducible and extensible first-principles workflow for graphene/CuS heterointerfaces and provides publication-ready analysis infrastructure for adsorption-selectivity research. The framework is designed to produce transparent convergence evidence, standardized figure assets, and directly comparable descriptors (energetics, DOS/PDOS, band features, and charge transfer). After insertion of computed values, the manuscript can support a high-impact narrative focused on interface-resolved mechanistic novelty rather than generic composite-property reporting.

---

## 5. Data and Code Availability
All scripts, model-building tools, and plotting/export utilities should be made public in the associated GitHub repository (preferably archived with a Zenodo DOI). The reproducibility package should include:
- Structures (`.xyz`, `.traj`)
- Energies (`energies.txt`, `paper_summary_metrics.csv`)
- Output index (`output_manifest.csv`)
- Electronic plots (`dos_total.png`, `pdos_cu_d.png`, `band_structure.png`)
- Convergence artifacts (`convergence_scan.csv`, `convergence_scan.png`)
- Runtime metadata (`run_metadata.txt`)

Large binary restart/checkpoint files (`.gpw`) and very large intermediate logs/trajectories may be hosted in an external data archive (Zenodo/Figshare/OSF) and referenced in the paper, or provided upon reasonable request.

---

## 6. Author Contributions (Template)
- Conceptualization: [Author A, Author B]
- Methodology: [Author A]
- Software: [Author A]
- Validation: [Author A, Author C]
- Formal analysis: [Author A, Author B]
- Writing – original draft: [Author A]
- Writing – review & editing: [All authors]
- Supervision: [Author B]

---

## 7. Conflict of Interest
The authors declare no competing financial interest.

---

## 8. Acknowledgments (Template)
The authors acknowledge computational resources from [Institution/Cloud Program], and thank [funding agency/grant number] for support.

---

## 9. Figure and Table Plan (Journal Quality)

### Figures
- **Figure 1**: Relaxed structures (graphene, CuS slab, composite, adsorbed systems), consistent color/scale bars.
- **Figure 2**: Convergence plot (k-point/cutoff) + uncertainty annotation.
- **Figure 3**: Total DOS + Cu-d/S-p PDOS comparison (clean typography, aligned Fermi level at 0 eV).
- **Figure 4**: Band structure with high-symmetry path labels.
- **Figure 5**: Charge-density-difference slices/isovalue maps for pristine and adsorbed interfaces.
- **Figure 6**: Adsorption-energy comparison chart across sites/species (Pb²⁺, Cd²⁺).

### Tables
- **Table 1**: Structural model parameters (supercell, layers, vacuum, mismatch).
- **Table 2**: Convergence matrix (ecut, k-points, energies, deltas).
- **Table 3**: Key energetics (binding, adsorption, optional zero-point/solvation corrections).
- **Table 4**: Charge-transfer descriptors (Bader/work-function where available).
- **Table 5**: Comparison with prior literature values and methodological differences.

---

## 10. References (Starter List to Expand)
1. ASE documentation and methodology papers.
2. GPAW documentation and method references.
3. Quantum ESPRESSO foundational references.
4. Graphene–CuS composite literature (thermoelectric and interface studies).
5. Covellite surface DFT literature.
6. Heavy-metal adsorption DFT benchmark papers.

> Replace this starter list with journal-formatted references after final literature curation.
