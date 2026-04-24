# Data & Code Sharing Recommendations for Your Paper

Short answer: **Yes, share code publicly**, and share **processed/essential data publicly**; share very large raw checkpoints selectively.

## What should be public (recommended)
1. **Code repository (public GitHub/Zenodo DOI)**
   - `src/`, `scripts/`, notebook, and exact run commands.
2. **Essential reproducibility outputs**
   - Final structures: `.xyz`, `.traj`
   - Key scalar outputs: `energies.txt`, `paper_summary_metrics.csv`
   - Figures and plotting scripts
   - Convergence table/plot: `convergence_scan.csv`, `convergence_scan.png`
   - Runtime metadata: `run_metadata.txt`
3. **Input settings metadata**
   - XC functional, cutoff, k-points, smearing, force criteria, slab/vacuum settings.

## What can be optional / repository-external
1. **Large binary restart files (`.gpw`)**
   - Optional in main repo if size is high.
   - Prefer archive release (Zenodo/Figshare/OSF) and link from paper.
2. **All intermediate trajectories/logs**
   - Upload if journal mandates; otherwise provide on request or in external archive.

## Recommended paper wording
Use in Data Availability section:

> "All scripts required to reproduce this study are available in a public GitHub repository (archived with DOI). Final structures, summary energies, convergence data, and plotting assets are publicly released. Large binary checkpoint files and full intermediate trajectories are archived externally and are available upon reasonable request."

## Practical publication checklist
- [ ] Make GitHub repo public (or anonymous review link during peer review)
- [ ] Tag release used in manuscript
- [ ] Archive release on Zenodo to obtain DOI
- [ ] Add `CITATION.cff` and software license
- [ ] Ensure figure generation scripts run from raw CSV inputs

## Suggested policy for your current files
- Public in GitHub: `.py`, `.md`, `.ipynb`, `.xyz`, `.traj`, `.csv`, `.png`, `energies.txt`, `run_metadata.txt`
- External archive / optional: `.gpw`, huge trajectories/log bundles
