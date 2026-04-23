#!/usr/bin/env python3
"""Automated runner for graphene/CuS composite studies (Colab + publication profile).

Usage examples
--------------
Quick profile (Colab):
  python scripts/run_from_github.py --output-dir /path/to/outputs --profile quick --engine gpaw

Publication profile (larger basis + convergence scans):
  python scripts/run_from_github.py --output-dir /path/to/outputs --profile publish --engine gpaw

Optional Quantum ESPRESSO engine (requires pw.x + ASE espresso support):
  python scripts/run_from_github.py --output-dir /path/to/outputs --engine qe
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Ensure repository root is importable when running `python scripts/run_from_github.py`
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write
from gpaw import GPAW

try:
    from src.gpaw_cus_graphene_pipeline import (
        add_adsorbate_to_composite,
        build_covellite_slab,
        build_graphene_nanoplate,
        compute_adsorption_energy,
        compute_band_structure,
        compute_binding_energy,
        compute_dos,
        compute_pdos,
        create_graphene_cus_composite,
        make_gpaw_calculator,
        plot_xy,
        relax_structure,
        save_artifacts,
        save_structure_images,
        single_point_energy,
    )
except ModuleNotFoundError:
    # Fallback for environments where `src` isn't treated as a package.
    SRC_DIR = REPO_ROOT / 'src'
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from gpaw_cus_graphene_pipeline import (
        add_adsorbate_to_composite,
        build_covellite_slab,
        build_graphene_nanoplate,
        compute_adsorption_energy,
        compute_band_structure,
        compute_binding_energy,
        compute_dos,
        compute_pdos,
        create_graphene_cus_composite,
        make_gpaw_calculator,
        plot_xy,
        relax_structure,
        save_artifacts,
        save_structure_images,
        single_point_energy,
    )


def detect_accelerator() -> str:
    """Return CPU/GPU/TPU hint for run metadata."""
    if Path('/sys/class/tpu').exists() or 'TPU_NAME' in os.environ:
        return 'TPU-detected'
    if shutil.which('nvidia-smi'):
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
            first = out.strip().splitlines()[0]
            return f'GPU-detected: {first}'
        except Exception:
            return 'GPU-detected: unknown model'
    return 'CPU-only'


def choose_profile(profile: str) -> Dict[str, object]:
    if profile == 'publish':
        return {
            'kpts': (5, 5, 1),
            'ecut': 520,
            'fmax': 0.015,
            'steps': 280,
            'dos_npts': 1800,
            'dos_width': 0.10,
        }
    return {
        'kpts': (3, 3, 1),
        'ecut': 450,
        'fmax': 0.02,
        'steps': 200,
        'dos_npts': 1000,
        'dos_width': 0.15,
    }


def _set_pub_plot_style() -> None:
    plt.rcParams.update(
        {
            'figure.dpi': 200,
            'savefig.dpi': 400,
            'font.size': 12,
            'axes.labelsize': 13,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'lines.linewidth': 2.2,
        }
    )


def run_convergence_scan(composite: Atoms, out_dir: Path, xc: str = 'PBE') -> Path:
    """Small ecut/k-point convergence scan for publication tables."""
    rows = []
    scans = [
        ((3, 3, 1), 420),
        ((3, 3, 1), 500),
        ((5, 5, 1), 420),
        ((5, 5, 1), 520),
    ]
    for kpts, ecut in scans:
        calc = make_gpaw_calculator(kpts=kpts, ecut=float(ecut), xc=xc, txt=str(out_dir / f'conv_k{kpts}_e{ecut}.log'))
        e = single_point_energy(composite, calc)
        rows.append({'kpts': str(kpts), 'ecut_eV': ecut, 'energy_eV': float(e)})

    csv_path = out_dir / 'convergence_scan.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['kpts', 'ecut_eV', 'energy_eV'])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k in ['(3, 3, 1)', '(5, 5, 1)']:
        xs = [r['ecut_eV'] for r in rows if r['kpts'] == k]
        ys = [r['energy_eV'] for r in rows if r['kpts'] == k]
        ax.plot(xs, ys, marker='o', label=f'k={k}')
    ax.set_xlabel('Cutoff energy (eV)')
    ax.set_ylabel('Total energy (eV)')
    ax.set_title('Convergence scan (composite)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'convergence_scan.png')
    plt.close(fig)
    return csv_path


def run_qe_single_point(atoms: Atoms, out_prefix: str, out_dir: Path, kpts=(3, 3, 1), ecutwfc=70.0) -> float:
    """Optional QE backend for single-point total energies.

    Requires installed `pw.x` and ASE espresso calculator support.
    """
    try:
        from ase.calculators.espresso import Espresso
    except Exception as exc:
        raise RuntimeError('ASE Espresso interface not available. Install QE-compatible ASE extras.') from exc

    pw = shutil.which('pw.x')
    if not pw:
        raise RuntimeError('Quantum ESPRESSO binary pw.x not found in PATH.')

    pseudo_dir = Path('/content/pseudos')
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    # User should provide pseudopotentials in /content/pseudos for production use.
    pseudopotentials = {'C': 'C.pbe-n-kjpaw_psl.1.0.0.UPF', 'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF', 'S': 'S.pbe-n-kjpaw_psl.1.0.0.UPF', 'Pb': 'Pb.pbe-dn-kjpaw_psl.1.0.0.UPF', 'Cd': 'Cd.pbe-dn-kjpaw_psl.1.0.0.UPF'}
    calc = Espresso(
        command=f'{pw} -in PREFIX.pwi > PREFIX.pwo',
        pseudopotentials=pseudopotentials,
        pseudo_dir=str(pseudo_dir),
        kpts=kpts,
        input_data={
            'control': {'calculation': 'scf', 'prefix': out_prefix, 'tprnfor': True, 'tstress': True},
            'system': {'ecutwfc': ecutwfc, 'occupations': 'smearing', 'smearing': 'mv', 'degauss': 0.01},
            'electrons': {'conv_thr': 1.0e-8},
        },
    )
    a = atoms.copy()
    a.calc = calc
    e = a.get_potential_energy()
    write(out_dir / f'{out_prefix}_qe.xyz', a)
    return float(e)



def _cell_lengths_xy(atoms: Atoms) -> Tuple[float, float]:
    import numpy as np

    return float(np.linalg.norm(atoms.cell[0])), float(np.linalg.norm(atoms.cell[1]))


def _find_best_supercells(requested_graphene_n: int, max_strain: float = 0.08) -> Tuple[int, Tuple[int, int, int], float]:
    """Search small graphene/CuS supercells to reduce in-plane mismatch."""
    best = None
    for g_n in range(max(2, requested_graphene_n - 2), requested_graphene_n + 5):
        g = build_graphene_nanoplate(size=(g_n, g_n, 1), vacuum=18.0)
        g_a, g_b = _cell_lengths_xy(g)
        for rep in [(1, 1, 1), (2, 2, 1), (3, 3, 1)]:
            c = build_covellite_slab(layers=4, vacuum=18.0, supercell=rep)
            c_a, c_b = _cell_lengths_xy(c)
            sx = abs(1.0 - g_a / c_a)
            sy = abs(1.0 - g_b / c_b)
            score = max(sx, sy)
            cand = (g_n, rep, score)
            if best is None or score < best[2]:
                best = cand
            if score <= max_strain:
                return cand
    assert best is not None
    return best

def run(output_dir: Path, graphene_n: int, spacing: float, adsorbate: str, profile: str, engine: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_pub_plot_style()
    run_cfg = choose_profile(profile)

    metadata = {
        'accelerator': detect_accelerator(),
        'profile': profile,
        'engine': engine,
        'kpts': run_cfg['kpts'],
        'ecut': run_cfg['ecut'],
    }
    (output_dir / 'run_metadata.txt').write_text('\n'.join(f'{k}: {v}' for k, v in metadata.items()) + '\n', encoding='utf-8')

    max_strain = 0.08
    best_graphene_n, best_rep, best_score = _find_best_supercells(graphene_n, max_strain=max_strain)
    graphene = build_graphene_nanoplate(size=(best_graphene_n, best_graphene_n, 1), vacuum=18.0)
    cus_slab = build_covellite_slab(
        layers=5 if profile == 'publish' else 4,
        vacuum=18.0,
        supercell=best_rep,
    )
    composite, mismatch = create_graphene_cus_composite(graphene, cus_slab, spacing=spacing, max_strain=max_strain)
    (output_dir / 'lattice_mismatch.txt').write_text(
        f"{mismatch}\nselected_graphene_n={best_graphene_n}, selected_cus_rep={best_rep}, residual_max_strain={best_score:.4f}\n",
        encoding='utf-8',
    )

    if profile == 'publish':
        run_convergence_scan(composite, output_dir)

    if engine == 'qe':
        e_comp = run_qe_single_point(composite, 'composite', output_dir, kpts=run_cfg['kpts'], ecutwfc=70.0)
        e_graphene = run_qe_single_point(graphene, 'graphene', output_dir, kpts=run_cfg['kpts'], ecutwfc=70.0)
        e_cus = run_qe_single_point(cus_slab, 'cus', output_dir, kpts=run_cfg['kpts'], ecutwfc=70.0)
        e_bind = compute_binding_energy(e_comp, e_graphene, e_cus)

        energies_report = {
            'E_composite': e_comp,
            'E_graphene': e_graphene,
            'E_cus': e_cus,
            'E_binding': e_bind,
        }
        structures = {'graphene': graphene, 'cus_slab': cus_slab, 'composite_init': composite}
        save_artifacts(output_dir, structures, energies_report)
        print('QE single-point workflow finished. For full relax/band/DOS in QE, extend run_qe_single_point to QE NSCF/BANDS tasks.')
        return

    calc_relax = make_gpaw_calculator(
        kpts=run_cfg['kpts'],
        ecut=float(run_cfg['ecut']),
        xc='PBE',
        txt=str(output_dir / 'relax.log'),
    )
    composite_relaxed, e_comp = relax_structure(
        composite,
        calc_relax,
        traj_path=str(output_dir / 'composite_relax.traj'),
        fmax=float(run_cfg['fmax']),
        steps=int(run_cfg['steps']),
    )
    calc_relax.write(str(output_dir / 'composite_relaxed.gpw'), mode='all')

    calc_g = make_gpaw_calculator(kpts=run_cfg['kpts'], ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'graphene_sp.log'))
    calc_c = make_gpaw_calculator(kpts=run_cfg['kpts'], ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'cus_sp.log'))
    e_graphene = single_point_energy(graphene, calc_g, gpw_out=str(output_dir / 'graphene.gpw'))
    e_cus = single_point_energy(cus_slab, calc_c, gpw_out=str(output_dir / 'cus.gpw'))
    e_bind = compute_binding_energy(e_comp, e_graphene, e_cus)

    calc_loaded = GPAW(str(output_dir / 'composite_relaxed.gpw'), txt=None)
    energies, dos = compute_dos(calc_loaded, npts=int(run_cfg['dos_npts']), width=float(run_cfg['dos_width']))
    cu_indices = [i for i, a in enumerate(composite_relaxed) if a.symbol == 'Cu']
    energies_p, pdos_cu_d = compute_pdos(
        calc_loaded, atom_indices=cu_indices, angular='d', npts=int(run_cfg['dos_npts']), width=float(run_cfg['dos_width'])
    )
    plot_xy(energies, dos, 'Energy - E_F (eV)', 'DOS (states/eV)', 'Total DOS', str(output_dir / 'dos_total.png'))
    plot_xy(energies_p, pdos_cu_d, 'Energy - E_F (eV)', 'PDOS (states/eV)', 'Cu-d PDOS', str(output_dir / 'pdos_cu_d.png'))

    bs = compute_band_structure(calc_loaded, path='GMKG', npoints=140 if profile == 'publish' else 80)
    bs.plot(filename=str(output_dir / 'band_structure.png'), show=False, emin=-6, emax=4)

    ads_system = add_adsorbate_to_composite(composite_relaxed, adsorbate=adsorbate, height=2.4)
    calc_ads = make_gpaw_calculator(kpts=run_cfg['kpts'], ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'ads_relax.log'))
    ads_relaxed, e_ads_total = relax_structure(
        ads_system,
        calc_ads,
        traj_path=str(output_dir / 'adsorption_relax.traj'),
        fmax=float(run_cfg['fmax']),
        steps=max(160, int(run_cfg['steps']) - 20),
    )

    ads_atom = Atoms('Pb' if adsorbate == 'Pb2+' else 'Cd', positions=[(0, 0, 0)], cell=[15, 15, 15], pbc=False)
    calc_adsorbate = make_gpaw_calculator(kpts=(1, 1, 1), ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'adsorbate_sp.log'))
    e_adsorbate = single_point_energy(ads_atom, calc_adsorbate, gpw_out=str(output_dir / 'adsorbate.gpw'))
    e_ads = compute_adsorption_energy(e_ads_total, e_comp, e_adsorbate)

    save_structure_images(graphene, str(output_dir / 'graphene_init.png'))
    save_structure_images(cus_slab, str(output_dir / 'cus_slab_init.png'))
    save_structure_images(composite, str(output_dir / 'composite_init.png'))
    save_structure_images(composite_relaxed, str(output_dir / 'composite_relaxed.png'))
    save_structure_images(ads_relaxed, str(output_dir / 'ads_relaxed.png'))

    energies_report = {
        'E_composite': e_comp,
        'E_graphene': e_graphene,
        'E_cus': e_cus,
        'E_binding': e_bind,
        'E_ads_total': e_ads_total,
        'E_adsorbate': e_adsorbate,
        'E_adsorption': e_ads,
    }
    structures = {
        'graphene': graphene,
        'cus_slab': cus_slab,
        'composite_init': composite,
        'composite_relaxed': composite_relaxed,
        'ads_relaxed': ads_relaxed,
    }
    save_artifacts(output_dir, structures, energies_report)

    summary_csv = output_dir / 'paper_summary_metrics.csv'
    with summary_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['binding_energy_eV', e_bind])
        writer.writerow(['adsorption_energy_eV', e_ads])
        writer.writerow(['accelerator', metadata['accelerator']])
        writer.writerow(['kpts', run_cfg['kpts']])
        writer.writerow(['ecut_eV', run_cfg['ecut']])

    print('Done. Results in:', output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run graphene/CuS publication-ready workflow from cloned repo')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adsorbate', type=str, default='Pb2+', choices=['Pb2+', 'Cd2+'])
    parser.add_argument('--graphene-n', type=int, default=4, help='NxN graphene supercell (N x N x 1)')
    parser.add_argument('--spacing', type=float, default=2.5, help='Initial CuS-graphene gap (Å)')
    parser.add_argument('--profile', type=str, default='quick', choices=['quick', 'publish'])
    parser.add_argument('--engine', type=str, default='gpaw', choices=['gpaw', 'qe'])
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        graphene_n=args.graphene_n,
        spacing=args.spacing,
        adsorbate=args.adsorbate,
        profile=args.profile,
        engine=args.engine,
    )


if __name__ == '__main__':
    main()
