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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Ensure repository root is importable when running `python scripts/run_from_github.py`
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE, LBFGS
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


def write_output_manifest(out_dir: Path) -> Path:
    """Create a CSV manifest of generated outputs (tables/images/data/checkpoints)."""
    category_map = {
        '.png': 'figure',
        '.csv': 'table',
        '.txt': 'text',
        '.xyz': 'structure',
        '.traj': 'trajectory',
        '.gpw': 'checkpoint',
        '.cube': 'charge_density',
        '.md': 'report',
        '.log': 'log',
    }
    rows = []
    for p in sorted(out_dir.glob('*')):
        if p.is_file():
            ext = p.suffix.lower()
            rows.append(
                {
                    'file': p.name,
                    'category': category_map.get(ext, 'other'),
                    'size_kb': round(p.stat().st_size / 1024.0, 2),
                }
            )
    manifest = out_dir / 'output_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'category', 'size_kb'])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def sync_outputs_to_google_drive(output_dir: Path, gdrive_dir: Path) -> Path:
    """Mirror all outputs/logs to Google Drive (Colab-friendly)."""
    output_dir = output_dir.resolve()
    gdrive_dir = gdrive_dir.expanduser().resolve()

    if not gdrive_dir.exists():
        raise RuntimeError(
            f"Google Drive directory does not exist: {gdrive_dir}. "
            "Mount Drive first (e.g., /content/drive) or pass --gdrive-dir to an existing folder."
        )

    # If user already writes directly into Drive, no extra copy is needed.
    if str(output_dir).startswith(str(gdrive_dir)):
        return output_dir

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    dest = gdrive_dir / f'{output_dir.name}_{stamp}'
    shutil.copytree(output_dir, dest, dirs_exist_ok=True)
    return dest


def choose_profile(profile: str) -> Dict[str, object]:
    if profile == 'publish':
        return {
            'kpts': (5, 5, 1),
            'ecut': 520,
            'fmax': 0.015,
            'steps': 240,
            'dos_npts': 1600,
            'dos_width': 0.10,
            'mode_type': 'pw',
            'pre_relax_lcao': True,
            'vacuum': 10.0,
            'layers': 3,
            'total_vacuum': 20.0,
            'max_cell_z': 60.0,
            'optimizer': 'LBFGS',
            'energy_conv': 1e-6,
            'maxstep': 0.04,
        }
    return {
        'kpts': (3, 3, 1),
        'ecut': 420,
        'fmax': 0.03,
        'steps': 120,
        'dos_npts': 900,
        'dos_width': 0.18,
        'mode_type': 'lcao',
        'pre_relax_lcao': False,
        'vacuum': 7.5,
        'layers': 2,
        'total_vacuum': 15.0,
        'max_cell_z': 40.0,
        'optimizer': 'LBFGS',
        'energy_conv': 1e-6,
        'maxstep': 0.05,
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
        calc = make_gpaw_calculator(kpts=kpts, ecut=float(ecut), xc=xc, txt=str(out_dir / f'conv_k{kpts}_e{ecut}.log'), mode_type='pw')
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


def _find_best_supercells(
    requested_graphene_n: int,
    max_strain: float = 0.08,
    layers: int = 3,
    vacuum: float = 10.0,
) -> Tuple[int, Tuple[int, int, int], float]:
    """Search small graphene/CuS supercells to reduce in-plane mismatch."""
    best = None
    for g_n in range(max(2, requested_graphene_n - 2), requested_graphene_n + 5):
        g = build_graphene_nanoplate(size=(g_n, g_n, 1), vacuum=18.0)
        g_a, g_b = _cell_lengths_xy(g)
        for rep in [(1, 1, 1), (2, 2, 1), (3, 3, 1)]:
            c = build_covellite_slab(layers=layers, vacuum=vacuum, supercell=rep)
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


def _enforce_total_vacuum(atoms: Atoms, total_vacuum: float = 15.0, max_cell_z: float | None = None) -> Dict[str, float]:
    """Center structure and enforce a compact z-cell for faster FFT grids."""
    import numpy as np

    z = atoms.positions[:, 2]
    thickness = float(np.max(z) - np.min(z))
    target_cell_z = thickness + float(total_vacuum)
    if max_cell_z is not None:
        target_cell_z = min(target_cell_z, float(max_cell_z))
    target_cell_z = max(target_cell_z, thickness + 10.0)  # keep a safe minimum total vacuum

    cell = atoms.cell.array.copy()
    cell[2, 2] = target_cell_z
    atoms.set_cell(cell, scale_atoms=False)
    atoms.center(axis=2)
    return {'thickness': thickness, 'cell_z': float(atoms.cell[2, 2]), 'total_vacuum': float(atoms.cell[2, 2] - thickness)}


def _optimizer_class(name: str):
    n = name.lower()
    if n == 'bfgs':
        return BFGS
    if n == 'lbfgs':
        return LBFGS
    if n == 'fire':
        return FIRE
    raise ValueError(f'Unsupported optimizer: {name}')


class _EarlyStopRelax(Exception):
    """Internal signal used to stop ASE optimizers from monitor callbacks."""


def monitored_relax(
    atoms: Atoms,
    calc,
    traj_path: Path,
    fmax: float,
    steps: int,
    optimizer_name: str = 'LBFGS',
    maxstep: float = 0.05,
):
    """Relax with instability detection and best-geometry capture."""
    atoms = atoms.copy()
    atoms.calc = calc
    opt_cls = _optimizer_class(optimizer_name)
    opt = opt_cls(
        atoms,
        trajectory=str(traj_path),
        logfile=str(traj_path).replace('.traj', '.opt.log'),
        maxstep=maxstep,
    )

    state = {
        'energies': [],
        'fmax': [],
        'best_e': float('inf'),
        'best_atoms': atoms.copy(),
        'unstable': False,
        'reason': '',
    }

    def monitor():
        e = atoms.get_potential_energy()
        fm = float(np.abs(atoms.get_forces()).max())
        state['energies'].append(e)
        state['fmax'].append(fm)
        if e < state['best_e']:
            state['best_e'] = e
            state['best_atoms'] = atoms.copy()

        if len(state['fmax']) >= 6:
            recent_min = min(state['fmax'][:-1])
            cur = state['fmax'][-1]
            if recent_min < 0.08 and cur > 0.15 and cur > 2.5 * recent_min:
                state['unstable'] = True
                state['reason'] = 'force_spike_after_near_convergence'
                raise _EarlyStopRelax(state['reason'])

        # Catastrophic divergence guard
        if len(state['fmax']) >= 3:
            cur_f = state['fmax'][-1]
            if cur_f > 5.0 or e > (state['best_e'] + 8.0):
                state['unstable'] = True
                state['reason'] = 'catastrophic_divergence'
                raise _EarlyStopRelax(state['reason'])

    opt.attach(monitor, interval=1)
    try:
        opt.run(fmax=fmax, steps=steps)
    except _EarlyStopRelax:
        # Deliberate early exit from the monitor callback.
        pass
    return state


def _pick_restart_geometry(traj_path: Path, force_threshold: float = 0.08) -> tuple[Atoms | None, str]:
    """Pick a stable restart frame from a prior relaxation trajectory.

    Preference order:
      1) Lowest-force frame with fmax <= force_threshold.
      2) Otherwise the global lowest-force frame.
    """
    if not traj_path.exists():
        return None, ''

    frames = Trajectory(str(traj_path))
    if len(frames) == 0:
        return None, ''

    best_under = None  # (fmax, energy, index, atoms)
    best_any = None
    for i, atoms_i in enumerate(frames):
        try:
            e = float(atoms_i.get_potential_energy())
            f = float(np.abs(atoms_i.get_forces()).max())
        except Exception:
            continue
        rec = (f, e, i, atoms_i.copy())
        if best_any is None or (f, e) < (best_any[0], best_any[1]):
            best_any = rec
        if f <= force_threshold and (best_under is None or (f, e) < (best_under[0], best_under[1])):
            best_under = rec

    chosen = best_under or best_any
    if chosen is None:
        return None, ''

    fval, eval_, idx, atoms_out = chosen
    note = (
        f"Loaded restart geometry from {traj_path.name} frame {idx} "
        f"(fmax={fval:.4f} eV/Å, E={eval_:.6f} eV)."
    )
    return atoms_out, note

def run(
    output_dir: Path,
    graphene_n: int,
    spacing: float,
    adsorbate: str,
    profile: str,
    engine: str,
    isolated_vacuum: float = 6.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_pub_plot_style()
    run_cfg = choose_profile(profile)

    metadata = {
        'accelerator': detect_accelerator(),
        'profile': profile,
        'engine': engine,
        'kpts': run_cfg['kpts'],
        'ecut': run_cfg['ecut'],
        'mode_type': run_cfg['mode_type'],
        'energy_conv': run_cfg['energy_conv'],
        'maxstep': run_cfg['maxstep'],
    }
    (output_dir / 'run_metadata.txt').write_text('\n'.join(f'{k}: {v}' for k, v in metadata.items()) + '\n', encoding='utf-8')

    max_strain = 0.08
    best_graphene_n, best_rep, best_score = _find_best_supercells(
        graphene_n,
        max_strain=max_strain,
        layers=int(run_cfg['layers']),
        vacuum=float(run_cfg['vacuum']),
    )
    graphene = build_graphene_nanoplate(size=(best_graphene_n, best_graphene_n, 1), vacuum=float(run_cfg['vacuum']))
    cus_slab = build_covellite_slab(
        layers=int(run_cfg['layers']),
        vacuum=float(run_cfg['vacuum']),
        supercell=best_rep,
    )
    composite, mismatch = create_graphene_cus_composite(graphene, cus_slab, spacing=spacing, max_strain=max_strain)
    vac_report = _enforce_total_vacuum(
        composite,
        total_vacuum=float(run_cfg['total_vacuum']),
        max_cell_z=float(run_cfg['max_cell_z']),
    )
    (output_dir / 'lattice_mismatch.txt').write_text(
        (
            f"{mismatch}\n"
            f"selected_graphene_n={best_graphene_n}, selected_cus_rep={best_rep}, residual_max_strain={best_score:.4f}\n"
            f"thickness={vac_report['thickness']:.3f} A, cell_z={vac_report['cell_z']:.3f} A, "
            f"total_vacuum={vac_report['total_vacuum']:.3f} A\n"
        ),
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

    working = composite
    restart_note = ''
    for restart_traj in [output_dir / 'composite_relax_restart.traj', output_dir / 'composite_relax.traj']:
        picked, note = _pick_restart_geometry(restart_traj, force_threshold=max(0.08, float(run_cfg['fmax']) * 2.0))
        if picked is not None:
            working = picked
            restart_note = note
            break
    if restart_note:
        (output_dir / 'restart_seed_note.txt').write_text(restart_note + '\n', encoding='utf-8')

    if bool(run_cfg.get('pre_relax_lcao', False)):
        calc_pre = make_gpaw_calculator(
            kpts=run_cfg['kpts'],
            ecut=float(run_cfg['ecut']),
            xc='PBE',
            txt=str(output_dir / 'pre_relax_lcao.log'),
            mode_type='lcao',
            basis='dzp',
            energy_convergence=float(run_cfg['energy_conv']),
        )
        working, _ = relax_structure(
            working,
            calc_pre,
            traj_path=str(output_dir / 'pre_relax_lcao.traj'),
            fmax=max(0.04, float(run_cfg['fmax']) * 2.0),
            steps=min(90, int(run_cfg['steps']) // 2),
        )

    calc_relax = make_gpaw_calculator(
        kpts=run_cfg['kpts'],
        ecut=float(run_cfg['ecut']),
        xc='PBE',
        txt=str(output_dir / 'relax.log'),
        mode_type=str(run_cfg['mode_type']),
        energy_convergence=float(run_cfg['energy_conv']),
    )
    relax_state = monitored_relax(
        working,
        calc_relax,
        traj_path=output_dir / 'composite_relax.traj',
        fmax=float(run_cfg['fmax']),
        steps=int(run_cfg['steps']),
        optimizer_name=str(run_cfg['optimizer']),
        maxstep=float(run_cfg['maxstep']),
    )
    if relax_state['unstable']:
        (output_dir / 'optimizer_restart_note.txt').write_text(
            f"Instability detected in primary relax ({relax_state['reason']}); restarting from best geometry with FIRE and tighter SCF.\n",
            encoding='utf-8',
        )
        restart_calc = make_gpaw_calculator(
            kpts=run_cfg['kpts'],
            ecut=float(run_cfg['ecut']),
            xc='PBE',
            txt=str(output_dir / 'relax_restart.log'),
            mode_type=str(run_cfg['mode_type']),
            energy_convergence=min(float(run_cfg['energy_conv']), 1e-6),
        )
        relax_state = monitored_relax(
            relax_state['best_atoms'],
            restart_calc,
            traj_path=output_dir / 'composite_relax_restart.traj',
            fmax=min(0.02, float(run_cfg['fmax'])),
            steps=max(80, int(run_cfg['steps']) // 2),
            optimizer_name='FIRE',
            maxstep=0.03,
        )
        calc_relax = restart_calc

    composite_relaxed = relax_state['best_atoms'].copy()
    composite_relaxed.calc = calc_relax
    e_comp = composite_relaxed.get_potential_energy()
    calc_relax.write(str(output_dir / 'composite_relaxed.gpw'), mode='all')

    calc_g = make_gpaw_calculator(kpts=run_cfg['kpts'], ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'graphene_sp.log'), mode_type=str(run_cfg['mode_type']), energy_convergence=float(run_cfg['energy_conv']))
    calc_c = make_gpaw_calculator(kpts=run_cfg['kpts'], ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'cus_sp.log'), mode_type=str(run_cfg['mode_type']), energy_convergence=float(run_cfg['energy_conv']))
    e_graphene = single_point_energy(graphene, calc_g, gpw_out=str(output_dir / 'graphene.gpw'))
    e_cus = single_point_energy(cus_slab, calc_c, gpw_out=str(output_dir / 'cus.gpw'))
    e_bind = compute_binding_energy(e_comp, e_graphene, e_cus)

    calc_loaded = GPAW(str(output_dir / 'composite_relaxed.gpw'), txt=None)
    energies, dos = compute_dos(calc_loaded, npts=int(run_cfg['dos_npts']), width=float(run_cfg['dos_width']))
    cu_indices = [i for i, a in enumerate(composite_relaxed) if a.symbol == 'Cu']
    if not cu_indices:
        raise RuntimeError('No Cu atoms found in relaxed composite; cannot compute Cu-d PDOS.')
    (output_dir / 'pdos_indices.txt').write_text(
        'Cu atom indices used for PDOS: ' + ', '.join(map(str, cu_indices)) + '\n',
        encoding='utf-8',
    )
    energies_p, pdos_cu_d = compute_pdos(
        calc_loaded, atom_indices=cu_indices, angular='d', npts=int(run_cfg['dos_npts']), width=float(run_cfg['dos_width'])
    )
    np.savetxt(output_dir / 'dos_total.csv', np.column_stack((energies, dos)), delimiter=',', header='energy_eV,dos', comments='')
    np.savetxt(output_dir / 'pdos_cu_d.csv', np.column_stack((energies_p, pdos_cu_d)), delimiter=',', header='energy_eV,pdos_cu_d', comments='')
    plot_xy(energies, dos, 'Energy - E_F (eV)', 'DOS (states/eV)', 'Total DOS', str(output_dir / 'dos_total.png'))
    plot_xy(energies_p, pdos_cu_d, 'Energy - E_F (eV)', 'PDOS (states/eV)', 'Cu-d PDOS', str(output_dir / 'pdos_cu_d.png'))

    if str(run_cfg['mode_type']) == 'lcao':
        path = composite_relaxed.cell.bandpath('GMKG', npoints=80)
        calc_bands = calc_loaded.fixed_density(kpts=path, symmetry='off', txt=str(output_dir / 'bandstructure.log'))
        bs = calc_bands.band_structure()
        bs.plot(filename=str(output_dir / 'band_structure.png'), show=False, emin=-6, emax=4)
    else:
        bs = compute_band_structure(calc_loaded, path='GMKG', npoints=140 if profile == 'publish' else 80)
        bs.plot(filename=str(output_dir / 'band_structure.png'), show=False, emin=-6, emax=4)

    ads_system = add_adsorbate_to_composite(composite_relaxed, adsorbate=adsorbate, height=2.4)
    calc_ads = make_gpaw_calculator(kpts=run_cfg['kpts'], ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'ads_relax.log'), mode_type=str(run_cfg['mode_type']), energy_convergence=float(run_cfg['energy_conv']))
    ads_state = monitored_relax(
        ads_system,
        calc_ads,
        traj_path=output_dir / 'adsorption_relax.traj',
        fmax=float(run_cfg['fmax']),
        steps=max(120, int(run_cfg['steps']) - 20),
        optimizer_name='LBFGS',
        maxstep=float(run_cfg['maxstep']),
    )
    ads_relaxed = ads_state['best_atoms'].copy()
    ads_relaxed.calc = calc_ads
    e_ads_total = ads_relaxed.get_potential_energy()

    ads_atom = Atoms('Pb' if adsorbate == 'Pb2+' else 'Cd', positions=[(0, 0, 0)], cell=[15, 15, 15], pbc=False)
    # Keep isolated adsorbate safely away from non-periodic cell boundaries.
    ads_atom.center(vacuum=max(0.0, float(isolated_vacuum)))
    calc_adsorbate = make_gpaw_calculator(kpts=(1, 1, 1), ecut=float(run_cfg['ecut']), xc='PBE', txt=str(output_dir / 'adsorbate_sp.log'), mode_type=str(run_cfg['mode_type']), energy_convergence=float(run_cfg['energy_conv']))
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

    write_output_manifest(output_dir)
    print('Done. Results in:', output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run graphene/CuS publication-ready workflow from cloned repo')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adsorbate', type=str, default='Pb2+', choices=['Pb2+', 'Cd2+'])
    parser.add_argument('--graphene-n', type=int, default=4, help='NxN graphene supercell (N x N x 1)')
    parser.add_argument('--spacing', type=float, default=2.5, help='Initial CuS-graphene gap (Å)')
    parser.add_argument('--profile', type=str, default='quick', choices=['quick', 'publish'])
    parser.add_argument('--engine', type=str, default='gpaw', choices=['gpaw', 'qe'])
    parser.add_argument(
        '--isolated-vacuum',
        type=float,
        default=6.0,
        help='Vacuum padding (Å) used when centering isolated adsorbate reference calculations.',
    )
    parser.add_argument(
        '--gdrive-dir',
        type=Path,
        default=Path('/content/drive/MyDrive/GrapheneCuS_outputs'),
        help='Google Drive folder where all outputs/logs are mirrored.',
    )
    parser.add_argument(
        '--no-gdrive-sync',
        action='store_true',
        help='Disable post-run mirroring to Google Drive.',
    )
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        graphene_n=args.graphene_n,
        spacing=args.spacing,
        adsorbate=args.adsorbate,
        profile=args.profile,
        engine=args.engine,
        isolated_vacuum=args.isolated_vacuum,
    )
    if not args.no_gdrive_sync:
        drive_path = sync_outputs_to_google_drive(args.output_dir, args.gdrive_dir)
        print('Google Drive mirror complete:', drive_path)


if __name__ == '__main__':
    main()
