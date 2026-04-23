#!/usr/bin/env python3
"""Post-process GPAW outputs into publication-ready figures/tables.

This script does not fabricate results; it computes figures from completed calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from gpaw import GPAW
from gpaw.dos import DOSCalculator

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.gpaw_cus_graphene_pipeline import (
    compute_binding_energy,
    density_difference,
    make_gpaw_calculator,
    single_point_energy,
)


def _parse_kpts(raw: str):
    vals = raw.strip().strip('()').split(',')
    return tuple(int(v.strip()) for v in vals if v.strip())


def _parse_energies(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            out[k.strip()] = float(v.strip().split()[0])
    return out


def elemental_pdos(calc: GPAW, atoms, npts=1600, width=0.12):
    dos_calc = DOSCalculator.from_calculator(calc)
    e = dos_calc.get_energies(npoints=npts)

    species = ['C', 'Cu', 'S']
    data = {}
    for sp in species:
        idx = [i for i, a in enumerate(atoms) if a.symbol == sp]
        y = np.zeros_like(e)
        if not idx:
            data[sp] = y
            continue
        for i in idx:
            for l in ['s', 'p', 'd']:
                try:
                    y += dos_calc.raw_pdos(e, a=i, l=l, width=width)
                except Exception:
                    pass
        data[sp] = y
    return e, data


def plot_elemental_pdos(e, data, out_png: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(e, data['C'], label='C (graphene)', color='black')
    plt.plot(e, data['Cu'], label='Cu', color='tab:orange')
    plt.plot(e, data['S'], label='S', color='tab:blue')
    plt.axvline(0.0, color='red', ls='--', lw=1.0)
    plt.xlim(-6, 6)
    plt.xlabel('Energy - Ef (eV)')
    plt.ylabel('pDOS (states/eV)')
    plt.title('Element-resolved pDOS (Graphene/CuS)')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=350)
    plt.close()


def plot_charge_contour_2d(diff, atoms, out_png: Path):
    slice_idx = diff.shape[1] // 2
    charge_slice = diff[:, slice_idx, :]

    cell = atoms.cell
    x = np.linspace(0, cell[0, 0], diff.shape[0])
    z = np.linspace(0, cell[2, 2], diff.shape[2])
    X, Z = np.meshgrid(x, z)

    plt.figure(figsize=(10, 6))
    vmax = np.max(np.abs(charge_slice))
    levels = np.linspace(-vmax, vmax, 80)
    c = plt.contourf(X, Z, charge_slice.T, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(c, label=r'$\Delta\rho$ (a.u.)')

    for atom in atoms:
        if abs(atom.position[1] - atoms.cell[1, 1] / 2.0) < 1.0:
            plt.scatter(atom.position[0], atom.position[2], c='k', s=8)

    plt.xlabel('X (Å)')
    plt.ylabel('Z (Å)')
    plt.title('2D charge-density-difference contour (mid-Y slice)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=350)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--mode-type', choices=['lcao', 'pw'], default='lcao')
    ap.add_argument('--kpts', default='(3,3,1)')
    ap.add_argument('--ecut', type=float, default=420.0)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    comp_xyz = out / 'composite_relaxed.xyz'
    comp_gpw = out / 'composite_relaxed.gpw'
    if not comp_xyz.exists() or not comp_gpw.exists():
        raise FileNotFoundError('Need composite_relaxed.xyz and composite_relaxed.gpw in output-dir.')

    atoms = read(comp_xyz)
    calc = GPAW(str(comp_gpw), txt=None)

    # Elemental pDOS
    e, pd = elemental_pdos(calc, atoms)
    plot_elemental_pdos(e, pd, out / 'pdos_elements_C_Cu_S.png')

    # Recompute component densities in same cell for charge-difference
    kpts = _parse_kpts(args.kpts)
    g = atoms[[a.symbol == 'C' for a in atoms]]
    c = atoms[[a.symbol != 'C' for a in atoms]]
    for part in (g, c):
        part.set_cell(atoms.cell)
        part.pbc = (True, True, True)
        part.center(axis=2)

    gcalc = make_gpaw_calculator(kpts=kpts, ecut=args.ecut, mode_type=args.mode_type, txt=str(out / 'dens_graphene.log'))
    ccalc = make_gpaw_calculator(kpts=kpts, ecut=args.ecut, mode_type=args.mode_type, txt=str(out / 'dens_cus.log'))
    single_point_energy(g, gcalc, gpw_out=str(out / 'graphene_samecell.gpw'))
    single_point_energy(c, ccalc, gpw_out=str(out / 'cus_samecell.gpw'))

    diff = density_difference(calc, gcalc, ccalc)
    write(out / 'charge_difference.cube', atoms, data=diff)
    plot_charge_contour_2d(diff, atoms, out / 'charge_contour_2d.png')

    # Binding energy summary
    energies = _parse_energies(out / 'energies.txt')
    if {'E_composite', 'E_graphene', 'E_cus'}.issubset(energies):
        energies['E_binding_recomputed'] = compute_binding_energy(
            energies['E_composite'], energies['E_graphene'], energies['E_cus']
        )

    report = out / 'publication_results.md'
    report.write_text(
        "\n".join([
            '# Publication Results Summary',
            '',
            f"- Elemental pDOS figure: `{(out / 'pdos_elements_C_Cu_S.png').name}`",
            f"- Charge density cube: `{(out / 'charge_difference.cube').name}`",
            f"- Charge contour figure: `{(out / 'charge_contour_2d.png').name}`",
            f"- Energies parsed: `{json.dumps(energies, indent=2)}`",
        ]) + '\n',
        encoding='utf-8',
    )
    print('Post-processing complete:', out)


if __name__ == '__main__':
    main()
