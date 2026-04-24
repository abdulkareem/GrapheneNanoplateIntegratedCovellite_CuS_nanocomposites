#!/usr/bin/env python3
"""Auto-healing runner for GPAW pipeline jobs in Colab/CLI environments.

This wrapper runs ``scripts/run_from_github.py`` and applies bounded retries for
common failures, while recording all actions in ``correction_log.txt``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AttemptResult:
    attempt: int
    command: List[str]
    returncode: int
    fixes_applied: List[str]
    log_file: str
    pdos_max: float | None = None
    accepted: bool = False
    failure_excerpt: str = ''


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def run_attempt(
    cmd: List[str],
    output_dir: Path,
    attempt: int,
    cwd: Path,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f'auto_heal_attempt_{attempt}.log'
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    merged = [
        f'[{utc_now()}] COMMAND: {" ".join(cmd)}',
        '',
        '--- STDOUT ---',
        proc.stdout,
        '',
        '--- STDERR ---',
        proc.stderr,
        '',
    ]
    log_path.write_text('\n'.join(merged), encoding='utf-8')
    return proc, log_path


def detect_failure(stderr: str) -> str:
    s = stderr or ''
    if 'Google Drive directory does not exist' in s or 'Mount Drive first' in s:
        return 'gdrive_missing'
    if "ModuleNotFoundError: No module named 'gpaw'" in s:
        return 'missing_gpaw'
    if "ModuleNotFoundError: No module named 'ase'" in s:
        return 'missing_ase'
    if 'GridBoundsError' in s:
        return 'grid_bounds'
    if 'ConvergenceError' in s or 'Convergence failure' in s:
        return 'scf_convergence'
    if 'MPI_ABORT' in s or 'Calling MPI_Abort' in s:
        return 'mpi_abort'
    if 'MemoryError' in s or 'Out of memory' in s:
        return 'oom'
    return 'unknown'


def _first_error_excerpt(stdout: str, stderr: str) -> str:
    text = '\n'.join([(stderr or ''), (stdout or '')]).strip()
    if not text:
        return ''
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if any(k in ln for k in ['Error', 'Exception', 'Traceback', 'ModuleNotFoundError', 'RuntimeError']):
            return ln[:220]
    return text.splitlines()[0][:220]


def pdos_max_from_csv(output_dir: Path) -> float | None:
    pdos_csv = output_dir / 'pdos_cu_d.csv'
    if not pdos_csv.exists():
        return None
    vals: list[float] = []
    with pdos_csv.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2:
                continue
            try:
                vals.append(abs(float(row[1])))
            except ValueError:
                continue
    if not vals:
        return None
    return float(max(vals))


def write_publication_report(output_dir: Path, accepted_attempt: AttemptResult | None) -> Path:
    report = output_dir / 'Publication_Ready_Summary.txt'
    lines = [
        '--- SCIENTIFIC SUMMARY REPORT ---',
        'System: Graphene-CuS Nanocomposite',
        f'Generated: {utc_now()}',
        '',
    ]
    if accepted_attempt is None:
        lines.extend(
            [
                'STATUS: NOT READY',
                'Reason: no successful and quality-accepted attempt.',
            ]
        )
    else:
        lines.extend(
            [
                f'STATUS: READY FOR REVIEW (attempt {accepted_attempt.attempt})',
                f'PDOS max(|Cu-d|): {accepted_attempt.pdos_max}',
                f'Log file: {accepted_attempt.log_file}',
                f'Fixes applied: {", ".join(accepted_attempt.fixes_applied) if accepted_attempt.fixes_applied else "none"}',
            ]
        )
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Auto-healing wrapper for scripts/run_from_github.py')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--adsorbate', type=str, default='Pb2+', choices=['Pb2+', 'Cd2+'])
    parser.add_argument('--graphene-n', type=int, default=4)
    parser.add_argument('--spacing', type=float, default=2.5)
    parser.add_argument('--profile', type=str, default='quick', choices=['quick', 'publish'])
    parser.add_argument('--engine', type=str, default='gpaw', choices=['gpaw', 'qe'])
    parser.add_argument('--gdrive-dir', type=Path, default=Path('/content/drive/MyDrive/GrapheneCuS_outputs'))
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--pdos-min-peak', type=float, default=1.0e-3)
    parser.add_argument(
        '--strict-exit',
        action='store_true',
        help='Return non-zero exit code if no quality-accepted run is produced.',
    )
    args = parser.parse_args()

    correction_log = args.output_dir / 'correction_log.txt'
    history_path = args.output_dir / 'auto_heal_history.json'

    attempts: list[AttemptResult] = []
    profile = args.profile
    isolated_vacuum = 6.0
    spacing = float(args.spacing)
    accepted: AttemptResult | None = None
    enable_gdrive_sync = True

    for attempt in range(1, int(args.max_attempts) + 1):
        cmd = [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'run_from_github.py'),
            '--output-dir',
            str(args.output_dir),
            '--adsorbate',
            args.adsorbate,
            '--graphene-n',
            str(args.graphene_n),
            '--spacing',
            str(spacing),
            '--profile',
            profile,
            '--engine',
            args.engine,
            '--isolated-vacuum',
            str(isolated_vacuum),
        ]
        if enable_gdrive_sync:
            cmd += ['--gdrive-dir', str(args.gdrive_dir)]
        else:
            cmd += ['--no-gdrive-sync']
        proc, logfile = run_attempt(cmd, args.output_dir, attempt, cwd=REPO_ROOT)
        fixes: list[str] = []
        result = AttemptResult(
            attempt=attempt,
            command=cmd,
            returncode=proc.returncode,
            fixes_applied=fixes,
            log_file=str(logfile),
            failure_excerpt=_first_error_excerpt(proc.stdout, proc.stderr),
        )

        if proc.returncode == 0:
            result.pdos_max = pdos_max_from_csv(args.output_dir)
            if result.pdos_max is None:
                fixes.append('PDOS CSV missing; accepting run but flagged for review.')
                result.accepted = True
                accepted = result
                attempts.append(result)
                break
            if result.pdos_max < float(args.pdos_min_peak):
                # Quality-based corrective retry.
                fixes.append(f'Flat PDOS detected (max={result.pdos_max:.4e}); switching to publish profile and narrower DOS width via profile defaults.')
                profile = 'publish'
                isolated_vacuum = max(isolated_vacuum, 8.0)
                attempts.append(result)
                continue

            result.accepted = True
            accepted = result
            attempts.append(result)
            break

        failure = detect_failure(proc.stderr)
        if failure == 'grid_bounds':
            isolated_vacuum = min(isolated_vacuum + 2.0, 14.0)
            fixes.append(f'Detected GridBoundsError; increased isolated vacuum to {isolated_vacuum:.1f} Å.')
        elif failure == 'scf_convergence':
            profile = 'publish'
            fixes.append('Detected SCF convergence issue; switched to publish profile (PW + tighter settings).')
        elif failure in {'mpi_abort', 'oom'}:
            profile = 'quick'
            spacing = min(spacing + 0.3, 3.2)
            fixes.append(f'Detected {failure}; using quick profile and increased spacing to {spacing:.2f} Å.')
        elif failure == 'gdrive_missing':
            enable_gdrive_sync = False
            fixes.append('Google Drive mount not detected; retrying with --no-gdrive-sync.')
        elif failure in {'missing_gpaw', 'missing_ase'}:
            pkg = 'gpaw gpaw-data ase mpi4py numpy scipy matplotlib' if failure == 'missing_gpaw' else 'ase'
            fixes.append(f'Missing Python package(s) detected; install with: pip install {pkg}')
            attempts.append(result)
            break
        else:
            excerpt = result.failure_excerpt or 'no stderr/stdout excerpt captured'
            fixes.append(f'Unknown failure type; no automatic patch available. First error line: {excerpt}')
            attempts.append(result)
            break

        attempts.append(result)

    correction_lines = [f'[{utc_now()}] Auto-heal session started.']
    for a in attempts:
        correction_lines.append(
            f'Attempt {a.attempt}: rc={a.returncode}, accepted={a.accepted}, '
            f'pdos_max={a.pdos_max}, fixes={"; ".join(a.fixes_applied) if a.fixes_applied else "none"}, '
            f'log={a.log_file}, excerpt={a.failure_excerpt}'
        )
    correction_lines.append(f'[{utc_now()}] Auto-heal session finished.')
    correction_log.write_text('\n'.join(correction_lines) + '\n', encoding='utf-8')
    history_path.write_text(json.dumps([asdict(a) for a in attempts], indent=2), encoding='utf-8')
    write_publication_report(args.output_dir, accepted)

    if accepted is None:
        msg = 'Auto-heal did not produce a quality-accepted run. See correction_log.txt and auto_heal_history.json.'
        if args.strict_exit:
            raise SystemExit(msg)
        print(msg)
        return

    print(f'Auto-heal succeeded on attempt {accepted.attempt}.')


if __name__ == '__main__':
    main()
