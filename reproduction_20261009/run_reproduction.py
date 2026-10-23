"""Run the supported analysis from packaged inputs in this directory."""
from pathlib import Path
import json
import subprocess
import sys
import time

from verify_package import verify

ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT / 'package_reproduction_verification.json'
    record = {'status': 'running', 'passed': False, 'runs': [],
              'input_scope': 'packaged standardized and source-matched inputs',
              'requires_companion': False, 'scientific_validation_implied': False}
    destination.write_text(json.dumps(record, indent=2) + '\n')
    try:
        verify()
        commands = ['rebuild_backbone.py', 'default_weight_update/run_scenarios.py',
                    'support/run_support.py', 'support/price/audit_zero_load.py',
                    'support/run_physical_supplements.py', 'build_figure_data.py',
                    'alignment/boundary_sensitivity.py',
                    'alignment/sync_reader_exports.py', 'export_2025_pv_comparison.py',
                    'default_weight_update/refresh_exports.py', 'physical_intervals/export_current.py',
                    'default_weight_update/run_diagnostics.py', 'default_weight_update/audit_current.py',
                    'followup_20261013/regional_checks.py']
        for command in commands:
            start = time.perf_counter()
            run = subprocess.run([sys.executable, '-B', str(ROOT / command)], cwd=ROOT)
            record['runs'].append({'command': command, 'exit_code': run.returncode,
                                   'seconds': time.perf_counter() - start})
            destination.write_text(json.dumps(record, indent=2) + '\n')
            if run.returncode:
                raise RuntimeError('Reproduction stage failed: ' + command)
        record.update(status='passed', passed=True)
    except Exception as error:
        record.update(status='failed', passed=False, error=repr(error))
        raise
    finally:
        destination.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
