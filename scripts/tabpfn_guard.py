"""
Exp D guard - MUST be imported BEFORE `tabpfn`. Enforces the plan rules (2026-11-14-ml-repo-i-tabpfn.md):
  - TabPFN v2 only, local weights only, NEVER the hosted API, NEVER an automatic weight download.
  - telemetry off, CPU, large-dataset CPU override.
If the checkpoint file is missing the process stops with instructions (nothing is downloaded).

Checkpoint location (first hit wins):
  1. env TABPFN_V2_CKPT (full path to the .ckpt file)
  2. <portfolio>/models/tabpfn/tabpfn-v2-classifier.ckpt
"""
import os
import sys
from pathlib import Path

CKPT_NAME = 'tabpfn-v2-classifier.ckpt'
CKPT_EXPECTED_BYTES = 29_016_968
CKPT_URL = f'https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/main/{CKPT_NAME}'
DEFAULT_CKPT = Path(__file__).resolve().parent.parent / 'models' / 'tabpfn' / CKPT_NAME

os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = 'true'
os.environ['HF_HUB_OFFLINE'] = '1'               # a missing file can never trigger a download
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ.pop('TABPFN_TOKEN', None)             # no API client credentials in this process
os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(DEFAULT_CKPT.parent))


def ckpt_path(require=True):
    p = Path(os.environ.get('TABPFN_V2_CKPT', DEFAULT_CKPT))
    if require and not p.is_file():
        sys.exit(f'\nTabPFN v2 weights NOT found: {p}\n'
                 f'Download them manually (Marin): {CKPT_URL}\n'
                 f'(~{CKPT_EXPECTED_BYTES / 1e6:.1f} MB), save as the path above or set TABPFN_V2_CKPT.\n'
                 f'Nothing is downloaded automatically. See requirements-tabpfn.txt / README Exp D.')
    if require and p.stat().st_size < 1_000_000:
        sys.exit(f'{p} is too small ({p.stat().st_size} B) - incomplete download or an HTML error page.')
    return p
