#!/usr/bin/env bash
# End-to-end: install -> download EMBER sample -> train -> explain -> sample reports (~10 min, ~900 MB download)
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
.venv/bin/python tests/test_features.py
.venv/bin/python src/download.py
.venv/bin/python src/data.py
.venv/bin/python src/train.py
.venv/bin/python src/explain.py
.venv/bin/python src/report.py --n 3
