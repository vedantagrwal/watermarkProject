# Explainable AI-Based Malware Classification System

Windows PE malware classifier on the public **EMBER-2018** dataset that doesn't just say *"malware"*: every verdict
comes with a **SHAP** + **LIME** explanation and a human-readable report.

Team member: Vedant Agarwal · Full write-up: [`REPORT.md`](REPORT.md) / [`report/Malware_XAI_Report.docx`](report/Malware_XAI_Report.docx)

## Pipeline

```
EMBER feature vectors (2381 / file)
  ├─ PE header   (general info, COFF/optional header, sections, data directories)
  ├─ Opcode proxy (byte histogram + byte-entropy histogram)
  ├─ API calls   (hashed import table, decoded back to Windows API names where possible)
  └─ Strings     (paths, URLs, registry keys, printable-char distribution)
        │
        ▼
 Logistic regression  ·  MLP (deep-learning baseline)  ·  LightGBM (main model)
        │
        ▼
 SHAP (TreeSHAP: global + per-file)   LIME (model-agnostic, per-file)
        │
        ▼
 Quality checks: SHAP-vs-LIME agreement, LIME stability, deletion (faithfulness) test
        │
        ▼
 reports/<sha>.md  -  verdict, risk, top reasons in plain English, family breakdown, LIME cross-check, waterfall plot
```

## Run it

```bash
./run_all.sh                       # everything, ~10 min, downloads ~900 MB of EMBER shards
# or step by step
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/download.py   # 11 EMBER shards (8 train / 3 test) from the HuggingFace mirror cw1521/ember2018
.venv/bin/python src/data.py       # jsonl -> data/processed.npz
.venv/bin/python src/train.py      # models/, results/metrics.json, results/roc.png
.venv/bin/python src/explain.py    # results/shap_*.png, explain_metrics.json, top_features.csv
.venv/bin/python src/report.py --n 3        # reports/*.md  (3 detected malware + 1 benign)
.venv/bin/python src/report.py --index 42   # report for one specific test row
```

Trained models are committed in `models/`, so `explain.py` / `report.py` work after `download.py` + `data.py` without retraining.

## Results (6 000 held-out EMBER test files, 12 241 training files)

| Model | AUC | Accuracy | TPR @ 1 % FPR |
|---|---|---|---|
| Logistic regression | 0.914 | 85.6 % | 12.1 % |
| MLP (deep-learning baseline) | 0.946 | 89.7 % | 51.1 % |
| **LightGBM** | **0.975** | **91.7 %** | **71.1 %** |

Explanation quality (LightGBM, 40 detected malware files): SHAP pushes the score down by 0.18 when its top-10 features are
neutralised vs. ≈0 for random features; vanilla LIME on all 2381 columns fails (surrogate R² ≈ 0.05), LIME on the 60 most
important columns works (R² 0.47, sign agreement with SHAP 85 %, top-10 overlap only 0.34). See the report for details.

## Honest limitations

* **Opcodes**: EMBER stores no disassembly, so "opcode frequencies" are approximated by byte-histogram bins labelled with the
  single-byte x86 opcode they usually represent (e.g. `0xE8` = CALL). Real opcode n-grams need the raw binaries.
* **API names are hints**: imports are feature-hashed (1024 bins). `src/features.py` inverts the hash for ~200 well-known APIs,
  but collisions exist and the decoding was not independently validated against raw import tables.
* **Sample, not full dataset**: ~2 % of the labelled EMBER-2018 files (the full set is 1 M files / 1.6 GB+). Shards are not shuffled, so the
  sample is spread across the dataset but not perfectly random. Single seed, no confidence intervals.
* **Shortcut features**: `hdr_timestamp` and "no signature" are the top features; they are real signal in EMBER but partly
  dataset artefacts that may not hold in the wild.
* Explanations describe the *model*, not causal malware behaviour.
* Scoring a brand-new `.exe` needs EMBER's `lief`-based extractor (not included).

## Layout

```
src/        download.py data.py features.py train.py explain.py report.py
tests/      test_features.py
models/     lgbm.txt logreg.joblib mlp.joblib
results/    metrics.json explain_metrics.json top_features.csv *.png logs
reports/    sample per-file malware reports (+ SHAP waterfall)
report/     Malware_XAI_Report.docx (submission document)
```
