# PathSnap — Asphalt-Anarchy

Noisy GPS -> candidate features -> trained classifier -> graph HMM/Viterbi -> route chunks -> FastAPI -> Leaflet client.

Use branch **main** for the integrated submission. The saved model is `artifacts/local_training/model_final.joblib`; its scaler, feature schema, metadata and integrity manifest form one inference bundle. The older root artifact and preliminary model tables are historical, not the current deployment.

## Submission

Submit these two deliverables:

1. **Public project repository:** https://github.com/DeepthiVenkatesh2006/Asphalt-Anarchy (branch `main`).
2. **One consolidated technical PDF:** [PathSnap_Technical_Report.pdf](reports/final/PathSnap_Technical_Report.pdf).

The repository includes data/feature source, five-model training and Optuna tuning, saved inference artifacts, HMM/Viterbi, API/UI, configurations, tests and measured results. The supplied datasets remain separate. Confirm that evaluators can access the public repository and upload the PDF through the teacher's submission form.

For presentation preparation, use the [simple project study guide](reports/final/PathSnap_Whole_Project_Simple_Study_Guide.docx), [team guide](reports/final/PathSnap_Team_Guide.docx) and [Member 1 notes](reports/final/Member1_Presentation_Notes.docx).

## Install and run

Use **Python >=3.12** for the pinned model runtime. The saved model requires scikit-learn 1.9.1; mismatches fail explicitly.

```bash
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-runtime-lock.txt
python scripts/run_demo.py --role local --data-root "/path/to/Student folder"
```

Open `http://localhost:8501`. For server deployment and two-laptop networking, see [DEMO_READY.md](DEMO_READY.md). Evaluation mode exposes only the ten evaluator trips, validates references server-side and freezes final controls. Training data are not needed to serve a saved model when the complete evaluator partition is available.

## Two-laptop demonstration

Install the dependencies above on both laptops and connect them to the same network.

**Laptop 1 — model/API:**

```bash
python scripts/run_demo.py --role api --data-root "/path/to/Student folder"
```

**Laptop 2 — evaluator interface:** replace `LAPTOP1_IP` with the API laptop's current IPv4 address.

```bash
python scripts/run_demo.py --role ui --api-url http://LAPTOP1_IP:8000
```

On Laptop 2, first check `http://LAPTOP1_IP:8000/health`, then open `http://localhost:8501`. Select an evaluator trip, run evaluation and show the received request in Laptop 1's terminal. See [the complete deployment guide](DEMO_READY.md) for the Windows path example and network setup.

## Project folders

| Folder | Contents |
| --- | --- |
| `src/data`, `src/features` | Input validation, reference alignment and candidate features |
| `src/models` | Training, comparison, tuning, artifact verification and scoring |
| `src/decoder` | Road graph, HMM/Viterbi and route reconstruction |
| `src/api`, `ui` | FastAPI backend, evaluator clients and Leaflet maps |
| `artifacts/local_training` | Final trained model and deployment metadata |
| `configs`, `scripts`, `tests` | Settings, launch/reproduction commands and regression checks |
| `reports` | Dataset notes, experiment evidence and final submission documents |

## Data and reproduction

Raw data are supplied separately: `train/` (80 trips), `student_test/` (10), `evaluator_test/` (10). Each trip has `.track`, `.nodes`, `.arcs` and a server-side `.route` reference. Raw/processed datasets and virtual environments are excluded from Git and handoff bundles.

```bash
python -m src.features.build_features --data-dir "/path/to/Student folder/train" --output-dir data/processed/v1/train --include-labels
python -m src.features.build_features --data-dir "/path/to/Student folder/student_test" --output-dir data/processed/v1/student_test --include-labels
python -m src.models.local_training --data-root data/processed/v1 --output-dir reports/member2/reproduced_run --folds 5 --tuning-folds 3 --trials 10
```

Use fresh output directories; existing experiments are protected from overwrite. Do not train, tune or select features using evaluator trips. KNN's private fitted cache retains training examples and is omitted from the public repo; it can be regenerated from the original training features.

## Verification and actual results

```bash
python -m pytest
node --test tests/ui/legacy_ui.test.cjs
python scripts/compare_decoder.py --output-dir reports/decoder/reproduced_comparison
python scripts/audit_model_metrics.py --output-dir reports/audit/reproduced_model_metrics
```

Student-test classifier Top-1 is **89.07%**; HMM returned-point accuracy is **89.53%**. Observation Macro F1 improves from **0.7731 to 0.8098**. Point metrics cover **14,215 / 19,160 approximate aligned labels (74.19%)**. HMM can worsen individual trips and adds runtime; these are not independently surveyed whole-trip point labels.

- [Consolidated technical PDF](reports/final/PathSnap_Technical_Report.pdf)
- [Data and feature contract](reports/member1/README.md)
- [Authoritative five-fold training results](reports/member2/local_training/RESULTS.md)
- [Measured classifier-versus-HMM comparison](reports/decoder/student_test/summary.csv)
- [Audit checks and input/model validation](reports/audit/)
- [Actual Folium development map](reports/final/assets/student_test_00000007_folium.html)

The live map is native Leaflet with bundled assets; Folium supplies the required development/report map. Reference geometry is never returned by evaluator endpoints. Verification evidence records 91 passing Python tests and four passing JavaScript tests. During the final two-laptop demonstration, capture the actual receiving-server request logs.
