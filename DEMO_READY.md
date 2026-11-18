# Two-laptop demonstration guide

The integrated project includes trained model scoring, graph-aware HMM/Viterbi reconstruction, FastAPI and interactive Leaflet route maps.

## Prepare both laptops

Use Python >=3.12. From the repository folder run:

```bash
python -m pip install -r requirements-runtime-lock.txt
```

Keep the four final bundle files together in `artifacts/local_training`: `model_final.joblib`, `model_metadata.json`, `feature_columns.json` and `artifact_manifest.json`.

## Laptop 1: trained model and API

Set the data-root argument to the actual folder containing `evaluator_test`. On Windows, for example:

```bash
python scripts/run_demo.py --role api --data-root "C:\PathSnap\Student folder"
```

The API listens on port 8000. Evaluation mode exposes the ten evaluator trips, validates references server-side and fixes the final inference settings. Find this laptop's current IPv4 address with `ipconfig` on Windows or Network settings on macOS. Keep its terminal running.

## Laptop 2: evaluator interface

Connect both laptops to the same network. Open `http://LAPTOP1_IP:8000/health` on Laptop 2 and confirm readiness. Replace LAPTOP1_IP with Laptop 1's actual IPv4 address.

```bash
python scripts/run_demo.py --role ui --api-url http://LAPTOP1_IP:8000
```

Open `http://localhost:8501` on Laptop 2. Select a trip, run evaluation and show the incoming request in Laptop 1's terminal. Allow the Python server through the host firewall on the trusted demonstration network if prompted.

## What to present

- Grey lines: supplied road network. Green lines: predicted route. Blue dots: original noisy GPS.
- The classifier scores possible roads; HMM/Viterbi selects a connected route.
- Classifier Top-1 and decoded point accuracy are separate metrics; the interface also reports label coverage and sequence quality.
- Student-test comparison: observation F1 improves from 0.7731 to 0.8098 and graph continuity from 97.12% to 99.92% with HMM.
- Reference routes stay server-side during evaluator predictions.

Verified regression evidence: 91 Python tests and four JavaScript tests pass. See `reports/audit` and the consolidated technical PDF. Record final screenshots and receiving-server logs during the two-laptop session.

For a local rehearsal, use `python scripts/run_demo.py --role local --data-root STUDENT_DATA_ROOT` and open `http://localhost:8501`. Ctrl+C stops the launched services.
