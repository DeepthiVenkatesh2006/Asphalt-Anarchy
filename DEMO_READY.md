# PathSnap local rehearsal and two-laptop handoff

## On this Mac: local rehearsal

The updated project is at `/Users/pranav/Downloads/PathSnap_student_materials/Asphalt-Anarchy`.

```bash
cd /Users/pranav/Downloads/PathSnap_student_materials/Asphalt-Anarchy
.venv/bin/python scripts/run_demo.py --role local
```

Open **http://localhost:8501**. Both API and dashboard run locally for this rehearsal. Choose `evaluator_test`, choose a trip, and click **Predict Route**. The sidebar should show a connected model. Ctrl+C in the launcher stops both processes. Stop the existing rehearsal before restarting; the launcher checks for occupied ports.

## Laptop 1: Windows API

Obtain this integration branch and use the project's Python environment. Install dependencies with `python -m pip install -r requirements.txt`.

Place the teacher's unmodified `evaluator_test` folder next to `train` and `student_test`. The dataset root must contain those three folders. Do not put evaluator trips into training or tune the model against them.

```powershell
python scripts/run_demo.py --role api --data-root "C:\YOUR\Student folder"
```

Replace that path with the actual dataset folder. The selected model is `artifacts/local_training/model_final.joblib` and its adjacent metadata/feature schema must be present. This is the tuned 64/32 MLP with its scaler, not the old root artifact. Model loading and schema errors stop the service; no geometric substitute is silently used. Use `http://localhost:8000/health` on Windows to check the server.

## Laptop 2: this Mac, dashboard only

After stopping the local rehearsal, connect to the same network as Windows. Substitute Windows's actual reachable IPv4 address:

```bash
.venv/bin/python scripts/run_demo.py --role ui --api-url http://WINDOWS_IP:8000
```

Open **http://localhost:8501**. The model and datasets stay on Windows. The dashboard connects over HTTP. Verify `/health` from the Mac before making predictions. Do not use Mac localhost:8000 to address the Windows server.

## What the integration now does

- Discovers 80 train, 10 student-test and 10 teacher evaluator trips on this Mac.
- Generates candidates using Member 1's pure GPS/road pipeline and the saved model's exact ten features.
- Uses Member 2's verified candidate scorer and normalized candidate probabilities.
- Uses Member 3's road graph, projected candidate positions, network-distance transitions and Viterbi.
- Splits sequences at large GPS gaps or missing candidates; restarts impossible chains and reports restart counts.
- Preserves missing observation predictions as `-1`, instead of inventing road 0.
- With Viterbi off, selects the highest model score.
- Loads references only after prediction for approximate aligned-reference evaluation; reports coverage and missing-reference metrics as unavailable.
- Returns no raw reference geometry to the UI. Road-set F1 is order-insensitive, and is labelled accordingly.
- Uses separate road segments for plotting rather than drawing artificial connectors; no external map tile requests are required. JavaScript/CSS components may still require internet on a fresh browser.
- Sends the chosen search radius and clears stale displayed results when backend/partition/settings change.

## Evidence and limits

`reports/demo/test_results.txt` contains test output. `reports/demo/evaluator_test_verification.json` and `evaluator_execution.log` contain actual all-ten teacher-trip API checks. `first_evaluator_check.json` contains one real response. `runtime_requirements.txt` records the Mac environment; use `requirements-training-lock.txt` for the exact training environment when transporting sklearn artifacts. Dependency/model compatibility on a different Windows environment still needs its own startup check.

The response keeps observation-level road assignments in `predicted_arcs` and inserts shortest directed connecting roads into `expanded_route` and the map geometry. Missing observations, large gaps and impossible-chain restarts break the route; no connector is inserted across those boundaries. Continuity measures the fraction of observation transitions that stay on or connect through the directed road network, with gaps/missing points treated as breaks. Reference point labels are approximate, not independent surveyed truth. No teacher-trip results are used for retraining or parameter selection. These integration fixes change the original demo metrics: use the new measured results, not numbers copied from old demonstration reports.

Verify real Windows-to-Mac communication and save the actual server request log during the final two-laptop demonstration; a local rehearsal does not establish that network test.
