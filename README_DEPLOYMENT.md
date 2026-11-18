# Two-laptop deployment

Follow [DEMO_READY.md](DEMO_READY.md) for the complete commands. Laptop 1 runs the trained model and FastAPI server; Laptop 2 runs the evaluator interface on the same network.

Use Python >=3.12 and `python -m pip install -r requirements-runtime-lock.txt`. The final model bundle includes embedded preprocessing and verified artifact hashes. Evaluation mode serves the ten evaluator trips with fixed inference settings.

The [technical report](reports/final/PathSnap_Technical_Report.pdf) explains the pipeline, measured improvements and reproducibility. Clean installation and regression evidence are in `reports/audit`. Capture receiving-server logs during the final two-laptop session.
