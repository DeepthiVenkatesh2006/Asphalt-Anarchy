
# PathSnap - Map-Matching & Route Reconstruction (Asphalt-Anarchy)

> **Current demo:** use [DEMO_READY.md](DEMO_READY.md) for the verified local rehearsal, evaluator trips and Windows/Mac launch commands. Older demonstration reports describe earlier code and metrics.

Noisy GPS points -> candidate road arcs -> ML classifier -> HMM/Viterbi -> coherent route.
FastAPI backend (Laptop 1) + evaluator UI (Laptop 2).

## Team
| Member | Area | Branch |
|---|---|---|
| M1 | Data, EDA, features | feat/data-features |
| M2 | Models, CV, metrics | feat/models-training |
| M3 | Bayesian opt + HMM/Viterbi | feat/tuning-hmm |
| M4 | API, UI, Folium, deployment | feat/api-ui |

## Setup
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Data goes in data/raw/ (not committed).

## Completed Member 1 and Member 2 work

- [Data pipeline, feature dictionary and reproduction commands](reports/member1/README.md)
- [Actual data-quality findings](reports/member1/DATA_QUALITY.md)
- [Local training and Bayesian optimisation instructions](reports/member2/LOCAL_TRAINING_README.md)
- [Five-model CV and student-testing results](reports/member2/local_training/RESULTS.md)
- [Selected classifier and loading metadata](artifacts/local_training/README.md)

The completed run compares Logistic Regression, KNN, Random Forest, Histogram Gradient Boosting and MLP with all five trip-aware outer folds. The selected MLP was tuned using ten Optuna TPE trials on training-trip CV. Student-test Top-1 is 89.07% on 14,215 aligned derived reference observations, with 74.19% label coverage. These metrics do not include HMM decoding or establish independently verified physical ground truth.

Install this component's dependencies with `python -m pip install -r requirements-training.txt`. Run `python -m pytest tests/models tests/data -q` for the regression suite. The map generator uses supplied road geometry instead of blocked background map tiles; Leaflet assets still need browser network access.

Raw and full processed datasets are not committed. Keep local data under `data/`, following the preparation instructions. The selected model is provided alongside the earlier artifact so teammates can explicitly choose the compatible version.

The KNN fitted cache is kept out of the public repository because it retains training examples. KNN code, configurations and measured results are included; its fitted cache can be regenerated or shared within the team.

Complete Member 2 source, model clarification, scoring API, test evidence and integration instructions are documented in [MEMBER2_HANDOFF.md](MEMBER2_HANDOFF.md).
