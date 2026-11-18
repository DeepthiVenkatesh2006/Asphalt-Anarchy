
# PathSnap - Map-Matching & Route Reconstruction (Asphalt-Anarchy)

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
