# Selected model from the local five-model comparison

This directory contains the selected tuned StandardScaler + MLP pipeline and its exact feature/preparation metadata. It preserves the older top-level artifact.

Load model_final.joblib with joblib.load. Pass the ten features in feature_columns.json in exactly that order; predict_proba(X)[:, 1] provides candidate scores. Preserve trip_id, obs_idx, candidate_idx and arc_id outside model inputs. Normalize scores within an observation if categorical candidate probabilities are required. Handle empty candidates explicitly.

Student-test Top-1 is 89.07% on 14,215 aligned derived reference observations (74.19% label coverage). These are approximate labels, not independently verified physical point truth. The tuned final fit reached the configured 35-epoch cap.

Actual comparison results, all five baseline artifacts, Optuna trial history, charts and reproduction metadata are in ../../reports/member2/local_training/. Raw and processed datasets are supplied separately.
