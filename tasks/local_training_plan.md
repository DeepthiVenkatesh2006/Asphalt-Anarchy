# Local model training and Bayesian tuning

Authorized scope: complete all model training and tuning on this laptop. Preserve existing teammate scripts and artifacts.

1. Test candidate-to-observation metrics, absent candidates, group splits, batch prediction, and tuning/factory interfaces.
2. Add a separate local runner. Compare Logistic Regression, KNN, Random Forest, HistGradientBoosting and MLP with all five GroupKFold splits; use only training trips for selection. KNN fitting uses a recorded, fixed-seed sample of complete observation candidate groups for bounded runtime.
3. Select by mean per-trip observation Macro F1 on confidently aligned reference labels, retaining candidate misses in that denominator. Report Top-1, coverage, conditional normalized candidate log loss, candidate binary metrics, times and per-fold variation. All labels remain approximate.
4. Tune the selected model with Optuna TPE using group-aware training CV and saved SQLite trial history. Reevaluate tuned and untuned with identical folds and evaluate student testing only after choices are frozen. A measured failure to improve is a valid result.
5. Export fitted model, schema, preparation settings and metadata; generate comparison plots, tables, tuning records and training histories where actually available. Review code and validate saved-artifact predictions.

Tests: .venv/bin/python -m pytest tests/models tests/data -q. Artifacts and reports use dedicated local_training directories. No API/UI/decoder work is included. Do not overwrite earlier model_final.joblib or claim old single-fold results are five-fold CV.
