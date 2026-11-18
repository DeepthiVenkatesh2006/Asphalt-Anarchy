# Local training and Bayesian tuning

This is a separate reproducible experiment run. It preserves the older benchmark and saved artifact. It does not implement the route decoder, API or UI.

## Run

```bash
python -m pip install -r requirements-training.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u -m src.models.local_training \
  --data-root data/processed/v1 \
  --output-dir reports/member2/local_training \
  --folds 5 --tuning-folds 3 --trials 10
```

Use a fresh output directory for each independent run. A `COMPLETE` marker means all comparison, tuning, final fitting and export stages finished. Results are checkpointed after each validation fold and after each completed model; Optuna records every trial in SQLite.

## What is compared

- Logistic Regression with fold-fitted StandardScaler.
- Distance-weighted KNN with fold-fitted StandardScaler and a KD-tree. For bounded runtime, fitting uses at most 10,000 complete observation candidate groups, balanced by training-trip quotas, with fixed seed 42. Actual candidate-row fitting counts are recorded; other models use all eligible fold-training rows. This is a practical-budget comparison, not identical fitting-row budgets.
- Random Forest with 80 trees and depth 12 initially.
- Histogram Gradient Boosting with up to 120 boosting iterations initially.
- MLP with initial layers (32, 16), early stopping and a 35-epoch cap.

All five outer GroupKFold splits are evaluated. No trip crosses the fitting and validation partitions. All learned preprocessing is fitted within each training fold. MLP/HGB internal early stopping uses row-level holdouts from within fitting trips; the outer trip validation remains untouched. Probability predictions are batched and CPU thread counts are bounded to reduce oversubscription. The models are fitted sequentially, so the laptop does not run five memory-heavy jobs at once.

## Model selection and Bayesian optimisation

The initial model family is selected by mean cross-validation observation Macro F1 on the training partition. Optuna's TPE sampler runs ten trials with three trip-aware training folds. The best searched configuration is then compared against the original model on the same five comparison folds. The selected final variant is frozen by training CV before the student-testing labels are loaded. If the searched variant does not improve five-fold CV, the baseline is retained; tuning is not guaranteed to improve performance.

This is selection/tuning CV, not an unbiased nested-CV estimate. Student-testing is used after selection for a measured final comparison. No evaluator-testing data is used.

## Metrics and limitations

The primary evaluation uses reference observations marked `aligned` by Member 1's approximate reference alignment. Candidate misses remain in the denominator; missing predictions use road ID -1. Label coverage is reported and the metrics are **not** full-trajectory verified physical accuracy.

Observation Macro F1 is computed separately for each trip, over the union of predicted and reference arc IDs (including unmatched -1), then averaged over trips. The result summaries average those values over the five held-out folds, and report fold standard deviation. Top-1 accuracy compares the highest-scoring candidate to the derived reference road.

Categorical candidate log loss is calculated by clipping positive-class scores to 1e-12 and normalizing over the candidates of each observation. It is conditional on the derived reference being present; log-loss coverage and candidate miss rate are separately reported. Candidate binary Macro F1/log loss are secondary diagnostics on the eligible candidate table. These are different from observation/route metrics and must not be substituted for them.

Member 3 still needs to add classifier-plus-decoder route continuity and sequence evaluation; classifier comparison does not complete the HMM requirement.

## Output files

Within `reports/member2/local_training/`:

- `cv_comparison.csv`: five model families, mean and standard deviation over all five folds.
- `fold_results.jsonl`: held-out trip IDs, fitting row count, timings, metrics and fold-fit warnings.
- `run_config.json`: seeds, configurations, folds, versions and feature order.
- `bayesian_search.sqlite3` and `bayesian_trials.csv`: persistent optimisation study and completed trial details.
- `selection.json`: baseline/tuned CV comparison and the frozen selection decision.
- `student_test_results.csv`: all five baselines plus the tuned model, with coverage and timings.
- `models/`: separately saved full-fitting baseline/tuned artifacts.
- `model_final.joblib`, `model_metadata.json`, `feature_columns.json`: selected inference bundle.
- Preparation manifests: exact dataset processing settings and per-trip projections.
- MLP history JSON and plots: actual final-fit epoch histories, where available. No artificial curves for non-epoch models.

Do not overwrite the team's top-level champion until the team has reviewed this run's results and agreed the new metric definitions. Model weights remain excluded from ordinary Git commits and can be shared as a local artifact bundle.

## Generate the final report figures

After the COMPLETE marker exists, run `python -m src.models.training_report reports/member2/local_training`. This generates RESULTS.md, student-test Macro F1/log-loss/timing comparisons, actual Bayesian search history and real MLP training/validation-accuracy plots. Validation accuracy in these histories is candidate-level internal early-stopping accuracy, not observation route accuracy.
