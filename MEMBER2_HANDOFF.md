> Current audit update: the scorer now checks artifact hashes and exact scikit-learn compatibility. Canonical training writes the manifest automatically. Use the pinned runtime, current source/tests and consolidated PDF; earlier attached test counts refer to their historical package snapshots.

# Complete Member 2 implementation and results

This handoff contains the completed local training run, source, tests, inference pipeline and Member 3 scoring interface. It contains no raw/processed dataset files or virtual environment. Extract it at a repository root or merge its directories into your own branch without discarding your changes.

## Exact model clarification

| Algorithm | Trained in the completed local run? |
|---|---|
| LogisticRegression | Yes, all five comparison folds and final fitting |
| KNeighborsClassifier | Yes, all five comparison folds and final fitting; fitting uses a recorded fixed-seed sample of complete candidate groups |
| DecisionTreeClassifier | No |
| RandomForestClassifier | Yes, all five comparison folds and final fitting |
| HistGradientBoostingClassifier | Yes, all five comparison folds and final fitting |
| GradientBoostingClassifier | No; HistGradientBoostingClassifier is a different estimator |
| MLPClassifier | Yes, all five comparison folds, tuning and final fitting |

The five actually trained families are Logistic Regression, KNN, Random Forest, Histogram Gradient Boosting and MLP. Do not claim the exact list containing Decision Tree and GradientBoostingClassifier was completed. The supplied assignment examples were not a mandatory fixed list, but any later team/organiser requirement for those exact estimators would require additional experiments.

Optuna TPE tuned only the CV-selected MLP, using ten trials on three group-aware training folds. The best searched variant was compared with the baseline on the same five comparison folds. Tuning settings included layer sizes, alpha and learning_rate_init. All configurations used max_iter=35. Results and warnings are actual recorded outputs.

## Evaluation and model choice

All 80 training trips were used for fitting/cross-validation. Student-test labels were loaded after model and parameter selection was frozen. Final comparison used the separate ten student-test trips; no evaluator-testing data was available or used.

Selected model: tuned StandardScaler + MLP pipeline. Student-test Top-1 is 0.8906788604 (89.07%) and per-trip observation Macro F1 is 0.7731251852. Evaluation covers 14,215 confidently aligned derived reference observations out of 19,160 GPS observations (74.19% label coverage). Candidate misses remain in that accuracy/F1 denominator. Reference labels are approximate, not independently verified physical point truth. Conditional candidate log loss is 0.2859007220; its coverage is recorded separately. Candidate binary metrics differ from observation metrics.

The final tuned fit reached its configured 35-epoch cap. Do not claim full convergence. Route continuity and HMM/Viterbi comparisons remain Member 3 work.

## Source map

- src/models/local_training.py: five-model factories, grouped sampling, five-fold comparison, metrics, Optuna tuning, final fitting and model export.
- src/models/training_report.py: final tables, student-test/tuning charts, actual epoch histories, and fitting diagnostics.
- src/models/scoring.py: model loading, score_candidates interface and real-trip example CLI.
- src/models/benchmark_models.py: retained legacy benchmark; it used one group holdout and different metric conventions. It is not the authoritative completed five-fold run.
- tests/models/: all Member 2 regression tests, requiring no external dataset.

## Setup and reproduction

Use a separate virtual environment with Python 3.10 or later. Exact versions used for the actual run are in run_config.json and requirements-training-lock.txt. The lock file records that run environment; compatible wheels may not exist for every Python/platform combination.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt
python -m pytest tests/models -q
```

The existing Member2_Processed_Data.zip should extract into a directory containing train/ and student_test/. Point --data-root at that directory. Do not move it into this ZIP or commit it to the public repo.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u -m src.models.local_training \
  --data-root /path/to/extracted/processed-data \
  --output-dir reports/member2/new_training_run \
  --folds 5 --tuning-folds 3 --trials 10 --seed 42
python -m src.models.training_report reports/member2/new_training_run
```

Choose a fresh output directory. The supplied local_training directory is the completed evidence and should not be overwritten. run_config.json records seeds, all actual model parameters, held-out trip IDs and versions. configs/member2_models.json and configs/member2_tuning.json provide machine-readable model and search-space snapshots. Input checksums and copied processing manifests identify the source data without embedding it. bayesian_search.sqlite3 preserves the actual study. Models are saved with joblib; the scaler is already inside the final pipeline, so no separately fitted scaler file is required.

## Member 3 loading and scoring

Use only trusted team-generated joblib files. Load once and reuse the scorer for requests:

```python
import pandas as pd
from src.models.scoring import CandidateScorer

scorer = CandidateScorer.from_directory('artifacts/local_training')
candidates = pd.read_parquet('/existing/student_test/candidates.parquet',
                             filters=[('trip_id', '==', '00000000')])
scores = scorer.score_candidates(candidates)
```

Input: a DataFrame with trip_id, obs_idx, candidate_idx, arc_id and all ten features listed in feature_columns.json. Other columns are ignored; target/reference/alignment columns are never model inputs. Data must be prepared with the recorded radius/count/gap settings in model_metadata.json.

Output: the same row order/index and four identity columns, plus candidate_score (raw positive-class classifier probability) and candidate_probability (scores clipped at 1e-12 and normalized within each observation). Both columns are in [0,1], and candidate_probability sums to one for each observation. Normalize the complete candidate set together; the internal prediction batches are reassembled before normalization.

An empty input returns an empty table. Observations with no candidates remain in Member 1's observations ledger; the scorer cannot invent a road for them. Use (trip_id, arc_id) for candidate-to-road mapping. HMM/Viterbi consumes the candidate scores, graph and timing information separately.

## Real-trip example

The example was executed on student-test trip 00000000 using the final saved model and no reference labels. Output is in reports/member2/local_training/member3_handoff/. The CSV contains scores for the first five observations only; the summary records the total full-trip candidate/observation counts. It is a prediction preview, not a decoded route or dataset copy.

```bash
python -m src.models.scoring \
  --artifact-dir artifacts/local_training \
  --candidates /existing/student_test/candidates.parquet \
  --trip-id 00000000 --preview-observations 5 \
  --output-dir reports/member2/my_scoring_example
```

## Artifacts and integrity

artifacts/local_training/ contains the selected model, feature schema, model metadata, selection record, verification and fitting diagnostics. artifact_manifest.json records sizes and SHA256 checksums for those inference files. reports/member2/local_training/models/ retains the other saved baseline/tuned models except the KNN fitted cache, which retains fitted training examples and is intentionally excluded from this data-free handoff. The KNN cache can be regenerated using the supplied code and your existing dataset.

Test execution output, machine-readable JUnit results, command and return status are under reports/member2/local_training/test_execution/. Package-wide file checksums and exact Git branch/source commit are in PACKAGE_MANIFEST.json at the ZIP root. The experiment was originally published in commit d93f91b0c919882a4c1b831b1bb4a6f828d52ff5 on feature/m1-m2-training-complete. The package manifest identifies the later commit adding this scoring interface and handoff documentation.
