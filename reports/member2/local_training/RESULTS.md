# Completed local model training and tuning

Five model families were compared using all 5 outer GroupKFold splits on 80 training trips. KNN used a recorded fitting sample of complete observation groups; other models used all eligible fitting rows. Selection used training CV, before loading student-test labels.

## Training cross-validation

| model | observation_macro_f1_mean | observation_macro_f1_std | top1_accuracy_mean |
| --- | --- | --- | --- |
| logistic | 0.7227 | 0.0528 | 0.8301 |
| knn | 0.7215 | 0.0553 | 0.8275 |
| forest | 0.7445 | 0.0511 | 0.8364 |
| histboost | 0.7508 | 0.0490 | 0.8373 |
| mlp | 0.7508 | 0.0526 | 0.8379 |

Macro F1 is computed per trip over the union of predicted and reference road IDs, including unmatched -1, then averaged across trips within each fold. The table summarizes fold means.

## Student-testing results

| model | variant | top1_accuracy | observation_macro_f1 | candidate_log_loss | reference_coverage | candidate_miss_rate | predict_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | baseline | 0.8792 | 0.7375 | 0.4477 | 0.7419 | 0.0008 | 0.0021 |
| knn | baseline | 0.8790 | 0.7494 | 0.8811 | 0.7419 | 0.0008 | 2.9409 |
| forest | baseline | 0.8874 | 0.7657 | 0.3107 | 0.7419 | 0.0008 | 0.2736 |
| histboost | baseline | 0.8874 | 0.7688 | 0.2914 | 0.7419 | 0.0008 | 0.1858 |
| mlp | baseline | 0.8865 | 0.7698 | 0.2903 | 0.7419 | 0.0008 | 0.0102 |
| mlp | tuned | 0.8907 | 0.7731 | 0.2859 | 0.7419 | 0.0008 | 0.0160 |

Scores use confidently aligned, approximately derived labels. Candidate misses are counted in Top-1 and F1. Log loss is conditional on the reference road being in the candidate set; inspect log_loss_observation_coverage in the CSV. These are not scores against independently verified physical point truth, and are not HMM-decoded route results.

## Bayesian optimisation

Selected family: mlp. Completed TPE trials: 10. Search used 3 trip-aware training folds; tuned/untuned comparison used the same 5 folds.

Best searched parameters:

```json
{
  "hidden_layer_sizes": [
    64,
    32
  ],
  "alpha": 0.0006358358856676254,
  "learning_rate_init": 0.0021992715112745254,
  "max_iter": 35
}
```

Selected variant: tuned. An improvement is judged on training CV, not by student-test results.

## Fitting limits

| model | epochs | epoch_cap | reached_epoch_cap |
| --- | --- | --- | --- |
| mlp_baseline | 26 | 35 | False |
| mlp_tuned | 35 | 35 | True |
Reaching the configured epoch cap is a budget-limited fit; do not claim full convergence. The configuration was selected by measured training CV, not guaranteed optimisation convergence.

## Reproduction and evidence

- run_config.json records feature names, seed, parameters, versions and held-out trip IDs.
- fold_results.jsonl records fitting time, row count, coverage, metrics and fitting warnings for each comparison fold.
- bayesian_search.sqlite3 and bayesian_trials.csv retain actual optimisation history.
- input_checksums.json and preparation manifests identify the processed data used.
- model_final.joblib, model_metadata.json and feature_columns.json form the inference bundle.
- Epoch histories and plots exist only for MLP fits that produced them; no artificial curves were generated.
- Early stopping used internal row holdouts within fitting trips; the outer trip-wise validation remained untouched.
- This is model-selection CV, not nested CV. Member 3 still needs route continuity and HMM/Viterbi comparisons.
