# PathSnap: technical report

Team: Asphalt-Anarchy. Date: 18 November 2026. This consolidated report follows the supplied dataset, training, submission and assessment PDFs. All numerical results below come from saved experiments; local verification is distinguished from cross-device demonstration.

## 1. Problem and system

Given noisy GPS observations, predict the supplied road segments travelled. The classifier scores nearby candidate roads; a graph-based HMM/Viterbi decoder chooses a coherent sequence. The API performs inference and separate reference evaluation on the server. A browser or Streamlit client displays returned predictions and aggregate metrics.

Pipeline: raw GPS and road files -> metric projection -> candidate features -> trained classifier -> graph transitions and Viterbi -> expanded road chunks -> server-side evaluation -> API and Leaflet UI.

| Member | Implemented responsibility |
| --- | --- |
| 1 | Data validation, EDA, reference alignment, candidates and ten features |
| 2 | Five-model comparison, grouped CV, Bayesian tuning, model export |
| 3 | Road graph, transition model, Viterbi and route reconstruction |
| 4 | FastAPI, client interfaces, map and deployment integration |

The model checksum is 84fab7ce6d3607346204e3d849caec8c630d1f5d9a2f08ca7cfa1ff132977e16. The trained weights were not changed during this audit. Correctness repairs were validated against synthetic graph/data cases; performance comparisons use student_test, not the evaluator partition.

## 2.2 Dataset understanding and exploratory analysis

| Partition | Trips | GPS observations | Purpose |
| --- | --- | --- | --- |
| train | 80 | 208,417 | Fitting and grouped CV |
| student_test | 10 | 19,160 | Development comparisons |
| evaluator_test | 10 | 19,674 | Final API evaluation only |

Every trip has .track (longitude, latitude, seconds), .nodes (longitude, latitude), .arcs (directed from/to node IDs), and a reference .route where supplied. Node and arc IDs are zero-based row positions within a trip. A route entry is a travelled arc, not a GPS observation; the two row counts need not match.

All 100 supplied trips passed strict input and reference validation; their IDs are disjoint across partitions. The original 90-trip EDA contains zero rejected trips or nonfinite input values. The processing code validates shape, coordinate ranges, strictly increasing timestamps and referenced IDs. It does not silently fill missing values, sort bad timestamps, delete GPS observations or shift coordinates to a background map.

A per-trip azimuthal equidistant projection supplies metre distances. At most eight complete road segments within 50 m are candidates for each observation. Graph connectors use all supplied roads, not only candidates. Gaps over 60 seconds split motion/decoder segments. Invalid direction and gap-adjacent speed are represented by explicit flags; a zero placeholder does not prove stationary motion.

| Development output | train | student_test |
| --- | --- | --- |
| Candidate rows | 1,084,832 | 113,639 |
| Eligible fitting-format candidate rows | 755,694 | 76,234 |
| Eligible observations for fitting-format rows | 143,533 | 14,203 |
| Aligned observations used for primary evaluation | 144,443 | 14,215 |
| Observations without candidates | 18,230 | 0 |

### Features and feature selection

| Feature | Meaning / reason |
| --- | --- |
| distance_m | Candidate proximity in metres |
| heading_delta_deg | Directed difference between motion and road bearing |
| projection_ratio | Position along the candidate, from 0 to 1 |
| distance_rank | Relative proximity among shortlisted candidates |
| segment_length_m | Road geometry scale |
| speed_mps | Previous metric displacement / elapsed time when valid |
| dt_s | Time since the preceding GPS observation |
| heading_valid | Whether the motion bearing is reliable |
| gap_before | Whether this observation follows a large time gap |
| zero_length_arc | Explicit degenerate-road indicator |

These are manually chosen geometric/motion/quality features, not the result of automatic feature elimination. IDs remain join keys. Reference IDs, target, alignment costs, raw projected coordinates and other diagnostic columns are not model inputs. StandardScaler is fitted only for Logistic Regression, KNN and MLP, inside each fitting fold. No learned imputation or feature log transformation is used.

### EDA interpretation

The trip-size histogram shows a right-skewed distribution: long trips can dominate row-weighted fitting. Whole-trip CV and per-trip Macro F1 address evaluation dependence, although most models still fit every eligible row. The road-count scatter shows varying candidate-search difficulty, motivating spatial indexing. The timestamp-gap plot spans several orders of magnitude, motivating motion-validity flags and sequence breaks. Only that plot uses a logarithmic y-axis; the stored feature values are unchanged.

The maximum training time gap is 153,064 s; the largest student-test gap is 210 s. Training trip 00000054 contributes 16,519 of the 18,230 missing-candidate observations, with median nearest-road distance about 84.2 m. The fixed 50 m radius is therefore an explicit coverage limitation, not a guarantee every GPS point can be matched.

![Actual development EDA plots](reports/member1/dataset_summary.png)

### Approximate reference labels and coverage

The reference arc sequence does not provide one road label per GPS point. A separate monotone spatial alignment derives approximate labels, preserving route occurrences and splitting infeasible chains. Ambiguous, distant, restarted and unmatched labels remain in the ledger but are excluded from the primary aligned-label subset. Fitting also excludes candidate misses; evaluation retains them as incorrect predictions. On student_test, primary accuracy/F1 use 14,215 of 19,160 observations (74.19%). These are approximate references, not independently surveyed point truth.

## 3. Model experimentation and comparison

The authoritative experiment fits all five algorithm families on 80 training trips, evaluates every outer GroupKFold split, and holds each training trip out once. No observation from a held-out trip enters that fold's fitting/scaler data. MLP and HistGradientBoosting use internal row-level early-stopping holdouts within fitting trips; these are distinct from the outer trip-wise validation.

| Model | CV Macro F1 (SD) | Student Top-1 | Student Macro F1 | Direct continuity | Log loss | Predict seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7227 +/- 0.0528 | 87.92% | 0.7375 | 0.8593 | 0.4477 | 0.0021 |
| Distance-weighted KNN | 0.7215 +/- 0.0553 | 87.90% | 0.7494 | 0.8520 | 0.8811 | 2.9409 |
| Random Forest | 0.7445 +/- 0.0511 | 88.74% | 0.7657 | 0.8734 | 0.3107 | 0.2736 |
| HistGradientBoosting | 0.7508 +/- 0.0490 | 88.74% | 0.7688 | 0.8684 | 0.2914 | 0.1858 |
| MLP | 0.7508 +/- 0.0526 | 88.65% | 0.7698 | 0.8675 | 0.2903 | 0.0102 |

Student scores use the same 74.19% aligned-reference coverage. Candidate log loss is categorical loss after positive-score clipping and within-observation normalisation, conditional on the reference being in the candidate set (99.9156% conditional coverage). Direct continuity measures same-road forward motion or directly adjacent chosen road states before connector insertion; it is not the expanded graph-continuity metric used later. Prediction seconds cover all 113,639 student candidates on the measured Mac and exclude feature extraction/decoding. KNN fits 59,961 candidate rows in its budgeted sample; other models fit 755,694.

![Five-fold model comparison](reports/member2/local_training/model_comparison.png)

### 3.2 Questions for each model: suitability, configuration, loss and actual results

### Logistic Regression (Linear/probabilistic)

Suitability and limitations: A fast interpretable baseline for geometric and motion clues. Linear decision boundaries can miss interactions between heading, distance and quality flags.

Initial configuration: C=1; lbfgs; max_iter=500; seed=42; StandardScaler. Fitting objective: Candidate binary log loss with L2 regularisation.

Measured student-test Top-1 87.92%, observation Macro F1 0.7375, macro precision 0.7663, macro recall 0.7216, conditional log loss 0.4477; final fitting 0.216 s, candidate prediction 0.0021 s. CV Top-1 83.01%, CV F1 0.7227 with SD 0.0528.

Over/underfitting evidence: CV and student performance are measured, but a higher student score is not proof against overfitting because trips differ in difficulty. A matched training-set point-accuracy curve was not retained for this model, so no unsupported train-versus-test conclusion is made. Complexity controls and grouped validation reduce risk rather than proving its absence.

### Distance-weighted KNN (Instance-based)

Suitability and limitations: Similar candidate clues can have similar labels, without assuming a linear boundary. Its stored examples make prediction and model storage more expensive; the fitting budget was capped at 10,000 complete observation groups.

Initial configuration: k=9; distance weights; KD-tree; StandardScaler; sample seed=42. Fitting objective: No gradient-based fitting loss: similarity-weighted neighbour voting.

Measured student-test Top-1 87.90%, observation Macro F1 0.7494, macro precision 0.7780, macro recall 0.7333, conditional log loss 0.8811; final fitting 0.014 s, candidate prediction 2.9409 s. CV Top-1 82.75%, CV F1 0.7215 with SD 0.0553.

Over/underfitting evidence: CV and student performance are measured, but a higher student score is not proof against overfitting because trips differ in difficulty. A matched training-set point-accuracy curve was not retained for this model, so no unsupported train-versus-test conclusion is made. Complexity controls and grouped validation reduce risk rather than proving its absence.

### Random Forest (Tree ensemble)

Suitability and limitations: Captures nonlinear feature interactions and reduces single-tree variance. Tree depth/leaf controls constrain complexity, but probability calibration and inference cost still need checking.

Initial configuration: 80 trees; depth=12; leaf minimum=3; sqrt feature sampling; seed=42. Fitting objective: Gini-impurity split criterion; averaged tree probabilities.

Measured student-test Top-1 88.74%, observation Macro F1 0.7657, macro precision 0.7921, macro recall 0.7513, conditional log loss 0.3107; final fitting 27.277 s, candidate prediction 0.2736 s. CV Top-1 83.64%, CV F1 0.7445 with SD 0.0511.

Over/underfitting evidence: CV and student performance are measured, but a higher student score is not proof against overfitting because trips differ in difficulty. A matched training-set point-accuracy curve was not retained for this model, so no unsupported train-versus-test conclusion is made. Complexity controls and grouped validation reduce risk rather than proving its absence.

### HistGradientBoosting (Boosting)

Suitability and limitations: Histogram-based trees efficiently model nonlinear tabular features. Early stopping limits iterations, but its internal row holdout is less independent than the outer trip-wise validation.

Initial configuration: 120 iterations; learning rate=.08; 31 leaves; early stopping; seed=42. Fitting objective: Candidate binary log loss, improved by successive trees.

Measured student-test Top-1 88.74%, observation Macro F1 0.7688, macro precision 0.7936, macro recall 0.7547, conditional log loss 0.2914; final fitting 4.410 s, candidate prediction 0.1858 s. CV Top-1 83.73%, CV F1 0.7508 with SD 0.0490.

Over/underfitting evidence: CV and student performance are measured, but a higher student score is not proof against overfitting because trips differ in difficulty. A matched training-set point-accuracy curve was not retained for this model, so no unsupported train-versus-test conclusion is made. Complexity controls and grouped validation reduce risk rather than proving its absence.

### MLP (Neural network)

Suitability and limitations: Learns flexible combinations of scaled distance, direction and quality clues. It is sensitive to scaling, optimisation settings and fitting budget; an epoch cap is not evidence of convergence.

Initial configuration: Initial layers=(32,16); Adam; ReLU; alpha=.0001; lr=.001; batch=512; cap=35; seed=42. Fitting objective: Candidate binary cross-entropy plus L2 regularisation.

Measured student-test Top-1 88.65%, observation Macro F1 0.7698, macro precision 0.7954, macro recall 0.7552, conditional log loss 0.2903; final fitting 8.798 s, candidate prediction 0.0102 s. CV Top-1 83.79%, CV F1 0.7508 with SD 0.0526.

Over/underfitting evidence: the baseline stopped after 26 of 35 allowed epochs; the tuned fit reached 35. Actual training loss and internal validation accuracy histories are retained. The tuned epoch cap limits convergence claims. Internal row-holdout accuracy is not an independent trip-validation score.

![Actual candidate confusion matrices for all fitted variants](reports/audit/model_metrics/candidate_confusion_matrices.png)

These confusion matrices use binary candidate targets on eligible derived-label rows at threshold >0.5. They diagnose candidate classification, not per-observation road-ID accuracy. No artificial loss histories were created for algorithms without epoch histories.

### 3.3 Selection and preliminary experiments

MLP had the highest recorded mean CV observation Macro F1, but was nearly tied with HistGradientBoosting; this is not evidence of a statistically significant victory. Its rapid prediction and compatibility with candidate probabilities also support deployment. The older reports/member2/model_comparison files contain a preliminary single grouped fold with different configurations and candidate-binary F1; they are retained as historical evidence and are not interchangeable with the final five-fold observation-level table. Preliminary RF used 100 trees, HistGradientBoosting 150 iterations, and MLP layers (64,32) with a 25-epoch cap. No standalone DecisionTreeClassifier or standard GradientBoostingClassifier is claimed as trained in the final run.

## 4. Bayesian hyperparameter optimisation

Only the selected MLP family was tuned. Optuna TPE used seed 42, three startup trials, ten completed trials and three group-aware training folds per trial. Search space: architectures 32, 32/16 or 64/32; alpha 1e-5 to .01 (log scale); learning rate .0003 to .005 (log scale); maximum 35 epochs. The objective was mean aligned-label observation Macro F1. The best trial was compared against the baseline on the same five outer folds; the final variant was chosen before student-test labels were loaded. This is model-selection CV, not nested CV.

Best settings: layers (64,32), alpha 0.0006358358856676254, learning rate 0.0021992715112745254, 35 epochs. Other fitting settings include Adam, ReLU, batch 512 and early stopping. The final model embeds its fitted StandardScaler.

| Metric | Untuned MLP | Tuned MLP |
| --- | --- | --- |
| CV Macro F1 | 0.7508 | 0.7552 |
| CV Top-1 | 83.79% | 83.71% |
| Student Top-1 | 88.65% | 89.07% |
| Student Macro F1 | 0.7698 | 0.7731 |
| Student log loss | 0.2903 | 0.2859 |
| Student prediction seconds | 0.0102 | 0.0160 |

Tuning modestly improved the declared CV F1 objective. CV Top-1 and CV log loss did not improve; student-test Top-1/F1/log loss did. Do not claim every metric improved or that the fit fully converged. bayesian_trials.csv, the SQLite study, fold_results.jsonl and configurations retain the actual search and computational cost.

![Actual tuned MLP loss and internal validation accuracy](reports/member2/local_training/mlp_tuned_history_details.png)

## 5. HMM and Viterbi route reconstruction

Hidden states are candidate directed road segments for each observation. Candidate-to-road keys retain trip_id, obs_idx, candidate_idx and arc_id. Clipped and normalised positive classifier scores act as discriminative emission potentials; they are not claimed to be calibrated generative P(GPS|road) likelihoods.

Transition potential is exp(-abs(network distance - observed geodesic distance)/beta), beta=50 m. Network distance includes the remaining previous arc, the shortest directed connector, and the travelled part of the current arc. Forward movement on one arc uses the difference in projection fractions. Returning to a smaller fraction requires a real directed return path. Impossible transitions have zero potential; connector searches use a bounded distance cutoff. Transition rows are normalised over current candidates. The initial state prior is uniform.

Viterbi recurrence: delta[t,j] = log E[t,j] + max_i(delta[t-1,i] + log T[t-1,i,j]). Backpointers recover the globally best path within each continuous chunk. Missing candidates are -1, not invented road 0. Large gaps and impossible chains split/restart chunks. Expanded routes preserve repeated traversals and intermediate connector arcs; no connector bridges a recorded break.

Decoder parameters remain the documented fixed settings; no evaluator-based tuning was performed. The required development experiment compares classifier-only and HMM assignments on all ten student-test trips with identical reference coverage and the same frozen trained model.

| Variant | Point accuracy | Observation F1 | Road-set F1 | Sequence similarity | Graph continuity | Mean inference ms |
| --- | --- | --- | --- | --- | --- | --- |
| classifier_only | 89.07% | 0.7731 | 0.7162 | 0.4360 | 97.12% | 584.8 |
| hmm | 89.53% | 0.8098 | 0.7888 | 0.5423 | 99.92% | 7364.8 |

Classifier Top-1 stays 89.07% because classifier scores do not change. Returned HMM point accuracy is a different metric: 89.53%. Observation F1 rises from .7731 to .8098, while ordered sequence similarity rises from .4360 to .5423. Road-set F1 ignores order; ordered similarity uses SequenceMatcher with autojunk disabled to retain frequent road IDs. Graph continuity includes valid connector paths; gaps/missing points remain breaks. Runtime increases substantially, and timings exclude server-side reference evaluation and network transfer.

The improvement is not uniform: student trip 00000007 drops in point accuracy from 92.66% to 91.80%, while sequence similarity improves from .5525 to .6221. Trip 00000000 also has a small observation-F1 decline. All per-trip results are retained, rather than showing only successful examples.

![Measured classifier-versus-HMM comparison](reports/decoder/student_test/decoder_comparison.png)

## 6. Geographical visualisation using Folium

The mandatory development visualisation is generated with Folium from actual student-test trip 00000007. Red GPS observations, green development reference roads and dark-blue predicted roads are separate toggleable layers. Road segments are drawn independently rather than joining gaps with fabricated straight lines. Reference geometry is included only in this development report map, never in an evaluator response. The interactive HTML and generator are included in the repository.

![Actual Folium development map: student-test trip 00000007](reports/final/assets/student_test_00000007_folium.png)

The live client uses native Leaflet with locally bundled JavaScript/CSS, grey supplied roads, green predictions, blue raw GPS and projected first/last matched observation markers. Street tiles are optional contextual geometry and may differ from the supplied graph. The application never shifts predictions cosmetically to match a basemap.

## 7. API, UI and deployment

FastAPI exposes /health, /trips, /predict, /evaluate and /map/{trip_id}, with /api/v1 aliases. Evaluation mode lists only the ten evaluator_test trips and freezes final inference controls. Startup verifies model hashes/version, input schemas and server-side references. An empty/invalid evaluation catalogue is not reported as ready. Invalid IDs/radii/partitions are rejected, and missing trips return 404. Reference validation/evaluation is separated from model features; JSON and evaluator maps contain no raw reference route.

The Streamlit and browser /ui/ clients share this API. Both clear old results before retries/context changes; nullable/subset metrics show N/A and label coverage. Asynchronous browser responses are tied to request generations. Predicted point assignments and expanded route chunks are separate. UI latency is inference time, not end-to-end network latency.

Verified achievements: clean dependency installation, 91 passing Python regression tests, four passing JavaScript regressions, real model/decoder processing and rendered route maps. Deployment follows the assignment architecture: Laptop 1 serves the trained model/API and Laptop 2 runs the evaluator client. Capture the receiving-server request logs during the final two-laptop demonstration.

Laptop 1: python scripts/run_demo.py --role api --data-root STUDENT_DATA_ROOT. Laptop 2: python scripts/run_demo.py --role ui --api-url http://LAPTOP1_IP:8000. Join the same network, confirm /health from Laptop 2, then open http://localhost:8501 there. Use Python >=3.12 and requirements-runtime-lock.txt on both machines.

## 8. Audit, limitations, reproducibility and integrity

Audit repairs cover stale UI success, omitted evaluator trips, unsafe error HTML, nullable metrics, invalid readiness, misleading configuration, model integrity/export compatibility, return-loop reconstruction, break preservation and order-sensitive scoring. The current code records separate classifier and decoded metrics. See reports/audit for regression outputs, input/model checksums, extra model metrics and the requirement matrix.

Limitations remain explicit: reference labels are approximate with partial coverage; model-selection CV is not nested; early stopping uses correlated fitting-row holdouts; the tuned MLP reached its epoch cap; candidate coverage can fail at a fixed radius; HMM adds computation and can hurt individual trips; graph paths and basemaps do not prove the true physical road. Cross-device hardware/network evidence is not inferred from unit tests.

Reproduce data preparation with src.features.build_features on train and student_test using --include-labels, writing fresh directories. Reproduce the five-model/Optuna run with src.models.local_training and a fresh output directory. scripts/compare_decoder.py uses student_test only. scripts/audit_model_metrics.py verifies every saved model reproduces its original primary metrics; a private KNN cache can be regenerated from training data. The final inference bundle contains model_final.joblib, feature_columns.json, model_metadata.json and artifact_manifest.json; scaler preprocessing is embedded.

AI assistance was used for parts of implementation, testing and documentation. Numerical claims are backed by actual generated files; AI text is not substituted for experimentation. Team members should review and understand these results. No claim of unaided work or independently surveyed reference truth is made.

Teacher sources: Main Guidelines (2 pages); Dataset Exploration (4 pages); Models and Training Guidelines (6 pages); Deliverables (7 pages); Assessment and Plagiarism Policy (3 pages). Key evidence: reports/member2/local_training; artifacts/local_training; reports/decoder/student_test; reports/audit; reports/final/assets. Source repository: https://github.com/DeepthiVenkatesh2006/Asphalt-Anarchy, integration branch feature/demo-integration-ready.
