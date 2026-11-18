# Full project audit

Goal: verify the complete implementation and produced documentation against the five original teacher PDFs, repair demonstrated defects, and generate reproducible evidence and corrected deliverables.

## Requirements and evidence
- [x] Read all five canonical PDFs and inventory the produced Markdown/Word documentation.
- [x] Trace data/features, candidate labels, all five training models, CV/tuning and artifact identity.
- [x] Adversarially test decoder gaps, missing candidates, restarts, projections, connector reconstruction and metric definitions.
- [x] Verify API schemas, lifecycle/configuration, UI variants, model-loading and input/reference separation.
- [x] Produce classifier-only versus HMM experiments on all ten student-test trips with point and sequence metrics, plots and real Folium maps.
- [x] Repair stale or incorrect reports/guides, including the Word study guide; produce one consolidated technical PDF with actual evidence and explicit limits.
- [x] Verify installation/runtime, all required artifacts, GitHub state; handoff archives remain separate from the source submission.
- [x] Re-run appropriate tests and browser checks, identify any externally unverified two-laptop requirement, and complete a requirement-by-requirement final audit.

Model fitting and tuning remain fixed to the existing training run. Further decoder/model comparisons use train/student_test only; evaluator labels/results must not guide changes. Historical teacher-trip system checks are retained as final-evaluation evidence, not selection evidence. A local rehearsal does not prove Windows/Mac network deployment.
