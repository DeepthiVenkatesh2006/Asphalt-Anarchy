# Integration audit and verified improvements

The audit checked the five assignment PDFs against the implemented data, training, decoder, API, UI and report pipeline.

| Area | Completed work and evidence |
| --- | --- |
| Data/features | Strict input validation; 100 trips checked; ten feature columns; reference labels excluded from prediction inputs. See validation_summary.json. |
| Training | Five classifier families, trip-grouped cross-validation and ten completed Optuna trials. Saved-model primary metrics reproduced; see model_metrics and reports/member2/local_training. |
| Decoder | Directed return loops, repeated traversals, gap boundaries and missing candidates preserved; sequence scoring retains frequent road IDs. |
| API | Validated YAML settings, model hashes/version, evaluation-only catalogue and server-side reference readiness. |
| UI/maps | Stale responses/results cleared, nullable metrics handled, responsive layout, native Leaflet and bounded display-map reuse. |
| Experiments | All ten student-test trips compared with and without HMM; observation F1 .7731 to .8098, ordered similarity .4360 to .5423. Per-trip outcomes retained. |
| Reports | Consolidated PDF, Folium development map and two-laptop deployment guide. |
| Tests | Clean installation, 91 Python tests passed; four JavaScript tests passed. See captured test outputs. |

Final hardware demonstration: run Laptop 1 as API/model host and Laptop 2 as evaluator client using DEMO_READY.md, recording actual receiving-server logs. Physical two-laptop success is established by that session, separately from automated regression checks.

Metrics retain their actual coverage and definitions. HMM improves aggregate route quality while individual trips can differ; the technical report preserves these details.
