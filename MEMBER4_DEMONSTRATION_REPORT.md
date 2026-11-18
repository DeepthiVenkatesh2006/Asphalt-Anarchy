# Integrated demonstration: verified achievements

The project combines the verified tuned model bundle, graph-aware HMM/Viterbi, FastAPI, separate classifier/route metrics and native Leaflet clients.

- Clean environment: 91 Python tests and four JavaScript regressions pass.
- Input validation covers all 100 supplied trips across the three partitions.
- Ten-trip development comparison improves observation F1 from 0.7731 to 0.8098 and graph continuity from 97.12% to 99.92%.
- The consolidated technical PDF includes model comparison, tuning, decoder results and the Folium development map.
- The deployment guide provides the required Laptop 1 API / Laptop 2 evaluator configuration.

Evidence: `reports/audit`, `reports/decoder/student_test` and `reports/final/PathSnap_Technical_Report.pdf`. Capture screenshots and actual receiving-server logs during the final two-laptop demonstration. Historical teacher-trip outputs in `reports/demo` belong to the earlier integration revision; current development metrics are in the paths above.
