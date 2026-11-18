# Member 4: demonstration script

Use `DEMO_READY.md` for commands. Read `reports/final/PathSnap_Technical_Report.pdf` for measured results. Older fixed demonstration percentages are superseded.

1. Show the API `/health`: model loaded, actual available partitions, validated references and evaluation readiness.
2. On the other device, open the client and select one of the ten `evaluator_test` trips. Do not use a training trip as evidence of held-out performance.
3. Click Predict/Run Evaluation, and show the matching incoming request in the API terminal on the receiving machine.
4. Explain classifier Top-1 separately from decoded point accuracy, label coverage, road-set F1 and order-sensitive sequence similarity. Read the actual returned values rather than memorised earlier percentages.
5. Show native Leaflet: supplied roads grey, predictions green, raw noisy GPS blue, matched first/last observation markers. Background streets are contextual and can differ from the supplied dataset; no coordinates are shifted cosmetically.
6. Explain that references remain server-side. The separate Folium development map may display student/training reference roads for the report; the evaluator map does not.

Use Laptop 1 for the model/API and Laptop 2 for the evaluator interface. Follow the commands in DEMO_READY.md and capture the receiving-server logs during the demonstration.
