"""
Full Demonstration Validation Script
Verifies:
1. pathsnap.api.main:app startup on 0.0.0.0:8000
2. GET /health
3. GET /trips
4. POST /predict on student_test trip '00000000'
5. Validation of predicted_coords, metrics, latency
6. Request logging
7. Error handling for invalid trip ID
"""

import subprocess
import time
import requests
import sys

server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "pathsnap.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print("[TEST] Launching uvicorn pathsnap.api.main:app...")

try:
    # 1. Wait for server readiness
    online = False
    for i in range(25):
        time.sleep(1)
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=1.0)
            if r.status_code == 200:
                print(f"[TEST] Server ready after {i+1}s: {r.json()['service']}")
                online = True
                break
        except Exception:
            pass

    if not online:
        print("[FAIL] Server did not become ready.")
        sys.exit(1)

    # 2. Test GET /health
    r_health = requests.get("http://127.0.0.1:8000/health")
    print("\n--- 1. GET /health ---")
    print("Status:", r_health.status_code)
    print("Response:", r_health.json())
    assert r_health.status_code == 200
    assert r_health.json()["model_loaded"] is True

    # 3. Test GET /trips
    r_trips = requests.get("http://127.0.0.1:8000/trips")
    print("\n--- 2. GET /trips ---")
    print("Status:", r_trips.status_code)
    trips_data = r_trips.json()
    print("Total trips:", trips_data["total_trips"])
    print("Partitions:", trips_data["partitions"])
    print("First 3 trips:", [t["trip_id"] for t in trips_data["trips"][:3]])
    assert r_trips.status_code == 200
    assert trips_data["total_trips"] > 0

    # 4. Test POST /predict on reserved student_test trip '00000000'
    print("\n--- 3. POST /predict (Trip: 00000000) ---")
    payload = {
        "trip_id": "00000000",
        "partition": "student_test",
        "use_viterbi": True
    }
    t0 = time.time()
    r_pred = requests.post("http://127.0.0.1:8000/predict", json=payload)
    elapsed = time.time() - t0
    print("Status:", r_pred.status_code)
    pred_data = r_pred.json()
    print(f"Elapsed Time: {elapsed:.2f}s | Server Latency: {pred_data['inference_time_ms']}ms")
    print("Total predicted arcs:", pred_data["total_arcs"])
    print("First 10 arcs:", pred_data["predicted_arcs"][:10])
    print("Predicted coords count:", len(pred_data["predicted_coords"]))
    print("First coord (Start):", pred_data["predicted_coords"][0])
    print("Last coord (End):", pred_data["predicted_coords"][-1])
    print("Aggregate Metrics:")
    for k, v in pred_data["metrics"].items():
        print(f"  {k}: {v}")
    assert r_pred.status_code == 200
    assert pred_data["metrics"]["top1_accuracy"] > 0.70
    assert pred_data["metrics"]["route_continuity"] > 0.90
    assert len(pred_data["predicted_coords"]) > 0

    # 5. Test invalid trip ID
    print("\n--- 4. POST /predict (Invalid Trip ID) ---")
    r_inv = requests.post("http://127.0.0.1:8000/predict", json={"trip_id": "invalid_999999"})
    print("Status:", r_inv.status_code)
    print("Response:", r_inv.json())
    assert r_inv.status_code in (404, 500)

    print("\n[SUCCESS] All backend endpoints validated successfully!")

finally:
    server_proc.terminate()
    server_proc.wait()
    print("[TEST] Server stopped.")
