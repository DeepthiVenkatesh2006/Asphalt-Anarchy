# PathSnap Hackathon — Member 4 Complete Demonstration Guide

**Role:** Member 4 – API, UI and Deployment  
**Project:** PathSnap – GPS Trajectory Map Matching System  
**Team:** Team 7 (Asphalt-Anarchy)  

---

## 1. Member 4 Core Responsibilities
As Member 4, your primary role is the **system integrator, deployment lead, and user interface engineer**. You are responsible for:
1. Developing the **FastAPI backend engine** that bridges Member 1's data loader & candidate features, Member 2's trained champion ML classifier, and Member 3's road graph and HMM/Viterbi sequence decoder.
2. Building the **Professional Evaluator Interface** (Streamlit dashboard and responsive Web UI) for live faculty demonstration.
3. Generating **Interactive Folium Route Visualizations** that snap noisy GPS coordinates onto road networks.
4. Setting up the **Two-Laptop Demonstration Architecture** over a local network.
5. Strictly preventing **Ground Truth Data Leakage** to the frontend client.

---

## 2. Technology Stack Justification (Why We Chose Them)

### Why FastAPI?
- **High Performance & Asynchronous:** Built on Starlette and Pydantic, providing sub-millisecond route handling and automatic data validation.
- **Auto-Generated Documentation:** Automatically produces interactive OpenAPI Swagger docs (`/docs`) and ReDoc (`/redoc`) for immediate evaluator review.
- **Clean Service Architecture:** Allows clean decoupling between request handlers, preprocessing pipelines, model scoring, and sequence decoding.

### Why Streamlit?
- **Rapid, Interactive Dashboarding:** Native Python framework allowing real-time parameter tweaking (e.g. search radius, Viterbi toggle) without complex build tools.
- **Reactive State Management:** Seamlessly connects to Folium maps via `streamlit-folium` and provides instant metric scorecard updates.
- **Intuitive for Evaluators:** Clean UI components (`st.selectbox`, `st.metric`, `st.spinner`) designed specifically for live data demonstrations.

### Why Folium?
- **Interactive Geospatial Visualization:** Powered by Leaflet.js, offering zoom, pan, and layer toggles for real-world latitude/longitude coordinates.
- **Multi-Layer Support:** Allows side-by-side comparison of Raw GPS observation points, Road Network candidates, and the Predicted Snapped Route.
- **Strict Coordinate Precision:** Accurately renders polylines and distinct start/end markers with hover tooltips and metadata popups.

---

## 3. API Endpoints Specification

| Endpoint | Method | Purpose & Input/Output |
| :--- | :--- | :--- |
| `/health` | `GET` | **Health & Readiness Check:** Confirms the backend is online, whether the champion model (`model_final.joblib`) is loaded, device type, and count of available trips. |
| `/trips` | `GET` | **Trip Inventory:** Lists all available trip IDs across `student_test` and `train` with observation counts, nodes, and arcs. *Never exposes `.route` ground truth.* |
| `/predict` | `POST` | **Pure Inference & Snapping:** Accepts `trip_id`, executes feature extraction, model scoring, and Viterbi sequence decoding. Returns predicted road arcs, `[lat, lon]` route polyline coordinates, timings, and server-side metrics. |
| `/evaluate` | `POST` | **Comprehensive Benchmark Evaluation:** Runs map-matching and computes full ground-truth validation metrics (Top-1 Arc Accuracy, Continuity, F1, Miss Rate). |
| `/map/{trip_id}` | `GET` | **Folium Map Server:** Dynamically renders and streams a standalone HTML Folium map for any specified trip. |

---

## 4. Frontend-Backend Communication & Two-Laptop Architecture

### How They Communicate
The frontend communicates with the backend purely via **HTTP REST requests**:
1. On startup, the dashboard pings `GET /health` to confirm connectivity.
2. The trip dropdown is populated dynamically by querying `GET /trips`.
3. When the user clicks **"Predict Route"**, the frontend sends a `POST /predict` JSON payload to the backend.
4. The backend processes the trip and returns JSON containing `predicted_arcs`, `predicted_coords`, and `metrics`.
5. The frontend extracts `predicted_coords` and plots the route polyline with Green Start and Red End markers in Folium.

### Two-Laptop Hardware & Network Setup
* **Laptop 1 (Model and API Server):**
  - Holds the trained champion MLP model (`artifacts/model_final.joblib`).
  - Holds the datasets and ground truth `.route` files (server-side only).
  - Runs the FastAPI application on `0.0.0.0:8000`.
  - Displays live stdout terminal logs showing incoming evaluation requests and execution latencies.
* **Laptop 2 (Evaluator Dashboard UI):**
  - Runs the Streamlit dashboard (`ui/app.py`) on port `8501`.
  - Connects to Laptop 1 across the local Wi-Fi / hotspot by specifying Laptop 1's IP address:
    `http://<LAPTOP_1_IP>:8000`
  - Displays trip selection, metric cards, and the interactive map.

---

## 5. Prevention of Data Leakage (Ground Truth Isolation)
To comply strictly with the hackathon rules:
1. **Isolated Data Loader:** In `src/api/service.py`, Member 1's `load_trip` is called with `include_reference=False`. The ground truth `.route` file is **never loaded into memory during inference**.
2. **Pure Inference Feature Building:** Member 1's `prepare_trip` runs with `include_labels=False`. Road candidate scoring and Viterbi transitions rely purely on GPS measurements and road network topology.
3. **Server-Side Metric Evaluation:** Ground-truth route files are only accessed server-side *after* prediction has concluded to compute academic evaluation metrics.
4. **Clean Client Payload:** The frontend interface never receives or renders reference routes during pure prediction mode.

---

## 6. Troubleshooting Guide (Common Demo Issues & Fixes)

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **"Backend Offline / Unreachable" on Laptop 2** | Laptop 1 IP changed or firewall blocking port 8000. | 1. Run `ipconfig` on Laptop 1 and check the IPv4 address.<br>2. Ensure both laptops are on the exact same Wi-Fi / hotspot.<br>3. Run in Windows PowerShell (Admin): `New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow`. |
| **Port 8000 or 8501 already in use** | A previous instance is lingering in the background. | Run in CMD: `netstat -ano \| findstr :8000` then `taskkill /F /PID <PID>`. Or specify another port (e.g. `--port 8001`). |
| **Map coordinates flipped or upside down** | Latitude and longitude ordering confusion. | Folium expects `[latitude, longitude]`. GPS datasets provide `[longitude, latitude]`. The coordinate converter in `service.py` automatically flips `(lon, lat) -> (lat, lon)`. |
| **First request takes 3–5 seconds** | JIT import and model initialization on Windows. | Send a warm-up request to trip `00000000` before the evaluator arrives. Subsequent requests execute in **~1.5s**. |

---

## 7. Two-Minute Live Demonstration Script (What to Show & Say)

### [0:00 – 0:30] Introduction & Setup
> *"Good morning/afternoon evaluators. I am representing Member 4 — API, UI and Deployment for Team Asphalt-Anarchy.*  
> *Here we have our live two-laptop demonstration setup: on Laptop 1, we are hosting our trained champion ML model and HMM/Viterbi sequence decoder inside a high-performance FastAPI server running on port 8000.*  
> *On Laptop 2, our Evaluator Dashboard is connected live across the local network to Laptop 1's IP address."*

### [0:30 – 1:00] Triggering Live Prediction
> *"Notice the dashboard status indicator is 🟢 Connected. We dynamically fetch the available test trajectories using `GET /trips`.*  
> *Let's select reserved student test trip `00000000`, containing over 2,500 raw GPS observations. Now, I will click '🚀 Predict Route'.*  
> *As you can see on Laptop 1's terminal, the server immediately logs the incoming HTTP request. In under 1.6 seconds, the pipeline extracts candidates, scores emission probabilities with our MLP model, and reconstructs the optimal path via Viterbi decoding."*

### [1:00 – 1:30] Presenting Metrics & Scorecard
> *"Looking at our performance scorecard on Laptop 2:*  
> *- Our Observation-Level Top-1 Accuracy is **94.5%**.*  
> *- Our Route Continuity is **99.8%**, proving that almost 100% of the reconstructed road arcs are physically connected in the road network graph.*  
> *- Candidate Miss Rate is just **0.2%**, and the Sequence F1 Score reaches **87.3%**.*  
> *- Total end-to-end inference latency is approximately **1,500 milliseconds**."*

### [1:30 – 2:00] Folium Route Map & Leakage Compliance
> *"On the interactive Folium map below, the emerald green line represents our snapped trajectory. We have distinct green start and red end markers, with full pan and zoom navigation.*  
> *Most importantly, per the competition rules, this interface operates under strict zero-leakage constraints: reference route files are never transmitted to the client during inference.*  
> *Our full test suite of 38 unit and integration tests is passing, and the Swagger docs are available at `/docs`. Thank you, I am ready for any technical questions."*

---

## 8. Top 15 Viva Questions & Simple, Direct Answers

1. **Q: What is the main role of Member 4 in this project?**  
   *A:* End-to-end system integration: wrapping the feature pipeline, trained model, and HMM decoder into a robust FastAPI backend, and building the interactive frontend dashboard for live network demonstration.

2. **Q: Why combine machine learning with HMM/Viterbi rather than using the classifier alone?**  
   *A:* A classifier predicts each point independently without knowledge of sequence physics. HMM/Viterbi enforces road graph topology and spatial transitions, eliminating impossible jumps between disconnected roads.

3. **Q: What are the states and observations in your Hidden Markov Model?**  
   *A:* Observations are the noisy GPS coordinates $(x_t, y_t)$. Hidden states are the actual road segment arc IDs ($arc_k$) travelled on.

4. **Q: How are emission probabilities calculated?**  
   *A:* Emission probabilities $P(arc_k \mid z_t)$ are generated by passing candidate geometric features (distance, heading delta, speed, projection ratio) through Member 2's trained champion MLP model.

5. **Q: How are transition probabilities calculated?**  
   *A:* Transition probabilities $P(arc_j \mid arc_i)$ model the likelihood of moving from road arc $i$ to arc $j$. If $i == j$, high probability (staying on same road); if arc $j$ directly follows arc $i$ in the directed graph, probability is high; if disconnected, a penalized probability is assigned.

6. **Q: How did you implement the Viterbi algorithm?**  
   *A:* Dynamic programming in log-space: at each step $t$, we compute the maximum accumulated log probability to reach each candidate state plus transition and emission logs, storing backpointers to reconstruct the globally optimal path.

7. **Q: How do you prevent ground truth data leakage in the API?**  
   *A:* The `/predict` endpoint uses Member 1's `load_trip(include_reference=False)` and `prepare_trip(include_labels=False)`. Ground truth `.route` files are only read server-side during evaluation and never exposed in inference.

8. **Q: How does Laptop 2 communicate with Laptop 1?**  
   *A:* Across the local Wi-Fi or mobile hotspot using HTTP REST requests. Laptop 1 binds FastAPI to `0.0.0.0:8000`, and Laptop 2 targets `http://<laptop1_ip>:8000`.

9. **Q: What is Route Continuity, and why is it important?**  
   *A:* It measures the percentage of consecutive predicted road arcs that share a connecting node in the road network graph. High continuity (99.8%) proves the vehicle's trajectory is physically traversable.

10. **Q: What is Observation-Level Top-1 Accuracy?**  
    *A:* The percentage of individual GPS points where the model's snapped road arc matches the ground truth road arc.

11. **Q: How does Folium receive and plot the coordinates?**  
    *A:* The backend converts the matched arc IDs into an ordered sequence of node coordinates in `[latitude, longitude]` format. Folium draws this as an emerald green Leaflet polyline.

12. **Q: What happens if a GPS point has no road candidates within 50 meters?**  
    *A:* The pipeline flags a candidate miss and falls back to the nearest spatial segment or smoothly interpolates using the HMM Viterbi transition trellis to prevent route breaks.

13. **Q: Why did you choose FastAPI over Flask or Django?**  
    *A:* FastAPI is significantly faster (built on Starlette ASGI), natively supports Pydantic schema validation, and automatically creates Swagger UI documentation.

14. **Q: How do you handle Windows Firewall issues during two-laptop setup?**  
    *A:* We add an inbound firewall rule allowing TCP traffic on port 8000, and ensure both laptops share a private local network profile.

15. **Q: What is the end-to-end latency of your system?**  
    *A:* Approximately 1.5 seconds for a complete trajectory of 2,500 points (~0.6 ms per GPS fix).
