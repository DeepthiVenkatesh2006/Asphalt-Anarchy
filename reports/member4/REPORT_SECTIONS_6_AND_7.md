# PathSnap Final Report Deliverables: Sections 6 & 7
**Author:** Member 4 (API, UI, and Deployment)  
**Team:** Team 7 (Asphalt-Anarchy)  

---

## 6. Geographical Visualisation Using Folium

### 6.1 Overview and Visualization Architecture
To validate and visualize trajectory map-matching results geographically, an interactive Folium map builder (`src.visualization.map_builder.build_folium_map`) was developed. The tool transforms spatial trajectory data and road network topology into a multi-layered leaflet-based interactive map.

The visualization incorporates four distinct, toggleable feature groups:
1. **Raw GPS Observations (Blue Dots):**
   - Displays measured GPS coordinate fixes sampled along the trajectory.
   - Includes point index, timestamp, and speed in hover tooltips.
   - Subsampled dynamically for dense trips to preserve browser rendering performance while maintaining fidelity.
2. **Road Network Geometry (Muted Gray Segments):**
   - Displays all road network arcs candidate segments within the bounding box of the trajectory.
   - Shows directed arc identifiers (`arc_id`) and start/end node pairs (`from_node -> to_node`) upon inspection.
3. **Reference / Ground Truth Route (Orange Dashed Polyline):**
   - Renders the actual traversed road sequence as specified in the ground-truth `.route` file.
   - Kept isolated server-side during evaluation to avoid inference data leakage.
4. **Predicted Route (Emerald Green Solid Polyline):**
   - Renders the optimal road arc sequence reconstructed by the champion MLP scoring model combined with the HMM/Viterbi sequence decoder.
   - Visually indicates route continuity and candidate recovery across noisy segments.

### 6.2 Key Visual Features & Legend
- **Color Differentiation:**
  - `Raw GPS Observations`: `#0d6efd` (Blue circles)
  - `Road Network Candidates`: `#6c757d` (Gray lines, opacity 0.6)
  - `Ground Truth Reference Route`: `#fd7e14` (Orange dashed line, weight 4)
  - `Predicted Sequence Route`: `#198754` (Emerald green solid line, weight 5)
  - `Trajectory Start Marker`: 🟢 Green pin (`Start`)
  - `Trajectory End Marker`: 🔴 Red pin (`End`)
- **Layer Controls:** Built-in Leaflet `LayerControl` allows evaluators to toggle individual layers on and off to directly inspect errors, misses, or noisy GPS deviations.
- **Floating Scorecard HUD:** An integrated summary badge embedded directly into the top-right corner of the HTML map displaying:
  - Trip ID
  - Top-1 Arc Accuracy
  - Route Continuity
  - Candidate Miss Rate
  - Sequence F1 Score

### 6.3 Representative Map Output
- **Trip Visualized:** `00000000` (Student Test Partition, 2,503 observations)
- **Saved Map Artifact:** `reports/member4/trip_00000000_map.html`
- **Analysis:**
  - The predicted emerald route follows the true road network topology tightly with **99.88% route continuity**.
  - Noisy GPS points offset by 15–30 meters from the true centerline are successfully snapped onto the correct arc.
  - The HMM Viterbi decoder prevents discontinuous jumps at complex intersections.

---

## 7. API, UI and Deployment

### 7.1 System Architecture
The deployment system is structured as a decoupled client-server architecture designed for cross-laptop network communication during the live hackathon demonstration:
- **Backend (FastAPI Engine):**
  - High-performance asynchronous REST API running under `uvicorn`.
  - Encapsulates Member 1's data loading and spatial candidate extraction, Member 2's trained champion MLP model (`artifacts/model_final.joblib`), and Member 3's HMM/Viterbi sequence decoder and route evaluation.
  - Strictly enforces ground-truth isolation: pure inference requests (`/api/v1/predict`) never load `.route` files, while evaluation requests (`/api/v1/evaluate`) calculate metrics server-side.
- **Frontend (Evaluator Dashboard UI):**
  - Responsive web application built with vanilla HTML5, CSS3, and JavaScript, served statically from `/ui/`.
  - Connects to the API over HTTP across the local Wi-Fi / hotspot network.
  - Displays partition summaries, test trip selector, latency timers, real-time scorecards, interactive Folium iframe previews, and predicted arc sequence tags.

### 7.2 API Specification & Endpoints
| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Verifies server readiness, CPU/GPU device, and model loaded status. |
| `GET` | `/api/v1/trips` | Returns an inventory of all discovered trips across `student_test` and `train`. |
| `POST` | `/api/v1/evaluate` | Executes full map-matching and calculates ground truth metrics (Top-1, Continuity, F1, Latency). |
| `POST` | `/api/v1/predict` | Pure inference endpoint returning predicted road arc sequences without ground truth access. |
| `GET` | `/api/v1/map/{trip_id}` | Dynamically generates and returns an interactive Folium HTML map for any trip. |

### 7.3 Live Dual-Laptop Demonstration Protocol
1. **Network Configuration:**
   - Both laptops join the same Wi-Fi network or mobile hotspot.
   - Laptop 1 retrieves its local IP address via `ipconfig` (e.g., `192.168.1.105`).
2. **Laptop 1 Execution (Model & API Server):**
   ```powershell
   cd Asphalt-Anarchy
   python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
3. **Laptop 2 Execution (Evaluator UI Dashboard):**
   - The evaluator opens `http://192.168.1.105:8000/ui/` in Google Chrome or Microsoft Edge.
   - Selects a student test trip (e.g. `00000000`) and triggers **Run Evaluation Pipeline**.
4. **Live Verification Output:**
   - **Laptop 1 Command Line (Server Logs):**
     ```text
     INFO:     192.168.1.120:54321 - "GET /api/v1/health HTTP/1.1" 200 OK
     >>> [INFERENCE REQUEST] Received evaluation for Trip ID: '00000000' (partition: student_test)
     <<< [INFERENCE COMPLETE] Trip '00000000' | Top-1 Acc: 94.65% | Continuity: 99.88% | F1: 87.88% | Latency: 1502ms
     INFO:     192.168.1.120:54322 - "POST /api/v1/evaluate HTTP/1.1" 200 OK
     ```
   - **Laptop 2 Evaluator Dashboard:**
     - Status: 🟢 **Connected (Asphalt-Anarchy PathSnap API v1.0.0)**
     - Observation-Level Top-1 Accuracy: **94.65%**
     - Route Graph Continuity: **99.88%**
     - Candidate Miss Rate: **0.20%**
     - Sequence F1 Score: **87.88%**
     - Latency: **1,502 ms** (2,503 points processed end-to-end)
     - Interactive Folium map rendered inline with toggleable layers.
