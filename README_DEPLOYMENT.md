# Asphalt-Anarchy: PathSnap Deployment & Evaluator Demo Guide

This document outlines the deployment, execution, and live demonstration setup for **Member 4 (API, UI, and Deployment)**.

---

## 🖥️ Live Demonstration Setup (Two-Laptop Protocol)

* **Laptop 1 (Model and API Server):**
  - Runs the FastAPI inference engine (`src.api.app:app`) on port `8000`.
  - Hosts the trained champion MLP pipeline (`artifacts/model_final.joblib`).
  - Holds server-side reference route data (`.route`) for evaluation.
  - Displays live incoming HTTP request logs on the command line for the evaluator.

* **Laptop 2 (Evaluator Dashboard UI):**
  - Connects across the local network (Wi-Fi or hotspot) to Laptop 1.
  - Displays the 10 reserved student test evaluation trips.
  - Sends evaluation requests to Laptop 1 and renders real-time scorecards and Folium interactive maps.

---

## 🚀 Step 1: Starting the Server on Laptop 1

1. Open PowerShell or Command Prompt.
2. Navigate to the repository root:
   ```cmd
   cd C:\Users\DELL\.gemini\antigravity\scratch\Asphalt-Anarchy
   ```
3. Set `PYTHONPATH`:
   ```powershell
   $env:PYTHONPATH="."
   ```
   *(or `set PYTHONPATH=.` in CMD)*
4. Launch the FastAPI server:
   ```cmd
   python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
5. Note Laptop 1's local IP address using `ipconfig` (e.g. `192.168.1.105`).

---

## 🌐 Step 2: Opening the Evaluator UI on Laptop 2

1. Open a browser (Chrome, Edge, or Firefox) on Laptop 2.
2. Navigate to:
   ```text
   http://<LAPTOP_1_IP>:8000/ui/
   ```
   *(e.g. `http://192.168.1.105:8000/ui/`)*
3. Confirm the status dot turns 🟢 **Connected (Asphalt-Anarchy PathSnap API v1.0.0)**.

---

## 🎯 Step 3: Conducting the Live Evaluation Demo

1. In the UI dropdown, select any of the 10 reserved student test trips (e.g., `00000000`, `00000007`, `00000052`).
2. Click **Run Evaluation Pipeline**.
3. **Show Laptop 1 CMD:** Point out the live request stream:
   ```text
   >>> [INFERENCE REQUEST] Received evaluation for Trip ID: '00000000' (partition: student_test)
   <<< [INFERENCE COMPLETE] Trip '00000000' | Top-1 Acc: 95.8% | Continuity: 99.8% | Latency: 4120ms
   ```
4. **Show Laptop 2 UI:** Inspect the metrics scorecard:
   - **Observation-Level Top-1 Accuracy**
   - **Route Continuity**
   - **Interactive Folium Map** (Toggleable layers for Raw GPS, Ground Truth, Predicted Route, Road Arcs)
   - **Predicted Arc Sequence**
