"""
PathSnap – GPS Trajectory Map Matching System
Professional Streamlit Dashboard for Hackathon Demonstration (Member 4)
"""

import os
import sys
import time
import requests
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(
    page_title="PathSnap – GPS Trajectory Map Matching",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0284c7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .status-badge-online {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-offline {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .arc-tag {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
        padding: 0.15rem 0.45rem;
        margin: 0.15rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR: Backend Configuration & Trip Selector
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/route.png", width=64)
    st.title("PathSnap Console")
    st.caption("Member 4: API, UI & Deployment Interface")
    
    st.markdown("---")
    st.subheader("🌐 Laptop 1 Backend Connection")
    backend_url = st.text_input(
        "FastAPI Server URL",
        value=st.session_state.get("backend_url", os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")),
        help="On Laptop 2, enter Laptop 1's LAN IP address, e.g. http://192.168.1.105:8000"
    ).rstrip("/")
    st.session_state["backend_url"] = backend_url
    
    # Check Backend Health
    backend_online = False
    health_info = {}
    try:
        resp = requests.get(f"{backend_url}/health", timeout=3.0)
        if resp.status_code == 200:
            backend_online = True
            health_info = resp.json()
    except Exception:
        # Try fallback /api/v1/health
        try:
            resp = requests.get(f"{backend_url}/api/v1/health", timeout=3.0)
            if resp.status_code == 200:
                backend_online = True
                health_info = resp.json()
        except Exception:
            backend_online = False

    if backend_online:
        st.markdown(
            f'<div class="status-badge-online">🟢 Connected: {health_info.get("service", "PathSnap API")}</div>',
            unsafe_allow_html=True
        )
        st.caption(f"**Model:** {health_info.get('model_name', 'Loaded')} | **Device:** {health_info.get('device', 'CPU')}")
    else:
        st.markdown(
            '<div class="status-badge-offline">🔴 Backend Offline / Unreachable</div>',
            unsafe_allow_html=True
        )
        st.warning(
            f"Cannot connect to `{backend_url}`.\n\n"
            "**Troubleshooting:**\n"
            "1. Confirm Laptop 1 started: `python -m uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000`\n"
            "2. Confirm both laptops are on the same Wi-Fi / hotspot.\n"
            "3. Verify Laptop 1 IP via `ipconfig`."
        )

    st.markdown("---")
    st.subheader("🚗 GPS Trip Selector")
    
    # Fetch available trips from backend
    trips_list = []
    if backend_online:
        try:
            trips_resp = requests.get(f"{backend_url}/trips", timeout=5.0)
            if trips_resp.status_code != 200:
                trips_resp = requests.get(f"{backend_url}/api/v1/trips", timeout=5.0)
            if trips_resp.status_code == 200:
                trips_data = trips_resp.json()
                trips_list = trips_data.get("trips", [])
        except Exception as e:
            st.error(f"Failed to fetch trips: {e}")

    if not trips_list:
        st.info("No trips are available. Check the API connection and its dataset folder.")
        st.stop()
    partitions = [p for p in ["evaluator_test", "student_test", "train"] if any(t.get("partition") == p for t in trips_list)]
    chosen_partition = st.radio("Dataset Partition", options=partitions, index=0)

    filtered_trips = [t for t in trips_list if t.get("partition") == chosen_partition]

    trip_options = [t["trip_id"] for t in filtered_trips]
    selected_trip_id = st.selectbox(
        "Select GPS Trajectory Trip ID",
        options=trip_options,
        index=0 if trip_options else 0
    )

    # Selected trip metadata
    selected_meta = next((t for t in filtered_trips if t["trip_id"] == selected_trip_id), None)
    if selected_meta:
        st.markdown("#### 📋 Trip Summary")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write(f"**Points:** `{selected_meta.get('points', 'N/A')}`")
            st.write(f"**Nodes:** `{selected_meta.get('nodes', 'N/A')}`")
        with col_m2:
            st.write(f"**Arcs:** `{selected_meta.get('arcs', 'N/A')}`")
            st.write(f"**Duration:** `{selected_meta.get('duration_sec', 0.0):.0f}s`")

    st.markdown("---")
    use_viterbi = st.checkbox("Enable HMM / Viterbi Sequence Decoder", value=True)
    search_radius = st.slider("Candidate Search Radius (meters)", min_value=20, max_value=100, value=50, step=5)

# -------------------------------------------------------------
# MAIN PANEL: Demonstration & Predictions
# -------------------------------------------------------------
st.markdown('<div class="main-header">PathSnap – GPS Trajectory Map Matching System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Machine learning candidate scoring coupled with an HMM/Viterbi sequence decoder '
    'for map-matching noisy GPS observations onto road networks.'
    '</div>',
    unsafe_allow_html=True
)

# Action bar
action_col1, action_col2 = st.columns([1, 4])
with action_col1:
    predict_clicked = st.button("🚀 Predict Route", type="primary", use_container_width=True)

request_key = (backend_url, chosen_partition, selected_trip_id, use_viterbi, search_radius)

# State container for prediction results
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "last_trip_id" not in st.session_state:
    st.session_state["last_trip_id"] = None

if predict_clicked:
    if not backend_online:
        st.error(f"Cannot execute prediction: Backend at `{backend_url}` is unreachable. Please verify Laptop 1.")
    else:
        with st.spinner(f"Snapping GPS Trajectory for Trip '{selected_trip_id}' (ML Candidate Scoring → HMM/Viterbi Decoding)..."):
            t_start = time.time()
            try:
                # Call /predict endpoint
                payload = {
                    "trip_id": selected_trip_id,
                    "partition": chosen_partition,
                    "search_radius_m": search_radius,
                    "use_viterbi": use_viterbi
                }
                
                resp = requests.post(f"{backend_url}/predict", json=payload, timeout=300.0)
                if resp.status_code == 404:
                    resp = requests.post(f"{backend_url}/api/v1/predict", json=payload, timeout=300.0)
                
                if resp.status_code == 200:
                    res_data = resp.json()
                    st.session_state["prediction_result"] = res_data
                    st.session_state["last_trip_id"] = request_key
                    st.session_state["run_time"] = time.time() - t_start
                else:
                    st.error(f"Prediction failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request error during map-matching: {e}")

# Display Results
res = st.session_state.get("prediction_result")
if res and st.session_state.get("last_trip_id") == request_key:
    metrics = res.get("metrics") or {}
    total_arcs = res.get("total_arcs", len(res.get("predicted_arcs", [])))
    latency_ms = res.get("inference_time_ms", 0.0)
    
    st.success(
        f"✅ **Prediction completed successfully for Trip `{selected_trip_id}`!** "
        f"(Processed in **{latency_ms:.1f} ms**)"
    )

    # ---------------------------------------------------------
    # Aggregate Metrics HUD
    # ---------------------------------------------------------
    st.markdown("### 📊 Performance & Evaluation Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    top1_acc = metrics.get("top1_accuracy")
    top1_display = f"{top1_acc * 100:.2f}%" if top1_acc is not None else "N/A"
    
    continuity = metrics.get("route_continuity") or res.get("route_continuity", 0.0)
    cont_display = f"{continuity * 100:.2f}%"
    
    miss_rate = metrics.get("candidate_miss_rate")
    miss_display = f"{miss_rate * 100:.2f}%" if miss_rate is not None else "N/A"
    
    seq_f1 = metrics.get("sequence_f1")
    f1_display = f"{seq_f1 * 100:.2f}%" if seq_f1 is not None else "N/A"
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top1_display}</div>
            <div class="metric-label">Top-1 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cont_display}</div>
            <div class="metric-label">Route Continuity</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{miss_display}</div>
            <div class="metric-label">Candidate Miss Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{f1_display}</div>
            <div class="metric-label">Road-set F1</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latency_ms:.0f} ms</div>
            <div class="metric-label">Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)

    coverage = metrics.get("evaluation_coverage")
    if coverage is not None:
        st.caption(f"Accuracy uses {metrics.get('evaluation_observations', 0):,} aligned observations ({coverage:.1%} coverage). Labels are approximate. Road-set F1 ignores order.")
    st.caption(f"Unmatched GPS points: {metrics.get('unmatched_observations', 0)} · Decoder restarts: {metrics.get('decoder_restarts', 0)}")
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Folium Map Visualization
    # ---------------------------------------------------------
    st.markdown("### 🗺️ Interactive Route Map")
    
    coords = res.get("predicted_coords", [])
    
    if coords and len(coords) >= 2:
        # Determine center
        avg_lat = sum(p[0] for p in coords) / len(coords)
        avg_lon = sum(p[1] for p in coords) / len(coords)

        # Create Folium Map with OpenStreetMap basemap
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=14,
            tiles=None
        )

        # 1. Predicted Snapped Route (Emerald Green Polyline)
        folium.PolyLine(
            res.get("predicted_segments") or coords,
            color="#10b981",
            weight=5,
            opacity=0.9,
            tooltip=f"Predicted Route: {total_arcs} road segments"
        ).add_to(m)

        # 2. Start Marker (Green Pin / Circle)
        start_pt = coords[0]
        folium.Marker(
            location=start_pt,
            popup=f"<b>Trip Start</b><br>Lat: {start_pt[0]:.6f}<br>Lon: {start_pt[1]:.6f}",
            tooltip="Trip Start",
            icon=folium.Icon(color="green", icon="play", prefix="fa")
        ).add_to(m)

        # 3. End Marker (Red Pin / Circle)
        end_pt = coords[-1]
        folium.Marker(
            location=end_pt,
            popup=f"<b>Trip Destination</b><br>Lat: {end_pt[0]:.6f}<br>Lon: {end_pt[1]:.6f}",
            tooltip="Trip End",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
        ).add_to(m)

        # Fit bounds to trajectory extent
        m.fit_bounds([[min(p[0] for p in coords), min(p[1] for p in coords)],
                      [max(p[0] for p in coords), max(p[1] for p in coords)]])

        # Add fullscreen control
        plugins.Fullscreen(position="topleft").add_to(m)

        # Render Folium in Streamlit
        st_folium(m, width=None, height=520)
    else:
        st.info("Coordinates not returned in simplified format; attempting backend map embed...")
        map_url = backend_url + res.get("map_url", f"/map/{selected_trip_id}?partition={chosen_partition}")
        st.components.v1.iframe(map_url, height=520, scrolling=True)

    # ---------------------------------------------------------
    # Road Arc Sequence & Technical Verification Details
    # ---------------------------------------------------------
    st.markdown("### 🛣️ Traversed Road Arc Sequence")
    predicted_arcs = res.get("predicted_arcs", [])
    
    # Condense consecutive duplicate arcs for clear sequence inspection
    condensed_arcs = []
    for arc in res.get("expanded_route") or predicted_arcs:
        if arc < 0:
            continue
        if not condensed_arcs or condensed_arcs[-1] != arc:
            condensed_arcs.append(arc)

    st.write(f"**Total Traversed Road Segments:** `{len(condensed_arcs)}` (from {len(predicted_arcs)} raw point-snaps)")
    
    # Display preview tags
    preview_tags = "".join([f'<span class="arc-tag">Arc #{a}</span>' for a in condensed_arcs[:35]])
    if len(condensed_arcs) > 35:
        preview_tags += f'<span class="arc-tag">... +{len(condensed_arcs)-35} more</span>'
    st.markdown(preview_tags, unsafe_allow_html=True)

    with st.expander("🔍 View Raw JSON Response (Backend Verification)"):
        st.json({
            "trip_id": res.get("trip_id"),
            "partition": res.get("partition"),
            "status": res.get("status"),
            "total_arcs": total_arcs,
            "metrics": metrics,
            "sample_predicted_arcs": predicted_arcs[:25]
        })
else:
    # Idle initial screen before clicking Predict
    st.info("👉 Select a GPS Trajectory in the sidebar and click **'🚀 Predict Route'** to run the live map-matching pipeline.")
    
    # Quick architecture explainer box
    st.markdown("""
    ---
    #### 🏗️ System Workflow Architecture
    ```
    [GPS Coordinates] ──► [Candidate Arc Search] ──► [ML Candidate Classifier] ──► [HMM / Viterbi Decoder] ──► [Folium Map & Scorecard]
         (Track)               (Radius = 50m)             (Score: P(arc|point))          (Sequence Optimization)        (Laptop 2 Dashboard)
    ```
    """)
