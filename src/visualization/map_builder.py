import os
import folium
from typing import Dict, Any

def build_folium_map(
    trip_result: Dict[str, Any],
    output_html_path: str = None
) -> folium.Map:
    track = trip_result["track"]
    nodes = trip_result["nodes"]
    arcs = trip_result["arcs"]
    route = trip_result["route"]
    predicted_route = trip_result["predicted_route"]
    metrics = trip_result.get("metrics", {})
    trip_id = trip_result.get("trip_id", "Unknown")

    if not track:
        return folium.Map(location=[0, 0], zoom_start=2)

    lats = [p[1] for p in track]
    lons = [p[0] for p in track]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # 1. Road Network Layer
    relevant_arc_ids = set(predicted_route) | set(route)
    road_layer = folium.FeatureGroup(name="1. Road Network (Local Arcs)", show=False)
    for arc_id in list(relevant_arc_ids)[:2000]:
        if arc_id < len(arcs):
            u, v = arcs[arc_id]
            u_pt = [nodes[u][1], nodes[u][0]]
            v_pt = [nodes[v][1], nodes[v][0]]
            folium.PolyLine(
                [u_pt, v_pt],
                color="#6c757d",
                weight=2,
                opacity=0.6,
                tooltip=f"Arc ID: {arc_id} ({u} -> {v})"
            ).add_to(road_layer)
    road_layer.add_to(m)

    # 2. Raw GPS Observations Layer
    gps_layer = folium.FeatureGroup(name="2. Raw GPS Observations", show=True)
    step = 1 if len(track) <= 600 else int(len(track) / 600)
    for i in range(0, len(track), step):
        lon, lat, ts = track[i]
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#e63946",
            fill=True,
            fill_color="#e63946",
            fill_opacity=0.8,
            popup=f"Obs #{i}<br>Lon: {lon:.5f}<br>Lat: {lat:.5f}<br>Time: {ts:.1f}s",
            tooltip=f"GPS #{i}"
        ).add_to(gps_layer)

    gps_coords = [[p[1], p[0]] for p in track]
    folium.PolyLine(
        gps_coords,
        color="#e63946",
        weight=1.5,
        dash_array="4, 6",
        opacity=0.5
    ).add_to(gps_layer)
    gps_layer.add_to(m)

    # 3. Reference Route Layer
    if route:
        gt_layer = folium.FeatureGroup(name="3. Reference Ground Truth Route", show=True)
        gt_segments = []
        for arc_id in route:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                gt_segments.append([nodes[u][1], nodes[u][0]])
                gt_segments.append([nodes[v][1], nodes[v][0]])
        if gt_segments:
            folium.PolyLine(
                gt_segments,
                color="#2a9d8f",
                weight=5,
                opacity=0.85,
                tooltip="Ground Truth Route"
            ).add_to(gt_layer)
        gt_layer.add_to(m)

    # 4. Predicted Route Layer
    pred_layer = folium.FeatureGroup(name="4. PathSnap Predicted Route", show=True)
    pred_segments = []
    prev_arc = None
    for arc_id in predicted_route:
        if arc_id != prev_arc and arc_id < len(arcs):
            u, v = arcs[arc_id]
            pred_segments.append([nodes[u][1], nodes[u][0]])
            pred_segments.append([nodes[v][1], nodes[v][0]])
            prev_arc = arc_id

    if pred_segments:
        folium.PolyLine(
            pred_segments,
            color="#1d3557",
            weight=4,
            opacity=0.9,
            dash_array="6, 8" if route else None,
            tooltip=f"Predicted Route (Trip {trip_id})"
        ).add_to(pred_layer)
    pred_layer.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # Floating Summary Card
    acc = metrics.get("top1_accuracy", 0.0) * 100.0
    cont = metrics.get("route_continuity", 0.0) * 100.0
    lat_ms = metrics.get("inference_time_ms", 0.0)
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.95);
        padding: 14px 18px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 13px;
        line-height: 1.6;
        min-width: 240px;
        border-left: 5px solid #1d3557;
    ">
        <h4 style="margin: 0 0 8px 0; font-size: 14px; color: #1d3557;">Trip {trip_id} Summary</h4>
        <div><span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#e63946; margin-right:6px;"></span> Raw GPS Observations ({len(track)})</div>
        <div><span style="display:inline-block; width:18px; height:4px; background:#2a9d8f; margin-right:6px; vertical-align:middle;"></span> Reference Route (Ground Truth)</div>
        <div><span style="display:inline-block; width:18px; height:4px; background:#1d3557; margin-right:6px; vertical-align:middle;"></span> PathSnap Predicted Route</div>
        <div><span style="display:inline-block; width:18px; height:2px; background:#6c757d; margin-right:6px; vertical-align:middle;"></span> Local Road Arcs</div>
        <hr style="margin: 8px 0; border:0; border-top: 1px solid #ddd;">
        <div><b>Top-1 Accuracy:</b> <span style="color:#2a9d8f; font-weight:bold;">{acc:.1f}%</span></div>
        <div><b>Route Continuity:</b> <span style="color:#1d3557; font-weight:bold;">{cont:.1f}%</span></div>
        <div><b>Inference Latency:</b> {lat_ms:.1f} ms</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_html_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
        m.save(output_html_path)

    return m
