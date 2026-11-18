"""Self-contained native Leaflet HTML with bundled JS/CSS and no reference data."""
import json
from pathlib import Path

UI_ROOT = Path(__file__).resolve().parents[2] / 'ui'


def render_leaflet_map(response):
    payload = {key: response.get(key, default) for key, default in [
        ('trip_id', ''), ('predicted_segments', []), ('network_segments', []), ('matched_coords', []), ('gps_coords', [])]}
    data = json.dumps(payload, allow_nan=False).replace('&', r'\u0026').replace('<', r'\u003c').replace('>', r'\u003e').replace('\u2028', r'\u2028').replace('\u2029', r'\u2029')
    css = (UI_ROOT / 'vendor/leaflet.css').read_text() + '\n' + (UI_ROOT / 'route-map.css').read_text()
    library = (UI_ROOT / 'vendor/leaflet.js').read_text()
    script = (UI_ROOT / 'route-map.js').read_text()
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>PathSnap Leaflet route map</title><style>' + css + '</style></head><body>'
            '<section id="map-panel"><header class="map-bar"><span id="map-title">Route map</span>'
            '<div class="map-actions"><button id="fit-route" type="button">Fit route</button>'
            '<button id="full-screen" type="button">Fullscreen</button></div></header>'
            '<div id="route-map" role="region" aria-label="Predicted roads and GPS observations"></div>'
            '<footer class="map-footer"><span class="key"><i class="swatch"></i>Predicted roads</span>'
            '<span class="key"><i class="dot"></i>Raw GPS observations</span><span class="map-note">Matching uses supplied roads; online streets may differ.</span>'
            '<span class="map-note" id="map-status">Street background · offline fallback available in layer controls</span>'
            '</footer></section><script id="map-data" type="application/json">' + data + '</script>'
            '<script>' + library + '\n</script><script>' + script + '</script></body></html>')
