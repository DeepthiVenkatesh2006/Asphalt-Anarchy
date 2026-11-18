import json
from src.visualization.leaflet_map import render_leaflet_map


def test_native_leaflet_uses_separate_segments_and_bundled_assets():
    response=dict(trip_id='00000023', predicted_segments=[[[1,2],[1,3]],[[5,6],[5,7]]], gps_coords=[[1,2],[5,7]], route=['PRIVATE_REFERENCE'])
    html=render_leaflet_map(response)
    assert 'L.map(' in html
    assert 'L.polyline(segment' in html
    assert 'L.control.layers(' in html
    assert 'Fit route' in html
    assert 'PRIVATE_REFERENCE' not in html
    assert '<script src=' not in html
    assert 'tile.openstreetmap.org' not in html
    assert 'leaflet-container' in html


def test_map_embedded_data_cannot_end_script_tag():
    html=render_leaflet_map(dict(trip_id='</script><script>alert(1)</script>', predicted_segments=[], gps_coords=[]))
    assert '</script><script>alert(1)</script>' not in html
    assert '\\u003c/script' in html
