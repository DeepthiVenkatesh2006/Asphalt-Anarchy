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


def test_dataset_road_layer_preserves_route_coordinates():
    data=dict(trip_id='00000023',predicted_segments=[[[1,2],[1,3]]],network_segments=[[[1,2],[1,3]],[[1,3],[1,4]]],gps_coords=[[1.0001,2.0001]])
    html=render_leaflet_map(data)
    assert 'network_segments' in html
    assert 'Supplied road network' in html
    assert 'supplied roads' in html
    assert '[[1, 2], [1, 3]]' in html


def test_development_folium_reference_is_explicit_and_never_allowed_for_evaluator():
    import pytest
    from src.visualization.map_builder import build_folium_map
    data=dict(trip_id='00000001',partition='student_test',track=[[0,0,1]],nodes=[[0,0],[.001,0]],arcs=[[0,1]],route=[0],predicted_route=[0])
    assert 'Reference Ground Truth Route' in build_folium_map(data,include_reference=True).get_root().render()
    data['partition']='evaluator_test'
    with pytest.raises(ValueError,match='development'):
        build_folium_map(data,include_reference=True)
