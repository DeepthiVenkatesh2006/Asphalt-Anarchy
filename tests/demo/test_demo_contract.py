from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.api.service import MapMatchingService
from src.api.schemas import PredictRequest


def test_evaluator_partition_and_data_override(tmp_path, monkeypatch):
    directory = tmp_path / 'evaluator_test' / '00000015'
    directory.mkdir(parents=True)
    (directory / '00000015.track').write_text('0 0 1\n0 0 11\n')
    (directory / '00000015.nodes').write_text('0 0\n.001 0\n')
    (directory / '00000015.arcs').write_text('0 1\n')
    monkeypatch.setenv('PATHSNAP_DATA_ROOT', str(tmp_path))
    service = MapMatchingService()
    trips = service.list_available_trips()
    assert len(trips) == 1
    assert trips[0]['partition'] == 'evaluator_test'
    assert trips[0]['duration_sec'] == 10


def test_default_model_is_verified_tuned_artifact():
    service = MapMatchingService()
    assert Path(service.model_path).parent.name == 'local_training'
    assert service.champion_model is not None


def test_predict_request_validates_identifiers_and_radius():
    assert PredictRequest(trip_id='00000015', partition='evaluator_test', search_radius_m=75).search_radius_m == 75
    for values in [dict(trip_id='../train'), dict(trip_id='00000015', partition='../train'), dict(trip_id='00000015', search_radius_m=-1)]:
        with pytest.raises(ValueError):
            PredictRequest(**values)


def test_decoder_keeps_missing_observation_and_uses_highest_score():
    from src.decoder.online import decode_candidates
    from types import SimpleNamespace
    observations = pd.DataFrame(dict(obs_idx=[0,1,2], gap_before=[False]*3, longitude=[0.0]*3, latitude=[0.0]*3))
    candidates = pd.DataFrame(dict(obs_idx=[0,0,2], arc_id=[0,1,0], candidate_score=[.1,.9,.8]))
    prepared = SimpleNamespace(observations=observations, candidates=candidates)
    decoded, _ = decode_candidates(prepared, use_viterbi=False)
    assert decoded == [1,-1,0]


def test_map_does_not_publish_reference_geometry():
    from src.visualization.map_builder import build_folium_map
    result = dict(track=[[0,0,1]], nodes=[[0,0],[1,0],[20,20],[21,20]], arcs=[[0,1],[2,3]], route=[1], predicted_route=[0], metrics={})
    html = build_folium_map(result).get_root().render()
    assert 'Reference Ground Truth Route' not in html
    assert 'Ground Truth Route' not in html
    assert '20.0, 21.0' not in html


def test_live_viterbi_multi_candidate_and_gap():
    from types import SimpleNamespace
    from src.decoder.online import decode_candidates
    observations = pd.DataFrame(dict(obs_idx=[0,1,2], gap_before=[False,False,True], longitude=[0.0001,0.0002,0.0001], latitude=[0.0]*3))
    candidates = pd.DataFrame(dict(trip_id=['00000001']*6, obs_idx=[0,0,1,1,2,2], arc_id=[0,1]*3, candidate_score=[.9,.1,.9,.1,.1,.9], candidate_probability=[.9,.1,.9,.1,.1,.9], projection_ratio=[.1,.9,.2,.8,.1,.9]))
    roads = pd.DataFrame(dict(trip_id=['00000001']*2, arc_id=[0,1], from_node_id=[0,1], to_node_id=[1,0], from_lon=[0.0,.001], from_lat=[0.0]*2, to_lon=[.001,0.0], to_lat=[0.0]*2, length_m=[111.0]*2, heading_deg=[90.0,270.0]))
    prepared=SimpleNamespace(observations=observations,candidates=candidates,road_segments=roads)
    decoded,restarts=decode_candidates(prepared)
    assert decoded == [0,0,1]
    assert restarts == []


def test_route_expansion_includes_connectors_without_bridging_gaps():
    from types import SimpleNamespace
    from src.decoder.online import expand_selected_route
    roads = pd.DataFrame(dict(trip_id=['00000001']*3, arc_id=[0,1,2], from_node_id=[0,1,2], to_node_id=[1,2,3], from_lon=[0,.001,.002], from_lat=[0.0]*3, to_lon=[.001,.002,.003], to_lat=[0.0]*3, length_m=[111.0]*3, heading_deg=[90.0]*3))
    observations=pd.DataFrame(dict(obs_idx=[0,1],gap_before=[False,False]))
    prepared=SimpleNamespace(road_segments=roads, observations=observations)
    route, connected=expand_selected_route(prepared,[0,2],[])
    assert route == [0,1,2]
    assert connected == 1
    prepared.observations.loc[1,'gap_before']=True
    assert expand_selected_route(prepared,[0,2],[])[0] == [0,2]
    prepared.observations.loc[1,'gap_before']=False
    assert expand_selected_route(prepared,[0,2],[1])[0] == [0,2]
