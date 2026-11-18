from types import SimpleNamespace
import math
import networkx as nx
import pandas as pd
import pytest
from src.decoder.route_evaluation import evaluate_route
from src.decoder.transition_model import calculate_transition_probability
from src.decoder.online import expand_selected_route


def fixture(cycle=False):
    roads=pd.DataFrame(dict(trip_id=['00000001']*(2 if cycle else 1),arc_id=[0,1] if cycle else [0],from_node_id=[0,1] if cycle else [0],to_node_id=[1,0] if cycle else [1],from_lon=[0,.001] if cycle else [0],from_lat=[0,0] if cycle else [0],to_lon=[.001,0] if cycle else [.001],to_lat=[0,0] if cycle else [0],length_m=[111,111] if cycle else [111],heading_deg=[90,270] if cycle else [90]))
    obs=pd.DataFrame(dict(obs_idx=[0,1],gap_before=[False,False],longitude=[.0009,.0001],latitude=[0,0]))
    candidates=pd.DataFrame(dict(obs_idx=[0,1],arc_id=[0,0],projection_ratio=[.9,.1]))
    return SimpleNamespace(road_segments=roads,observations=obs,candidates=candidates)


def test_sequence_metric_does_not_discard_frequent_road_ids():
    reference=[0,1]*100+[2,3]
    predicted=[2,3]+[0,1]*100
    assert evaluate_route(predicted,reference)['sequence_similarity'] > .98


def test_backward_one_way_state_is_not_reported_connected():
    expanded,connected=expand_selected_route(fixture(),[0,0],[])
    assert connected == 0
    assert expanded == [0,0]  # Separate chunks, rather than silently collapsing.


def test_same_arc_return_cycle_is_a_valid_transition():
    graph=nx.MultiDiGraph()
    graph.add_edge(0,1,key=0,arc_id=0,length_m=111)
    graph.add_edge(1,0,key=1,arc_id=1,length_m=111)
    score=calculate_transition_probability(graph,0,0,88.8,previous_projection_ratio=.9,current_projection_ratio=.1)
    assert score == pytest.approx(math.exp(-44.4/50))
    route,connected=expand_selected_route(fixture(cycle=True),[0,0],[])
    assert route == [0,1,0]
    assert connected == 1


def test_restart_preserves_identical_arc_as_separate_chunks():
    route,connected=expand_selected_route(fixture(cycle=True),[0,0],[1])
    assert route == [0,0]
    assert connected == 0
