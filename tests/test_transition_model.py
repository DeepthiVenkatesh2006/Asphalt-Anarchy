
import networkx as nx
import pytest

from src.decoder.transition_model import (
    calculate_transition_probability,
)


def make_graph():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        0, 1, key=10,
        arc_id=10, length_m=100.0
    )

    graph.add_edge(
        1, 2, key=20,
        arc_id=20, length_m=100.0
    )

    return graph


def test_connected_road_transition():
    graph = make_graph()

    probability = calculate_transition_probability(
        graph, 10, 20, 200.0
    )

    assert probability == pytest.approx(1.0)


def test_disconnected_road_transition():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        0, 1, key=10,
        arc_id=10, length_m=100.0
    )

    graph.add_edge(
        5, 6, key=20,
        arc_id=20, length_m=100.0
    )

    probability = calculate_transition_probability(
        graph, 10, 20, 200.0
    )

    assert probability == 0.0


def test_partial_segment_transition():
    graph = make_graph()

    probability = calculate_transition_probability(
        graph,
        10,
        20,
        100.0,
        previous_projection_ratio=0.5,
        current_projection_ratio=0.5,
    )

    # 50 m remaining + 0 m connector + 50 m travelled.
    assert probability == pytest.approx(1.0)


def test_same_segment_transition():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        0, 1, key=10,
        arc_id=10, length_m=100.0
    )

    probability = calculate_transition_probability(
        graph,
        10,
        10,
        40.0,
        previous_projection_ratio=0.2,
        current_projection_ratio=0.6,
    )

    assert probability == pytest.approx(1.0)


def test_reverse_movement_on_same_directed_segment():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        0, 1, key=10,
        arc_id=10, length_m=100.0
    )

    probability = calculate_transition_probability(
        graph,
        10,
        10,
        40.0,
        previous_projection_ratio=0.8,
        current_projection_ratio=0.2,
    )

    assert probability == 0.0


def test_invalid_projection_ratio():
    graph = make_graph()

    with pytest.raises(ValueError):
        calculate_transition_probability(
            graph,
            10,
            20,
            100.0,
            previous_projection_ratio=1.5,
        )
