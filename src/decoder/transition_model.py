
import math
import networkx as nx


def _get_transition_cache(graph):
    """Cache arc indexes and single-source shortest-path results."""

    if "_transition_cache" not in graph.graph:
        arc_index = {}

        for u, v, key, data in graph.edges(
            keys=True, data=True
        ):
            arc_id = int(data.get("arc_id", key))
            arc_index.setdefault(arc_id, []).append(
                (u, v, data)
            )

        graph.graph["_transition_cache"] = {
            "arc_index": arc_index,
            "source_distances": {},
        }

    return graph.graph["_transition_cache"]


def _shortest_connector_distance(
    graph, source, target, cache, cutoff
):
    """
    Reuse single-source Dijkstra results for multiple targets.
    Distances beyond cutoff are treated as unreachable.
    """

    cache_key = (source, cutoff)

    if cache_key not in cache["source_distances"]:
        try:
            distances = nx.single_source_dijkstra_path_length(
                graph,
                source=source,
                weight="length_m",
                cutoff=cutoff,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            distances = {}

        cache["source_distances"][cache_key] = distances

    return cache["source_distances"][cache_key].get(
        target, math.inf
    )


def calculate_transition_probability(
    graph: nx.MultiDiGraph,
    previous_arc_id: int,
    current_arc_id: int,
    observed_distance_m: float,
    beta: float = 50.0,
    previous_projection_ratio: float = 0.0,
    current_projection_ratio: float = 1.0,
) -> float:

    if beta <= 0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and greater than zero")

    if observed_distance_m < 0 or not math.isfinite(observed_distance_m):
        raise ValueError("observed_distance_m must be finite and non-negative")

    for name, ratio in (
        ("previous_projection_ratio", previous_projection_ratio),
        ("current_projection_ratio", current_projection_ratio),
    ):
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    cache = _get_transition_cache(graph)
    arc_index = cache["arc_index"]

    previous_edges = arc_index.get(int(previous_arc_id), [])
    current_edges = arc_index.get(int(current_arc_id), [])

    if not previous_edges or not current_edges:
        return 0.0

    best_probability = 0.0

    # A connector farther than this produces a score below exp(-10).
    cutoff = observed_distance_m + 10.0 * beta

    for previous_from, previous_to, previous_data in previous_edges:
        previous_length = float(previous_data["length_m"])

        for current_from, current_to, current_data in current_edges:
            current_length = float(current_data["length_m"])

            if (
                previous_arc_id == current_arc_id
                and previous_from == current_from
                and previous_to == current_to
                and current_projection_ratio >= previous_projection_ratio
            ):
                road_distance = (
                    current_projection_ratio
                    - previous_projection_ratio
                ) * previous_length

            else:
                previous_remaining = (
                    1.0 - previous_projection_ratio
                ) * previous_length

                current_travelled = (
                    current_projection_ratio * current_length
                )

                connector_distance = _shortest_connector_distance(
                    graph,
                    previous_to,
                    current_from,
                    cache,
                    cutoff,
                )

                if not math.isfinite(connector_distance):
                    continue

                road_distance = (
                    previous_remaining
                    + connector_distance
                    + current_travelled
                )

            error = abs(road_distance - observed_distance_m)
            probability = math.exp(-error / beta)

            best_probability = max(best_probability, probability)

    return best_probability
