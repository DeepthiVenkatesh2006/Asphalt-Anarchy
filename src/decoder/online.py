"""Connect live candidate scoring to Member 3's graph transitions and Viterbi."""
import numpy as np
from pyproj import Geod
from src.road_network.graph_builder import build_trip_graph
from src.decoder.transition_model import calculate_transition_probability
from src.decoder.hmm import prepare_hmm_inputs
from src.decoder.viterbi import viterbi_decode


def decode_candidates(prepared, *, use_viterbi=True):
    groups = {int(i): g.reset_index(drop=True) for i, g in prepared.candidates.groupby('obs_idx', sort=True)}
    result = [-1] * len(prepared.observations)
    if not use_viterbi:
        for i, group in groups.items():
            result[i] = int(group.loc[group.candidate_score.idxmax(), 'arc_id'])
        return result, []
    if not groups:
        return result, []
    graph = build_trip_graph(prepared.road_segments, str(prepared.candidates.iloc[0].trip_id))
    geod = Geod(ellps='WGS84')
    chain, matrices, reachable = [], [], None
    restarts = []

    def flush():
        if chain:
            emissions = [g.candidate_probability.astype(float).tolist() for _, g in chain]
            log_e, log_t = prepare_hmm_inputs(emissions, matrices)
            selected = viterbi_decode(log_e, log_t)
            for (i, g), selected_idx in zip(chain, selected):
                result[i] = int(g.iloc[selected_idx].arc_id)
        chain.clear()
        matrices.clear()

    previous = None
    for obs in prepared.observations.itertuples(index=False):
        i = int(obs.obs_idx)
        group = groups.get(i)
        if group is None or bool(obs.gap_before):
            flush()
            previous = None
            reachable = None
        if group is None:
            continue
        if chain:
            prev_group = chain[-1][1]
            _, _, distance = geod.inv(previous.longitude, previous.latitude, obs.longitude, obs.latitude)
            mat = np.zeros((len(prev_group), len(group)))
            for a, prev in enumerate(prev_group.itertuples(index=False)):
                for b, curr in enumerate(group.itertuples(index=False)):
                    mat[a, b] = calculate_transition_probability(
                        graph, int(prev.arc_id), int(curr.arc_id), float(distance),
                        previous_projection_ratio=float(prev.projection_ratio),
                        current_projection_ratio=float(curr.projection_ratio))
            # Cache source searches within a timestep; do not retain every GPS-specific cutoff.
            graph.graph.get('_transition_cache', {}).get('source_distances', {}).clear()
            next_reachable = ((mat > 0) & reachable[:, None]).any(axis=0)
            if not next_reachable.any():
                flush()
                restarts.append(i)
                reachable = np.ones(len(group), dtype=bool)
            else:
                matrices.append(mat)
                reachable = next_reachable
        else:
            reachable = np.ones(len(group), dtype=bool)
        chain.append((i, group))
        previous = obs
    flush()
    return result, restarts


def expand_selected_route(prepared, predicted, restart_indices):
    """Insert shortest directed connector arcs, preserving observation/gap boundaries."""
    import networkx as nx
    roads = prepared.road_segments.set_index('arc_id')
    graph = build_trip_graph(prepared.road_segments, str(prepared.road_segments.iloc[0].trip_id))
    expanded, connected, previous, cache = [], 0, None, {}
    restarts = set(restart_indices)
    for i, arc in enumerate(predicted):
        if arc < 0:
            previous = None
            continue
        if bool(prepared.observations.iloc[i].gap_before) or i in restarts:
            previous = None
        if previous is not None:
            if previous == arc:
                connected += 1
            else:
                key = (previous, arc)
                if key not in cache:
                    source = int(roads.loc[previous, 'to_node_id'])
                    target = int(roads.loc[arc, 'from_node_id'])
                    try:
                        nodes = nx.shortest_path(graph, source, target, weight='length_m')
                        cache[key] = [int(min(graph[u][v].items(), key=lambda item: (item[1]['length_m'], item[0]))[0]) for u,v in zip(nodes,nodes[1:])]
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        cache[key] = None
                connectors = cache[key]
                if connectors is not None:
                    connected += 1
                    for connector in connectors:
                        if not expanded or expanded[-1] != connector:
                            expanded.append(connector)
        if not expanded or expanded[-1] != arc:
            expanded.append(arc)
        previous = arc
    return expanded, connected
