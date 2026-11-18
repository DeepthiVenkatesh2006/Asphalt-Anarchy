"""Connect live candidate scoring to Member 3's graph transitions and Viterbi."""
import numpy as np
from pyproj import Geod
from src.road_network.graph_builder import build_trip_graph
from src.decoder.transition_model import calculate_transition_probability
from src.decoder.hmm import prepare_hmm_inputs
from src.decoder.viterbi import viterbi_decode


def decode_candidates(prepared, *, use_viterbi=True, beta=50.0):
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
                        graph, int(prev.arc_id), int(curr.arc_id), float(distance), beta=beta,
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


def expand_route_chunks(prepared, predicted, restart_indices, *, beta=50.0):
    """Expand valid directed transitions, retaining breaks and return loops.

    Continuity uses the same projected-position transition validity as decoding.
    A repeated state with a smaller projection requires a real return connector.
    """
    import networkx as nx
    if len(predicted) != len(prepared.observations):
        raise ValueError('Predictions must preserve every observation')
    roads = prepared.road_segments.set_index('arc_id')
    graph = build_trip_graph(prepared.road_segments, str(prepared.road_segments.iloc[0].trip_id))
    ratios = {}
    if hasattr(prepared, 'candidates') and 'projection_ratio' in prepared.candidates:
        ratios = {(int(r.obs_idx), int(r.arc_id)): float(r.projection_ratio)
                  for r in prepared.candidates.itertuples(index=False)}
    chunks, current, connected, previous, cache = [], [], 0, None, {}
    restarts = set(restart_indices)
    geod = Geod(ellps='WGS84')

    def flush():
        if current:
            chunks.append(current.copy())
            current.clear()

    for i, raw_arc in enumerate(predicted):
        arc = int(raw_arc)
        if arc == -1:
            flush()
            previous = None
            continue
        if arc not in roads.index:
            raise ValueError(f'Unknown predicted arc {arc}')
        if bool(prepared.observations.iloc[i].gap_before) or i in restarts:
            flush()
            previous = None
        if previous is not None:
            previous_i, previous_arc = previous
            prev_ratio = ratios.get((previous_i, previous_arc), 0.0)
            curr_ratio = ratios.get((i, arc), 1.0)
            valid = True
            if {'longitude', 'latitude'}.issubset(prepared.observations.columns):
                a, b = prepared.observations.iloc[previous_i], prepared.observations.iloc[i]
                _, _, distance = geod.inv(a.longitude, a.latitude, b.longitude, b.latitude)
                valid = calculate_transition_probability(graph, previous_arc, arc, float(distance), beta=beta,
                    previous_projection_ratio=prev_ratio, current_projection_ratio=curr_ratio) > 0
                graph.graph.get('_transition_cache', {}).get('source_distances', {}).clear()
            connectors = []
            if valid and (previous_arc != arc or curr_ratio < prev_ratio):
                source = int(roads.loc[previous_arc, 'to_node_id'])
                target = int(roads.loc[arc, 'from_node_id'])
                key = (source, target)
                if key not in cache:
                    try:
                        nodes = nx.shortest_path(graph, source, target, weight='length_m')
                        cache[key] = [int(min(graph[u][v].items(), key=lambda item: (item[1]['length_m'], item[0]))[0])
                                      for u, v in zip(nodes, nodes[1:])]
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        cache[key] = None
                connectors = cache[key]
                valid = connectors is not None
            if valid:
                connected += 1
                current.extend(connectors)
            else:
                flush()
        if not current or current[-1] != arc:
            current.append(arc)
        previous = (i, arc)
    flush()
    return chunks, connected


def expand_selected_route(prepared, predicted, restart_indices, *, beta=50.0):
    """Compatibility interface; chunk boundaries remain available in expand_route_chunks."""
    chunks, connected = expand_route_chunks(prepared, predicted, restart_indices, beta=beta)
    return [arc for chunk in chunks for arc in chunk], connected
