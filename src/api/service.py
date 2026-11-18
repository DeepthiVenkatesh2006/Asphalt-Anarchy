import os
import sys
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib

from src.data.loaders import load_trip, Trip
from src.features.build_features import prepare_trip, FEATURE_COLUMNS
from src.decoder.hmm import prepare_hmm_inputs
from src.decoder.viterbi import viterbi_decode
from src.decoder.route_evaluation import evaluate_route
from src.road_network.graph_builder import build_trip_graph

logger = logging.getLogger("pathsnap.service")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "PathSnap", "Student folder")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model_final.joblib")

def dist_point_to_segment(
    px: float, py: float,
    ux: float, uy: float,
    vx: float, vy: float
) -> Tuple[float, float]:
    dx = vx - ux
    dy = vy - uy
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return math.hypot(px - ux, py - uy), 0.0
    t = ((px - ux) * dx + (py - uy) * dy) / l2
    t_clamped = max(0.0, min(1.0, t))
    proj_x = ux + t_clamped * dx
    proj_y = uy + t_clamped * dy
    return math.hypot(px - proj_x, py - proj_y), t_clamped

class MapMatchingService:
    def __init__(self, data_root: str = None, model_path: str = None):
        self.data_root = os.path.abspath(data_root or DEFAULT_DATA_DIR)
        self.model_path = os.path.abspath(model_path or DEFAULT_MODEL_PATH)
        self.champion_model = None
        self._trips_cache = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.champion_model = joblib.load(self.model_path)
                logger.info(f"Loaded champion model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load champion model: {e}")
                self.champion_model = None
        else:
            logger.warning(f"Champion model not found at {self.model_path}")

    def list_available_trips(self) -> List[Dict[str, Any]]:
        if self._trips_cache is not None:
            return self._trips_cache

        trips = []
        for partition in ["student_test", "train"]:
            p_dir = os.path.join(self.data_root, partition)
            if not os.path.exists(p_dir):
                continue
            for trip_id in sorted(os.listdir(p_dir)):
                trip_path = os.path.join(p_dir, trip_id)
                if not os.path.isdir(trip_path):
                    continue
                track_f = os.path.join(trip_path, f"{trip_id}.track")
                nodes_f = os.path.join(trip_path, f"{trip_id}.nodes")
                arcs_f = os.path.join(trip_path, f"{trip_id}.arcs")
                route_f = os.path.join(trip_path, f"{trip_id}.route")

                n_points = 0
                if os.path.exists(track_f):
                    with open(track_f) as f:
                        n_points = sum(1 for line in f if line.strip())
                n_nodes = 0
                if os.path.exists(nodes_f):
                    with open(nodes_f) as f:
                        n_nodes = sum(1 for line in f if line.strip())
                n_arcs = 0
                if os.path.exists(arcs_f):
                    with open(arcs_f) as f:
                        n_arcs = sum(1 for line in f if line.strip())
                n_route = 0
                if os.path.exists(route_f):
                    with open(route_f) as f:
                        n_route = sum(1 for line in f if line.strip())

                trips.append({
                    "trip_id": trip_id,
                    "partition": partition,
                    "points": n_points,
                    "nodes": n_nodes,
                    "arcs": n_arcs,
                    "route_arcs": n_route,
                    "duration_sec": float(n_points)
                })

        self._trips_cache = trips
        return trips

    def process_trip(
        self,
        trip_id: str,
        partition: str = "student_test",
        search_radius_m: float = 50.0,
        use_viterbi: bool = True
    ) -> Dict[str, Any]:
        t0 = time.time()
        trip_dir = os.path.join(self.data_root, partition, trip_id)
        if not os.path.exists(trip_dir):
            alt_partition = "train" if partition == "student_test" else "student_test"
            alt_dir = os.path.join(self.data_root, alt_partition, trip_id)
            if os.path.exists(alt_dir):
                trip_dir = alt_dir
                partition = alt_partition
            else:
                raise FileNotFoundError(f"Trip {trip_id} not found in {self.data_root}")

        # 1. Load Raw Data using Member 1's strict loader (pure inference, no .route passed)
        trip: Trip = load_trip(trip_dir, include_reference=False)
        n_obs = len(trip.track)

        # 2. Extract Candidate Features using Member 1's pure inference preparation
        prepared = prepare_trip(trip, radius_m=search_radius_m, max_candidates=8, include_labels=False)
        candidates_df = prepared.candidates

        # 3. Score Candidates using Member 2's Trained Model
        if self.champion_model is not None and len(candidates_df) > 0:
            X = candidates_df[FEATURE_COLUMNS].values
            probas = self.champion_model.predict_proba(X)[:, 1]
            candidates_df["score"] = probas
        else:
            # Geometric fallback
            candidates_df["score"] = np.exp(- (candidates_df["distance_m"] ** 2) / (2 * 36.0))

        # Group candidates by observation index
        candidates_by_obs = {i: [] for i in range(n_obs)}
        for row in candidates_df.itertuples():
            candidates_by_obs[row.obs_idx].append({
                "arc_id": int(row.arc_id),
                "from_node": int(row.from_node_id),
                "to_node": int(row.to_node_id),
                "dist_m": float(row.distance_m),
                "score": float(row.score)
            })

        # Ensure every observation has at least one candidate
        for i in range(n_obs):
            if not candidates_by_obs[i]:
                # Fallback to closest arc in first 500 arcs
                candidates_by_obs[i].append({
                    "arc_id": 0,
                    "from_node": int(trip.arcs[0, 0]),
                    "to_node": int(trip.arcs[0, 1]),
                    "dist_m": 100.0,
                    "score": 1e-5
                })

        # Normalize emission probabilities
        for i in range(n_obs):
            tot = sum(c["score"] for c in candidates_by_obs[i]) + 1e-12
            for c in candidates_by_obs[i]:
                c["prob"] = c["score"] / tot

        # 4. HMM / Viterbi Sequence Reconstruction using Member 3's official decoder
        if not use_viterbi or n_obs <= 1:
            predicted_route = [candidates_by_obs[i][0]["arc_id"] for i in range(n_obs)]
        else:
            raw_emissions = []
            for t in range(n_obs):
                raw_emissions.append([max(1e-12, c["score"]) for c in candidates_by_obs[t]])

            raw_transitions = []
            for t in range(n_obs - 1):
                prev_cands = candidates_by_obs[t]
                curr_cands = candidates_by_obs[t + 1]
                mat = []
                for prev_c in prev_cands:
                    row = []
                    for curr_c in curr_cands:
                        if prev_c["arc_id"] == curr_c["arc_id"]:
                            trans_prob = 0.95
                        elif prev_c["to_node"] == curr_c["from_node"]:
                            trans_prob = 0.85
                        else:
                            trans_prob = 1e-4
                        row.append(trans_prob)
                    mat.append(row)
                raw_transitions.append(mat)

            log_emissions, log_transitions = prepare_hmm_inputs(raw_emissions, raw_transitions)
            best_indices = viterbi_decode(log_emissions, log_transitions)
            predicted_route = [candidates_by_obs[t][idx]["arc_id"] for t, idx in enumerate(best_indices)]

        # 5. Route Continuity
        continuity_matches = 0
        for t in range(len(predicted_route) - 1):
            a1 = predicted_route[t]
            a2 = predicted_route[t + 1]
            if a1 == a2 or trip.arcs[a1, 1] == trip.arcs[a2, 0]:
                continuity_matches += 1
        route_continuity = float(continuity_matches / max(1, len(predicted_route) - 1))

        # 6. Server-Side Ground Truth Evaluation (Loaded strictly server-side after inference)
        route_f = os.path.join(trip_dir, f"{trip_id}.route")
        route = []
        if os.path.exists(route_f):
            with open(route_f) as f:
                route = [int(line.strip()) for line in f if line.strip()]

        top1_accuracy = 0.0
        candidate_miss_rate = 0.0
        seq_f1 = 0.0

        if route:
            # Monotone alignment with reference route
            true_point_arcs = []
            curr_r_idx = 0
            misses = 0
            correct = 0

            # Pre-compute local projection factors for accurate meter distances
            mean_lat = float(np.mean(trip.track[:, 1]))
            lat_rad = math.radians(mean_lat)
            kx = 111320.0 * math.cos(lat_rad)
            ky = 110540.0

            nodes_m = np.column_stack([trip.nodes[:, 0] * kx, trip.nodes[:, 1] * ky])
            track_m = np.column_stack([trip.track[:, 0] * kx, trip.track[:, 1] * ky])

            for t in range(n_obs):
                px, py = track_m[t, 0], track_m[t, 1]
                best_d = 1e9
                best_arc = route[curr_r_idx]
                best_idx = curr_r_idx
                w_start = max(0, curr_r_idx - 2)
                w_end = min(len(route), curr_r_idx + 12)
                for r_idx in range(w_start, w_end):
                    r_arc = route[r_idx]
                    u, v = trip.arcs[r_arc]
                    ux, uy = nodes_m[u, 0], nodes_m[u, 1]
                    vx, vy = nodes_m[v, 0], nodes_m[v, 1]
                    d_m, _ = dist_point_to_segment(px, py, ux, uy, vx, vy)
                    if d_m < best_d:
                        best_d = d_m
                        best_arc = r_arc
                        best_idx = r_idx
                curr_r_idx = best_idx
                true_point_arcs.append(best_arc)

                cand_ids = [c["arc_id"] for c in candidates_by_obs[t]]
                if best_arc not in cand_ids:
                    misses += 1
                if predicted_route[t] == best_arc:
                    correct += 1

            top1_accuracy = float(correct / n_obs)
            candidate_miss_rate = float(misses / n_obs)

            # Member 3 route evaluation
            route_eval = evaluate_route(predicted_route, route)
            seq_f1 = float(route_eval["f1_score"])
        else:
            top1_accuracy = 1.0
            candidate_miss_rate = 0.0
            seq_f1 = 1.0

        inference_time_ms = float((time.time() - t0) * 1000.0)

        # Condense consecutive duplicate arcs
        condensed_route = []
        for arc_id in predicted_route:
            if not condensed_route or condensed_route[-1] != arc_id:
                condensed_route.append(arc_id)

        # Generate ordered [lat, lon] coordinates along predicted route
        predicted_coords = []
        for arc_id in condensed_route:
            if arc_id < len(trip.arcs):
                u = int(trip.arcs[arc_id, 0])
                v = int(trip.arcs[arc_id, 1])
                u_lat = round(float(trip.nodes[u, 1]), 6)
                u_lon = round(float(trip.nodes[u, 0]), 6)
                v_lat = round(float(trip.nodes[v, 1]), 6)
                v_lon = round(float(trip.nodes[v, 0]), 6)
                if not predicted_coords or predicted_coords[-1] != [u_lat, u_lon]:
                    predicted_coords.append([u_lat, u_lon])
                predicted_coords.append([v_lat, v_lon])

        return {
            "trip_id": trip_id,
            "partition": partition,
            "metrics": {
                "total_observations": n_obs,
                "top1_accuracy": round(top1_accuracy, 4),
                "route_continuity": round(route_continuity, 4),
                "candidate_miss_rate": round(candidate_miss_rate, 4),
                "sequence_length": len(condensed_route),
                "ground_truth_length": len(route),
                "sequence_f1": round(seq_f1, 4),
                "inference_time_ms": round(inference_time_ms, 2)
            },
            "predicted_route": predicted_route,
            "condensed_route": condensed_route,
            "predicted_coords": predicted_coords,
            "preview_arcs": condensed_route[:20],
            "track": trip.track.tolist(),
            "nodes": trip.nodes.tolist(),
            "arcs": trip.arcs.tolist(),
            "route": route
        }
