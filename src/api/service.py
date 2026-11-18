"""Live inference uses only GPS/roads; reference evaluation runs afterwards."""
import logging
import os
import time
from pathlib import Path
import numpy as np
from src.data.loaders import load_trip
from src.features.build_features import prepare_trip, FEATURE_COLUMNS
from src.features.spatial import RoadIndex
from src.data.alignment import align_reference
from src.models.scoring import CandidateScorer
from src.decoder.online import decode_candidates, expand_selected_route
from src.decoder.route_evaluation import evaluate_route

logger = logging.getLogger('pathsnap.service')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = ('evaluator_test', 'student_test', 'train')


def default_data_root():
    for root in (PROJECT_ROOT.parent / 'Student folder', PROJECT_ROOT.parent.parent / 'Student folder', PROJECT_ROOT.parent / 'PathSnap' / 'Student folder'):
        if (root / 'train').is_dir() or (root / 'student_test').is_dir():
            return root
    return PROJECT_ROOT / 'data' / 'raw'


class MapMatchingService:
    def __init__(self, data_root=None, model_path=None):
        self.data_root = str(Path(data_root or os.environ.get('PATHSNAP_DATA_ROOT') or default_data_root()).resolve())
        self.model_path = str(Path(model_path or os.environ.get('PATHSNAP_MODEL_PATH') or PROJECT_ROOT / 'artifacts/local_training/model_final.joblib').resolve())
        self.scorer = CandidateScorer.from_directory(Path(self.model_path).parent)
        # An explicit model filename must match the artifact directory contract.
        if Path(self.model_path).name != 'model_final.joblib':
            raise ValueError('Model artifact must be named model_final.joblib alongside its metadata')
        self.champion_model = self.scorer.model
        if self.scorer.feature_columns != FEATURE_COLUMNS:
            raise ValueError('Model and feature-builder column orders differ')
        self._trips_cache = None

    def list_available_trips(self):
        if self._trips_cache is not None:
            return self._trips_cache
        trips = []
        for partition in PARTITIONS:
            root = Path(self.data_root) / partition
            if not root.exists():
                continue
            for track in sorted(root.glob('*/*.track')):
                tid = track.stem
                points = np.loadtxt(track, ndmin=2)
                def count(suffix):
                    path = track.with_suffix(suffix)
                    return sum(bool(line.strip()) for line in path.open()) if path.exists() else 0
                trips.append(dict(trip_id=tid, partition=partition, points=len(points), nodes=count('.nodes'), arcs=count('.arcs'), duration_sec=float(points[-1, 2] - points[0, 2])))
        self._trips_cache = trips
        return trips

    def process_trip(self, trip_id, partition='student_test', search_radius_m=50.0, use_viterbi=True):
        if partition not in PARTITIONS or len(trip_id) != 8 or not trip_id.isascii() or not trip_id.isdigit():
            raise ValueError('Use a known partition and an eight-digit trip ID')
        started = time.perf_counter()
        trip_dir = Path(self.data_root) / partition / trip_id
        if not trip_dir.is_dir():
            raise FileNotFoundError(f'Trip {trip_id} is absent from partition {partition}')
        trip = load_trip(trip_dir, include_reference=False)
        prepared = prepare_trip(trip, radius_m=search_radius_m, max_candidates=8, include_labels=False)
        scores = self.scorer.score_candidates(prepared.candidates)
        prepared.candidates['candidate_score'] = scores.candidate_score.to_numpy()
        prepared.candidates['candidate_probability'] = scores.candidate_probability.to_numpy()
        predicted, restarts = decode_candidates(prepared, use_viterbi=use_viterbi)
        condensed, connected = expand_selected_route(prepared, predicted, restarts)
        inference_ms = (time.perf_counter() - started) * 1000
        valid = [a for a in predicted if a >= 0]
        continuity = connected / max(1, len(predicted) - 1)
        # Evaluation happens after all prediction decisions are frozen.
        accuracy = miss_rate = coverage = f1 = None
        label_count = 0
        route = []
        if (trip_dir / f'{trip_id}.route').exists():
            reference_trip = load_trip(trip_dir, include_reference=True)
            route = reference_trip.route.tolist()
            labels = align_reference(RoadIndex(reference_trip), prepared.observations)
            eligible = labels[labels.alignment_status == 'aligned']
            label_count = len(eligible)
            coverage = label_count / len(predicted)
            candidate_ids = prepared.candidates.groupby('obs_idx').arc_id.agg(set).to_dict()
            if label_count:
                accuracy = sum(predicted[int(row.obs_idx)] == int(row.reference_arc_id) for row in eligible.itertuples()) / label_count
                miss_rate = sum(int(row.reference_arc_id) not in candidate_ids.get(int(row.obs_idx), set()) for row in eligible.itertuples()) / label_count
            f1 = evaluate_route(condensed, route)['f1_score'] if condensed and route else (0.0 if route else None)
        segments = []
        for a in condensed:
            u, v = trip.arcs[a]
            segments.append([[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])], [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]])
        return dict(trip_id=trip_id, partition=partition, metrics=dict(
            total_observations=len(predicted), top1_accuracy=accuracy, route_continuity=continuity,
            candidate_miss_rate=miss_rate, sequence_length=len(condensed), ground_truth_length=len(route),
            sequence_f1=f1, inference_time_ms=inference_ms, evaluation_observations=label_count,
            evaluation_coverage=coverage, unmatched_observations=predicted.count(-1), decoder_restarts=len(restarts),
            evaluation_note='Approximate aligned reference labels; road-set F1 is order-insensitive.'),
            predicted_route=predicted, condensed_route=condensed, predicted_segments=segments,
            predicted_coords=[point for segment in segments for point in segment],
            preview_arcs=condensed[:20], track=trip.track.tolist(), nodes=trip.nodes.tolist(), arcs=trip.arcs.tolist(), route=[])
