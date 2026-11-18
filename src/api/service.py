"""Live inference uses only GPS/roads; reference evaluation runs afterwards."""
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.loaders import load_trip, load_reference
from src.features.build_features import prepare_trip, FEATURE_COLUMNS
from src.features.spatial import RoadIndex
from src.data.alignment import align_reference
from src.models.scoring import CandidateScorer
from src.decoder.online import decode_candidates, expand_route_chunks
from src.api.config import load_inference_settings
from src.api.evaluation import point_metrics
from src.models.local_training import evaluate_routes
from src.decoder.route_evaluation import evaluate_route

logger = logging.getLogger('pathsnap.service')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = ('evaluator_test', 'student_test', 'train')


def default_data_root():
    for root in (PROJECT_ROOT.parent / 'Student folder', PROJECT_ROOT.parent.parent / 'Student folder', PROJECT_ROOT.parent / 'PathSnap' / 'Student folder'):
        if any((root / partition).is_dir() for partition in PARTITIONS):
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
        self.settings = load_inference_settings()
        mode = os.environ.get("PATHSNAP_MODE", "development")
        if mode not in {"development", "evaluation"}:
            raise ValueError("PATHSNAP_MODE must be development or evaluation")
        self.evaluation_mode = mode == "evaluation"
        self.catalog_errors = []
        self.reference_errors = []
        self._trips_cache = None

    def list_available_trips(self):
        if self._trips_cache is not None:
            return self._trips_cache
        trips = []
        self.catalog_errors = []
        self.reference_errors = []
        for partition in PARTITIONS:
            root = Path(self.data_root) / partition
            if not root.exists():
                continue
            for folder in sorted(p for p in root.iterdir() if p.is_dir()):
                tid = folder.name
                try:
                    if len(tid) != 8 or not tid.isascii() or not tid.isdigit():
                        raise ValueError('Trip folder must have an eight-digit ID')
                    trip = load_trip(folder, include_reference=False)
                    reference_available = (folder / f'{tid}.route').is_file()
                    reference_valid = False
                    if reference_available:
                        try:
                            load_reference(folder, len(trip.arcs))
                            reference_valid = True
                        except (OSError, ValueError) as error:
                            self.reference_errors.append(dict(partition=partition, trip_id=tid, error=str(error)))
                    trips.append(dict(trip_id=tid, partition=partition, points=len(trip.track),
                        nodes=len(trip.nodes), arcs=len(trip.arcs),
                        reference_available=reference_available, reference_valid=reference_valid,
                        duration_sec=float(trip.track[-1, 2] - trip.track[0, 2])))
                except (OSError, ValueError, IndexError) as error:
                    logger.error('Unusable trip %s/%s: %s', partition, tid, error)
                    self.catalog_errors.append(dict(partition=partition, trip_id=tid, error=str(error)))
        self._trips_cache = trips
        return trips

    def process_trip(self, trip_id, partition='student_test', search_radius_m=None, use_viterbi=True):
        if partition not in PARTITIONS or len(trip_id) != 8 or not trip_id.isascii() or not trip_id.isdigit():
            raise ValueError('Use a known partition and an eight-digit trip ID')
        if self.evaluation_mode and partition != 'evaluator_test':
            raise ValueError('Evaluation mode only accepts the evaluator_test partition')
        search_radius_m = self.settings.search_radius_meters if search_radius_m is None else float(search_radius_m)
        if partition == 'evaluator_test':
            expected = self.scorer.metadata['preparation']
            if (search_radius_m != expected['radius_m'] or not use_viterbi
                    or self.settings.max_candidates_per_point != expected['max_candidates']
                    or self.settings.gap_threshold_s != expected['gap_s']):
                raise ValueError('Final evaluator inference settings are frozen; use student_test for development experiments')
        started = time.perf_counter()
        trip_dir = Path(self.data_root) / partition / trip_id
        if not trip_dir.is_dir():
            raise FileNotFoundError(f'Trip {trip_id} is absent from partition {partition}')
        trip = load_trip(trip_dir, include_reference=False)
        prepared = prepare_trip(trip, radius_m=search_radius_m, max_candidates=self.settings.max_candidates_per_point, gap_s=self.settings.gap_threshold_s, include_labels=False)
        scores = self.scorer.score_candidates(prepared.candidates)
        prepared.candidates['candidate_score'] = scores.candidate_score.to_numpy()
        prepared.candidates['candidate_probability'] = scores.candidate_probability.to_numpy()
        classifier_predicted, _ = decode_candidates(prepared, use_viterbi=False)
        if use_viterbi:
            predicted, restarts = decode_candidates(prepared, use_viterbi=True, beta=self.settings.beta_transition)
        else:
            predicted, restarts = classifier_predicted, []
        chunks, connected = expand_route_chunks(prepared, predicted, restarts, beta=self.settings.beta_transition)
        condensed = [arc for chunk in chunks for arc in chunk]
        inference_ms = (time.perf_counter() - started) * 1000
        continuity = connected / max(1, len(predicted) - 1)
        # Evaluation happens after all prediction decisions are frozen.
        accuracy = classifier_accuracy = miss_rate = coverage = f1 = sequence_similarity = log_loss = log_loss_coverage = None
        point = dict(macro_precision=None, macro_recall=None, macro_f1=None)
        classifier_point = dict(macro_f1=None)
        label_count = 0
        route = []
        if (trip_dir / f'{trip_id}.route').exists():
            reference_trip = load_trip(trip_dir, include_reference=True)
            route = reference_trip.route.tolist()
            labels = align_reference(RoadIndex(reference_trip), prepared.observations)
            eligible = labels[labels.alignment_status == 'aligned']
            label_count = len(eligible)
            coverage = label_count / len(predicted)
            if label_count:
                point = point_metrics(predicted, labels)
                classifier_point = point_metrics(classifier_predicted, labels)
                accuracy, classifier_accuracy = point['accuracy'], classifier_point['accuracy']
                candidate_metrics = evaluate_routes(prepared.candidates, scores.candidate_score.to_numpy(), labels)
                miss_rate = candidate_metrics['candidate_miss_rate']
                log_loss = candidate_metrics['candidate_log_loss']
                log_loss_coverage = candidate_metrics['log_loss_observation_coverage']
            if route:
                route_scores = evaluate_route(condensed, route) if condensed else dict(f1_score=0.0, sequence_similarity=0.0)
                f1, sequence_similarity = route_scores['f1_score'], route_scores['sequence_similarity']
        matched_coords = []
        from pyproj import Transformer
        inverse = Transformer.from_crs(prepared.projection, 'EPSG:4326', always_xy=True)
        locations = {(int(r.obs_idx), int(r.arc_id)): (r.projected_x_m, r.projected_y_m)
                     for r in prepared.candidates.itertuples(index=False)}
        for i, arc in enumerate(predicted):
            if arc < 0:
                matched_coords.append(None)
            else:
                lon, lat = inverse.transform(*locations[(i, arc)])
                matched_coords.append([float(lat), float(lon)])
        segments = []
        for a in condensed:
            u, v = trip.arcs[a]
            segments.append([[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])], [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]])
        # Display context comes from the supplied graph, never reference labels.
        # Cap geometry for browser responsiveness; inference still uses every road.
        roads = prepared.road_segments
        lo, hi = trip.track[:, :2].min(axis=0) - .002, trip.track[:, :2].max(axis=0) + .002
        visible = roads[((roads.from_lon.between(lo[0], hi[0]) & roads.from_lat.between(lo[1], hi[1])) |
                         (roads.to_lon.between(lo[0], hi[0]) & roads.to_lat.between(lo[1], hi[1])))]
        selected_roads = roads[roads.arc_id.isin(condensed)].iloc[:5000]
        context = visible[~visible.arc_id.isin(selected_roads.arc_id)]
        budget = 5000 - len(selected_roads)
        if len(context) > budget:
            context = context.iloc[np.linspace(0, len(context)-1, budget, dtype=int)] if budget else context.iloc[:0]
        visible = pd.concat([selected_roads, context])
        network_segments = [[[float(r.from_lat), float(r.from_lon)], [float(r.to_lat), float(r.to_lon)]] for r in visible.itertuples()]
        return dict(trip_id=trip_id, partition=partition, search_radius_m=search_radius_m, metrics=dict(
            total_observations=len(predicted), top1_accuracy=classifier_accuracy, point_accuracy=accuracy,
            observation_macro_f1=point["macro_f1"], observation_macro_precision=point["macro_precision"],
            observation_macro_recall=point["macro_recall"], classifier_observation_macro_f1=classifier_point["macro_f1"],
            candidate_log_loss=log_loss, log_loss_observation_coverage=log_loss_coverage,
            sequence_similarity=sequence_similarity, route_set_f1=f1, route_continuity=continuity,
            candidate_miss_rate=miss_rate, sequence_length=len(condensed), ground_truth_length=len(route),
            sequence_f1=f1, inference_time_ms=inference_ms, evaluation_observations=label_count,
            evaluation_coverage=coverage, unmatched_observations=predicted.count(-1), decoder_restarts=len(restarts),
            evaluation_note='Top-1 is classifier argmax; point accuracy/F1 use returned assignments on approximate aligned labels. Road-set F1 ignores order; sequence similarity is order-sensitive with autojunk disabled.'),
            predicted_route=predicted, route_chunks=chunks, condensed_route=condensed, predicted_segments=segments, network_segments=network_segments,
            predicted_coords=[point for segment in segments for point in segment],
            matched_coords=matched_coords, gps_coords=trip.track[:, [1, 0]].tolist(), preview_arcs=condensed[:20], track=trip.track.tolist(), nodes=trip.nodes.tolist(), arcs=trip.arcs.tolist(), route=[])
