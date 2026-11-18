"""Aggregate point metrics without confusing observation and trip denominators."""
import numpy as np


def summarize(records):
    if not records:
        raise ValueError('No trip results to summarize')
    total = sum(r['total_observations'] for r in records)
    labels = sum(r['evaluation_observations'] for r in records)
    def mean(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    def weighted(key, weight):
        rows = [r for r in records if r.get(key) is not None]
        denominator = sum(weight(r) for r in rows)
        return sum(r[key]*weight(r) for r in rows)/denominator if denominator else None
    return dict(trips=len(records), total_observations=total, evaluation_observations=labels,
        evaluation_coverage=labels/total if total else None,
        classifier_top1_accuracy=weighted('top1_accuracy',lambda r:r['evaluation_observations']),
        point_accuracy=weighted('point_accuracy',lambda r:r['evaluation_observations']),
        observation_macro_f1=mean('observation_macro_f1'),
        observation_macro_precision=mean('observation_macro_precision'),
        observation_macro_recall=mean('observation_macro_recall'),
        route_set_f1=mean('route_set_f1'), sequence_similarity=mean('sequence_similarity'),
        route_continuity=weighted('route_continuity',lambda r:max(0,r['total_observations']-1)),
        candidate_miss_rate=weighted('candidate_miss_rate',lambda r:r['evaluation_observations']),
        unmatched_observations=sum(r['unmatched_observations'] for r in records),
        decoder_restarts=sum(r['decoder_restarts'] for r in records),
        mean_inference_ms=mean('inference_time_ms'))
