import pandas as pd
from src.api.evaluation import point_metrics
from src.api.schemas import EvaluationMetrics


def test_classifier_and_decoded_point_accuracy_are_distinct():
    labels=pd.DataFrame(dict(obs_idx=[0,1,2],alignment_status=['aligned','aligned','ambiguous'],reference_arc_id=[0,0,1]))
    classifier=point_metrics([0,1,1],labels)
    decoded=point_metrics([0,0,1],labels)
    assert classifier['accuracy']==.5
    assert decoded['accuracy']==1
    assert classifier['observations']==decoded['observations']==2


def test_missing_candidate_remains_in_accuracy_denominator():
    labels=pd.DataFrame(dict(obs_idx=[0,1],alignment_status=['aligned']*2,reference_arc_id=[0,0]))
    assert point_metrics([0,-1],labels)['accuracy']==.5
