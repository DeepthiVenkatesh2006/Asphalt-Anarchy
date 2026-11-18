import pytest
from src.evaluation.summary import summarize


def test_trip_f1_and_point_accuracy_use_their_declared_denominators():
    common=dict(total_observations=100,evaluation_coverage=.5,top1_accuracy=.5,observation_macro_precision=.5,observation_macro_recall=.5,route_set_f1=.5,sequence_similarity=.5,route_continuity=1,candidate_miss_rate=0,unmatched_observations=0,decoder_restarts=0,inference_time_ms=1)
    a=dict(common,evaluation_observations=10,point_accuracy=1,observation_macro_f1=1)
    b=dict(common,evaluation_observations=90,point_accuracy=0,observation_macro_f1=0)
    result=summarize([a,b])
    assert result['point_accuracy']==pytest.approx(.1)
    assert result['observation_macro_f1']==pytest.approx(.5)
    assert result['evaluation_coverage']==pytest.approx(.5)
