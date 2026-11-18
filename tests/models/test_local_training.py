import numpy as np
import pandas as pd
import pytest
from src.models.local_training import evaluate_routes, grouped_sample, make_model, probabilities


def test_missing_candidates_count_as_incorrect_observations():
    candidates=pd.DataFrame({'trip_id':['a','a'], 'obs_idx':[0,0], 'arc_id':[10,20]})
    labels=pd.DataFrame({'trip_id':['a','a'], 'obs_idx':[0,1], 'reference_arc_id':[10,30], 'alignment_status':['aligned','aligned']})
    result=evaluate_routes(candidates,np.array([.8,.2]),labels)
    assert result['top1_accuracy']==.5
    assert result['evaluated_observations']==2
    assert result['candidate_miss_rate']==.5
    assert result['observation_macro_f1']==pytest.approx(1/3)


def test_equal_probabilities_have_stable_first_candidate_tie_break():
    c=pd.DataFrame({'trip_id':['a','a'], 'obs_idx':[0,0], 'arc_id':[10,20]})
    l=pd.DataFrame({'trip_id':['a'], 'obs_idx':[0], 'reference_arc_id':[10], 'alignment_status':['aligned']})
    r=evaluate_routes(c,np.array([.5,.5]),l)
    assert r['top1_accuracy']==1
    assert r['candidate_log_loss']==pytest.approx(np.log(2))


def test_labels_do_not_change_candidate_probability_normalization():
    c=pd.DataFrame({'trip_id':['a','a','a'], 'obs_idx':[0,0,0], 'arc_id':[10,20,30]})
    l=pd.DataFrame({'trip_id':['a'], 'obs_idx':[0], 'reference_arc_id':[20], 'alignment_status':['aligned']})
    assert evaluate_routes(c,np.array([0.,0.,0.]),l)['candidate_log_loss']==pytest.approx(np.log(3))


def test_training_sample_preserves_complete_candidate_groups():
    f=pd.DataFrame({'trip_id':['a']*8,'obs_idx':np.repeat(np.arange(4),2),'target':[0,1]*4})
    sampled=grouped_sample(f,2,42)
    assert len(sampled)==4
    assert sampled.groupby(['trip_id','obs_idx']).size().tolist()==[2,2]
    assert sampled.equals(grouped_sample(f,2,42))


def test_probability_batching_matches_single_batch():
    from sklearn.linear_model import LogisticRegression
    X=np.arange(40).reshape(20,2);y=np.tile([0,1],10)
    m=LogisticRegression().fit(X,y)
    assert np.allclose(probabilities(m,X,batch_size=3),m.predict_proba(X)[:,1])


@pytest.mark.parametrize('name',['logistic','knn','forest','histboost','mlp'])
def test_all_five_models_accept_small_binary_training_set(name):
    rng=np.random.default_rng(42);X=rng.normal(size=(100,10));y=(X[:,0]>0).astype(int)
    m=make_model(name,42,quick=True);m.fit(X,y)
    p=probabilities(m,X)
    assert p.shape==(100,)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()


def test_five_folds_hold_each_trip_out_once_without_leakage():
    from types import SimpleNamespace
    from src.models.local_training import validation_folds
    rows=[]
    for trip in range(10):
        for obs in range(2):
            for target in [0,1]:
                rows.append({'trip_id':str(trip),'obs_idx':obs,'target':target})
    frame=pd.DataFrame(rows)
    labels=frame[['trip_id','obs_idx']].drop_duplicates().reset_index(drop=True)
    part=SimpleNamespace(training=frame,candidates=frame,labels=labels)
    held_out=[]
    for fit,valid,label_idx,trip_ids in validation_folds(part,5):
        assert not (set(frame.iloc[fit].trip_id) & set(trip_ids))
        assert set(frame.iloc[valid].trip_id)==set(labels.iloc[label_idx].trip_id)==set(trip_ids)
        held_out.extend(trip_ids)
    assert sorted(held_out)==sorted(frame.trip_id.unique())


def test_ambiguous_labels_are_outside_primary_metric_denominator():
    c=pd.DataFrame({'trip_id':['a','a'],'obs_idx':[0,1],'arc_id':[10,99]})
    l=pd.DataFrame({'trip_id':['a','a'],'obs_idx':[0,1],'reference_arc_id':[10,20],
                    'alignment_status':['aligned','ambiguous']})
    result=evaluate_routes(c,np.array([.8,.9]),l)
    assert result['top1_accuracy']==1
    assert result['evaluated_observations']==1
    assert result['reference_coverage']==.5


def test_reporting_rejects_incomplete_training_run(tmp_path):
    from src.models.training_report import create_report
    with pytest.raises(ValueError,match='not completed'):
        create_report(tmp_path)
