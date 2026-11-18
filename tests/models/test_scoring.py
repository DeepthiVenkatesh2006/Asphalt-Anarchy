"""Member 3 scoring contract tests using a tiny synthetic fitted classifier."""
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.scoring import CandidateScorer


@pytest.fixture
def scorer(tmp_path):
    model=make_pipeline(StandardScaler(),LogisticRegression(random_state=42))
    model.fit(np.array([[0,0],[1,0],[9,1],[10,1]]),np.array([1,1,0,0]))
    joblib.dump(model,tmp_path/'model_final.joblib')
    (tmp_path/'feature_columns.json').write_text(json.dumps(['distance_m','heading_delta_deg']))
    (tmp_path/'model_metadata.json').write_text(json.dumps({'feature_columns':['distance_m','heading_delta_deg']}))
    names=['model_final.joblib','feature_columns.json','model_metadata.json']
    (tmp_path/'artifact_manifest.json').write_text(json.dumps({'files':[{'path':name,'sha256':hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()} for name in names]}))
    return CandidateScorer.from_directory(tmp_path)


@pytest.fixture
def candidates():
    return pd.DataFrame({'trip_id':['00000001']*3,'obs_idx':[0,0,1],
        'candidate_idx':[0,1,0],'arc_id':[10,20,30],
        'distance_m':[1.,9.,2.],'heading_delta_deg':[0.,1.,0.]},index=[5,8,11])


def test_scoring_preserves_keys_order_and_normalizes_per_observation(scorer,candidates):
    output=scorer.score_candidates(candidates,batch_size=1)
    assert output.index.tolist()==[5,8,11]
    assert output.arc_id.tolist()==[10,20,30]
    assert np.allclose(output.groupby(['trip_id','obs_idx']).candidate_probability.sum(),1)
    assert output.iloc[0].candidate_score>output.iloc[1].candidate_score
    assert np.allclose(output.candidate_score,scorer.model.predict_proba(candidates[scorer.feature_columns].to_numpy())[:,1])


def test_extra_reference_columns_do_not_affect_scores(scorer,candidates):
    before=scorer.score_candidates(candidates)
    candidates['target']=[0,1,0]
    candidates['reference_arc_id']=[20,20,999]
    assert scorer.score_candidates(candidates).equals(before)


def test_missing_feature_and_duplicate_candidate_are_rejected(scorer,candidates):
    with pytest.raises(ValueError,match='Missing'):
        scorer.score_candidates(candidates.drop(columns='distance_m'))
    with pytest.raises(ValueError,match='Duplicate'):
        scorer.score_candidates(pd.concat([candidates,candidates.iloc[:1]]))


def test_nonfinite_values_and_invalid_batch_size_are_rejected(scorer,candidates):
    with pytest.raises(ValueError,match='batch_size'):
        scorer.score_candidates(candidates,batch_size=0)
    candidates.loc[5,'distance_m']=float('nan')
    with pytest.raises(ValueError,match='finite'):
        scorer.score_candidates(candidates)


def test_empty_candidate_frame_returns_explicit_empty_output(scorer,candidates):
    output=scorer.score_candidates(candidates.iloc[:0])
    assert output.empty
    assert list(output.columns)==['trip_id','obs_idx','candidate_idx','arc_id','candidate_score','candidate_probability']


def test_modified_model_artifact_is_rejected_before_deserialization(tmp_path, scorer):
    # scorer fixture created a valid bundle in the same directory.
    with (tmp_path/'model_final.joblib').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        CandidateScorer.from_directory(tmp_path)


def test_fresh_export_manifest_is_accepted_by_scorer(tmp_path):
    from src.models.artifacts import write_artifact_manifest
    model=make_pipeline(StandardScaler(),LogisticRegression(random_state=42))
    model.fit([[0],[1],[9],[10]],[1,1,0,0])
    joblib.dump(model,tmp_path/'model_final.joblib')
    (tmp_path/'feature_columns.json').write_text('["distance_m"]')
    (tmp_path/'model_metadata.json').write_text('{"feature_columns":["distance_m"]}')
    write_artifact_manifest(tmp_path)
    assert CandidateScorer.from_directory(tmp_path).feature_columns == ['distance_m']
