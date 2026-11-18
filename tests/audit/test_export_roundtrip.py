from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.features.build_features import FEATURE_COLUMNS
from src.models.local_training import run
from src.models.scoring import CandidateScorer


def test_fresh_training_export_loads_without_manual_packaging(tmp_path,monkeypatch):
    root=tmp_path/'processed'
    for partition,ntrips in [('train',4),('student_test',2)]:
        folder=root/partition;folder.mkdir(parents=True)
        candidates=[];labels=[]
        for trip in range(ntrips):
            tid=f'{trip:08d}'
            for obs in range(8):
                labels.append(dict(trip_id=tid,obs_idx=obs,reference_arc_id=0,alignment_status='aligned'))
                for arc in [0,1]:
                    row=dict(trip_id=tid,obs_idx=obs,candidate_idx=arc,arc_id=arc,target=int(arc==0))
                    row.update({name:float(arc+1) for name in FEATURE_COLUMNS})
                    row['distance_m']=1.0 if arc==0 else 20.0
                    candidates.append(row)
        frame=pd.DataFrame(candidates)
        frame.drop(columns='target').to_parquet(folder/'candidates.parquet',index=False)
        frame.to_parquet(folder/'train_features.parquet',index=False)
        pd.DataFrame(labels).to_parquet(folder/'reference_labels.parquet',index=False)
        (folder/'feature_columns.json').write_text(json.dumps(FEATURE_COLUMNS))
        (folder/'manifest.json').write_text(json.dumps(dict(radius_m=50,max_candidates=8,gap_s=60,reference_radius_m=100,alignment_config={},trips=[dict(trip_id=f'{i:08d}',observations=8) for i in range(ntrips)])))
    # Plot styling is unrelated to the export->inference contract under test.
    monkeypatch.setattr('src.models.local_training.make_plots',lambda *args:None)
    output=tmp_path/'fresh_training'
    args=SimpleNamespace(data_root=str(root),output_dir=str(output),folds=2,tuning_folds=2,trials=1,knn_observations=100,seed=42)
    with threadpool_limits(limits=2):run(args)
    assert (output/'COMPLETE').is_file()
    scorer=CandidateScorer.from_directory(output)
    records=pd.read_parquet(root/'student_test/candidates.parquet')
    result=scorer.score_candidates(records)
    assert np.allclose(result.groupby(['trip_id','obs_idx']).candidate_probability.sum(),1)
