"""Measure direct (unexpanded) classifier road-state continuity on student_test."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.models.local_training import probabilities
from src.models.artifacts import file_sha256


def main():
    root=Path('data/processed/v1/student_test')
    out=Path('reports/audit/model_metrics/direct_continuity.csv')
    if out.exists():raise FileExistsError(out)
    c=pd.read_parquet(root/'candidates.parquet')
    observations=pd.read_parquet(root/'observations.parquet')
    features=json.loads((root/'feature_columns.json').read_text())
    expected=json.loads(Path('reports/audit/model_metrics/verification.json').read_text())['model_hashes']
    rows=[]
    with threadpool_limits(limits=2):
        for name in expected:
            model_path=Path('reports/member2/local_training/models')/name
            if file_sha256(model_path)!=expected[name]:raise ValueError('Model changed since the primary-metric audit')
            scores=probabilities(joblib.load(model_path),c[features].to_numpy(dtype=float))
            work=c[['trip_id','obs_idx','arc_id','from_node_id','to_node_id','projection_ratio']].copy()
            work['score']=scores
            chosen=work.loc[work.groupby(['trip_id','obs_idx']).score.idxmax()]
            ledger=observations.merge(chosen,on=['trip_id','obs_idx'],how='left',validate='one_to_one')
            for tid,g in ledger.groupby('trip_id'):
                g=g.sort_values('obs_idx')
                previous=g.shift(1)
                same_forward=(g.arc_id==previous.arc_id)&(g.projection_ratio>=previous.projection_ratio)
                adjacent=(g.arc_id!=previous.arc_id)&(previous.to_node_id==g.from_node_id)
                connected=(same_forward|adjacent)&~g.gap_before&g.arc_id.notna()&previous.arc_id.notna()
                rows.append(dict(model_variant=model_path.stem,trip_id=tid,observation_pairs=max(0,len(g)-1),connected_pairs=int(connected.sum())))
    frame=pd.DataFrame(rows);frame['direct_continuity']=frame.connected_pairs/frame.observation_pairs
    frame.to_csv(out,index=False)
    total=frame.groupby('model_variant')[['connected_pairs','observation_pairs']].sum()
    total['direct_continuity']=total.connected_pairs/total.observation_pairs
    total.to_csv(out.with_name('direct_continuity_summary.csv'))
    print(total.to_string())


if __name__=='__main__':main()
