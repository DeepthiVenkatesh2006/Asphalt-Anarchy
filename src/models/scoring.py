"""Member 3 inference interface: candidate identities in, mapped model scores out."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.local_training import probabilities

KEY_COLUMNS=['trip_id','obs_idx','candidate_idx','arc_id']


class CandidateScorer:
    """Load only a trusted team-generated joblib artifact; it contains the scaler."""
    def __init__(self,model,feature_columns,metadata):
        if not feature_columns or len(set(feature_columns)) != len(feature_columns):
            raise ValueError('Feature names must be a nonempty unique list')
        if metadata.get('feature_columns') != feature_columns:
            raise ValueError('Feature order differs between feature_columns.json and metadata')
        if getattr(model,'n_features_in_',None) != len(feature_columns):
            raise ValueError('Fitted model feature count does not match feature schema')
        self.model=model
        self.feature_columns=feature_columns
        self.metadata=metadata

    @classmethod
    def from_directory(cls,artifact_dir):
        root=Path(artifact_dir)
        return cls(joblib.load(root/'model_final.joblib'),
                   json.loads((root/'feature_columns.json').read_text()),
                   json.loads((root/'model_metadata.json').read_text()))

    def score_candidates(self,records: pd.DataFrame,*,batch_size=20000) -> pd.DataFrame:
        """Return input keys plus raw binary scores and per-observation probabilities.

        Scores preserve input row order/index. No reference or target column is
        read. Normalize the COMPLETE candidate set for each observation together.
        Empty input remains empty; observations with zero candidates must be
        preserved separately by the caller's observations ledger.
        """
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
            raise ValueError('batch_size must be a positive integer')
        required=KEY_COLUMNS+self.feature_columns
        if records.columns.duplicated().any():
            raise ValueError('Duplicate input column names')
        missing=[name for name in required if name not in records]
        if missing:raise ValueError(f'Missing candidate columns: {missing}')
        if records[KEY_COLUMNS].isna().any().any():
            raise ValueError('Candidate mapping keys cannot be missing')
        for keys in [['trip_id','obs_idx','candidate_idx'],['trip_id','obs_idx','arc_id']]:
            if records.duplicated(keys).any():raise ValueError('Duplicate candidate identity or road')
        try:X=records[self.feature_columns].to_numpy(dtype=np.float64)
        except (TypeError,ValueError) as exc:
            raise ValueError('Model features must be numeric') from exc
        if not np.isfinite(X).all():raise ValueError('Model features must be finite')
        result=records[KEY_COLUMNS].copy()
        result['candidate_score']=probabilities(self.model,X,batch_size=batch_size)
        weights=np.maximum(result.candidate_score,1e-12)
        result['candidate_probability']=weights/weights.groupby([result.trip_id,result.obs_idx]).transform('sum')
        return result


def score_candidates(records,artifact_dir='artifacts/local_training',*,batch_size=20000):
    """Convenience entry point; reuse CandidateScorer for repeated trip requests."""
    return CandidateScorer.from_directory(artifact_dir).score_candidates(records,batch_size=batch_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',default='artifacts/local_training')
    parser.add_argument('--candidates',required=True,help='Existing Member 1 candidates.parquet')
    parser.add_argument('--trip-id',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--preview-observations',type=int,default=5)
    args=parser.parse_args()
    if args.preview_observations<1:raise ValueError('preview-observations must be positive')
    data=pd.read_parquet(args.candidates,filters=[('trip_id','==',args.trip_id)])
    if data.empty:raise ValueError(f'No candidates found for trip {args.trip_id}')
    scores=score_candidates(data,args.artifact_dir)
    first_ids=scores.obs_idx.drop_duplicates().iloc[:args.preview_observations]
    preview=scores[scores.obs_idx.isin(first_ids)]
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    preview.to_csv(out/'real_trip_scores_preview.csv',index=False)
    summary=dict(trip_id=args.trip_id,total_scored_candidates=len(scores),
        observations_with_candidates=int(scores.obs_idx.nunique()),
        preview_observations=len(first_ids),preview_candidate_rows=len(preview),
        probability_sums_equal_one=bool(np.allclose(scores.groupby(['trip_id','obs_idx']).candidate_probability.sum(),1)),
        reference_labels_used=False,model_directory=args.artifact_dir,
        note='CSV is an output preview, not the full trip candidate dataset or decoded route.')
    (out/'real_trip_scoring_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
