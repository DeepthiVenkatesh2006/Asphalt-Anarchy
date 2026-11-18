"""Re-evaluate saved models on student_test; no fitting or evaluator data access."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from src.models.local_training import load_partition, probabilities, evaluate_routes, fit_model
from src.models.artifacts import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed-root',default='data/processed/v1')
    parser.add_argument('--models-dir',default='reports/member2/local_training/models')
    parser.add_argument('--output-dir',default='reports/audit/model_metrics')
    args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    model_root=Path(args.models_dir)
    features=json.loads(Path('artifacts/local_training/feature_columns.json').read_text())
    partition=load_partition(Path(args.processed_root)/'student_test',features)
    authoritative=pd.read_csv('reports/member2/local_training/student_test_results.csv')
    rows=[];matrices={};hashes={}
    with threadpool_limits(limits=2):
        for name,variant in [('logistic','baseline'),('knn','baseline'),('forest','baseline'),('histboost','baseline'),('mlp','baseline'),('mlp','tuned')]:
            path=model_root/f'{name}_{variant}.joblib'
            if not path.exists() and name == 'knn':
                fitting=pd.read_parquet(Path(args.processed_root)/'train/train_features.parquet')
                model,info=fit_model('knn',fitting,features,42,{},10000)
                joblib.dump(model,path)
                (output/'knn_regeneration.json').write_text(json.dumps(dict(seed=42,complete_observation_cap=10000,
                    fit_candidate_rows=info['fit_candidate_rows'],reason='KNN cache omitted from public repository because it retains training examples'),indent=2))
            else:
                model=joblib.load(path)
            hashes[path.name]=file_sha256(path)
            scores=probabilities(model,partition.X_candidates)
            metrics=evaluate_routes(partition.candidates,scores,partition.labels)
            recorded=authoritative[(authoritative.model==name)&(authoritative.variant==variant)].iloc[0]
            for key in ['top1_accuracy','observation_macro_f1','candidate_log_loss']:
                if not np.isclose(metrics[key],recorded[key],atol=1e-8):
                    raise AssertionError(f'Saved {name} {variant} no longer reproduces {key}')
            candidates=partition.candidates[['trip_id','obs_idx','arc_id']].reset_index(drop=True).copy()
            candidates['score']=scores
            best=candidates.loc[candidates.groupby(['trip_id','obs_idx']).score.idxmax()].rename(columns={'arc_id':'predicted_arc_id'})
            truth=partition.labels[partition.labels.alignment_status=='aligned'][['trip_id','obs_idx','reference_arc_id']]
            joined=truth.merge(best,on=['trip_id','obs_idx'],how='left')
            joined.predicted_arc_id=joined.predicted_arc_id.fillna(-1).astype(int)
            per_trip=[precision_recall_fscore_support(g.reference_arc_id.astype(int),g.predicted_arc_id,average='macro',zero_division=0)[:3]
                      for _,g in joined.groupby('trip_id')]
            prec,rec,f1=np.asarray(per_trip).mean(axis=0)
            eligible=partition.training
            binary=probabilities(model,eligible[features].to_numpy(dtype=float))>.5
            matrix=confusion_matrix(eligible.target,binary,labels=[0,1])
            matrices[f'{name}_{variant}']=matrix.tolist()
            rows.append(dict(model=name,variant=variant,macro_precision=float(prec),macro_recall=float(rec),**metrics))
            print(name,variant,'reproduced',metrics['top1_accuracy'],flush=True)
    pd.DataFrame(rows).to_csv(output/'student_test_metrics.csv',index=False)
    (output/'confusion_matrices.json').write_text(json.dumps(matrices,indent=2))
    (output/'verification.json').write_text(json.dumps(dict(model_hashes=hashes,all_original_primary_metrics_reproduced=True,
        partition='student_test',model_fitting_performed=False,evaluator_data_read=False,
        confusion_matrix_scope='binary candidate rows with eligible derived reference labels; threshold >0.5',
        macro_scope='per-trip road-ID precision/recall/F1 then average over ten trips'),indent=2))
    make_plot(output,matrices)


def make_plot(output,matrices):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,(name,values) in zip(axes.flat,matrices.items()):
        a=np.asarray(values);ax.imshow(a,cmap='Blues')
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{a[i,j]:,}',ha='center',va='center',color='white' if a[i,j]>a.max()/2 else 'black')
        ax.set(title=name,xlabel='Predicted candidate class',ylabel='Derived reference class',xticks=[0,1],yticks=[0,1])
    fig.suptitle('Student-test candidate confusion matrices; eligible-label subset only')
    fig.tight_layout();fig.savefig(output/'candidate_confusion_matrices.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
