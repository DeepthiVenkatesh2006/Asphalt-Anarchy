"""Reproducible five-model comparison and Bayesian tuning on training-trip folds.

Examples (from repository root):
  python -m src.models.local_training --data-root data/processed/v1 --trials 10
Results are written separately; legacy teammate artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODEL_NAMES = ['logistic', 'knn', 'forest', 'histboost', 'mlp']


def make_model(name: str, seed: int, *, params: dict | None = None, quick=False):
    parameters = dict(params or {})
    if name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=500, C=parameters.get('C', 1.0), random_state=seed))
    if name == 'knn':
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            n_neighbors=parameters.get('n_neighbors', 9),
            weights=parameters.get('weights', 'distance'), algorithm='kd_tree', n_jobs=2))
    if name == 'forest':
        return RandomForestClassifier(n_estimators=parameters.get('n_estimators', 80 if not quick else 5),
            max_depth=parameters.get('max_depth', 12),
            min_samples_leaf=parameters.get('min_samples_leaf', 3),
            max_features=parameters.get('max_features', 'sqrt'), n_jobs=2, random_state=seed)
    if name == 'histboost':
        return HistGradientBoostingClassifier(max_iter=parameters.get('max_iter', 120 if not quick else 5),
            learning_rate=parameters.get('learning_rate', .08),
            max_leaf_nodes=parameters.get('max_leaf_nodes', 31),
            min_samples_leaf=parameters.get('min_samples_leaf', 20),
            l2_regularization=parameters.get('l2_regularization', 0.0),
            early_stopping=True, random_state=seed)
    if name == 'mlp':
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=tuple(parameters.get('hidden_layer_sizes', [32, 16])),
            max_iter=parameters.get('max_iter', 35 if not quick else 5),
            alpha=parameters.get('alpha', .0001),
            learning_rate_init=parameters.get('learning_rate_init', .001),
            batch_size=parameters.get('batch_size', 512),
            early_stopping=True, random_state=seed))
    raise ValueError(f'Unknown model: {name}')


def probabilities(model, X, *, batch_size=20000):
    """A single probability pass; class decisions derive from these same scores."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if len(X) == 0:
        return np.empty(0, dtype=float)
    classes = np.asarray(model.classes_)
    if classes.tolist() != [0, 1]:
        raise ValueError('Expected trained binary candidate classes [0, 1]')
    return np.concatenate([model.predict_proba(X[i:i+batch_size])[:, 1]
                           for i in range(0, len(X), batch_size)])


def grouped_sample(frame, max_observations, seed):
    """For KNN only: fixed seed, complete candidate groups, all training trips represented."""
    keys = frame[['trip_id', 'obs_idx']].drop_duplicates()
    if max_observations < 1:
        raise ValueError('max_observations must be positive')
    if len(keys) <= max_observations:
        return frame.copy()
    rng = np.random.default_rng(seed)
    # Equal per-trip quotas prevent one long trip consuming the entire fitting cap.
    groups = list(keys.groupby('trip_id', sort=True))
    if max_observations < len(groups):
        raise ValueError('KNN sample must include at least one observation per training trip')
    quota = max_observations // len(groups)
    selected = []
    for _, group in groups:
        chosen = rng.choice(len(group), size=min(quota, len(group)), replace=False)
        selected.append(group.iloc[chosen])
    return frame.merge(pd.concat(selected), on=['trip_id','obs_idx'], how='inner', validate='many_to_one')


def evaluate_routes(candidates, scores, labels):
    """Per-trip Macro F1 uses the union of true/predicted arc IDs, including -1 unmatched.

Only confidently aligned reference observations form the primary denominator.
Candidate misses are included as incorrect predictions. Log loss is conditional
on the reference being in the candidate set; its coverage is reported separately.
"""
    scores = np.asarray(scores, dtype=float)
    if len(candidates) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and map one-to-one to candidates')
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError('Candidate scores must be in [0, 1]')
    truth = labels.loc[(labels.alignment_status == 'aligned') & labels.reference_arc_id.notna(),
                       ['trip_id','obs_idx','reference_arc_id']].copy()
    if len(truth) == 0:
        raise ValueError('No confidently aligned reference observations to evaluate')
    if truth.duplicated(['trip_id','obs_idx']).any():
        raise ValueError('Reference observation keys must be unique')
    work = candidates[['trip_id','obs_idx','arc_id']].reset_index(drop=True).copy()
    work['score'] = scores
    if len(work):
        if work.duplicated(['trip_id','obs_idx','arc_id']).any():
            raise ValueError('Duplicate candidate road for an observation')
        best = work.loc[work.groupby(['trip_id','obs_idx'],sort=False).score.idxmax(),
                        ['trip_id','obs_idx','arc_id']].rename(columns={'arc_id':'predicted_arc_id'})
    else:
        best = pd.DataFrame(columns=['trip_id','obs_idx','predicted_arc_id'])
    joined = truth.merge(best,on=['trip_id','obs_idx'],how='left',validate='one_to_one')
    joined['predicted_arc_id'] = joined.predicted_arc_id.fillna(-1).astype('int64')
    joined['reference_arc_id'] = joined.reference_arc_id.astype('int64')
    f1s = [f1_score(g.reference_arc_id,g.predicted_arc_id,average='macro',zero_division=0)
           for _,g in joined.groupby('trip_id',sort=True)]
    work['positive'] = np.maximum(work.score, 1e-12)
    if len(work):
        work['normalized'] = work.positive / work.groupby(['trip_id','obs_idx']).positive.transform('sum')
        ref_scores = work.merge(truth,on=['trip_id','obs_idx'],validate='many_to_one')
        ref_scores = ref_scores[ref_scores.arc_id == ref_scores.reference_arc_id]
    else:
        ref_scores = pd.DataFrame(columns=['normalized'])
    return dict(top1_accuracy=float((joined.predicted_arc_id == joined.reference_arc_id).mean()),
        observation_macro_f1=float(np.mean(f1s)), evaluated_observations=len(joined),
        evaluated_trips=len(f1s), candidate_miss_rate=1-len(ref_scores)/len(truth),
        candidate_log_loss=float(-np.log(ref_scores.normalized).mean()) if len(ref_scores) else None,
        log_loss_observation_coverage=len(ref_scores)/len(truth),
        reference_coverage=len(truth)/len(labels))


@dataclass
class Partition:
    candidates: pd.DataFrame
    labels: pd.DataFrame
    training: pd.DataFrame
    X_candidates: np.ndarray
    feature_columns: list[str]
    manifest: dict


def load_partition(root: Path, features):
    candidates = pd.read_parquet(root/'candidates.parquet')
    labels = pd.read_parquet(root/'reference_labels.parquet')
    training = pd.read_parquet(root/'train_features.parquet')
    if training.empty or training.target.nunique() != 2:
        raise ValueError(f'{root}: requires nonempty binary training-format data')
    X = candidates[features].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError('Nonfinite candidate features')
    manifest=json.loads((root/'manifest.json').read_text())
    if len(labels) != sum(t['observations'] for t in manifest['trips']):
        raise ValueError('Labels must preserve every observation from the manifest')
    return Partition(candidates,labels,training,X,features,manifest)


def validation_folds(partition, count):
    folds = []
    frame = partition.training
    split = GroupKFold(n_splits=count)
    for fit, valid in split.split(frame,frame.target,groups=frame.trip_id):
        fit_ids = set(frame.iloc[fit].trip_id)
        valid_ids = set(frame.iloc[valid].trip_id)
        if fit_ids & valid_ids:
            raise AssertionError('Trip leakage')
        folds.append((fit, np.flatnonzero(partition.candidates.trip_id.isin(valid_ids)),
                      np.flatnonzero(partition.labels.trip_id.isin(valid_ids)), sorted(valid_ids)))
    return folds


def fit_model(name, frame, features, seed, params, knn_observations):
    fit_frame = grouped_sample(frame, knn_observations, seed) if name=='knn' else frame
    model = make_model(name,seed,params=params)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(fit_frame[features].to_numpy(dtype=np.float64),fit_frame.target.to_numpy(dtype=int))
    elapsed=time.perf_counter()-start
    history_model=model.steps[-1][1] if hasattr(model,'steps') else model
    history={key:getattr(history_model,key) for key in ['loss_curve_','validation_scores_']
             if hasattr(history_model,key)}
    return model,dict(fit_seconds=elapsed,fit_candidate_rows=len(fit_frame),
                      warnings=[str(w.message) for w in caught],history=history)


def cross_validate(name, partition, folds, seed, params, knn_observations, callback=None):
    records=[]
    for fold_idx,(fit,valid,label_idx,trips) in enumerate(folds,1):
        model,fit_info=fit_model(name,partition.training.iloc[fit],partition.feature_columns,seed,params,knn_observations)
        start=time.perf_counter()
        scores=probabilities(model,partition.X_candidates[valid])
        seconds=time.perf_counter()-start
        metrics=evaluate_routes(partition.candidates.iloc[valid],scores,partition.labels.iloc[label_idx])
        record=dict(model=name,fold=fold_idx,validation_trip_ids=trips,
                    fit_seconds=fit_info['fit_seconds'],predict_seconds=seconds,
                    fit_candidate_rows=fit_info['fit_candidate_rows'],warnings=fit_info['warnings'],**metrics)
        records.append(record)
        if callback:callback(record)
    return records


def summary(records):
    frame=pd.DataFrame(records)
    result={}
    for key in ['observation_macro_f1','top1_accuracy','candidate_log_loss','fit_seconds','predict_seconds']:
        result[key+'_mean']=float(frame[key].mean())
        result[key+'_std']=float(frame[key].std(ddof=0))
    return result


def trial_parameters(trial,name):
    if name=='logistic':return {'C':trial.suggest_float('C',.01,20,log=True)}
    if name=='knn':return dict(n_neighbors=trial.suggest_int('n_neighbors',3,15,step=2),
                             weights=trial.suggest_categorical('weights',['uniform','distance']))
    if name=='forest':return dict(n_estimators=trial.suggest_int('n_estimators',50,120,step=10),
        max_depth=trial.suggest_int('max_depth',8,18),min_samples_leaf=trial.suggest_int('min_samples_leaf',2,12))
    if name=='histboost':return dict(max_iter=trial.suggest_int('max_iter',60,180,step=20),
        learning_rate=trial.suggest_float('learning_rate',.03,.2,log=True),
        max_leaf_nodes=trial.suggest_categorical('max_leaf_nodes',[15,31,63]),
        l2_regularization=trial.suggest_float('l2_regularization',1e-4,2,log=True))
    if name=='mlp':return dict(hidden_layer_sizes=trial.suggest_categorical('architecture',['32','32,16','64,32']),
        alpha=trial.suggest_float('alpha',1e-5,.01,log=True),
        learning_rate_init=trial.suggest_float('learning_rate_init',.0003,.005,log=True),max_iter=35)
    raise ValueError(name)


def translate_parameters(params):
    params=dict(params)
    if 'hidden_layer_sizes' in params and isinstance(params['hidden_layer_sizes'],str):
        params['hidden_layer_sizes']=[int(x) for x in params['hidden_layer_sizes'].split(',')]
    # Optuna saves categorical architecture under this key.
    if 'architecture' in params:
        params['hidden_layer_sizes']=[int(x) for x in params.pop('architecture').split(',')]
    return params


def run(args):
    import optuna
    output=Path(args.output_dir)
    if output.exists():raise FileExistsError('Use a new result directory; existing experiments are preserved')
    if args.trials < 1:raise ValueError('At least one Bayesian optimisation trial is required')
    root=Path(args.data_root)
    features=json.loads((root/'train/feature_columns.json').read_text())
    train=load_partition(root/'train',features)
    output.mkdir(parents=True)
    (output/'models').mkdir()
    (output/'train_preparation_manifest.json').write_text(json.dumps(train.manifest,indent=2))
    folds=validation_folds(train,args.folds)
    all_records=[]
    configurations={name:make_model(name,args.seed).get_params() for name in MODEL_NAMES}
    # Text representation is deliberate: estimator/scaler objects are not JSON serializable.
    (output/'run_config.json').write_text(json.dumps(dict(seed=args.seed,folds=args.folds,
        tuning_folds=args.tuning_folds,trials=args.trials,knn_max_training_observations=args.knn_observations,
        feature_columns=features,selection_metric='mean CV observation Macro F1 on aligned labels',
        fold_validation_trips=[f[3] for f in folds],model_configurations={k:repr(v) for k,v in configurations.items()},
        versions={n:version(n) for n in ['numpy','pandas','scikit-learn','optuna']},
        label_warning='Derived approximate labels; aligned subset coverage must accompany metrics.'),indent=2))
    current_variant='baseline'
    def record(row):
        row=dict(row,variant=current_variant)
        all_records.append(row)
        with (output/'fold_results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    comparisons=[]
    for name in MODEL_NAMES:
        print(f'START MODEL {name}',flush=True)
        rows=cross_validate(name,train,folds,args.seed,{},args.knn_observations,callback=record)
        comparisons.append(dict(model=name,**summary(rows)))
        pd.DataFrame(comparisons).to_csv(output/'cv_comparison.csv',index=False)
    champion=max(comparisons,key=lambda r:r['observation_macro_f1_mean'])['model']
    print(f'CV SELECTED MODEL {champion}',flush=True)
    tuning_folds=validation_folds(train,args.tuning_folds)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=args.seed,n_startup_trials=3),
        storage='sqlite:///'+str((output/'bayesian_search.sqlite3').resolve()),study_name='pathsnap_local')
    def objective(trial):
        params=translate_parameters(trial_parameters(trial,champion))
        start=time.perf_counter()
        rows=cross_validate(champion,train,tuning_folds,args.seed,params,args.knn_observations)
        trial.set_user_attr('parameters',params)
        trial.set_user_attr('fold_results',rows)
        trial.set_user_attr('wall_seconds',time.perf_counter()-start)
        score=summary(rows)['observation_macro_f1_mean']
        print(f'TUNING TRIAL {trial.number} objective={score:.6f} params={params}',flush=True)
        return score
    study.optimize(objective,n_trials=args.trials,callbacks=[lambda s,t:s.trials_dataframe().to_csv(output/'bayesian_trials.csv',index=False)])
    tuned_params=study.best_trial.user_attrs['parameters']
    current_variant='tuned'
    tuned_rows=cross_validate(champion,train,folds,args.seed,tuned_params,args.knn_observations,callback=record)
    tuned_summary=summary(tuned_rows)
    baseline=next(r for r in comparisons if r['model']==champion)
    # Final choice is frozen using only training-trip CV, before student-test loading.
    use_tuned=tuned_summary['observation_macro_f1_mean']>baseline['observation_macro_f1_mean']
    chosen_parameters=tuned_params if use_tuned else {}
    decision=dict(model=champion,tuned_parameters=tuned_params,selected_tuned=use_tuned,
                  baseline_cv=baseline,tuned_cv=tuned_summary,
                  optimisation_objective='mean aligned-label observation Macro F1 on group-aware training CV',
                  tuning_folds=args.tuning_folds,comparison_folds=args.folds)
    (output/'selection.json').write_text(json.dumps(decision,indent=2))
    # Student-test is now used only for measured reporting, not model/tuning selection.
    test=load_partition(root/'student_test',features)
    (output/'student_test_preparation_manifest.json').write_text(json.dumps(test.manifest,indent=2))
    test_rows=[]
    for name,params,variant in [(n,{},'baseline') for n in MODEL_NAMES]+[(champion,tuned_params,'tuned')]:
        print(f'FINAL FIT {name} {variant}',flush=True)
        model,info=fit_model(name,train.training,features,args.seed,params,args.knn_observations)
        start=time.perf_counter();scores=probabilities(model,test.X_candidates);seconds=time.perf_counter()-start
        result=dict(model=name,variant=variant,fit_seconds=info['fit_seconds'],predict_seconds=seconds,
                    candidate_rows_per_second=len(scores)/seconds,fit_candidate_rows=info['fit_candidate_rows'],
                    **evaluate_routes(test.candidates,scores,test.labels))
        # Clearly labelled candidate-level diagnostics on eligible fitting-format rows.
        eligible=test.training
        e_scores=probabilities(model,eligible[features].to_numpy(dtype=np.float64))
        result['candidate_binary_macro_f1']=float(f1_score(eligible.target,(e_scores>.5).astype(int),average='macro'))
        result['candidate_binary_log_loss']=float(log_loss(eligible.target,e_scores,labels=[0,1]))
        test_rows.append(result)
        pd.DataFrame(test_rows).to_csv(output/'student_test_results.csv',index=False)
        joblib.dump(model,output/'models'/f'{name}_{variant}.joblib')
        if info['history']:(output/f'{name}_{variant}_history.json').write_text(json.dumps(info['history'],indent=2))
        if name==champion and variant==('tuned' if use_tuned else 'baseline'):
            joblib.dump(model,output/'model_final.joblib')
        print(json.dumps(result),flush=True)
    preparation={k:train.manifest[k] for k in ['radius_m','max_candidates','gap_s','reference_radius_m','alignment_config']}
    metadata=dict(decision,feature_columns=features,preparation=preparation,
        candidate_classes=[0,1],model_file='model_final.joblib',normalization='clip positive-class scores to 1e-12 and normalize within observation for candidate log loss',
        fold_records='fold_results.jsonl',evaluation_label_subset='alignment_status == aligned; candidate misses retained',
        macro_f1_definition='Per-trip macro over union of reference/predicted arc IDs, including unmatched -1; then mean across trips',
        log_loss_definition='Conditional categorical log loss only where aligned reference road is in candidates; coverage separately reported',
        versions={n:version(n) for n in ['numpy','pandas','scikit-learn','optuna']})
    (output/'model_metadata.json').write_text(json.dumps(metadata,indent=2))
    (output/'feature_columns.json').write_text(json.dumps(features,indent=2))
    make_plots(output,comparisons,test_rows)
    (output/'COMPLETE').write_text('Training, CV, Bayesian tuning, student-test evaluation and export completed.\n')
    print(f'COMPLETE {output}',flush=True)


def make_plots(output,comparisons,test_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    ax[0].bar([r['model'] for r in comparisons],[r['observation_macro_f1_mean'] for r in comparisons],
        yerr=[r['observation_macro_f1_std'] for r in comparisons],capsize=3)
    ax[0].set(title='Training-trip CV observation Macro F1',ylabel='Mean across folds with fold standard deviation',ylim=(0,1))
    ax[1].bar([r['model']+' '+r['variant'] for r in test_rows],[r['top1_accuracy'] for r in test_rows])
    ax[1].set(title='Student-test Top-1 on aligned references',ylabel='Accuracy; candidate misses included',ylim=(0,1))
    for a in ax:a.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(output/'model_comparison.png',dpi=150);plt.close(fig)
    for file in output.glob('*_history.json'):
        history=json.loads(file.read_text())
        fig,axis=plt.subplots(figsize=(7,4))
        axis.plot(np.arange(1,len(history['loss_curve_'])+1),history['loss_curve_'],label='Training loss')
        axis.set(xlabel='Epoch',ylabel='Training loss',title=file.stem)
        axis.legend();fig.tight_layout();fig.savefig(file.with_suffix('.png'),dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default='data/processed/v1')
    parser.add_argument('--output-dir',default='reports/member2/local_training')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--tuning-folds',type=int,default=3)
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--knn-observations',type=int,default=10000)
    args=parser.parse_args()
    with threadpool_limits(limits=2):run(args)


if __name__=='__main__':main()
