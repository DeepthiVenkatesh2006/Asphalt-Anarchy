"""Generate report-ready figures and a concise evidence summary from a completed run."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _table(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{value:.4f}' if isinstance(value,float) else str(value) for value in row)+' |')
    return '\n'.join(lines)


def create_report(root):
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    if not (root/'COMPLETE').exists():
        raise ValueError('Training has not completed; do not present partial results as final')
    cv=pd.read_csv(root/'cv_comparison.csv')
    student=pd.read_csv(root/'student_test_results.csv')
    selection=json.loads((root/'selection.json').read_text())
    config=json.loads((root/'run_config.json').read_text())
    manifest=json.loads((root/'train_preparation_manifest.json').read_text())
    trials=pd.read_csv(root/'bayesian_trials.csv')
    finished=trials[trials.state=='COMPLETE'].copy()
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    labels=student.model+' '+student.variant
    for ax,metric,title in zip(axes.flat,
        ['top1_accuracy','observation_macro_f1','candidate_log_loss','predict_seconds'],
        ['Student-test Top-1 accuracy','Student-test per-trip observation Macro F1',
         'Conditional candidate log loss (lower is better)','Full candidate prediction time (seconds)']):
        ax.bar(labels,student[metric]);ax.set_title(title)
        ax.tick_params(axis='x',rotation=35)
        if metric in ['top1_accuracy','observation_macro_f1']:ax.set_ylim(0,1)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Aligned derived references; candidate misses included in accuracy/F1; log-loss coverage separately reported')
    fig.tight_layout();fig.savefig(root/'student_test_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(finished.number,finished.value,'o-',label='Trial objective')
    axes[0].plot(finished.number,finished.value.cummax(),label='Best so far')
    axes[0].set(xlabel='Completed trial number',ylabel='Training-trip tuning CV observation Macro F1',title='Actual Bayesian optimisation history')
    axes[0].legend()
    axes[1].bar(['Untuned','Tuned'],[selection['baseline_cv']['observation_macro_f1_mean'],selection['tuned_cv']['observation_macro_f1_mean']],
        yerr=[selection['baseline_cv']['observation_macro_f1_std'],selection['tuned_cv']['observation_macro_f1_std']],capsize=3)
    axes[1].set(ylim=(0,1),ylabel='Observation Macro F1',title=f"Same {selection['comparison_folds']} training-trip comparison folds")
    fig.tight_layout();fig.savefig(root/'bayesian_tuning_comparison.png',dpi=150);plt.close(fig)
    # Real MLP histories: validation_scores_ are accuracy, never mislabeled loss.
    for file in root.glob('*_history.json'):
        history=json.loads(file.read_text())
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].plot(np.arange(1,len(history['loss_curve_'])+1),history['loss_curve_'])
        axes[0].set(xlabel='Epoch',ylabel='Training loss',title='Actual final-fit training history')
        validation=history.get('validation_scores_',[])
        axes[1].plot(np.arange(1,len(validation)+1),validation)
        axes[1].set(xlabel='Epoch',ylabel='Internal candidate binary validation accuracy',title='Internal fitting-data holdout (not outer trip CV)')
        fig.tight_layout();fig.savefig(root/(file.stem+'_details.png'),dpi=150);plt.close(fig)
    diagnostics=[]
    for file in sorted((root/'models').glob('mlp_*.joblib')):
        fitted=joblib.load(file).steps[-1][1]
        diagnostics.append(dict(model=file.stem,epochs=fitted.n_iter_,epoch_cap=fitted.max_iter,
                                reached_epoch_cap=fitted.n_iter_==fitted.max_iter))
    (root/'fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    lines=['# Completed local model training and tuning','',
        f"Five model families were compared using all {config['folds']} outer GroupKFold splits on {len(manifest['trips'])} training trips. "
        'KNN used a recorded fitting sample of complete observation groups; other models used all eligible fitting rows. '
        'Selection used training CV, before loading student-test labels.','',
        '## Training cross-validation','',_table(cv[['model','observation_macro_f1_mean','observation_macro_f1_std','top1_accuracy_mean']]),'',
        'Macro F1 is computed per trip over the union of predicted and reference road IDs, including unmatched -1, '
        'then averaged across trips within each fold. The table summarizes fold means.','',
        '## Student-testing results','',_table(student[['model','variant','top1_accuracy','observation_macro_f1','candidate_log_loss','reference_coverage','candidate_miss_rate','predict_seconds']]),'',
        'Scores use confidently aligned, approximately derived labels. Candidate misses are counted in Top-1 and F1. '
        'Log loss is conditional on the reference road being in the candidate set; inspect log_loss_observation_coverage in the CSV. '
        'These are not scores against independently verified physical point truth, and are not HMM-decoded route results.','',
        '## Bayesian optimisation','',f"Selected family: {selection['model']}. Completed TPE trials: {len(finished)}. "
        f"Search used {selection['tuning_folds']} trip-aware training folds; tuned/untuned comparison used the same {selection['comparison_folds']} folds.",'',
        'Best searched parameters:','', '```json',json.dumps(selection['tuned_parameters'],indent=2),'```','',
        f"Selected variant: {'tuned' if selection['selected_tuned'] else 'baseline (tuning did not improve the comparison objective)'}. "
        'An improvement is judged on training CV, not by student-test results.','',
        '## Fitting limits','',_table(pd.DataFrame(diagnostics)) if diagnostics else 'No epoch-based model artifacts.',
        'Reaching the configured epoch cap is a budget-limited fit; do not claim full convergence. '
        'The configuration was selected by measured training CV, not guaranteed optimisation convergence.','',
        '## Reproduction and evidence','',
        '- run_config.json records feature names, seed, parameters, versions and held-out trip IDs.',
        '- fold_results.jsonl records fitting time, row count, coverage, metrics and fitting warnings for each comparison fold.',
        '- bayesian_search.sqlite3 and bayesian_trials.csv retain actual optimisation history.',
        '- input_checksums.json and preparation manifests identify the processed data used.',
        '- model_final.joblib, model_metadata.json and feature_columns.json form the inference bundle.',
        '- Epoch histories and plots exist only for MLP fits that produced them; no artificial curves were generated.',
        '- Early stopping used internal row holdouts within fitting trips; the outer trip-wise validation remained untouched.',
        '- This is model-selection CV, not nested CV. Member 3 still needs route continuity and HMM/Viterbi comparisons.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(root/'RESULTS.md')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir')
    args=parser.parse_args();create_report(args.results_dir)


if __name__=='__main__':main()
