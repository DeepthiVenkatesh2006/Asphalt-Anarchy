"""Measured classifier versus HMM comparison on student_test only; never evaluator_test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from threadpoolctl import threadpool_limits
from src.api.service import MapMatchingService
from src.models.artifacts import file_sha256
from src.data.loaders import trip_directories
from src.evaluation.summary import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir',default='reports/decoder/student_test')
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    (output/'predictions').mkdir()
    service=MapMatchingService(data_root=args.data_root)
    trips=[{'trip_id':p.name} for p in trip_directories(Path(service.data_root)/'student_test')]
    if len(trips)!=10:raise ValueError('This comparison requires all ten student-test trips')
    config=dict(partition='student_test',trip_ids=[t['trip_id'] for t in trips],
        model_path=service.model_path,model_sha256=file_sha256(service.model_path),
        inference_settings=service.settings.model_dump(),model_fitting_performed=False,
        evaluator_data_used=False,created_utc=datetime.now(timezone.utc).isoformat(),
        metric_definitions=dict(top1='classifier argmax accuracy',point='returned assignments on aligned labels',
            macro='compute precision/recall/F1 per trip then average',sequence='SequenceMatcher ratio, autojunk=False',
            set_f1='order-insensitive F1 of visited road sets',continuity='valid projected-position road transitions / all observation pairs'))
    (output/'run_config.json').write_text(json.dumps(config,indent=2))
    records=[]
    started=time.perf_counter()
    with threadpool_limits(limits=2):
        for trip in trips:
            for use_viterbi in [False,True]:
                result=service.process_trip(trip['trip_id'],partition='student_test',use_viterbi=use_viterbi)
                variant='hmm' if use_viterbi else 'classifier_only'
                record=dict(trip_id=trip['trip_id'],variant=variant,**result['metrics'])
                records.append(record)
                pd.DataFrame(records).to_csv(output/'trip_comparison.csv',index=False)
                (output/'predictions'/f"{trip['trip_id']}_{variant}.json").write_text(json.dumps(dict(
                    trip_id=trip['trip_id'],partition='student_test',predicted_arcs=result['predicted_route'],
                    route_chunks=result['route_chunks'],metrics=result['metrics']),indent=2))
                print(trip['trip_id'],variant,'point accuracy',record['point_accuracy'],'sequence',record['sequence_similarity'],flush=True)
    summaries=[dict(variant=variant,**summarize([r for r in records if r['variant']==variant]))
               for variant in ['classifier_only','hmm']]
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    make_plots(output,records,summaries)
    (output/'COMPLETE').write_text(f'Compared ten student-test trips, both variants. Wall seconds: {time.perf_counter()-started:.3f}\n')
    print('COMPLETE',output,flush=True)


def make_plots(output,records,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(records)
    keys=['point_accuracy','observation_macro_f1','sequence_similarity','route_continuity']
    titles=['Returned point accuracy','Observation Macro F1','Order-sensitive sequence similarity','Valid road-transition continuity']
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,key,title in zip(axes,keys,titles):
        ax.bar(['Classifier','HMM'],[r[key] for r in summaries],color=['#377eb8','#087f5b'])
        ax.set(title=title,ylim=(0,1));ax.tick_params(axis='x',rotation=15)
    fig.suptitle('Ten student-test trips; fixed trained model and identical reference coverage')
    fig.tight_layout();fig.savefig(output/'decoder_comparison.png',dpi=150);plt.close(fig)
    pivot=frame.pivot(index='trip_id',columns='variant',values='point_accuracy')
    pivot.plot.bar(figsize=(12,4),color=['#377eb8','#087f5b']);plt.ylabel('Point accuracy on aligned reference subset');plt.ylim(0,1)
    plt.title('Per-trip comparison: HMM may improve or worsen individual trips');plt.tight_layout();plt.savefig(output/'per_trip_accuracy.png',dpi=150);plt.close()


if __name__=='__main__':main()
