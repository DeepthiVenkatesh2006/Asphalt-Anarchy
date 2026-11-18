"""Actual development maps and validation summaries for the consolidated report."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.api.service import MapMatchingService
from src.data.loaders import load_trip
from src.models.artifacts import file_sha256
from src.visualization.map_builder import build_folium_map


def main():
    root=Path('reports/final/assets');root.mkdir(parents=True,exist_ok=True)
    service=MapMatchingService()
    # Student-test illustration; evaluator reference geometry is never exported.
    tid='00000007'
    result=service.process_trip(tid,partition='student_test',use_viterbi=True)
    reference=load_trip(Path(service.data_root)/'student_test'/tid,include_reference=True)
    result['route']=reference.route.tolist()
    build_folium_map(result,str(root/f'student_test_{tid}_folium.html'),include_reference=True)
    (root/'map_metrics.json').write_text(json.dumps(dict(trip_id=tid,partition='student_test',metrics=result['metrics']),indent=2))
    catalogue=service.list_available_trips()
    expected=json.loads(Path('reports/member2/local_training/input_checksums.json').read_text())
    checked={key:file_sha256(Path('data/processed/v1')/key)==value['sha256'] for key,value in expected.items()}
    if not all(checked.values()):raise AssertionError('Processed training/evaluation data differ from the original measured run')
    data=dict(partitions={p:sum(t['partition']==p for t in catalogue) for p in ['train','student_test','evaluator_test']},
        input_validation_errors=service.catalog_errors,reference_validation_errors=service.reference_errors,
        model_sha256=file_sha256(service.model_path),processed_input_checksums_match=checked,
        trip_ids_disjoint=len({t['trip_id'] for t in catalogue})==len(catalogue))
    Path('reports/audit/validation_summary.json').write_text(json.dumps(data,indent=2))
    print('Actual Folium development map and dataset/artifact audit generated.')


if __name__=='__main__':main()
