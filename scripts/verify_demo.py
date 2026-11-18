"""Verify real API responses on supplied evaluator trips without fitting any model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.routes import get_service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', default='evaluator_test', choices=['evaluator_test','student_test'])
    args = parser.parse_args()
    output = Path('reports/demo')
    output.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    health = client.get('/health')
    health.raise_for_status()
    catalog = client.get('/trips').json()
    service = get_service()
    model_hash = hashlib.sha256(Path(service.model_path).read_bytes()).hexdigest()
    rows = []
    for trip in catalog['trips']:
        if trip['partition'] != args.partition:
            continue
        started = time.perf_counter()
        response = client.post('/predict', json=dict(trip_id=trip['trip_id'], partition=args.partition, use_viterbi=True))
        response.raise_for_status()
        data = response.json()
        assert len(data['predicted_arcs']) == trip['points']
        assert 'route' not in data
        rows.append(dict(trip_id=trip['trip_id'], status=response.status_code, elapsed_s=time.perf_counter()-started, **data['metrics']))
        (output/f'{args.partition}_verification.json').write_text(json.dumps(dict(health=health.json(), partitions=catalog['partitions'], model_sha256=model_hash, results=rows), indent=2))
        print(trip['trip_id'], response.status_code, f"{rows[-1]['elapsed_s']:.2f}s", flush=True)
    assert len(rows) == 10, f'Expected 10 {args.partition} trips, got {len(rows)}'
    print('All ten trips returned real predictions; no model training or tuning was performed.', flush=True)


if __name__ == '__main__':
    main()
