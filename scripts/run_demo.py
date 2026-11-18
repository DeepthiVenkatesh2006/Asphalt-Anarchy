"""One command for local rehearsal, API laptop, or dashboard laptop."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def free_port(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as exc:
            raise SystemExit(f'Port {port} is already in use. Stop the old demo first.') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['local', 'api', 'ui'], default='local')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000')
    parser.add_argument('--mode', choices=['development','evaluation'], default=None)
    parser.add_argument('--data-root', help='Folder containing train, student_test and evaluator_test')
    args = parser.parse_args()
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    mode = args.mode or ('evaluation' if args.role == 'api' else 'development')
    env['PATHSNAP_MODE'] = mode
    env['PATHSNAP_API_URL'] = args.api_url.rstrip('/')
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('OPENBLAS_NUM_THREADS', '2')
    if args.data_root:
        env['PATHSNAP_DATA_ROOT'] = str(Path(args.data_root).resolve())
    if args.role in ['local', 'api']:
        sys.path.insert(0, str(ROOT))
        from src.api.service import MapMatchingService
        service = MapMatchingService(data_root=args.data_root)
        trips = service.list_available_trips()
        if not trips:
            raise SystemExit('No trips found. Supply --data-root pointing to the folder containing train/student_test/evaluator_test.')
        if mode == 'evaluation':
            evaluation_trips = [t for t in trips if t['partition']=='evaluator_test']
            if len(evaluation_trips) != 10 or not all(t['reference_valid'] for t in evaluation_trips):
                raise SystemExit('Evaluation mode requires ten valid evaluator_test trips with server-side references.')
        invalid = [e for e in service.catalog_errors if mode != 'evaluation' or e['partition']=='evaluator_test']
        if invalid:
            raise SystemExit(f'Invalid trip files found: {invalid}')
        print(f'Preflight passed: model loaded, {len(trips)} trips at {service.data_root}', flush=True)
    commands = []
    if args.role in ['local', 'api']:
        free_port(8000)
        commands.append([sys.executable, '-m', 'uvicorn', 'pathsnap.api.main:app', '--host', '0.0.0.0' if args.role=='api' else '127.0.0.1', '--port', '8000'])
    if args.role in ['local', 'ui']:
        free_port(8501)
        commands.append([sys.executable, '-m', 'streamlit', 'run', 'ui/app.py', '--server.address', '127.0.0.1', '--server.port', '8501', '--server.headless', 'true', '--browser.gatherUsageStats', 'false'])
    children = []
    try:
        for command in commands:
            children.append(subprocess.Popen(command, cwd=ROOT, env=env))
        print('Dashboard: http://localhost:8501' if args.role!='api' else 'API: http://localhost:8000/health', flush=True)
        print('Press Ctrl+C to stop this demo.', flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise SystemExit('A demo process stopped. See its error above.')
    except KeyboardInterrupt:
        print('\nStopping demo.', flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == '__main__':
    main()
