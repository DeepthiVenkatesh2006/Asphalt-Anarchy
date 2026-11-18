import runpy
from pathlib import Path


def test_importing_manual_demo_never_starts_a_server(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Import must not start a server')
    monkeypatch.setattr('subprocess.Popen',forbidden)
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'test_full_demo.py'),run_name='manual_demo_import')
