from pathlib import Path
from unittest.mock import Mock
from streamlit.testing.v1 import AppTest
import requests

ROOT=Path(__file__).resolve().parents[2]


def test_failed_retry_clears_old_success(monkeypatch):
    class Response:
        def __init__(self, code, data):
            self.status_code=code;self.data=data;self.text=str(data)
        def json(self):return self.data
    trip=dict(trip_id='00000001',partition='student_test',points=2,nodes=2,arcs=1,duration_sec=1)
    def get(url,**kwargs):
        return Response(200,{'trips':[trip]}) if url.endswith('/trips') else Response(200,dict(model_loaded=True,service='PathSnap',model_name='Test model'))
    payload=dict(trip_id='00000001',status='success',predicted_arcs=[0,0],expanded_route=[0],predicted_segments=[[[0,0],[0,.001]]],gps_coords=[[0,0],[0,.001]],metrics=dict(point_accuracy=1,top1_accuracy=1,route_continuity=1,candidate_miss_rate=0,sequence_f1=1,inference_time_ms=1))
    post=Mock(side_effect=[Response(200,payload),Response(500,{'detail':'failure'})])
    monkeypatch.setattr(requests,'get',get);monkeypatch.setattr(requests,'post',post)
    app=AppTest.from_file(str(ROOT/'ui/app.py'),default_timeout=20).run()
    app.button[0].click().run()
    assert len(app.success)==1
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error)==1
    assert len(app.success)==0
    assert app.session_state['prediction_result'] is None
