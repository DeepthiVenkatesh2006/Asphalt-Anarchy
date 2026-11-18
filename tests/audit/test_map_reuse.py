from types import SimpleNamespace
from unittest.mock import Mock
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.routes import get_service


def test_displaying_evaluated_map_does_not_repeat_inference():
    metrics=dict(total_observations=1,top1_accuracy=None,route_continuity=0,candidate_miss_rate=None,sequence_length=1,ground_truth_length=0,sequence_f1=None,inference_time_ms=1)
    result=dict(trip_id='00000001',partition='student_test',search_radius_m=50,metrics=metrics,predicted_route=[0],condensed_route=[0],preview_arcs=[0],predicted_segments=[[[0,0],[0,.001]]],gps_coords=[[0,.0005]])
    service=SimpleNamespace(settings=SimpleNamespace(search_radius_meters=50))
    # Use a real hashable service identity, as production MapMatchingService does.
    class Service:
        settings=service.settings
        process_trip=Mock(return_value=result)
    instance=Service()
    app=create_app();app.dependency_overrides[get_service]=lambda:instance
    client=TestClient(app)
    response=client.post('/evaluate',json=dict(trip_id='00000001',partition='student_test'))
    assert response.status_code==200
    mapped=client.get(response.json()['map_url'])
    assert mapped.status_code==200 and 'L.map(' in mapped.text
    assert instance.process_trip.call_count==1
