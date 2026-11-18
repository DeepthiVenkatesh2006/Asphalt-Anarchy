import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.routes import get_service
from src.api.service import MapMatchingService


def make_trip(root, tid='00000001', track='0 0 1\n0.00001 0 2\n'):
    folder=root/'student_test'/tid;folder.mkdir(parents=True)
    (folder/f'{tid}.track').write_text(track)
    (folder/f'{tid}.nodes').write_text('0 0\n0.001 0\n')
    (folder/f'{tid}.arcs').write_text('0 1\n')
    return folder


def test_empty_catalog_is_not_healthy(tmp_path):
    service=MapMatchingService(data_root=tmp_path)
    app=create_app();app.dependency_overrides[get_service]=lambda:service
    response=TestClient(app).get('/health')
    assert response.status_code == 503
    assert response.json()['model_loaded'] is True
    assert response.json()['status'] == 'not_ready'


def test_bad_trip_does_not_hide_valid_trips(tmp_path):
    make_trip(tmp_path)
    bad=make_trip(tmp_path,'00000002',track='not coordinates\n')
    service=MapMatchingService(data_root=tmp_path)
    trips=service.list_available_trips()
    assert [t['trip_id'] for t in trips] == ['00000001']
    assert service.catalog_errors[0]['trip_id']=='00000002'
    assert trips[0]['reference_available'] is False


def test_track_only_trip_is_rejected_from_catalog(tmp_path):
    folder=make_trip(tmp_path)
    (folder/'00000001.arcs').unlink()
    service=MapMatchingService(data_root=tmp_path)
    assert service.list_available_trips() == []
    assert service.catalog_errors


def test_final_evaluation_settings_cannot_be_changed(tmp_path):
    service=MapMatchingService(data_root=tmp_path)
    with pytest.raises(ValueError,match='frozen'):
        service.process_trip('00000015',partition='evaluator_test',search_radius_m=100)
    with pytest.raises(ValueError,match='frozen'):
        service.process_trip('00000015',partition='evaluator_test',use_viterbi=False)


@pytest.mark.parametrize('reference',[None,'999\n'])
def test_evaluator_readiness_requires_valid_references(tmp_path,monkeypatch,reference):
    for i in range(10):
        folder=make_trip(tmp_path,f'{i:08d}')
        evaluation=tmp_path/'evaluator_test';evaluation.mkdir(exist_ok=True)
        new=evaluation/folder.name;folder.rename(new)
        if reference is not None:(new/f'{i:08d}.route').write_text(reference)
    monkeypatch.setenv('PATHSNAP_MODE','evaluation')
    service=MapMatchingService(data_root=tmp_path)
    app=create_app();app.dependency_overrides[get_service]=lambda:service
    response=TestClient(app).get('/health')
    assert response.status_code==503
    assert response.json()['evaluator_ready'] is False
