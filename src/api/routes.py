import os
import logging
from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import HTMLResponse
from typing import Optional

from src.api.schemas import (
    TripSummary,
    TripsListResponse,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationMetrics,
    PredictRequest,
    PredictResponse,
    HealthResponse
)
from src.api.service import MapMatchingService, PARTITIONS
from src.visualization.leaflet_map import render_leaflet_map
from src.api import map_cache

logger = logging.getLogger("pathsnap.api")
router = APIRouter(tags=["PathSnap"])

_service: Optional[MapMatchingService] = None

def get_service() -> MapMatchingService:
    global _service
    if _service is None:
        try:
            _service = MapMatchingService()
        except (OSError, ValueError, KeyError) as error:
            logger.exception("Model or configuration startup failed")
            raise HTTPException(status_code=503, detail="Model or runtime configuration failed validation") from error
    return _service

@router.get('/health', response_model=HealthResponse)
def health_check(response: Response, service: MapMatchingService = Depends(get_service)):
    trips = service.list_available_trips()
    partitions = {p: sum(t['partition'] == p for t in trips) for p in PARTITIONS}
    evaluator_ready = (partitions['evaluator_test'] == 10
                       and all(t['reference_valid'] for t in trips if t['partition']=='evaluator_test')
                       and not any(e['partition']=='evaluator_test' for e in service.catalog_errors))
    ready = bool(trips) and (evaluator_ready if service.evaluation_mode else True)
    if not ready:
        response.status_code = 503
    visible = [t for t in trips if t['partition']=='evaluator_test'] if service.evaluation_mode else trips
    visible_partitions = {p: sum(t['partition']==p for t in visible) for p in PARTITIONS}
    return HealthResponse(status='ok' if ready else 'not_ready', service='Asphalt-Anarchy PathSnap API',
        version='1.0.0', device='CPU', model_loaded=service.champion_model is not None,
        model_name=f"Selected {service.scorer.metadata.get('model', 'classifier').upper()} + graph HMM/Viterbi", total_available_trips=len(visible),
        evaluator_ready=evaluator_ready, evaluation_mode=service.evaluation_mode,
        invalid_trips=service.catalog_errors, reference_errors=service.reference_errors, partitions=visible_partitions, loaded_partitions=partitions,
        inference_settings=service.settings.model_dump())

@router.get("/trips", response_model=TripsListResponse)
async def list_trips(service: MapMatchingService = Depends(get_service)):
    trips = service.list_available_trips()
    if service.evaluation_mode:
        trips = [t for t in trips if t["partition"] == "evaluator_test"]
    train_count = sum(1 for t in trips if t["partition"] == "train")
    test_count = sum(1 for t in trips if t["partition"] == "student_test")
    return TripsListResponse(
        total_trips=len(trips),
        partitions={p: sum(t["partition"] == p for t in trips) for p in PARTITIONS},
        trips=[TripSummary(**t) for t in trips]
    )

@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_trip(req: EvaluationRequest, service: MapMatchingService = Depends(get_service)):
    logger.info(f">>> [INFERENCE REQUEST] Received evaluation for Trip ID: '{req.trip_id}' (partition: {req.partition})")

    key=map_cache.map_key(service,req.partition,req.trip_id,req.search_radius_m,req.use_viterbi if req.use_viterbi is not None else True)
    map_cache.invalidate(key)
    try:
        result = service.process_trip(
            trip_id=req.trip_id,
            partition=req.partition,
            search_radius_m=req.search_radius_m,
            use_viterbi=req.use_viterbi if req.use_viterbi is not None else True
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Trip not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing trip {req.trip_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    map_cache.store(key,render_leaflet_map(result))

    metrics = result["metrics"]
    logger.info(
        f"<<< [INFERENCE COMPLETE] Trip '{req.trip_id}' | "
        f"Top-1 Acc: {metrics['top1_accuracy']} | "
        f"Continuity: {metrics['route_continuity']*100:.1f}% | "
        f"Latency: {metrics['inference_time_ms']:.1f}ms"
    )

    return EvaluationResponse(
        trip_id=req.trip_id,
        partition=req.partition,
        status="success",
        metrics=EvaluationMetrics(**metrics),
        predicted_route_preview=result["preview_arcs"],
        total_predicted_arcs=len(result["condensed_route"]),
        map_url=f"/api/v1/map/{req.trip_id}?partition={req.partition}&search_radius_m={result['search_radius_m']}&use_viterbi={str(req.use_viterbi if req.use_viterbi is not None else True).lower()}"
    )

@router.post("/predict", response_model=PredictResponse)
def predict_trip(req: PredictRequest, service: MapMatchingService = Depends(get_service)):
    logger.info(f">>> [PREDICT REQUEST] Received predict for Trip ID: '{req.trip_id}'")
    key=map_cache.map_key(service,req.partition,req.trip_id,req.search_radius_m,req.use_viterbi if req.use_viterbi is not None else True)
    map_cache.invalidate(key)
    try:
        result = service.process_trip(
            trip_id=req.trip_id,
            partition=req.partition or "student_test",
            search_radius_m=req.search_radius_m,
            use_viterbi=req.use_viterbi if req.use_viterbi is not None else True
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    map_cache.store(key,render_leaflet_map(result))
    metrics = result["metrics"]
    return PredictResponse(
        trip_id=req.trip_id,
        partition=req.partition or "student_test",
        status="success",
        predicted_arcs=result["predicted_route"],
        expanded_route=result["condensed_route"],
        route_chunks=result["route_chunks"],
        gps_coords=result["gps_coords"],
        matched_coords=result["matched_coords"],
        network_segments=result["network_segments"],
        predicted_coords=result.get("predicted_coords", []),
        predicted_segments=result.get("predicted_segments", []),
        total_arcs=len(result["predicted_route"]),
        route_continuity=metrics["route_continuity"],
        inference_time_ms=metrics["inference_time_ms"],
        metrics=EvaluationMetrics(**metrics),
        map_url=f"/api/v1/map/{req.trip_id}?partition={req.partition}&search_radius_m={result['search_radius_m']}&use_viterbi={str(req.use_viterbi if req.use_viterbi is not None else True).lower()}"
    )

@router.get("/map/{trip_id}", response_class=HTMLResponse)
def get_map_view(
    trip_id: str,
    partition: str = Query("student_test", pattern="^(train|student_test|evaluator_test)$"),
    search_radius_m: Optional[float] = Query(None, gt=0, le=1000),
    use_viterbi: bool = Query(True),
    service: MapMatchingService = Depends(get_service)
):
    if len(trip_id) != 8 or not trip_id.isascii() or not trip_id.isdigit():
        raise HTTPException(status_code=422, detail="Use an eight-digit trip ID")
    key=map_cache.map_key(service,partition,trip_id,search_radius_m,use_viterbi)
    cached=map_cache.lookup(key)
    if cached is not None:
        return HTMLResponse(content=cached,status_code=200)
    try:
        result = service.process_trip(trip_id=trip_id, partition=partition, search_radius_m=search_radius_m, use_viterbi=use_viterbi)
        markup=render_leaflet_map(result)
        map_cache.store(key,markup)
        return HTMLResponse(content=markup, status_code=200)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Map rendering failed")
        raise HTTPException(status_code=500, detail="Map rendering failed") from e
