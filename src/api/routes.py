import os
import logging
from fastapi import APIRouter, HTTPException, Query, Depends
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
from src.visualization.map_builder import build_folium_map
from src.visualization.leaflet_map import render_leaflet_map

logger = logging.getLogger("pathsnap.api")
router = APIRouter(tags=["PathSnap"])

_service: Optional[MapMatchingService] = None

def get_service() -> MapMatchingService:
    global _service
    if _service is None:
        _service = MapMatchingService()
    return _service

@router.get("/health", response_model=HealthResponse)
async def health_check(service: MapMatchingService = Depends(get_service)):
    trips = service.list_available_trips()
    return HealthResponse(
        status="ok",
        service="Asphalt-Anarchy PathSnap API",
        version="1.0.0",
        device="CPU (Optimized)",
        model_loaded=service.champion_model is not None,
        model_name="Champion MLP Pipeline + Viterbi Decoder" if service.champion_model else "Geometric Emission + Viterbi Decoder",
        total_available_trips=len(trips)
    )

@router.get("/trips", response_model=TripsListResponse)
async def list_trips(service: MapMatchingService = Depends(get_service)):
    trips = service.list_available_trips()
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

    try:
        result = service.process_trip(
            trip_id=req.trip_id,
            partition=req.partition,
            search_radius_m=req.search_radius_m or 50.0,
            use_viterbi=req.use_viterbi if req.use_viterbi is not None else True
        )
    except FileNotFoundError as e:
        logger.error(f"Trip not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing trip {req.trip_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # Save Folium map
    maps_dir = os.path.join(os.path.dirname(service.model_path), "maps")
    map_full_path = os.path.join(maps_dir, f"{req.trip_id}.html")
    try:
        build_folium_map(result, output_html_path=map_full_path)
    except Exception as e:
        logger.warning(f"Could not generate Folium map: {e}")

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
        total_predicted_arcs=len(result["predicted_route"]),
        map_url=f"/api/v1/map/{req.trip_id}?partition={req.partition}&search_radius_m={req.search_radius_m}&use_viterbi={str(req.use_viterbi if req.use_viterbi is not None else True).lower()}"
    )

@router.post("/predict", response_model=PredictResponse)
def predict_trip(req: PredictRequest, service: MapMatchingService = Depends(get_service)):
    logger.info(f">>> [PREDICT REQUEST] Received predict for Trip ID: '{req.trip_id}'")
    try:
        result = service.process_trip(
            trip_id=req.trip_id,
            partition=req.partition or "student_test",
            search_radius_m=req.search_radius_m,
            use_viterbi=req.use_viterbi if req.use_viterbi is not None else True
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    metrics = result["metrics"]
    return PredictResponse(
        trip_id=req.trip_id,
        partition=req.partition or "student_test",
        status="success",
        predicted_arcs=result["predicted_route"],
        expanded_route=result["condensed_route"],
        gps_coords=result["gps_coords"],
        predicted_coords=result.get("predicted_coords", []),
        predicted_segments=result.get("predicted_segments", []),
        total_arcs=len(result["predicted_route"]),
        route_continuity=metrics["route_continuity"],
        inference_time_ms=metrics["inference_time_ms"],
        metrics=EvaluationMetrics(**metrics),
        map_url=f"/api/v1/map/{req.trip_id}?partition={req.partition}&search_radius_m={req.search_radius_m}&use_viterbi={str(req.use_viterbi if req.use_viterbi is not None else True).lower()}"
    )

@router.get("/map/{trip_id}", response_class=HTMLResponse)
def get_map_view(
    trip_id: str,
    partition: str = Query("student_test", pattern="^(train|student_test|evaluator_test)$"),
    search_radius_m: float = Query(50.0, gt=0, le=1000),
    use_viterbi: bool = Query(True),
    service: MapMatchingService = Depends(get_service)
):
    if len(trip_id) != 8 or not trip_id.isascii() or not trip_id.isdigit():
        raise HTTPException(status_code=422, detail="Use an eight-digit trip ID")
    try:
        result = service.process_trip(trip_id=trip_id, partition=partition, search_radius_m=search_radius_m, use_viterbi=use_viterbi)
        return HTMLResponse(content=render_leaflet_map(result), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not render map for trip {trip_id}: {str(e)}")
