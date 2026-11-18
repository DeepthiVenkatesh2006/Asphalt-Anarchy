from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TripSummary(BaseModel):
    trip_id: str = Field(..., description="Unique trip identifier")
    partition: str = Field(..., description="Dataset partition: 'train' or 'student_test'")
    points: int = Field(..., description="Number of GPS observations")
    nodes: int = Field(..., description="Number of road network nodes")
    arcs: int = Field(..., description="Number of road network arcs")
    route_arcs: int = Field(0, description="Number of reference route arcs if available")
    duration_sec: float = Field(0.0, description="Duration of trajectory in seconds")

class TripsListResponse(BaseModel):
    total_trips: int
    partitions: Dict[str, int]
    trips: List[TripSummary]

class EvaluationRequest(BaseModel):
    trip_id: str = Field(..., description="Trip ID to evaluate (e.g. '00000000')")
    partition: str = Field("student_test", description="Dataset partition to fetch trip from")
    search_radius_m: Optional[float] = Field(50.0, description="Spatial candidate radius in meters")
    use_viterbi: Optional[bool] = Field(True, description="Whether to apply HMM/Viterbi route decoding")

class EvaluationMetrics(BaseModel):
    total_observations: int = Field(..., description="Total GPS observation count")
    top1_accuracy: float = Field(..., description="Observation-level Top-1 Arc Accuracy")
    route_continuity: float = Field(..., description="Percentage of consecutive predicted arcs connected in road graph")
    candidate_miss_rate: float = Field(..., description="Percentage of observations where true arc was not in candidate set")
    sequence_length: int = Field(..., description="Length of predicted sequence")
    ground_truth_length: int = Field(..., description="Length of ground truth route")
    sequence_f1: float = Field(..., description="Sequence-level F1 score against ground truth")
    inference_time_ms: float = Field(..., description="Total pipeline latency in milliseconds")

class EvaluationResponse(BaseModel):
    trip_id: str
    partition: str
    status: str = "success"
    metrics: EvaluationMetrics
    predicted_route_preview: List[int] = Field(..., description="First 20 predicted arc IDs")
    total_predicted_arcs: int
    map_url: str
    message: str = "Evaluation completed successfully."

class PredictRequest(BaseModel):
    trip_id: str = Field(..., description="Trip identifier")
    partition: Optional[str] = Field("student_test", description="Dataset partition")
    use_viterbi: Optional[bool] = Field(True, description="Enable Viterbi decoding")

class PredictResponse(BaseModel):
    trip_id: str
    partition: str = "student_test"
    status: str = "success"
    predicted_arcs: List[int]
    predicted_coords: List[List[float]] = Field(default_factory=list, description="Ordered [[lat, lon], ...] along predicted road segments")
    total_arcs: int
    route_continuity: float
    inference_time_ms: float
    metrics: Optional[EvaluationMetrics] = None
    map_url: str

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "Asphalt-Anarchy PathSnap API"
    version: str = "1.0.0"
    device: str
    model_loaded: bool
    model_name: str
    total_available_trips: int
