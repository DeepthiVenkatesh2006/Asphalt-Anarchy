from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class TripSummary(BaseModel):
    trip_id: str = Field(..., pattern=r'^[0-9]{8}$', description="Unique trip identifier")
    partition: str = Field(..., description="Dataset partition: 'train' or 'student_test'")
    points: int = Field(..., description="Number of GPS observations")
    nodes: int = Field(..., description="Number of road network nodes")
    arcs: int = Field(..., description="Number of road network arcs")
    reference_available: bool = False
    reference_valid: bool = False
    duration_sec: float = Field(0.0, description="Duration of trajectory in seconds")

class TripsListResponse(BaseModel):
    total_trips: int
    partitions: Dict[str, int]
    trips: List[TripSummary]

class EvaluationRequest(BaseModel):
    trip_id: str = Field(..., pattern=r'^[0-9]{8}$', description="Trip ID to evaluate (e.g. '00000000')")
    partition: Literal["train", "student_test", "evaluator_test"] = Field("student_test", description="Dataset partition to fetch trip from")
    search_radius_m: Optional[float] = Field(None, gt=0, le=1000, description="Spatial candidate radius in meters")
    use_viterbi: Optional[bool] = Field(True, description="Whether to apply HMM/Viterbi route decoding")

class EvaluationMetrics(BaseModel):
    point_accuracy: Optional[float] = None
    observation_macro_f1: Optional[float] = None
    observation_macro_precision: Optional[float] = None
    observation_macro_recall: Optional[float] = None
    classifier_observation_macro_f1: Optional[float] = None
    candidate_log_loss: Optional[float] = None
    log_loss_observation_coverage: Optional[float] = None
    route_set_f1: Optional[float] = None
    sequence_similarity: Optional[float] = None
    evaluation_observations: int = 0
    evaluation_coverage: Optional[float] = None
    unmatched_observations: int = 0
    decoder_restarts: int = 0
    evaluation_note: str = ""
    total_observations: int = Field(..., description="Total GPS observation count")
    top1_accuracy: Optional[float] = Field(None, description="Observation-level Top-1 Arc Accuracy")
    route_continuity: float = Field(..., description="Percentage of consecutive predicted arcs connected in road graph")
    candidate_miss_rate: Optional[float] = Field(None, description="Percentage of observations where true arc was not in candidate set")
    sequence_length: int = Field(..., description="Length of predicted sequence")
    ground_truth_length: int = Field(..., description="Length of ground truth route")
    sequence_f1: Optional[float] = Field(None, description="Deprecated alias of order-insensitive road-set F1; use sequence_similarity for ordering")
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
    search_radius_m: Optional[float] = Field(None, gt=0, le=1000)
    trip_id: str = Field(..., pattern=r'^[0-9]{8}$', description="Trip identifier")
    partition: Literal["train", "student_test", "evaluator_test"] = Field("student_test", description="Dataset partition")
    use_viterbi: Optional[bool] = Field(True, description="Enable Viterbi decoding")

class PredictResponse(BaseModel):
    trip_id: str
    partition: str = "student_test"
    status: str = "success"
    predicted_arcs: List[int]
    network_segments: List[List[List[float]]] = Field(default_factory=list)
    matched_coords: List[Optional[List[float]]] = Field(default_factory=list)
    gps_coords: List[List[float]] = Field(default_factory=list)
    route_chunks: List[List[int]] = Field(default_factory=list)
    expanded_route: List[int] = Field(default_factory=list)
    predicted_segments: List[List[List[float]]] = Field(default_factory=list)
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
    evaluator_ready: bool = False
    evaluation_mode: bool = False
    invalid_trips: List[Dict[str, str]] = Field(default_factory=list)
    reference_errors: List[Dict[str, str]] = Field(default_factory=list)
    partitions: Dict[str, int] = Field(default_factory=dict)
    loaded_partitions: Dict[str, int] = Field(default_factory=dict)
    inference_settings: Dict[str, Any] = Field(default_factory=dict)
