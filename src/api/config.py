"""Validated inference settings shared by the API and development experiments."""
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]


class InferenceSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)
    search_radius_meters: float = Field(50.0, gt=0, le=1000)
    max_candidates_per_point: int = Field(8, ge=1, le=64)
    gap_threshold_s: float = Field(60.0, gt=0)
    beta_transition: float = Field(50.0, gt=0)


def load_inference_settings(path=None):
    config_path = Path(path or ROOT / 'configs/api.yaml')
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict) or set(config) != {'inference'}:
        raise ValueError('configs/api.yaml must contain only the inference section')
    return InferenceSettings.model_validate(config['inference'])
