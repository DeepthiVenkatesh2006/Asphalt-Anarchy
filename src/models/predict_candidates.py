
import json
import os

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "artifacts/local_training/model_final.joblib"
METADATA_PATH = "artifacts/local_training/model_metadata.json"
CANDIDATES_PATH = "data/processed/v1/train/candidates.parquet"
OUTPUT_PATH = "artifacts/train_candidate_probabilities.parquet"

BATCH_SIZE = 100_000


def main():
    print("Loading trained MLP model...")

    from src.models.scoring import CandidateScorer
    scorer = CandidateScorer.from_directory(os.path.dirname(MODEL_PATH))
    model = scorer.model

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    features = metadata["feature_columns"]

    print("Model:", metadata["model"])
    print("Features:", len(features))
    print("Loading candidate dataset...")

    candidates = pd.read_parquet(CANDIDATES_PATH)

    missing = set(features) - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    if 1 not in model.classes_:
        raise ValueError("Model does not contain positive class 1")

    positive_class_index = list(model.classes_).index(1)

    output_columns = [
        "trip_id",
        "obs_idx",
        "timestamp_s",
        "segment_id",
        "candidate_idx",
        "arc_id",
    ]

    results = []

    total = len(candidates)

    print("Generating candidate probabilities...")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch = candidates.iloc[start:end]

        X = batch[features]

        probabilities = model.predict_proba(X)[
            :, positive_class_index
        ]

        result = batch[output_columns].copy()
        result["ml_probability"] = probabilities

        results.append(result)

        print(f"Processed {end:,} / {total:,} candidates")

    predictions = pd.concat(results, ignore_index=True)

    if len(predictions) != total:
        raise RuntimeError("Prediction row count mismatch")

    if not np.isfinite(predictions["ml_probability"]).all():
        raise ValueError("Non-finite probabilities detected")

    if not predictions["ml_probability"].between(0, 1).all():
        raise ValueError("Probabilities outside [0, 1]")

    os.makedirs("artifacts", exist_ok=True)

    predictions.to_parquet(OUTPUT_PATH, index=False)

    print("\nPREDICTION COMPLETED")
    print("Total candidates:", len(predictions))
    print("Unique trips:", predictions["trip_id"].nunique())
    print("Unique observations:", predictions[
        ["trip_id", "obs_idx"]
    ].drop_duplicates().shape[0])

    print("Mean probability:", predictions[
        "ml_probability"
    ].mean())

    print("Output saved to:", OUTPUT_PATH)

    print("\nSample predictions:")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
