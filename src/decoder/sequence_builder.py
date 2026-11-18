
import os
import numpy as np
import pandas as pd


OBSERVATIONS_PATH = "data/processed/v1/train/observations.parquet"
PREDICTIONS_PATH = "artifacts/train_candidate_probabilities.parquet"

OUTPUT_PATH = "artifacts/train_hmm_emissions.parquet"
SUMMARY_PATH = "artifacts/train_hmm_sequence_summary.csv"


def main():
    print("Loading observations and ML predictions...")

    observations = pd.read_parquet(OBSERVATIONS_PATH)
    predictions = pd.read_parquet(PREDICTIONS_PATH)

    observations = observations.sort_values(
        ["trip_id", "obs_idx"]
    ).reset_index(drop=True)

    predictions = predictions.sort_values(
        ["trip_id", "obs_idx", "candidate_idx"]
    ).reset_index(drop=True)

    # Check that candidate counts agree with the observation table.
    actual_counts = (
        predictions.groupby(["trip_id", "obs_idx"])
        .size()
        .rename("actual_candidates")
        .reset_index()
    )

    check = observations.merge(
        actual_counts,
        on=["trip_id", "obs_idx"],
        how="left",
    )

    check["actual_candidates"] = (
        check["actual_candidates"].fillna(0).astype(int)
    )

    if not (
        check["num_candidates"] == check["actual_candidates"]
    ).all():
        raise ValueError("Candidate counts do not match observations")

    print("Candidate-count consistency verified.")

    # Normalize classifier probabilities within each observation.
    predictions["probability_sum"] = (
        predictions.groupby(["trip_id", "obs_idx"])[
            "ml_probability"
        ].transform("sum")
    )

    if (predictions["probability_sum"] <= 0).any():
        raise ValueError("An observation has zero total ML probability")

    predictions["emission_probability"] = (
        predictions["ml_probability"]
        / predictions["probability_sum"]
    )

    # Build contiguous sequences. Empty observations break sequences.
    prediction_groups = {
        key: group.copy()
        for key, group in predictions.groupby(
            ["trip_id", "obs_idx"], sort=False
        )
    }

    output_rows = []
    summaries = []

    sequence_number = 0

    for trip_id, trip_obs in observations.groupby(
        "trip_id", sort=True
    ):
        current_sequence = []
        previous_obs_idx = None
        previous_segment_id = None

        def save_sequence():
            nonlocal sequence_number, current_sequence

            if not current_sequence:
                return

            sequence_number += 1
            sequence_id = f"{trip_id}_{sequence_number:05d}"

            sequence_rows = []

            for candidate_group in current_sequence:
                candidate_group = candidate_group.copy()
                candidate_group["sequence_id"] = sequence_id
                sequence_rows.append(candidate_group)

            sequence_data = pd.concat(
                sequence_rows, ignore_index=True
            )

            output_rows.append(sequence_data)

            summaries.append({
                "sequence_id": sequence_id,
                "trip_id": str(trip_id),
                "num_observations": len(current_sequence),
                "num_candidates": len(sequence_data),
                "first_obs_idx": int(
                    sequence_data["obs_idx"].min()
                ),
                "last_obs_idx": int(
                    sequence_data["obs_idx"].max()
                ),
            })

            current_sequence = []

        for row in trip_obs.itertuples(index=False):
            obs_idx = int(row.obs_idx)
            segment_id = row.segment_id

            key = (str(trip_id), obs_idx)
            candidate_group = prediction_groups.get(key)

            # No candidates: close the current sequence.
            if candidate_group is None:
                save_sequence()
                previous_obs_idx = obs_idx
                previous_segment_id = segment_id
                continue

            # Start a new sequence at a GPS gap, segment change,
            # or non-consecutive observation index.
            must_split = (
                previous_obs_idx is None
                or bool(row.gap_before)
                or segment_id != previous_segment_id
                or obs_idx != previous_obs_idx + 1
            )

            if must_split:
                save_sequence()

            current_sequence.append(candidate_group)

            previous_obs_idx = obs_idx
            previous_segment_id = segment_id

        save_sequence()

    if not output_rows:
        raise RuntimeError("No valid HMM sequences were created")

    emissions = pd.concat(output_rows, ignore_index=True)
    summary = pd.DataFrame(summaries)

    # Verify each observation's emissions sum to one.
    emission_sums = emissions.groupby(
        ["trip_id", "obs_idx"]
    )["emission_probability"].sum()

    if not np.allclose(emission_sums.values, 1.0, atol=1e-9):
        raise ValueError("Emission probabilities failed normalization")

    os.makedirs("artifacts", exist_ok=True)

    emissions.to_parquet(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nHMM SEQUENCE PREPARATION COMPLETED")
    print("Trips:", observations["trip_id"].nunique())
    print("Sequences:", len(summary))
    print("Observations in sequences:", len(emission_sums))
    print("Candidate rows:", len(emissions))
    print("Average observations per sequence:",
          round(summary["num_observations"].mean(), 2))
    print("Shortest sequence:",
          summary["num_observations"].min())
    print("Longest sequence:",
          summary["num_observations"].max())
    print("Emission sum range:",
          emission_sums.min(), "to", emission_sums.max())

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
