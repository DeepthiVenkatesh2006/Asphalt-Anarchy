
import os
import numpy as np
import pandas as pd
from pyproj import Geod

from src.road_network.graph_builder import build_trip_graph
from src.decoder.transition_model import calculate_transition_probability
from src.decoder.hmm import prepare_hmm_inputs
from src.decoder.viterbi import viterbi_decode


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EMISSIONS_PATH = "artifacts/train_hmm_emissions.parquet"
CANDIDATES_PATH = "data/processed/v1/train/candidates.parquet"
OBSERVATIONS_PATH = "data/processed/v1/train/observations.parquet"
ROADS_PATH = "data/processed/v1/train/road_segments.parquet"

OUTPUT_PATH = "artifacts/train_viterbi_predictions.parquet"
SUMMARY_PATH = "artifacts/train_viterbi_summary.csv"

BETA = 50.0

GEOD = Geod(ellps="WGS84")


# --------------------------------------------------
# BUILD TRANSITION MATRIX
# --------------------------------------------------

def build_transition_matrix(
    graph,
    previous,
    current,
    previous_observation,
    current_observation,
):
    """Build and normalize one transition matrix."""

    _, _, observed_distance = GEOD.inv(
        float(previous_observation.longitude),
        float(previous_observation.latitude),
        float(current_observation.longitude),
        float(current_observation.latitude),
    )

    matrix = np.zeros(
        (len(previous), len(current)),
        dtype=np.float64,
    )

    for i, prev in enumerate(previous.itertuples(index=False)):
        for j, curr in enumerate(current.itertuples(index=False)):

            matrix[i, j] = calculate_transition_probability(
                graph=graph,
                previous_arc_id=int(prev.arc_id),
                current_arc_id=int(curr.arc_id),
                observed_distance_m=float(observed_distance),
                beta=BETA,
                previous_projection_ratio=float(
                    prev.projection_ratio
                ),
                current_projection_ratio=float(
                    curr.projection_ratio
                ),
            )

    # Normalize each row independently.
    row_sums = matrix.sum(axis=1)
    valid_rows = row_sums > 0

    matrix[valid_rows] /= row_sums[valid_rows, None]

    # All-zero rows remain zero.
    return matrix


# --------------------------------------------------
# VITERBI DECODING
# --------------------------------------------------

def decode_chunk(
    candidate_groups,
    transition_matrices,
    sequence_id,
    chunk_number,
):
    """Decode one continuous sequence chunk using Viterbi."""

    if not candidate_groups:
        return []

    emissions = [
        group["emission_probability"].astype(float).tolist()
        for group in candidate_groups
    ]

    log_emissions, log_transitions = prepare_hmm_inputs(
        emissions,
        transition_matrices,
    )

    selected_indices = viterbi_decode(
        log_emissions,
        log_transitions,
    )

    rows = []

    for group, selected_index in zip(
        candidate_groups,
        selected_indices,
    ):
        selected = group.iloc[selected_index]

        rows.append({
            "sequence_id": str(sequence_id),
            "decode_chunk": int(chunk_number),
            "trip_id": str(selected["trip_id"]),
            "obs_idx": int(selected["obs_idx"]),
            "candidate_idx": int(selected["candidate_idx"]),
            "arc_id": int(selected["arc_id"]),
            "ml_probability": float(
                selected["ml_probability"]
            ),
            "emission_probability": float(
                selected["emission_probability"]
            ),
        })

    return rows


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    print("Loading input data...")

    emissions = pd.read_parquet(EMISSIONS_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)
    observations = pd.read_parquet(OBSERVATIONS_PATH)
    roads = pd.read_parquet(ROADS_PATH)

    print("Emissions:", len(emissions))
    print("Candidates:", len(candidates))
    print("Observations:", len(observations))

    keys = [
        "trip_id",
        "obs_idx",
        "candidate_idx",
    ]

    candidate_fields = [
        "from_node_id",
        "to_node_id",
        "projection_ratio",
        "segment_length_m",
    ]

    # --------------------------------------------------
    # JOIN CANDIDATE ROAD METADATA
    # --------------------------------------------------

    print("\nJoining candidate road metadata...")

    data = emissions.merge(
        candidates[keys + candidate_fields],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if data[candidate_fields].isna().any().any():
        raise ValueError(
            "Candidate metadata join contains missing values"
        )

    # --------------------------------------------------
    # JOIN GPS COORDINATES
    # --------------------------------------------------

    print("Joining GPS coordinates...")

    gps = observations[
        [
            "trip_id",
            "obs_idx",
            "longitude",
            "latitude",
        ]
    ]

    data = data.merge(
        gps,
        on=["trip_id", "obs_idx"],
        how="left",
        validate="many_to_one",
    )

    if data[
        ["longitude", "latitude"]
    ].isna().any().any():
        raise ValueError(
            "GPS coordinate join contains missing values"
        )

    data = data.sort_values(
        [
            "trip_id",
            "sequence_id",
            "obs_idx",
            "candidate_idx",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # INITIALIZE OUTPUTS
    # --------------------------------------------------

    predictions = []
    summaries = []

    total_sequences = data["sequence_id"].nunique()

    completed_sequences = 0
    total_chunks = 0
    total_zero_matrices = 0

    print("\nStarting trip-by-trip HMM decoding...")
    print("Total sequences:", total_sequences)

    # --------------------------------------------------
    # PROCESS ONE TRIP AT A TIME
    # --------------------------------------------------

    for trip_id, trip_data in data.groupby(
        "trip_id",
        sort=True,
    ):

        trip_roads = roads[
            roads["trip_id"].astype(str) == str(trip_id)
        ]

        graph = build_trip_graph(
            trip_roads,
            str(trip_id),
        )

        print(
            f"\nTrip {trip_id}: "
            f"{graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        # --------------------------------------------------
        # PROCESS EACH HMM SEQUENCE
        # --------------------------------------------------

        for sequence_id, sequence in trip_data.groupby(
            "sequence_id",
            sort=True,
        ):

            groups = [
                group.sort_values(
                    "candidate_idx"
                ).reset_index(drop=True)

                for _, group in sequence.groupby(
                    "obs_idx",
                    sort=True,
                )
            ]

            if not groups:
                continue

            # Track sequence-level statistics.
            sequence_prediction_start = len(predictions)
            sequence_chunk_start = total_chunks
            sequence_boundary_start = total_zero_matrices

            # GPS coordinate lookup.
            observation_lookup = {
                int(row.obs_idx): row
                for row in sequence[
                    [
                        "obs_idx",
                        "longitude",
                        "latitude",
                    ]
                ].drop_duplicates(
                    "obs_idx"
                ).itertuples(index=False)
            }

            # --------------------------------------------------
            # GENERATE TRANSITION MATRICES
            # --------------------------------------------------

            matrices = []

            for previous, current in zip(
                groups[:-1],
                groups[1:],
            ):

                previous_idx = int(
                    previous["obs_idx"].iloc[0]
                )

                current_idx = int(
                    current["obs_idx"].iloc[0]
                )

                if current_idx != previous_idx + 1:
                    raise ValueError(
                        f"Non-consecutive observations "
                        f"in sequence {sequence_id}"
                    )

                matrix = build_transition_matrix(
                    graph=graph,
                    previous=previous,
                    current=current,
                    previous_observation=(
                        observation_lookup[previous_idx]
                    ),
                    current_observation=(
                        observation_lookup[current_idx]
                    ),
                )

                matrices.append(matrix)

            # --------------------------------------------------
            # SPLIT WHEN NO TRANSITION IS REACHABLE
            # --------------------------------------------------

            chunk_groups = [groups[0]]
            chunk_matrices = []
            chunk_number = 0

            # Every candidate in the first observation
            # is initially reachable.
            reachable = np.ones(
                len(groups[0]),
                dtype=bool,
            )

            for index, matrix in enumerate(matrices):

                # Check transitions only from reachable states.
                current_reachable = np.any(
                    matrix[reachable, :] > 0,
                    axis=0,
                )

                if not current_reachable.any():

                    # No continuous path exists across this boundary.
                    total_zero_matrices += 1

                    # Decode the completed left-hand chunk.
                    predictions.extend(
                        decode_chunk(
                            chunk_groups,
                            chunk_matrices,
                            sequence_id,
                            chunk_number,
                        )
                    )

                    total_chunks += 1
                    chunk_number += 1

                    # Restart from the current observation.
                    chunk_groups = [
                        groups[index + 1]
                    ]

                    chunk_matrices = []

                    reachable = np.ones(
                        len(groups[index + 1]),
                        dtype=bool,
                    )

                else:

                    chunk_groups.append(
                        groups[index + 1]
                    )

                    chunk_matrices.append(matrix)

                    reachable = current_reachable

            # --------------------------------------------------
            # DECODE FINAL REMAINING CHUNK
            # --------------------------------------------------

            if chunk_groups:

                predictions.extend(
                    decode_chunk(
                        chunk_groups,
                        chunk_matrices,
                        sequence_id,
                        chunk_number,
                    )
                )

                total_chunks += 1

            # --------------------------------------------------
            # RECORD SEQUENCE SUMMARY
            # --------------------------------------------------

            decoded_count = (
                len(predictions) - sequence_prediction_start
            )

            expected_count = len(groups)

            sequence_chunks = (
                total_chunks - sequence_chunk_start
            )

            sequence_boundaries = (
                total_zero_matrices - sequence_boundary_start
            )

            fully_decoded = (
                decoded_count == expected_count
            )

            summaries.append({
                "trip_id": str(trip_id),
                "sequence_id": str(sequence_id),
                "observations": expected_count,
                "decoded_observations": decoded_count,
                "decoded_chunks": sequence_chunks,
                "impossible_boundaries": sequence_boundaries,
                "fully_decoded": fully_decoded,
            })

            if fully_decoded:
                completed_sequences += 1

        # Release this trip's graph before loading another.
        del graph

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    print("\nPreparing final prediction files...")

    if not predictions:
        raise RuntimeError(
            "No Viterbi predictions were generated"
        )

    result = pd.DataFrame(predictions)
    summary = pd.DataFrame(summaries)

    result = result.sort_values(
        [
            "trip_id",
            "sequence_id",
            "obs_idx",
        ]
    ).reset_index(drop=True)

    # Validate that each decoded observation appears once.
    if result.duplicated(
        ["trip_id", "obs_idx"]
    ).any():
        raise ValueError(
            "Duplicate observations found in Viterbi output"
        )

    # Validate prediction count against input emission observations.
    expected_observations = emissions[
        ["trip_id", "obs_idx"]
    ].drop_duplicates().shape[0]

    if len(result) != expected_observations:
        raise ValueError(
            f"Decoded {len(result)} observations, "
            f"expected {expected_observations}"
        )

    # Validate that every sequence has a summary.
    if len(summary) != total_sequences:
        raise ValueError(
            f"Generated {len(summary)} sequence summaries, "
            f"expected {total_sequences}"
        )

    os.makedirs("artifacts", exist_ok=True)

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n===================================")
    print("HMM + VITERBI DECODING COMPLETED")
    print("===================================")

    print("Sequences decoded:", completed_sequences)
    print("Total sequences:", total_sequences)
    print("Decoded chunks:", total_chunks)

    print(
        "Completely impossible boundaries:",
        total_zero_matrices,
    )

    print(
        "Predicted observations:",
        len(result),
    )

    print(
        "Unique trips:",
        result["trip_id"].nunique(),
    )

    print(
        "Unique predicted road arcs:",
        result["arc_id"].nunique(),
    )

    print(
        "Fully decoded sequences:",
        int(summary["fully_decoded"].sum()),
    )

    print("\nSaved files:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)

    print("\nSample predictions:")
    print(result.head(10).to_string(index=False))

    print("\nSequence summary:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
