
import numpy as np
import pandas as pd
from pyproj import Geod

from src.road_network.graph_builder import build_trip_graph
from src.decoder.transition_model import calculate_transition_probability


EMISSIONS_PATH = "artifacts/train_hmm_emissions.parquet"
CANDIDATES_PATH = "data/processed/v1/train/candidates.parquet"
OBSERVATIONS_PATH = "data/processed/v1/train/observations.parquet"
ROAD_SEGMENTS_PATH = "data/processed/v1/train/road_segments.parquet"

GEOD = Geod(ellps="WGS84")


def main():
    print("Loading data...")

    emissions = pd.read_parquet(EMISSIONS_PATH)
    candidates = pd.read_parquet(CANDIDATES_PATH)
    observations = pd.read_parquet(OBSERVATIONS_PATH)
    roads = pd.read_parquet(ROAD_SEGMENTS_PATH)

    keys = ["trip_id", "obs_idx", "candidate_idx"]

    candidate_fields = [
        "from_node_id",
        "to_node_id",
        "projection_ratio",
        "segment_length_m",
    ]

    # Restore road metadata for every candidate.
    data = emissions.merge(
        candidates[keys + candidate_fields],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if data[candidate_fields].isna().any().any():
        raise ValueError("Candidate metadata join contains missing values")

    # Select the first sequence containing at least two observations.
    lengths = (
        emissions.groupby("sequence_id")["obs_idx"]
        .nunique()
    )

    sequence_id = lengths[lengths >= 2].index[0]

    sequence = data[
        data["sequence_id"] == sequence_id
    ].copy()

    sequence = sequence.sort_values(
        ["obs_idx", "candidate_idx"]
    )

    trip_id = str(sequence["trip_id"].iloc[0])

    print("Selected sequence:", sequence_id)
    print("Trip:", trip_id)
    print("Observations:", sequence["obs_idx"].nunique())

    # Build only the selected trip's directed road graph.
    trip_roads = roads[
        roads["trip_id"].astype(str) == trip_id
    ]

    graph = build_trip_graph(trip_roads, trip_id)

    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    trip_observations = observations[
        observations["trip_id"].astype(str) == trip_id
    ].set_index("obs_idx")

    grouped = [
        group.sort_values("candidate_idx")
        for _, group in sequence.groupby("obs_idx", sort=True)
    ]

    transition_matrices = []
    zero_rows = 0
    zero_matrices = 0

    for previous, current in zip(grouped[:-1], grouped[1:]):

        previous_obs = trip_observations.loc[
            int(previous["obs_idx"].iloc[0])
        ]

        current_obs = trip_observations.loc[
            int(current["obs_idx"].iloc[0])
        ]

        # Actual GPS displacement in metres.
        _, _, observed_distance = GEOD.inv(
            float(previous_obs["longitude"]),
            float(previous_obs["latitude"]),
            float(current_obs["longitude"]),
            float(current_obs["latitude"]),
        )

        matrix = np.zeros(
            (len(previous), len(current)),
            dtype=np.float64,
        )

        for i, prev_candidate in enumerate(
            previous.itertuples(index=False)
        ):
            for j, curr_candidate in enumerate(
                current.itertuples(index=False)
            ):
                matrix[i, j] = calculate_transition_probability(
                    graph=graph,
                    previous_arc_id=int(prev_candidate.arc_id),
                    current_arc_id=int(curr_candidate.arc_id),
                    observed_distance_m=observed_distance,
                    beta=50.0,
                    previous_projection_ratio=float(
                        prev_candidate.projection_ratio
                    ),
                    current_projection_ratio=float(
                        curr_candidate.projection_ratio
                    ),
                )

        row_sums = matrix.sum(axis=1)

        zero_rows += int((row_sums == 0).sum())

        # Normalize each previous-candidate row.
        valid_rows = row_sums > 0
        matrix[valid_rows] /= row_sums[valid_rows, None]

        if not valid_rows.any():
            zero_matrices += 1

        transition_matrices.append(matrix)

    print("\nTRANSITION MATRIX TEST")
    print("Matrices generated:", len(transition_matrices))
    print("First matrix shape:", transition_matrices[0].shape)
    print("Zero transition rows:", zero_rows)
    print("Completely impossible matrices:", zero_matrices)

    all_values = np.concatenate([
        matrix.ravel() for matrix in transition_matrices
    ])

    print("Minimum transition probability:", all_values.min())
    print("Maximum transition probability:", all_values.max())

    print("\nFirst normalized transition matrix:")
    print(transition_matrices[0])

    # Validate that nonzero rows sum to one.
    for matrix in transition_matrices:
        sums = matrix.sum(axis=1)
        valid = sums > 0

        if not np.allclose(sums[valid], 1.0):
            raise ValueError("Transition matrix normalization failed")

    print("\nTransition matrix validation PASSED")


if __name__ == "__main__":
    main()
