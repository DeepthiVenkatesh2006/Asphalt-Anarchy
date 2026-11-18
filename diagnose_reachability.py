
import numpy as np
import pandas as pd
from pyproj import Geod

from src.road_network.graph_builder import build_trip_graph
from src.decoder.transition_model import calculate_transition_probability


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TRIP = "00000082"
SEQUENCE = "00000082_00235"
BETA = 50.0

CANDIDATES_PATH = "data/processed/v1/train/candidates.parquet"
ROADS_PATH = "data/processed/v1/train/road_segments.parquet"
OBSERVATIONS_PATH = "data/processed/v1/train/observations.parquet"
EMISSIONS_PATH = "artifacts/train_hmm_emissions.parquet"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading diagnostic data...")

candidates = pd.read_parquet(CANDIDATES_PATH)
roads = pd.read_parquet(ROADS_PATH)
observations = pd.read_parquet(OBSERVATIONS_PATH)
emissions = pd.read_parquet(EMISSIONS_PATH)

# Keep arc_id from emissions.
# Retrieve projection_ratio from candidates only.

data = emissions[
    (emissions["trip_id"].astype(str) == TRIP)
    & (emissions["sequence_id"].astype(str) == SEQUENCE)
].merge(
    candidates[
        [
            "trip_id",
            "obs_idx",
            "candidate_idx",
            "projection_ratio",
        ]
    ],
    on=["trip_id", "obs_idx", "candidate_idx"],
    how="left",
    validate="one_to_one",
)

if data.empty:
    raise ValueError("No emission data found for selected sequence")

if data["projection_ratio"].isna().any():
    raise ValueError("Missing candidate projection ratios")

data = data.merge(
    observations[
        [
            "trip_id",
            "obs_idx",
            "longitude",
            "latitude",
        ]
    ],
    on=["trip_id", "obs_idx"],
    how="left",
    validate="many_to_one",
)

if data[["longitude", "latitude"]].isna().any().any():
    raise ValueError("Missing GPS coordinates")

data = data.sort_values(
    ["obs_idx", "candidate_idx"]
).reset_index(drop=True)


# --------------------------------------------------
# BUILD SEQUENCE GROUPS
# --------------------------------------------------

groups = [
    group.sort_values(
        "candidate_idx"
    ).reset_index(drop=True)

    for _, group in data.groupby(
        "obs_idx",
        sort=True,
    )
]

print("Sequence:", SEQUENCE)
print("Observations:", len(groups))

# --------------------------------------------------
# BUILD ROAD GRAPH
# --------------------------------------------------

print("\nBuilding road graph...")

trip_roads = roads[
    roads["trip_id"].astype(str) == TRIP
]

graph = build_trip_graph(
    trip_roads,
    TRIP,
)

print("Graph nodes:", graph.number_of_nodes())
print("Graph edges:", graph.number_of_edges())


# --------------------------------------------------
# TRANSITION REACHABILITY DIAGNOSTIC
# --------------------------------------------------

geod = Geod(ellps="WGS84")

# Initially every candidate at observation 0 is reachable.

reachable = np.ones(
    len(groups[0]),
    dtype=bool,
)

print("\nChecking reachability...")
print("Starting at observation:", groups[0]["obs_idx"].iloc[0])


for index in range(len(groups) - 1):

    previous = groups[index]
    current = groups[index + 1]

    previous_idx = int(previous["obs_idx"].iloc[0])
    current_idx = int(current["obs_idx"].iloc[0])

    # GPS displacement between observations.

    previous_gps = previous.iloc[0]
    current_gps = current.iloc[0]

    _, _, distance = geod.inv(
        float(previous_gps["longitude"]),
        float(previous_gps["latitude"]),
        float(current_gps["longitude"]),
        float(current_gps["latitude"]),
    )

    distance = abs(float(distance))

    # Construct transition matrix.

    matrix = np.zeros(
        (len(previous), len(current)),
        dtype=np.float64,
    )

    for i, prev in enumerate(
        previous.itertuples(index=False)
    ):

        for j, curr in enumerate(
            current.itertuples(index=False)
        ):

            matrix[i, j] = calculate_transition_probability(
                graph=graph,
                previous_arc_id=int(prev.arc_id),
                current_arc_id=int(curr.arc_id),
                observed_distance_m=distance,
                beta=BETA,
                previous_projection_ratio=float(
                    prev.projection_ratio
                ),
                current_projection_ratio=float(
                    curr.projection_ratio
                ),
            )

    # Determine reachable next states.

    next_reachable = np.any(
        matrix[reachable, :] > 0,
        axis=0,
    )

    # Print diagnostics near the known boundary.

    if 74 <= previous_idx <= 80:

        print(
            f"\nTransition {previous_idx} -> {current_idx}"
        )

        print(
            "GPS distance:",
            round(distance, 3),
            "m",
        )

        print(
            "Previous reachable candidate indices:",
            np.where(reachable)[0].tolist(),
        )

        print(
            "Previous reachable arc IDs:",
            previous.loc[
                reachable, "arc_id"
            ].astype(int).tolist(),
        )

        print(
            "Current reachable candidate indices:",
            np.where(next_reachable)[0].tolist(),
        )

        print(
            "Current candidate arc IDs:",
            current["arc_id"].astype(int).tolist(),
        )

        print(
            "Nonzero matrix entries:",
            int(np.count_nonzero(matrix)),
        )

        print(
            "Nonzero entries from reachable states:",
            int(np.count_nonzero(matrix[reachable, :])),
        )

    # Detect the first impossible boundary.

    if not next_reachable.any():

        print("\n" + "=" * 55)
        print("FIRST IMPOSSIBLE BOUNDARY FOUND")
        print("=" * 55)

        print("Previous observation:", previous_idx)
        print("Current observation:", current_idx)

        print(
            "Previous reachable candidate indices:",
            np.where(reachable)[0].tolist(),
        )

        print(
            "Previous reachable arc IDs:",
            previous.loc[
                reachable, "arc_id"
            ].astype(int).tolist(),
        )

        print(
            "Current candidate arc IDs:",
            current["arc_id"].astype(int).tolist(),
        )

        print("\nTransition matrix:")
        print(np.round(matrix, 6))

        print("\nTransition matrix from reachable states:")
        print(
            np.round(
                matrix[reachable, :],
                6,
            )
        )

        break

    reachable = next_reachable

else:

    print("\nNo impossible boundary found in this sequence.")

print("\nDiagnostic completed.")
