from pathlib import Path

import pandas as pd

from baseline_common import make_user_groups


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SUPPORT_DIR = BASE_DIR / "data" / "supporting"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_dataset(name, data):
    users = data["user_id"].nunique()
    items = data["item_id"].nunique()
    interactions = len(data)
    possible_interactions = users * items

    return {
        "dataset": name,
        "users": users,
        "items": items,
        "interactions": interactions,
        "sparsity": (
            1 - interactions / possible_interactions
            if possible_interactions
            else 0.0
        ),
        "mean_rating": data["rating"].mean(),
    }


train = pd.read_csv(
    RAW_DIR / "interactions_train.csv",
    dtype={"user_id": str, "item_id": str},
)
validation = pd.read_csv(
    RAW_DIR / "interactions_validation.csv",
    dtype={"user_id": str, "item_id": str},
)
candidates = pd.read_csv(
    RAW_DIR / "candidate_items_warm.csv",
    dtype={"item_id": str},
)
activity = pd.read_csv(
    SUPPORT_DIR / "user_activity_train.csv",
    dtype={"user_id": str},
)

statistics = pd.DataFrame([
    summarize_dataset("train", train),
    summarize_dataset("validation", validation),
])
statistics["candidate_items_warm"] = candidates["item_id"].nunique()
statistics.to_csv(
    RESULT_DIR / "dataset_statistics.csv",
    index=False,
)

grouped_activity = make_user_groups(activity)
activity_distribution = (
    grouped_activity.groupby("activity_group", observed=True)
    .agg(
        users=("user_id", "nunique"),
        min_interactions=("train_interactions", "min"),
        mean_interactions=("train_interactions", "mean"),
        max_interactions=("train_interactions", "max"),
    )
    .reset_index()
)
activity_distribution.to_csv(
    RESULT_DIR / "user_activity_distribution.csv",
    index=False,
)

rating_distribution = (
    train.groupby("rating")
    .size()
    .rename("interactions")
    .reset_index()
    .sort_values("rating")
)
rating_distribution.to_csv(
    RESULT_DIR / "rating_distribution.csv",
    index=False,
)

print("Dataset statistics:")
print(statistics.to_string(index=False))
print("\nRating distribution:")
print(rating_distribution.to_string(index=False))
print("\nUser activity distribution:")
print(activity_distribution.to_string(index=False))
