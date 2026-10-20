
from pathlib import Path
import pandas as pd

from baseline_common import make_user_groups

BASE_DIR = Path(__file__).resolve().parent.parent
SUPPORT_DIR = BASE_DIR / "data" / "supporting"

activity = pd.read_csv(
    SUPPORT_DIR / "user_activity_train.csv"
)
evaluable_users = pd.read_csv(
    SUPPORT_DIR / "evaluable_users_validation_warm.csv"
)

activity = activity[
    activity["user_id"].isin(evaluable_users["user_id"])
].copy()

print("Jumlah pengguna evaluasi:", len(activity))
print("\nStatistik jumlah interaksi:")
print(activity["train_interactions"].describe())

activity = make_user_groups(activity)

print("\nJumlah pengguna per kelompok:")
print(activity["activity_group"].value_counts().sort_index())

print("\nRentang interaksi per kelompok:")
print(
    activity.groupby("activity_group", observed=True)[
        "train_interactions"
    ].agg(["min", "max", "mean"])
)
