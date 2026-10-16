
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SUPPORT_DIR = BASE_DIR / "data" / "supporting"

activity = pd.read_csv(
    SUPPORT_DIR / "user_activity_train.csv"
)

print("Jumlah pengguna:", len(activity))
print("\nStatistik jumlah interaksi:")
print(activity["train_interactions"].describe())

# Batas kelompok berdasarkan tertil
activity["activity_group"] = pd.qcut(
    activity["train_interactions"].rank(method="first"),
    q=3,
    labels=["Low", "Medium", "High"]
)

print("\nJumlah pengguna per kelompok:")
print(activity["activity_group"].value_counts().sort_index())

print("\nRentang interaksi per kelompok:")
print(
    activity.groupby("activity_group", observed=True)[
        "train_interactions"
    ].agg(["min", "max", "mean"])
)
