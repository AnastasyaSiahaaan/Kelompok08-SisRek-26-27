
from pathlib import Path
import pandas as pd

from baseline_common import make_user_groups

BASE_DIR = Path(__file__).resolve().parent.parent
SUPPORT_DIR = BASE_DIR / "data" / "supporting"
RESULT_DIR = BASE_DIR / "results"

# Baca hasil evaluasi popularity per pengguna
metrics = pd.read_csv(
    RESULT_DIR / "popularity_validation_user_metrics.csv"
)

# Baca aktivitas pengguna berdasarkan data training
activity = pd.read_csv(
    SUPPORT_DIR / "user_activity_train.csv"
)

# Gunakan definisi kelompok yang sama dengan seluruh baseline lainnya.
activity = make_user_groups(activity)

# Gabungkan metrik dan kelompok berdasarkan user_id
merged = metrics.merge(
    activity[["user_id", "train_interactions", "activity_group"]],
    on="user_id",
    how="inner",
    validate="one_to_one"
)

# Hitung rata-rata metrik untuk setiap kelompok
summary = (
    merged.groupby("activity_group", observed=True)
    .agg(
        users=("user_id", "nunique"),
        Recall_at_10=("Recall@10", "mean"),
        NDCG_at_10=("NDCG@10", "mean")
    )
    .reset_index()
)

summary.insert(0, "model", "popularity")
summary = summary[
    ["model", "activity_group", "users", "Recall_at_10", "NDCG_at_10"]
]

print("=== POPULARITY BASELINE BERDASARKAN AKTIVITAS ===")
print(summary.to_string(index=False))

output_path = RESULT_DIR / "popularity_by_activity_group.csv"
summary.to_csv(output_path, index=False)

print("\nHasil disimpan di:", output_path)
print("Jumlah pengguna yang cocok:", merged["user_id"].nunique())
