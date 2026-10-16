
from pathlib import Path
import pandas as pd

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

# Buat kelompok dengan metode yang sama seperti sebelumnya
activity["activity_group"] = pd.qcut(
    activity["train_interactions"].rank(method="first"),
    q=3,
    labels=["Low", "Medium", "High"]
)

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
        mean_interactions=("train_interactions", "mean"),
        Recall_at_10=("Recall@10", "mean"),
        NDCG_at_10=("NDCG@10", "mean")
    )
    .reset_index()
)

summary["mean_interactions"] = summary["mean_interactions"].round(2)
summary["Recall_at_10"] = summary["Recall_at_10"].round(4)
summary["NDCG_at_10"] = summary["NDCG_at_10"].round(4)

print("=== POPULARITY BASELINE BERDASARKAN AKTIVITAS ===")
print(summary.to_string(index=False))

output_path = RESULT_DIR / "popularity_by_activity_group.csv"
summary.to_csv(output_path, index=False)

print("\nHasil disimpan di:", output_path)
print("Jumlah pengguna yang cocok:", merged["user_id"].nunique())
