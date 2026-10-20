
import pandas as pd
from pathlib import Path

results_dir = Path("results")

models = ["popularity", "mf", "tfidf_cbf", "hybrid"]

# Gabungkan hasil evaluasi keseluruhan
overall = pd.concat(
    [
        pd.read_csv(results_dir / f"{model}_overall_metrics.csv")
        for model in models
    ],
    ignore_index=True
)

overall.to_csv(
    results_dir / "comparison_overall.csv",
    index=False
)

print("\n=== PERBANDINGAN KESELURUHAN ===")
print(overall.to_string(index=False))

# Gabungkan hasil berdasarkan kelompok aktivitas pengguna
by_group = pd.concat(
    [
        pd.read_csv(results_dir / f"{model}_by_activity_group.csv")
        for model in models
    ],
    ignore_index=True
)

by_group.to_csv(
    results_dir / "comparison_by_activity_group.csv",
    index=False
)

print("\n=== PERBANDINGAN BERDASARKAN AKTIVITAS ===")
print(by_group.to_string(index=False))

print("\nFile perbandingan berhasil disimpan di folder results.")
