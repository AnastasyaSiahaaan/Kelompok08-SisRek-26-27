"""
Buat results/tuning_log.csv dari experiment_log.csv dengan kolom tambahan:
  seed, role (core/appendix), selected (bool), hyperparameters (dari run_id)

Verifikasi: jumlah baris == jumlah baris experiment_log.csv (61)
"""
import re, sys
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
from baseline_common import RESULT_DIR

log = pd.read_csv(RESULT_DIR / "experiment_log.csv")
print(f"experiment_log.csv: {len(log)} baris")

# --- Ekstrak seed dari run_id ---
def extract_seed(run_id):
    m = re.search(r'seed(\d+)', str(run_id))
    return int(m.group(1)) if m else 42  # default seed=42 jika tidak ada label

# --- Tentukan role ---
CORE_MODELS = {"popularity", "mf", "tfidf_cbf", "hybrid"}
CORE_RUNS = {
    "mf_components32_seed42",
    "tfidf_cbf_maxfeat100000_seed42",
    "hybrid_alpha0.5_seed42",
}
# Popularity tidak ada di experiment_log (tidak dituning), jadi hanya 3 above + 1 baseline

def get_role(row):
    model = str(row["model"])
    run_id = str(row["run_id"])
    if model in {"popularity_recency", "item_knn", "hybrid_new"}:
        return "appendix"
    if model in CORE_MODELS:
        return "core"
    return "other"

def get_selected(row):
    """True jika run ini adalah konfigurasi final yang dipilih."""
    run_id = str(row["run_id"])
    selected_runs = {
        "mf_components32_seed42",         # MF final
        "tfidf_cbf_maxfeat100000_seed42",  # TF-IDF final
        "hybrid_alpha0.5_seed42",           # Hybrid final
        "popularity_recency_window_days180_seed42",  # Pop Recency final
        "item_knn_shrinkage500_seed42",     # Item-kNN final (berdasarkan experiment_log)
        "hybrid_new_wknn0.4_wcbf0.4_wpop0.2_seed42",  # Hybrid New final (reported)
    }
    return run_id in selected_runs

# --- Ekstrak hyperparameters dari params kolom ---
tuning = log.copy()
tuning["seed"] = tuning["run_id"].apply(extract_seed)
tuning["role"] = tuning.apply(get_role, axis=1)
tuning["selected"] = tuning.apply(get_selected, axis=1)

# Rename kolom agar konsisten
rename_map = {
    "recall_at_10": "Recall@10",
    "ndcg_at_10":   "NDCG@10",
    "recall_low":   "Recall@10_Low",
    "ndcg_low":     "NDCG@10_Low",
    "recall_medium":"Recall@10_Medium",
    "ndcg_medium":  "NDCG@10_Medium",
    "recall_high":  "Recall@10_High",
    "ndcg_high":    "NDCG@10_High",
}
tuning = tuning.rename(columns=rename_map)

# Kolom final
cols = ["run_id", "model", "params", "seed", "role", "selected",
        "Recall@10", "NDCG@10",
        "Recall@10_Low", "NDCG@10_Low",
        "Recall@10_Medium", "NDCG@10_Medium",
        "Recall@10_High", "NDCG@10_High",
        "n_evaluated_users", "notes", "timestamp", "git_commit"]
cols = [c for c in cols if c in tuning.columns]
tuning = tuning[cols]

out_path = RESULT_DIR / "tuning_log.csv"
tuning.to_csv(out_path, index=False)
print(f"tuning_log.csv: {len(tuning)} baris -> {out_path}")

# Verifikasi jumlah baris
assert len(tuning) == len(log), f"MISMATCH: {len(tuning)} vs {len(log)}"
print("✅ Jumlah baris tuning_log == experiment_log:", len(tuning))

# Ringkasan
print("\nDistribusi role:")
print(tuning["role"].value_counts().to_string())
print("\nSelected runs:")
print(tuning[tuning["selected"]][["run_id","model","seed","role","NDCG@10"]].to_string())
print("\nSemua model yang ada di log:")
print(tuning["model"].value_counts().to_string())
