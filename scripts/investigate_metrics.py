"""
Investigasi menyeluruh inkonsistensi metrik:
1. Hitung ulang dari per-user metrics → global + per-grup
2. Bandingkan hybrid_user_metrics vs hybrid_tune_a05_user_metrics
3. Cek semua model baseline (Popularity, MF, TF-IDF, Hybrid)
4. Verifikasi experiment_log.csv
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_common import RAW_DIR, RESULT_DIR

DATA_DIR = Path("data/processed")

# ---------------------------------------------------------------------------
# 1. Load ground-truth user list dan grup
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: User list & grup dari file resmi")
print("=" * 70)

# Cari file evaluable users
evalu_candidates = [
    DATA_DIR / "evaluable_users_validation_warm.csv",
    DATA_DIR / "users.csv",
    DATA_DIR / "user_activity_groups.csv",
    DATA_DIR / "user_activity_train.csv",
]
for f in evalu_candidates:
    print(f"  {'EXISTS' if f.exists() else 'MISSING'}: {f}")

# Load dari sumber yang ada
users_file = None
for f in [DATA_DIR / "users.csv", DATA_DIR / "evaluable_users_validation_warm.csv"]:
    if f.exists():
        users_file = f
        break

activity_file = None
for f in [DATA_DIR / "user_activity_train.csv", DATA_DIR / "user_activity_groups.csv"]:
    if f.exists():
        activity_file = f
        break

print(f"\nMenggunakan users: {users_file}")
print(f"Menggunakan activity: {activity_file}")

users_df = pd.read_csv(users_file, dtype={"user_id": str}) if users_file else None
activity_df = pd.read_csv(activity_file, dtype={"user_id": str}) if activity_file else None

if users_df is not None:
    print(f"users_df: {len(users_df)} rows, columns: {users_df.columns.tolist()}")
if activity_df is not None:
    print(f"activity_df: {len(activity_df)} rows, columns: {activity_df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 2. Fungsi hitung ulang metrik dari per-user CSV
# ---------------------------------------------------------------------------
def recompute_metrics(user_metrics_df, activity_df):
    """
    Hitung global mean dan per-grup mean dari per-user metrics.
    Gabungkan dengan activity_df untuk mendapat label grup.
    """
    df = user_metrics_df.copy()
    df["user_id"] = df["user_id"].astype(str)

    if activity_df is not None:
        act = activity_df.copy()
        act["user_id"] = act["user_id"].astype(str)
        # Cari kolom grup
        group_col = None
        for c in ["activity_group", "group", "activity_label"]:
            if c in act.columns:
                group_col = c
                break
        if group_col and group_col != "activity_group":
            act = act.rename(columns={group_col: "activity_group"})
        df = df.merge(act[["user_id", "activity_group"]], on="user_id", how="left",
                      suffixes=("", "_from_activity"))
        # Jika sudah ada kolom activity_group di df, pakai itu
        if "activity_group" not in df.columns or df["activity_group"].isna().all():
            if "activity_group_from_activity" in df.columns:
                df["activity_group"] = df["activity_group_from_activity"]

    global_recall = df["Recall@10"].mean()
    global_ndcg = df["NDCG@10"].mean()
    global_n = len(df)

    group_results = {}
    if "activity_group" in df.columns:
        for grp, gdf in df.groupby("activity_group"):
            group_results[grp] = {
                "n": len(gdf),
                "Recall@10": gdf["Recall@10"].mean(),
                "NDCG@10": gdf["NDCG@10"].mean(),
            }

    return global_recall, global_ndcg, global_n, group_results


# ---------------------------------------------------------------------------
# 3. Investigasi khusus Hybrid
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Investigasi Hybrid – hybrid_user_metrics vs hybrid_tune_a05")
print("=" * 70)

hu_orig = pd.read_csv(RESULT_DIR / "hybrid_user_metrics.csv", dtype={"user_id": str})
hu_tune = pd.read_csv(RESULT_DIR / "hybrid_tune_a05_user_metrics.csv", dtype={"user_id": str})

print(f"\nhybrid_user_metrics.csv:")
print(f"  Rows: {len(hu_orig)}, Columns: {hu_orig.columns.tolist()}")
print(f"  Recall mean: {hu_orig['Recall@10'].mean():.9f}")
print(f"  NDCG mean:   {hu_orig['NDCG@10'].mean():.9f}")

print(f"\nhybrid_tune_a05_user_metrics.csv:")
print(f"  Rows: {len(hu_tune)}, Columns: {hu_tune.columns.tolist()}")
print(f"  Recall mean: {hu_tune['Recall@10'].mean():.9f}")
print(f"  NDCG mean:   {hu_tune['NDCG@10'].mean():.9f}")

# Cek apakah user_id sama
orig_ids = set(hu_orig["user_id"].astype(str))
tune_ids = set(hu_tune["user_id"].astype(str))
print(f"\nUser IDs:")
print(f"  hybrid_user_metrics: {len(orig_ids)} unique")
print(f"  hybrid_tune_a05:     {len(tune_ids)} unique")
print(f"  Intersection:        {len(orig_ids & tune_ids)}")
print(f"  Only in orig:        {len(orig_ids - tune_ids)}")
print(f"  Only in tune:        {len(tune_ids - orig_ids)}")

# Merge dan bandingkan per-user
merged = hu_orig.merge(hu_tune, on="user_id", suffixes=("_orig", "_tune"))
print(f"\nMerged rows: {len(merged)}")
diff_ndcg = (merged["NDCG@10_orig"] - merged["NDCG@10_tune"]).abs()
diff_rec  = (merged["Recall@10_orig"] - merged["Recall@10_tune"]).abs()
print(f"  Max |NDCG diff|:   {diff_ndcg.max():.10f}")
print(f"  Max |Recall diff|: {diff_rec.max():.10f}")
print(f"  Users with NDCG diff > 1e-9: {(diff_ndcg > 1e-9).sum()}")

# Jika ada perbedaan — tampilkan contoh
if (diff_ndcg > 1e-9).any():
    print(f"\nContoh user dengan perbedaan:")
    print(merged[diff_ndcg > 1e-9][["user_id", "NDCG@10_orig", "NDCG@10_tune",
                                     "Recall@10_orig", "Recall@10_tune"]].head(10))

# Timestamps
import os
t_orig = os.path.getmtime(RESULT_DIR / "hybrid_user_metrics.csv")
t_tune = os.path.getmtime(RESULT_DIR / "hybrid_tune_a05_user_metrics.csv")
t_bag  = os.path.getmtime(RESULT_DIR / "hybrid_by_activity_group.csv")
t_ov   = os.path.getmtime(RESULT_DIR / "hybrid_overall_metrics.csv")
import datetime
fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
print(f"\nTimestamps:")
print(f"  hybrid_user_metrics.csv:          {fmt(t_orig)}")
print(f"  hybrid_tune_a05_user_metrics.csv: {fmt(t_tune)}")
print(f"  hybrid_by_activity_group.csv:     {fmt(t_bag)}")
print(f"  hybrid_overall_metrics.csv:       {fmt(t_ov)}")

# ---------------------------------------------------------------------------
# 4. Hitung ulang dari per-user ke per-group untuk hybrid
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Hitung ulang per-group dari hybrid_tune_a05_user_metrics.csv")
print("=" * 70)

# Gunakan kolom activity_group yang sudah ada di file (kalau ada)
print(f"Kolom di hu_tune: {hu_tune.columns.tolist()}")
if "activity_group" in hu_tune.columns:
    print("Activity group sudah ada di file tune.")
    for grp, gdf in hu_tune.groupby("activity_group"):
        print(f"  {grp}: n={len(gdf)}, Recall={gdf['Recall@10'].mean():.6f}, NDCG={gdf['NDCG@10'].mean():.6f}")
    total = hu_tune["Recall@10"].mean()
    total_ndcg = hu_tune["NDCG@10"].mean()
    print(f"  TOTAL (micro-avg): Recall={total:.6f}, NDCG={total_ndcg:.6f}")
else:
    print("Activity group TIDAK ada di file tune – perlu merge dari activity_df")
    if activity_df is not None:
        act = activity_df.copy()
        act["user_id"] = act["user_id"].astype(str)
        group_col = [c for c in act.columns if "group" in c.lower() or "label" in c.lower()]
        print(f"  Kolom activity_df: {act.columns.tolist()}")
        print(f"  Kandidat group col: {group_col}")

print(f"\nhybrid_by_activity_group.csv (dari run awal):")
hbag = pd.read_csv(RESULT_DIR / "hybrid_by_activity_group.csv")
print(hbag[["activity_group","users","Recall_at_10","NDCG_at_10"]].to_string(index=False))
print(f"hybrid_overall_metrics.csv (dari run awal): Recall={0.014496:.6f}, NDCG={0.009134:.6f}")

# ---------------------------------------------------------------------------
# 5. Semua model baseline – hitung ulang dari per-user
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Hitung ulang SEMUA model dari per-user metrics")
print("=" * 70)

model_files = {
    "popularity":        "popularity_user_metrics.csv",
    "mf":               "mf_user_metrics.csv",
    "tfidf_cbf":        "tfidf_cbf_user_metrics.csv",
    "hybrid":           "hybrid_user_metrics.csv",
    "hybrid_tune_a05":  "hybrid_tune_a05_user_metrics.csv",
    "pop_recency":      "popularity_recency_user_metrics.csv",
    "item_knn":         "item_knn_user_metrics.csv",
    "hybrid_new":       "hybrid_new_user_metrics.csv",
}

official_overall = {
    "popularity":   (0.0134985, 0.0092220),
    "mf":           (0.0130936, 0.0079174),
    "tfidf_cbf":    (0.0125253, 0.0073334),
    "hybrid":       (0.0144957, 0.0091342),
}

print(f"\n{'Model':<20} {'N':>6} {'Recall(recomputed)':>18} {'NDCG(recomputed)':>16} {'NDCG(official)':>14} {'MATCH':>6}")
print("-" * 90)
results_recomputed = {}
for model, fname in model_files.items():
    fpath = RESULT_DIR / fname
    if not fpath.exists():
        print(f"  {model:<20} MISSING: {fname}")
        continue
    df = pd.read_csv(fpath, dtype={"user_id": str})
    r = df["Recall@10"].mean()
    n = df["NDCG@10"].mean()
    official = official_overall.get(model, None)
    match = ""
    if official:
        match = "OK" if abs(n - official[1]) < 1e-5 else f"DIFF={n-official[1]:.6f}"
    results_recomputed[model] = {"n_users": len(df), "Recall@10": r, "NDCG@10": n}
    print(f"  {model:<20} {len(df):>6} {r:>18.9f} {n:>16.9f} {str(official[1]) if official else 'N/A':>14} {match:>6}")

# Per-group untuk model utama
print(f"\n{'Per-group recomputed (dari per-user metrics)':}")
for model, fname in model_files.items():
    fpath = RESULT_DIR / fname
    if not fpath.exists():
        continue
    df = pd.read_csv(fpath, dtype={"user_id": str})
    if "activity_group" not in df.columns:
        continue
    print(f"\n  {model}:")
    for grp in ["Low", "Medium", "High"]:
        gdf = df[df["activity_group"] == grp]
        if len(gdf) == 0:
            continue
        print(f"    {grp:6}: n={len(gdf):5d}  Recall={gdf['Recall@10'].mean():.6f}  NDCG={gdf['NDCG@10'].mean():.6f}")
    print(f"    {'Total':6}: n={len(df):5d}  Recall={df['Recall@10'].mean():.6f}  NDCG={df['NDCG@10'].mean():.6f}")

# ---------------------------------------------------------------------------
# 6. Verifikasi experiment_log
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: Verifikasi experiment_log.csv")
print("=" * 70)
log = pd.read_csv(RESULT_DIR / "experiment_log.csv")
print(f"Total rows: {len(log)}")

# Duplikat run_id
dupes = log[log.duplicated("run_id", keep=False)]
if len(dupes) > 0:
    print(f"DUPLIKAT run_id DITEMUKAN ({len(dupes)} rows):")
    print(dupes[["run_id", "timestamp", "model"]].to_string(index=False))
else:
    print("Tidak ada duplikat run_id.")

# Hybrid runs
hyb_rows = log[log["model"] == "hybrid"]
print(f"\nHybrid rows: {len(hyb_rows)} (expected 9)")
print(hyb_rows[["run_id", "recall_at_10", "ndcg_at_10"]].to_string(index=False))

# MF runs
mf_rows = log[log["model"] == "mf"]
print(f"\nMF rows: {len(mf_rows)} (expected 15)")
print(mf_rows[["run_id", "params", "recall_at_10", "ndcg_at_10"]].to_string(index=False))
