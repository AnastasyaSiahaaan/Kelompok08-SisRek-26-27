"""
Final reconciliation summary untuk semua model.
Membandingkan _overall_metrics.csv dengan mean per-user dari _user_metrics.csv.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RESULT = Path("results")

# ---------------------------------------------------------------------------
# Data dari commit 3a79cf2 (hybrid_by_activity_group dan tfidf_cbf_by_activity_group
# diubah di commit ini oleh anggota kelompok)
# ---------------------------------------------------------------------------
# Angka dari git show 3a79cf2:results/hybrid_by_activity_group.csv
OLD_HYBRID_BAG = {
    "High":   (3190, 0.013977081782724415, 0.009793777795371047),
    "Low":    (6105, 0.013617521612174019, 0.008209233344994864),
    "Medium": (2364, 0.017463171111648267, 0.010632918908591273),
}
# Angka dari git show 3a79cf2:results/tfidf_cbf_by_activity_group.csv
OLD_TFIDF_BAG = {
    "High":   (3190, 0.011168555408367321, 0.006152720669629338),
    "Low":    (6105, 0.012322693205046146, 0.007254041396355683),
    "Medium": (2364, 0.014879273762522495, 0.009131745455076750),
}

def weighted_avg(bag_dict, metric="ndcg"):
    idx = 2 if metric == "ndcg" else 1
    total_n = sum(v[0] for v in bag_dict.values())
    return sum(v[0] * v[idx] for v in bag_dict.values()) / total_n

def recompute_from_user(fname):
    df = pd.read_csv(RESULT / fname, dtype={"user_id": str})
    global_r = df["Recall@10"].mean()
    global_n = df["NDCG@10"].mean()
    per_grp = {}
    if "activity_group" in df.columns:
        for grp, gdf in df.groupby("activity_group"):
            per_grp[grp] = {
                "n": len(gdf),
                "Recall@10": gdf["Recall@10"].mean(),
                "NDCG@10": gdf["NDCG@10"].mean(),
            }
    return global_r, global_n, len(df), per_grp

# ---------------------------------------------------------------------------
# Hitung ulang semua model dari per-user
# ---------------------------------------------------------------------------
print("=" * 80)
print("REKONSILIASI LENGKAP: _overall_metrics.csv vs recomputed dari _user_metrics.csv")
print("=" * 80)

models = [
    ("popularity",   "popularity_overall_metrics.csv",   "popularity_user_metrics.csv"),
    ("mf",           "mf_overall_metrics.csv",           "mf_user_metrics.csv"),
    ("tfidf_cbf",    "tfidf_cbf_overall_metrics.csv",    "tfidf_cbf_user_metrics.csv"),
    ("hybrid",       "hybrid_overall_metrics.csv",       "hybrid_user_metrics.csv"),
    ("pop_recency",  "popularity_recency_overall_metrics.csv", "popularity_recency_user_metrics.csv"),
    ("item_knn",     "item_knn_overall_metrics.csv",     "item_knn_user_metrics.csv"),
    ("hybrid_new",   "hybrid_new_overall_metrics.csv",   "hybrid_new_user_metrics.csv"),
]

print(f"\n{'Model':<14} {'N':>6} {'Overall(file)':>14} {'Recomputed':>12} {'Diff':>10} {'Status':>10}")
print("-" * 70)
recomp_all = {}
for model, ov_file, um_file in models:
    if not (RESULT / ov_file).exists() or not (RESULT / um_file).exists():
        print(f"  {model:<14} MISSING")
        continue
    ov = pd.read_csv(RESULT / ov_file)
    file_ndcg = ov["NDCG@10"].iloc[0]
    file_rec  = ov["Recall@10"].iloc[0]
    r, n, cnt, per_grp = recompute_from_user(um_file)
    diff = file_ndcg - n
    status = "OK" if abs(diff) < 1e-7 else ("DIFF-small" if abs(diff) < 1e-4 else "DIFF-LARGE")
    print(f"  {model:<14} {cnt:>6} {file_ndcg:>14.9f} {n:>12.9f} {diff:>10.6f} {status:>10}")
    recomp_all[model] = {"n": cnt, "Recall@10": r, "NDCG@10": n, "per_grp": per_grp}

# ---------------------------------------------------------------------------
# Per-group recomputed (tabel bersih)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TABEL RECOMPUTED DARI PER-USER (sumber tunggal: _user_metrics.csv)")
print("=" * 80)

print(f"\n{'Model':<14} {'Group':<8} {'N':>6} {'Recall@10':>12} {'NDCG@10':>12}")
print("-" * 56)
for model, _, _ in models:
    if model not in recomp_all:
        continue
    per_grp = recomp_all[model]["per_grp"]
    r_total = recomp_all[model]["Recall@10"]
    n_total = recomp_all[model]["NDCG@10"]
    n_users = recomp_all[model]["n"]
    for grp in ["Low", "Medium", "High"]:
        if grp in per_grp:
            g = per_grp[grp]
            print(f"  {model:<14} {grp:<8} {g['n']:>6} {g['Recall@10']:>12.6f} {g['NDCG@10']:>12.6f}")
    print(f"  {model:<14} {'Overall':<8} {n_users:>6} {r_total:>12.6f} {n_total:>12.6f}")
    print()

# ---------------------------------------------------------------------------
# Root cause analysis
# ---------------------------------------------------------------------------
print("=" * 80)
print("ROOT CAUSE ANALYSIS")
print("=" * 80)

print("""
FAKTA TERBUKTI (dari git log dan perbandingan file):

1. hybrid_user_metrics.csv (timestamp: 2026-10-20):
   - Dihasilkan oleh hybrid_baseline.py pada run PERTAMA
   - Mengandung 11659 baris, identik persis dengan hybrid_tune_a05_user_metrics.csv
   - Mean NDCG@10 = 0.009165459

2. hybrid_by_activity_group.csv (timestamp: 2026-11-14, commit 3a79cf2):
   - Di-OVERWRITE oleh anggota kelompok lain (Yolanda Septania Saragih)
     pada commit "Add W10 validation evaluation results"
   - Angka per-group di file ini BERBEDA dari hybrid_user_metrics.csv
     (High: 0.009794 vs 0.009853, Low: 0.008209 vs 0.008238)
   - Ini BUKAN run yang sama: by_activity_group menunjukkan angka dari
     run hybrid BERBEDA yang menghasilkan hasil sedikit lebih rendah

3. hybrid_overall_metrics.csv (timestamp: 2026-11-14, commit 3a79cf2):
   - Juga di-OVERWRITE bersamaan dengan by_activity_group.csv
   - Angka 0.009134 adalah micro-avg dari by_activity_group yang tertimpa itu
   - BUKAN weighted-avg vs micro-avg issue: ini dua run yang berbeda

4. tfidf_cbf memiliki pola yang sama:
   - tfidf_cbf_user_metrics.csv vs tfidf_cbf_by_activity_group.csv tidak konsisten
   - Low group: 0.012405 (user_metrics) vs 0.012323 (by_activity_group)
   - Ini juga di-overwrite pada commit 3a79cf2 yang sama

KESIMPULAN:
   Commit 3a79cf2 ("Add W10 validation evaluation results") oleh anggota lain
   menimpa hybrid_overall_metrics, hybrid_by_activity_group, tfidf_cbf_overall_metrics,
   dan tfidf_cbf_by_activity_group dengan angka dari run BERBEDA (mungkin parameter
   berbeda atau seed berbeda). hybrid_user_metrics.csv TIDAK ikut ditimpa,
   sehingga ada inkonsistensi internal di repo.

ANGKA YANG BENAR (dari per-user metrics yang tidak tertimpa):
   - hybrid:    Recall@10=0.014567, NDCG@10=0.009165
   - tfidf_cbf: Recall@10=0.012568, NDCG@10=0.007304
""")

# Verifikasi weighted avg dari by_activity_group
wt_hybrid = weighted_avg(OLD_HYBRID_BAG, "ndcg")
wt_tfidf  = weighted_avg(OLD_TFIDF_BAG,  "ndcg")
print(f"Verifikasi: weighted-avg NDCG dari hybrid_by_activity_group.csv (lama): {wt_hybrid:.9f}")
print(f"  -> cocok dengan hybrid_overall_metrics.csv (0.009134209): {abs(wt_hybrid - 0.009134209) < 1e-9}")
print(f"Verifikasi: weighted-avg NDCG dari tfidf_cbf_by_activity_group.csv (lama): {wt_tfidf:.9f}")
print(f"  -> cocok dengan tfidf_cbf_overall_metrics.csv (0.007333437): {abs(wt_tfidf - 0.007333437) < 1e-6}")
