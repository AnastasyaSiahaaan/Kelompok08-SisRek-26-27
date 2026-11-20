"""
Track A - Analisis Lengkap (Bagian 12 Reproducibility Package)

Menghasilkan:
  results/track_a_table.csv          - metrik 4 model x 4 kelompok (Recall/NDCG/HitRate/MRR)
  results/track_a_bootstrap_ci.csv   - paired CI (Bonferroni-adjusted untuk 4 perbandingan konfirmatori)
  results/track_a_indep_ci.csv       - independent CI selisih High-Low per model per metrik
  results/track_a_popularity.csv     - rata-rata popularitas item Top-10 per model per kelompok
  results/track_a_overlap.csv        - overlap Top-10 Hybrid dengan MF dan TF-IDF per kelompok

Ketentuan rancangan:
  - Konfirmatori (Bonferroni k=4): Hybrid vs Pop dan Hybrid vs MF, Low dan High, NDCG@10
  - Semua lainnya: eksploratori (95% CI nominal, bukan adjusted)
  - Independent CI High-Low: bootstrap resampling terpisah per kelompok; JANGAN
    menyimpulkan "High lebih baik/buruk" bila CI memuat 0.
  - Random seed: 42, n_bootstrap: 10_000

Jalankan dari root proyek:
    python src/track_a_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from baseline_common import RESULT_DIR

RANDOM_SEED = 42
N_BOOTSTRAP = 10_000

# ---------------------------------------------------------------------------
# Daftar 4 model inti (urutan laporan)
# ---------------------------------------------------------------------------
MODELS = ["popularity", "mf", "tfidf_cbf", "hybrid"]
MODEL_LABELS = {
    "popularity": "Popularity",
    "mf": "MF (n=32)",
    "tfidf_cbf": "TF-IDF CBF",
    "hybrid": "Hybrid (a=0.5)",
}

GROUPS = ["Overall", "Low", "Medium", "High"]
METRICS = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]

# Batas kelompok (didefinisikan di baseline_common.make_user_groups):
#   Low    : train_interactions in [1, 5]
#   Medium : train_interactions in [6, 8]
#   High   : train_interactions >= 9
# Batas ini ditetapkan di awal proyek (sebelum melihat data validasi) dan
# diimplementasikan dengan pd.cut(bins=[0,5,8,inf], include_lowest=True).
GROUP_DEFINITIONS = {
    "Low":    "train_interactions in [1, 5]",
    "Medium": "train_interactions in [6, 8]",
    "High":   "train_interactions >= 9",
}

# ---------------------------------------------------------------------------
# Konfirmatori (Bonferroni): 4 sel
# ---------------------------------------------------------------------------
CONFIRMATORY = [
    # (model_a, model_b, group, metric)
    ("hybrid", "popularity", "Low",  "NDCG@10"),
    ("hybrid", "popularity", "High", "NDCG@10"),
    ("hybrid", "mf",         "Low",  "NDCG@10"),
    ("hybrid", "mf",         "High", "NDCG@10"),
]
BONFERRONI_K = len(CONFIRMATORY)   # 4
ALPHA_CONFIRMATORY = 0.05 / BONFERRONI_K   # 0.0125 -> CI 98.75%


# ===========================================================================
# BAGIAN 1 - Muat user_metrics semua model
# ===========================================================================

def load_user_metrics():
    """Muat user_metrics.csv untuk 4 model; tambahkan HitRate/MRR = NaN bila tidak ada."""
    dfs = {}
    for m in MODELS:
        path = RESULT_DIR / f"{m}_user_metrics.csv"
        df = pd.read_csv(path, dtype={"user_id": str})
        # Tambahkan kolom yang mungkin tidak ada
        for col in ["HitRate@10", "MRR@10"]:
            if col not in df.columns:
                df[col] = np.nan
        dfs[m] = df.set_index("user_id")
    return dfs


# ===========================================================================
# BAGIAN 2 - Tabel metrik lengkap (Tabel Track A)
# ===========================================================================

def build_track_a_table(dfs):
    """
    Bangun tabel 4 model x 4 kelompok dengan semua metrik.
    Kolom: model, label, group, n_users, Recall@10, NDCG@10, HitRate@10, MRR@10
    """
    rows = []
    activity_path = BASE_DIR / "data" / "supporting" / "user_activity_train.csv"
    activity = pd.read_csv(activity_path, dtype={"user_id": str})
    total_evaluable = None   # dihitung dari salah satu df

    for m in MODELS:
        df = dfs[m].reset_index()
        if total_evaluable is None:
            total_evaluable = len(df)

        # Overall
        row = {
            "model": m,
            "label": MODEL_LABELS[m],
            "group": "Overall",
            "n_users": len(df),
        }
        for metric in METRICS:
            row[metric] = df[metric].mean() if metric in df.columns else np.nan
        rows.append(row)

        # Per group
        for grp in ["Low", "Medium", "High"]:
            sub = df[df["activity_group"] == grp]
            r = {
                "model": m,
                "label": MODEL_LABELS[m],
                "group": grp,
                "n_users": len(sub),
            }
            for metric in METRICS:
                r[metric] = sub[metric].mean() if metric in sub.columns else np.nan
            rows.append(r)

    table = pd.DataFrame(rows)

    # Tambahkan kolom persentase pengguna (hanya untuk grup spesifik)
    n_total = table.loc[table["group"] == "Overall", "n_users"].iloc[0]
    table["pct_users"] = table.apply(
        lambda r: f"{100 * r['n_users'] / n_total:.1f}%"
        if r["group"] != "Overall" else "100.0%",
        axis=1,
    )

    return table


# ===========================================================================
# BAGIAN 3 - Bootstrap CI Paired (konfirmatori + eksploratori)
# ===========================================================================

PAIRED_COMPARISONS = [
    ("mf",        "popularity", "MF vs Popularity"),
    ("tfidf_cbf", "popularity", "TF-IDF vs Popularity"),
    ("hybrid",    "popularity", "Hybrid vs Popularity"),
    ("hybrid",    "mf",         "Hybrid vs MF"),
    ("hybrid",    "tfidf_cbf",  "Hybrid vs TF-IDF"),
]


def _bootstrap_paired(a: np.ndarray, b: np.ndarray, n: int, seed: int,
                      alpha: float = 0.05):
    """
    Paired bootstrap: mean(a - b).
    Returns (mean_diff, ci_lo, ci_hi) dengan level 1-alpha.
    """
    rng = np.random.default_rng(seed)
    diffs = a - b
    n_obs = len(diffs)
    boot = np.array([
        rng.choice(diffs, size=n_obs, replace=True).mean()
        for _ in range(n)
    ])
    q_lo = 100 * (alpha / 2)
    q_hi = 100 * (1 - alpha / 2)
    return diffs.mean(), np.percentile(boot, q_lo), np.percentile(boot, q_hi)


def run_paired_bootstrap(dfs):
    """
    Jalankan semua paired CI.
    Konfirmatori: alpha disesuaikan Bonferroni (0.0125 -> CI 98.75%).
    Eksploratori: alpha=0.05 -> CI 95%.
    """
    rows = []

    for model_a, model_b, comp_label in PAIRED_COMPARISONS:
        df_a = dfs[model_a]
        df_b = dfs[model_b]
        common = df_a.index.intersection(df_b.index)

        for grp in GROUPS:
            if grp == "Overall":
                idx = common
            else:
                # Filter berdasarkan activity_group di df_a (sama di semua model)
                mask = df_a.loc[common, "activity_group"] == grp
                idx = common[mask]

            if len(idx) == 0:
                continue

            sub_a = df_a.loc[idx]
            sub_b = df_b.loc[idx]

            for metric in ["Recall@10", "NDCG@10"]:
                if metric not in sub_a.columns:
                    continue

                a_vals = sub_a[metric].fillna(0).to_numpy(dtype=np.float64)
                b_vals = sub_b[metric].fillna(0).to_numpy(dtype=np.float64)

                # Tentukan apakah konfirmatori
                is_conf = (model_a, model_b, grp, metric) in CONFIRMATORY
                alpha_used = ALPHA_CONFIRMATORY if is_conf else 0.05
                ci_level = f"{100 * (1 - alpha_used):.2f}%"
                analysis_type = "confirmatory (Bonferroni)" if is_conf else "exploratory"

                mean_diff, ci_lo, ci_hi = _bootstrap_paired(
                    a_vals, b_vals, N_BOOTSTRAP, RANDOM_SEED, alpha=alpha_used
                )

                # Signifikansi: 0 di luar CI
                if ci_lo > 0:
                    sig = "Ya (+)"
                elif ci_hi < 0:
                    sig = "Ya (-)"
                else:
                    sig = "Tidak"

                rows.append({
                    "comparison": comp_label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "group": grp,
                    "metric": metric,
                    "mean_a": round(float(sub_a[metric].mean()), 6),
                    "mean_b": round(float(sub_b[metric].mean()), 6),
                    "diff": round(float(mean_diff), 6),
                    "ci_lo": round(float(ci_lo), 6),
                    "ci_hi": round(float(ci_hi), 6),
                    "ci_level": ci_level,
                    "analysis_type": analysis_type,
                    "significant": sig,
                    "n_users": len(idx),
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": RANDOM_SEED,
                })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 4 - Bootstrap CI Independen High - Low
# ===========================================================================

def _bootstrap_independent(a: np.ndarray, b: np.ndarray, n: int, seed: int):
    """
    Independent bootstrap: mean(a) - mean(b), dua sampel berbeda.
    Returns (observed_diff, ci_lo_95, ci_hi_95).
    Interpretasi: JANGAN simpulkan arah bila CI memuat 0.
    """
    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    boot = np.array([
        rng.choice(a, size=n_a, replace=True).mean() -
        rng.choice(b, size=n_b, replace=True).mean()
        for _ in range(n)
    ])
    return a.mean() - b.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def run_independent_high_low(dfs):
    """
    CI independen untuk selisih NDCG@10(High) - NDCG@10(Low) per model.
    Eksploratori (tidak Bonferroni-adjusted).
    """
    rows = []
    for m in MODELS:
        df = dfs[m].reset_index()
        high_vals = df.loc[df["activity_group"] == "High", "NDCG@10"].fillna(0).to_numpy(dtype=np.float64)
        low_vals  = df.loc[df["activity_group"] == "Low",  "NDCG@10"].fillna(0).to_numpy(dtype=np.float64)

        obs_diff, ci_lo, ci_hi = _bootstrap_independent(
            high_vals, low_vals, N_BOOTSTRAP, RANDOM_SEED
        )

        # Hanya klaim arah bila CI tidak memuat 0
        if ci_lo > 0:
            interpretation = "High > Low (CI tidak memuat 0)"
        elif ci_hi < 0:
            interpretation = "High < Low (CI tidak memuat 0)"
        else:
            interpretation = "Tidak dapat disimpulkan (CI memuat 0)"

        rows.append({
            "model": m,
            "label": MODEL_LABELS[m],
            "metric": "NDCG@10",
            "mean_high": round(float(high_vals.mean()), 6),
            "mean_low":  round(float(low_vals.mean()),  6),
            "diff_high_minus_low": round(float(obs_diff), 6),
            "ci_lo_95": round(float(ci_lo), 6),
            "ci_hi_95": round(float(ci_hi), 6),
            "n_high": len(high_vals),
            "n_low":  len(low_vals),
            "interpretation": interpretation,
            "note": "Bootstrap independen (user High dan Low berbeda). "
                    "Gunakan hati-hati: confounding dari item pool berbeda.",
            "n_bootstrap": N_BOOTSTRAP,
            "seed": RANDOM_SEED,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 5 - Popularitas rata-rata item Top-10 per model per kelompok
# ===========================================================================

def compute_avg_popularity(dfs):
    """
    Hitung rata-rata popularitas (positive_interactions dari popularity_ranking.csv)
    item yang direkomendasikan untuk setiap model dan kelompok aktivitas.

    Popularity model tidak punya recommendations_top10.csv; skor dihitung dari
    popularity_ranking.csv langsung (item 1-10 per user dikurangi seen items).
    """
    pop_rank = pd.read_csv(
        RESULT_DIR / "popularity_ranking.csv", dtype={"item_id": str}
    )
    pop_map = pop_rank.set_index("item_id")["positive_interactions"].to_dict()

    # Training seen - untuk popularity model
    train = pd.read_csv(
        BASE_DIR / "data" / "raw" / "interactions_train.csv",
        dtype={"user_id": str, "item_id": str},
    )
    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    top_items_pop = pop_rank["item_id"].tolist()

    rows = []

    for m in MODELS:
        df_metrics = dfs[m].reset_index()[["user_id", "activity_group"]]

        if m == "popularity":
            # Rekonstruksi top-10 popularity per user dari popularity_ranking.csv
            rec_rows = []
            for _, row in df_metrics.iterrows():
                uid = row["user_id"]
                seen = seen_by_user.get(uid, set())
                top10 = [item for item in top_items_pop if item not in seen][:10]
                for item in top10:
                    rec_rows.append({"user_id": uid, "item_id": item})
            recs = pd.DataFrame(rec_rows)
        else:
            recs_path = RESULT_DIR / f"{m}_recommendations_top10.csv"
            recs = pd.read_csv(recs_path, dtype={"user_id": str, "item_id": str})

        # Gabungkan dengan activity_group dan popularitas
        recs = recs.merge(df_metrics, on="user_id", how="left")
        recs["popularity_score"] = recs["item_id"].map(pop_map).fillna(0)

        # Overall
        rows.append({
            "model": m,
            "label": MODEL_LABELS[m],
            "group": "Overall",
            "avg_popularity": round(recs["popularity_score"].mean(), 4),
            "n_recs": len(recs),
        })

        # Per kelompok
        for grp in ["Low", "Medium", "High"]:
            sub = recs[recs["activity_group"] == grp]
            rows.append({
                "model": m,
                "label": MODEL_LABELS[m],
                "group": grp,
                "avg_popularity": round(sub["popularity_score"].mean(), 4) if len(sub) > 0 else np.nan,
                "n_recs": len(sub),
            })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 6 - Overlap Top-10 Hybrid dengan MF dan TF-IDF
# ===========================================================================

def compute_overlap(dfs):
    """
    Hitung rata-rata overlap item di Top-10 Hybrid dengan MF dan TF-IDF,
    per user, per kelompok aktivitas.
    Dilaporkan sebagai hipotesis eksploratori, bukan klaim kausal.
    """
    hybrid_recs = pd.read_csv(
        RESULT_DIR / "hybrid_recommendations_top10.csv",
        dtype={"user_id": str, "item_id": str},
    )
    mf_recs = pd.read_csv(
        RESULT_DIR / "mf_recommendations_top10.csv",
        dtype={"user_id": str, "item_id": str},
    )
    tfidf_recs = pd.read_csv(
        RESULT_DIR / "tfidf_cbf_recommendations_top10.csv",
        dtype={"user_id": str, "item_id": str},
    )

    # Buat set per user
    def to_sets(df):
        return df.groupby("user_id")["item_id"].apply(set).to_dict()

    hybrid_sets = to_sets(hybrid_recs)
    mf_sets     = to_sets(mf_recs)
    tfidf_sets  = to_sets(tfidf_recs)

    df_activity = dfs["hybrid"].reset_index()[["user_id", "activity_group"]]

    rows = []
    for grp in ["Overall", "Low", "Medium", "High"]:
        if grp == "Overall":
            users = df_activity["user_id"].tolist()
        else:
            users = df_activity.loc[df_activity["activity_group"] == grp, "user_id"].tolist()

        overlap_mf    = []
        overlap_tfidf = []
        for uid in users:
            h = hybrid_sets.get(uid, set())
            m = mf_sets.get(uid, set())
            t = tfidf_sets.get(uid, set())
            if len(h) > 0:
                overlap_mf.append(len(h & m) / len(h))
                overlap_tfidf.append(len(h & t) / len(h))

        rows.append({
            "group": grp,
            "n_users": len(users),
            "avg_overlap_hybrid_mf": round(np.mean(overlap_mf), 4) if overlap_mf else np.nan,
            "avg_overlap_hybrid_tfidf": round(np.mean(overlap_tfidf), 4) if overlap_tfidf else np.nan,
            "note": "Eksploratori. Overlap tinggi = rekomendasi Hybrid didominasi komponen itu. "
                    "Ini hipotesis, bukan bukti kausalitas.",
        })

    return pd.DataFrame(rows)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print("Track A - Analisis Lengkap")
    print("=" * 65)

    # 1. Muat data
    print("\n[1/6] Memuat user_metrics semua model...")
    dfs = load_user_metrics()
    print(f"  Loaded: {list(dfs.keys())}")

    # 2. Tabel lengkap
    print("\n[2/6] Membangun tabel metrik lengkap (4 model x 4 kelompok)...")
    table = build_track_a_table(dfs)
    table.to_csv(RESULT_DIR / "track_a_table.csv", index=False)
    print(table.to_string(index=False))

    # Cetak distribusi pengguna per kelompok
    print("\n--- Distribusi kelompok aktivitas ---")
    n_total = table.loc[table["group"] == "Overall", "n_users"].iloc[0]
    for grp, defn in GROUP_DEFINITIONS.items():
        n = table.loc[(table["model"] == "popularity") & (table["group"] == grp), "n_users"].iloc[0]
        pct = 100 * n / n_total
        print(f"  {grp:7s}: {n:5d} users ({pct:.1f}%)  [{defn}]")
    print(f"  {'Total':7s}: {n_total:5d} users")

    # 3. Bootstrap CI Paired
    print(f"\n[3/6] Bootstrap CI Paired (n={N_BOOTSTRAP}, seed={RANDOM_SEED})...")
    print(f"  Konfirmatori (Bonferroni k={BONFERRONI_K}): alpha={ALPHA_CONFIRMATORY:.4f} -> CI {100*(1-ALPHA_CONFIRMATORY):.2f}%")
    print(f"  Eksploratori: alpha=0.05 -> CI 95%")
    ci_df = run_paired_bootstrap(dfs)
    ci_df.to_csv(RESULT_DIR / "track_a_bootstrap_ci.csv", index=False)

    print("\n  === CI Paired lengkap ===")
    display_cols = ["comparison", "group", "metric", "mean_a", "mean_b",
                    "diff", "ci_lo", "ci_hi", "ci_level", "analysis_type", "significant"]
    print(ci_df[display_cols].to_string(index=False))

    # 4. Bootstrap CI Independen High-Low
    print(f"\n[4/6] Bootstrap CI Independen High-Low (n={N_BOOTSTRAP})...")
    indep_df = run_independent_high_low(dfs)
    indep_df.to_csv(RESULT_DIR / "track_a_indep_ci.csv", index=False)
    print(indep_df.to_string(index=False))

    # 5. Popularitas rata-rata item Top-10
    print("\n[5/6] Menghitung rata-rata popularitas item Top-10...")
    pop_df = compute_avg_popularity(dfs)
    pop_df.to_csv(RESULT_DIR / "track_a_popularity.csv", index=False)
    print(pop_df.to_string(index=False))

    # 6. Overlap Hybrid
    print("\n[6/6] Menghitung overlap Top-10 Hybrid dengan MF dan TF-IDF...")
    overlap_df = compute_overlap(dfs)
    overlap_df.to_csv(RESULT_DIR / "track_a_overlap.csv", index=False)
    print(overlap_df.to_string(index=False))

    print("\n" + "=" * 65)
    print("Selesai. File disimpan di results/track_a_*.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
