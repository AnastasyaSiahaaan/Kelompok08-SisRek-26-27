"""
Track A - Analisis Lengkap v3 (Bagian 12 Reproducibility Package)

Menghasilkan:
  results/track_a_table.csv          - 4 model x 4 kelompok x 4 metrik;
                                       kolom *_best diisi dengan argmax otomatis
  results/track_a_bootstrap_ci.csv   - paired CI; Bonferroni k=7 untuk konfirmatori
  results/track_a_indep_ci.csv       - independent CI selisih High-Low per model
  results/track_a_diff_of_diff.csv   - uji selisih-dari-selisih (High-Low) antar model
  results/track_a_popularity.csv     - avg popularitas item Top-10 yang direkomendasikan
  results/track_a_relevant_pop.csv   - popularitas item RELEVAN + cakupan Top-100
                                       + proporsi Top-100 global yang sudah seen
  results/track_a_overlap.csv        - overlap Top-10 Hybrid vs MF/TF-IDF
  results/track_a_sensitivity.csv    - sensitivitas batas kelompok (tertil & kuintil)
  results/track_a_ndcg_bar.png       - bar chart NDCG@10 dengan error bar CI 95%

Daftar konfirmatori (k=7, Bonferroni, CI 99.29%):
  C1-C6: Hybrid vs Popularity dan Hybrid vs MF, NDCG@10, Low/Medium/High
  C7:    Selisih-dari-selisih Hybrid-Popularity NDCG@10, High minus Low
  Dicatat di docs/FREEZE_CONFIG.md sebelum FinalTest.

Catatan tentang argmax:
  Kolom *_best di track_a_table.csv dihitung dengan idxmax() per kelompok.
  Fungsi print hanya membaca kolom tersebut -- tidak ada logika best di fungsi print.

Jalankan dari root proyek:
    python src/track_a_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from baseline_common import RESULT_DIR, make_user_groups

RANDOM_SEED  = 42
N_BOOTSTRAP  = 10_000

MODELS = ["popularity", "mf", "tfidf_cbf", "hybrid"]
MODEL_LABELS = {
    "popularity": "Popularity",
    "mf":         "MF (n=32)",
    "tfidf_cbf":  "TF-IDF CBF",
    "hybrid":     "Hybrid (a=0.5)",
}
GROUPS  = ["Overall", "Low", "Medium", "High"]
METRICS = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]

GROUP_DEFINITIONS = {
    "Low":    "train_interactions in [1, 5]",
    "Medium": "train_interactions in [6, 8]",
    "High":   "train_interactions >= 9",
}

# ---------------------------------------------------------------------------
# Konfirmatori: k=7
#   C1-C6: Hybrid vs {Popularity, MF}, NDCG@10, {Low, Medium, High}
#   C7:    DoD Hybrid-Popularity NDCG@10 (High - Low)
# alpha per sel = 0.05 / 7 = 0.007143  ->  CI 99.29%
# ---------------------------------------------------------------------------
BONFERRONI_K       = 7
ALPHA_CONFIRMATORY = 0.05 / BONFERRONI_K   # ~0.007143
CI_CONF_LEVEL      = 100 * (1 - ALPHA_CONFIRMATORY)  # ~99.29%

CONFIRMATORY_PAIRED = [
    ("hybrid", "popularity", "Low",    "NDCG@10"),
    ("hybrid", "popularity", "Medium", "NDCG@10"),
    ("hybrid", "popularity", "High",   "NDCG@10"),
    ("hybrid", "mf",         "Low",    "NDCG@10"),
    ("hybrid", "mf",         "Medium", "NDCG@10"),
    ("hybrid", "mf",         "High",   "NDCG@10"),
]
# C7 (DoD) diterapkan di run_diff_of_diff

PAIRED_COMPARISONS = [
    ("mf",        "popularity", "MF vs Popularity"),
    ("tfidf_cbf", "popularity", "TF-IDF vs Popularity"),
    ("hybrid",    "popularity", "Hybrid vs Popularity"),
    ("hybrid",    "mf",         "Hybrid vs MF"),
    ("hybrid",    "tfidf_cbf",  "Hybrid vs TF-IDF"),
]

DIFF_OF_DIFF_PAIRS = [
    ("hybrid", "popularity", "Hybrid-Popularity"),
    ("hybrid", "mf",         "Hybrid-MF"),
]


# ===========================================================================
# BAGIAN 0 - Muat user_metrics
# ===========================================================================

def load_user_metrics() -> dict:
    dfs = {}
    for m in MODELS:
        path = RESULT_DIR / f"{m}_user_metrics.csv"
        df   = pd.read_csv(path, dtype={"user_id": str})
        for col in ["HitRate@10", "MRR@10"]:
            if col not in df.columns:
                df[col] = np.nan
        dfs[m] = df.set_index("user_id")
    return dfs


# ===========================================================================
# BAGIAN 1 - Tabel metrik + penanda terbaik
# ===========================================================================

def build_track_a_table(dfs: dict) -> pd.DataFrame:
    """
    Hitung mean setiap metrik per model per kelompok.
    Kolom <metric>_best diisi idxmax() per kelompok -- bukan manual.
    Fungsi cetak hanya membaca kolom ini; tidak ada logika ulang di printer.
    """
    rows = []
    for m in MODELS:
        df = dfs[m].reset_index()
        # Overall
        row = {"model": m, "label": MODEL_LABELS[m], "group": "Overall",
               "n_users": len(df)}
        for metric in METRICS:
            row[metric] = df[metric].mean() if metric in df.columns else np.nan
        rows.append(row)
        # Per grup
        for grp in ["Low", "Medium", "High"]:
            sub = df[df["activity_group"] == grp]
            r = {"model": m, "label": MODEL_LABELS[m], "group": grp,
                 "n_users": len(sub)}
            for metric in METRICS:
                r[metric] = (sub[metric].mean()
                             if (metric in sub.columns and len(sub) > 0)
                             else np.nan)
            rows.append(r)

    table = pd.DataFrame(rows)

    # Persentase pengguna
    n_total = table.loc[table["group"] == "Overall", "n_users"].iloc[0]
    table["pct_users"] = table.apply(
        lambda r: f"{100 * r['n_users'] / n_total:.1f}%"
        if r["group"] != "Overall" else "100.0%", axis=1)

    # Penanda terbaik -- idxmax() per kelompok, satu kali, disimpan ke kolom boolean
    for metric in METRICS:
        col_best = f"{metric}_best"
        table[col_best] = False
        for grp in GROUPS:
            mask = table["group"] == grp
            vals = table.loc[mask, metric]
            if vals.notna().any():
                # idxmax() mengembalikan indeks DataFrame, bukan posisi lokal
                best_global_idx = vals.idxmax()
                table.loc[best_global_idx, col_best] = True

    return table


def print_table_full(table: pd.DataFrame) -> None:
    """
    Cetak 4 model x 4 kelompok x 4 metrik secara lengkap.
    Tanda * diambil langsung dari kolom <metric>_best -- tidak ada logika ulang.
    """
    hdr = (f"{'Model':<16} {'Kelompok':<8} {'n':>6} {'%':>6}  "
           f"{'Recall@10':>11} {'NDCG@10':>11} {'HitRate@10':>11} {'MRR@10':>11}")
    print(hdr)
    print("-" * len(hdr))
    prev_model = None
    for _, row in table.iterrows():
        if prev_model is not None and row["model"] != prev_model:
            print()   # blank line between models
        prev_model = row["model"]

        def fmt(metric: str) -> str:
            val  = row[metric]
            best = bool(row.get(f"{metric}_best", False))
            if pd.isna(val):
                return f"{'N/A':>11}"
            mark = "*" if best else " "
            return f"{val:10.6f}{mark}"

        print(f"{row['label']:<16} {row['group']:<8} {row['n_users']:>6} "
              f"{row['pct_users']:>6}  "
              f"{fmt('Recall@10')} {fmt('NDCG@10')} {fmt('HitRate@10')} {fmt('MRR@10')}")


# ===========================================================================
# BAGIAN 2 - Bootstrap CI Paired
# ===========================================================================

def _bootstrap_paired(a: np.ndarray, b: np.ndarray,
                      n: int, seed: int, alpha: float = 0.05):
    rng  = np.random.default_rng(seed)
    diff = a - b
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(n)])
    return (diff.mean(),
            np.percentile(boot, 100 * alpha / 2),
            np.percentile(boot, 100 * (1 - alpha / 2)))


def run_paired_bootstrap(dfs: dict) -> pd.DataFrame:
    rows = []
    for model_a, model_b, comp_label in PAIRED_COMPARISONS:
        df_a   = dfs[model_a]
        df_b   = dfs[model_b]
        common = df_a.index.intersection(df_b.index)

        for grp in GROUPS:
            idx = (common if grp == "Overall"
                   else common[df_a.loc[common, "activity_group"] == grp])
            if len(idx) == 0:
                continue
            sub_a, sub_b = df_a.loc[idx], df_b.loc[idx]

            for metric in ["Recall@10", "NDCG@10"]:
                if metric not in sub_a.columns:
                    continue
                a_v = sub_a[metric].fillna(0).to_numpy(dtype=np.float64)
                b_v = sub_b[metric].fillna(0).to_numpy(dtype=np.float64)

                is_conf   = (model_a, model_b, grp, metric) in CONFIRMATORY_PAIRED
                alpha_use = ALPHA_CONFIRMATORY if is_conf else 0.05
                ci_pct    = f"{100*(1-alpha_use):.2f}%"
                atype     = (f"confirmatory (Bonferroni k={BONFERRONI_K})"
                             if is_conf else "exploratory")

                md, ci_lo, ci_hi = _bootstrap_paired(a_v, b_v, N_BOOTSTRAP,
                                                      RANDOM_SEED, alpha_use)
                sig = ("Ya (+)" if ci_lo > 0 else "Ya (-)" if ci_hi < 0 else "Tidak")

                rows.append({
                    "comparison":    comp_label,
                    "model_a":       model_a,
                    "model_b":       model_b,
                    "group":         grp,
                    "metric":        metric,
                    "mean_a":        round(float(sub_a[metric].mean()), 6),
                    "mean_b":        round(float(sub_b[metric].mean()), 6),
                    "diff":          round(float(md), 6),
                    "ci_lo":         round(float(ci_lo), 6),
                    "ci_hi":         round(float(ci_hi), 6),
                    "ci_level":      ci_pct,
                    "analysis_type": atype,
                    "significant":   sig,
                    "n_users":       len(idx),
                    "n_bootstrap":   N_BOOTSTRAP,
                    "seed":          RANDOM_SEED,
                })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 3 - CI Independen High-Low
# ===========================================================================

def _bootstrap_independent(a: np.ndarray, b: np.ndarray, n: int, seed: int,
                            alpha: float = 0.05):
    rng  = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(a, size=len(a), replace=True).mean() -
        rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n)
    ])
    return (a.mean() - b.mean(),
            np.percentile(boot, 100 * alpha / 2),
            np.percentile(boot, 100 * (1 - alpha / 2)))


def run_independent_high_low(dfs: dict) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        df = dfs[m].reset_index()
        hi = df.loc[df["activity_group"] == "High", "NDCG@10"].fillna(0).to_numpy(np.float64)
        lo = df.loc[df["activity_group"] == "Low",  "NDCG@10"].fillna(0).to_numpy(np.float64)
        obs, ci_lo, ci_hi = _bootstrap_independent(hi, lo, N_BOOTSTRAP, RANDOM_SEED)

        if   ci_lo > 0: interp = "High > Low (CI tidak memuat 0)"
        elif ci_hi < 0: interp = "High < Low (CI tidak memuat 0)"
        else:           interp = "Tidak dapat disimpulkan (CI memuat 0)"

        rows.append({
            "model":               m, "label": MODEL_LABELS[m],
            "metric":              "NDCG@10",
            "mean_high":           round(float(hi.mean()), 6),
            "mean_low":            round(float(lo.mean()), 6),
            "diff_high_minus_low": round(float(obs), 6),
            "ci_lo_95":            round(float(ci_lo), 6),
            "ci_hi_95":            round(float(ci_hi), 6),
            "n_high":              len(hi), "n_low": len(lo),
            "interpretation":      interp,
            "analysis_type":       "exploratory",
            "note": ("Bootstrap independen (user High dan Low berbeda). "
                     "Hati-hati: confounding dari item pool berbeda antar kelompok."),
            "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 4 - Selisih-dari-selisih
# ===========================================================================

def run_diff_of_diff(dfs: dict) -> pd.DataFrame:
    """
    Stat: mean(d_High) - mean(d_Low), d = metric_a - metric_b per user.
    Bootstrap: paired dalam kelompok, independen antar kelompok.
    C7 (konfirmatori): Hybrid-Popularity NDCG@10, Bonferroni k=7.
    """
    rows = []
    for model_a, model_b, label in DIFF_OF_DIFF_PAIRS:
        df_a   = dfs[model_a]
        df_b   = dfs[model_b]
        common = df_a.index.intersection(df_b.index)

        for metric in ["NDCG@10", "Recall@10"]:
            hi_idx = common[df_a.loc[common, "activity_group"] == "High"]
            lo_idx = common[df_a.loc[common, "activity_group"] == "Low"]

            d_hi = (df_a.loc[hi_idx, metric].fillna(0).to_numpy(np.float64) -
                    df_b.loc[hi_idx, metric].fillna(0).to_numpy(np.float64))
            d_lo = (df_a.loc[lo_idx, metric].fillna(0).to_numpy(np.float64) -
                    df_b.loc[lo_idx, metric].fillna(0).to_numpy(np.float64))

            obs_stat = d_hi.mean() - d_lo.mean()

            # C7: Hybrid-Popularity NDCG@10 adalah konfirmatori
            is_c7 = (label == "Hybrid-Popularity" and metric == "NDCG@10")
            alpha_use = ALPHA_CONFIRMATORY if is_c7 else 0.05
            ci_pct    = f"{100*(1-alpha_use):.2f}%"
            atype     = (f"confirmatory (Bonferroni k={BONFERRONI_K}, C7)"
                         if is_c7 else "exploratory")

            rng  = np.random.default_rng(RANDOM_SEED)
            boot = np.array([
                rng.choice(d_hi, size=len(d_hi), replace=True).mean() -
                rng.choice(d_lo, size=len(d_lo), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo = np.percentile(boot, 100 * alpha_use / 2)
            ci_hi = np.percentile(boot, 100 * (1 - alpha_use / 2))

            if   ci_lo > 0: interp = "Interaksi ada: keuntungan model_a lebih besar di High"
            elif ci_hi < 0: interp = "Interaksi ada: keuntungan model_a lebih besar di Low"
            else:           interp = "Tidak dapat disimpulkan (CI memuat 0)"

            rows.append({
                "comparison":     label,
                "model_a":        model_a,
                "model_b":        model_b,
                "metric":         metric,
                "mean_diff_high": round(float(d_hi.mean()), 6),
                "mean_diff_low":  round(float(d_lo.mean()), 6),
                "diff_of_diff":   round(float(obs_stat), 6),
                "ci_lo":          round(float(ci_lo), 6),
                "ci_hi":          round(float(ci_hi), 6),
                "ci_level":       ci_pct,
                "n_high":         len(d_hi), "n_low": len(d_lo),
                "interpretation": interp,
                "analysis_type":  atype,
                "method": ("Paired dalam kelompok, independen antar kelompok."),
                "n_bootstrap":    N_BOOTSTRAP, "seed": RANDOM_SEED,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 5 - Popularitas item Top-10 yang direkomendasikan
# ===========================================================================

def compute_avg_popularity(dfs: dict) -> pd.DataFrame:
    pop_rank     = pd.read_csv(RESULT_DIR / "popularity_ranking.csv",
                               dtype={"item_id": str})
    pop_map      = pop_rank.set_index("item_id")["positive_interactions"].to_dict()
    train        = pd.read_csv(BASE_DIR / "data/raw/interactions_train.csv",
                               dtype={"user_id": str, "item_id": str})
    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    top_items    = pop_rank["item_id"].tolist()

    rows = []
    for m in MODELS:
        df_m = dfs[m].reset_index()[["user_id", "activity_group"]]
        if m == "popularity":
            rec_rows = []
            for _, row in df_m.iterrows():
                uid  = row["user_id"]
                seen = seen_by_user.get(uid, set())
                for item in [i for i in top_items if i not in seen][:10]:
                    rec_rows.append({"user_id": uid, "item_id": item})
            recs = pd.DataFrame(rec_rows)
        else:
            recs = pd.read_csv(RESULT_DIR / f"{m}_recommendations_top10.csv",
                               dtype={"user_id": str, "item_id": str})

        recs        = recs.merge(df_m, on="user_id", how="left")
        recs["pop"] = recs["item_id"].map(pop_map).fillna(0)

        for grp in ["Overall"] + ["Low", "Medium", "High"]:
            sub = recs if grp == "Overall" else recs[recs["activity_group"] == grp]
            rows.append({
                "model":          m,
                "label":          MODEL_LABELS[m],
                "group":          grp,
                "avg_popularity": round(sub["pop"].mean(), 4) if len(sub) > 0 else np.nan,
                "n_recs":         len(sub),
            })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 6 - Popularitas item RELEVAN di validation
# ===========================================================================

def compute_relevant_popularity(dfs: dict) -> pd.DataFrame:
    """
    Per kelompok aktivitas:
      - avg dan median popularitas training item relevan (rating val>=4, dalam kandidat,
        belum di training)
      - CI 95% bootstrap untuk rata-rata
      - % item relevan yang masuk Top-100 item paling populer training
      - % item Top-100 yang sudah ter-mask (ada di training user kelompok tsb)
    """
    pop_rank = pd.read_csv(RESULT_DIR / "popularity_ranking.csv", dtype={"item_id": str})
    pop_map  = pop_rank.set_index("item_id")["positive_interactions"].to_dict()
    top100   = set(pop_rank.head(100)["item_id"].tolist())

    train = pd.read_csv(BASE_DIR / "data/raw/interactions_train.csv",
                        dtype={"user_id": str, "item_id": str})
    val   = pd.read_csv(BASE_DIR / "data/supporting/interactions_validation_warm.csv",
                        dtype={"user_id": str, "item_id": str})
    users = pd.read_csv(BASE_DIR / "data/supporting/evaluable_users_validation_warm.csv",
                        dtype={"user_id": str})
    cands = pd.read_csv(BASE_DIR / "data/raw/candidate_items_warm.csv",
                        dtype={"item_id": str})
    act   = pd.read_csv(BASE_DIR / "data/supporting/user_activity_train.csv",
                        dtype={"user_id": str})

    candidate_items = set(cands["item_id"])
    evaluated_users = set(users["user_id"])
    seen            = train.groupby("user_id")["item_id"].apply(set).to_dict()

    act_grouped = make_user_groups(act)
    group_map   = act_grouped.set_index("user_id")["activity_group"].to_dict()

    # Item relevan per user (unik, sudah filter seen + kandidat)
    rel_rows = val[val["user_id"].isin(evaluated_users) & (val["rating"] >= 4)].copy()
    rel_rows["item_id"] = rel_rows["item_id"].astype(str)

    records = []
    for uid, grp_df in rel_rows.groupby("user_id"):
        u_seen = seen.get(uid, set())
        u_grp  = group_map.get(uid, "Unknown")
        for item in grp_df["item_id"]:
            if item in candidate_items and item not in u_seen:
                records.append({
                    "user_id":        uid,
                    "item_id":        item,
                    "activity_group": u_grp,
                    "pop":            pop_map.get(item, 0),
                    "in_top100":      item in top100,
                })
    df_rel = pd.DataFrame(records)

    # Efek seen-item masking pada Popularity: berapa user tiap grup punya >= 1 item Top-100
    # yang sudah ter-mask di training mereka
    rng = np.random.default_rng(RANDOM_SEED)

    def _ci_mean_bootstrap(vals: np.ndarray) -> tuple:
        if len(vals) == 0:
            return np.nan, np.nan
        boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                         for _ in range(N_BOOTSTRAP)])
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    # Hitung juga seen-masking per grup: proporsi Top-100 item yang di-mask tiap user
    # (item Top-100 yang ada di training user, dari populasi user per grup)
    seen_mask_records = []
    for uid, grp_df in train.groupby("user_id"):
        u_grp = group_map.get(str(uid), "Unknown")
        if u_grp == "Unknown":
            continue
        u_top100_seen = len(set(grp_df["item_id"].astype(str)) & top100)
        seen_mask_records.append({
            "user_id":        str(uid),
            "activity_group": u_grp,
            "n_top100_seen":  u_top100_seen,
        })
    df_seen_mask = pd.DataFrame(seen_mask_records)

    rows = []
    for grp in ["Overall", "Low", "Medium", "High"]:
        sub = df_rel if grp == "Overall" else df_rel[df_rel["activity_group"] == grp]
        if len(sub) == 0:
            rows.append({"group": grp}); continue

        pop_vals = sub["pop"].to_numpy(np.float64)
        ci_lo, ci_hi = _ci_mean_bootstrap(pop_vals)

        # Seen-masking: avg jumlah item Top-100 yang sudah di-mask per user kelompok
        sub_mask = df_seen_mask if grp == "Overall" else df_seen_mask[df_seen_mask["activity_group"] == grp]
        avg_top100_masked = sub_mask["n_top100_seen"].mean() if len(sub_mask) > 0 else np.nan

        rows.append({
            "group":              grp,
            "n_relevant_items":   len(sub),
            "n_unique_users":     sub["user_id"].nunique(),
            "avg_relevant_pop":   round(float(pop_vals.mean()), 4),
            "median_relevant_pop": round(float(np.median(pop_vals)), 4),
            "ci_lo_avg_pop_95":   round(float(ci_lo), 4) if not np.isnan(ci_lo) else np.nan,
            "ci_hi_avg_pop_95":   round(float(ci_hi), 4) if not np.isnan(ci_hi) else np.nan,
            "pct_in_top100":      round(100 * sub["in_top100"].mean(), 2),
            "avg_top100_masked_per_user": round(float(avg_top100_masked), 4),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 7 - Overlap Hybrid vs MF / TF-IDF
# ===========================================================================

def compute_overlap(dfs: dict) -> pd.DataFrame:
    def to_sets(path):
        df = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
        return df.groupby("user_id")["item_id"].apply(set).to_dict()

    h_sets = to_sets(RESULT_DIR / "hybrid_recommendations_top10.csv")
    m_sets = to_sets(RESULT_DIR / "mf_recommendations_top10.csv")
    t_sets = to_sets(RESULT_DIR / "tfidf_cbf_recommendations_top10.csv")

    df_act = dfs["hybrid"].reset_index()[["user_id", "activity_group"]]
    rows = []
    for grp in GROUPS:
        users = (df_act["user_id"].tolist() if grp == "Overall"
                 else df_act.loc[df_act["activity_group"] == grp, "user_id"].tolist())
        ov_mf, ov_tf = [], []
        for uid in users:
            h = h_sets.get(uid, set())
            if len(h) == 0:
                continue
            ov_mf.append(len(h & m_sets.get(uid, set())) / len(h))
            ov_tf.append(len(h & t_sets.get(uid, set())) / len(h))
        rows.append({
            "group":                    grp,
            "n_users":                  len(users),
            "avg_overlap_hybrid_mf":    round(np.mean(ov_mf),  4) if ov_mf else np.nan,
            "avg_overlap_hybrid_tfidf": round(np.mean(ov_tf), 4)  if ov_tf else np.nan,
            "note": ("Eksploratori. Hipotesis: overlap tinggi dengan TF-IDF menunjukkan "
                     "skor CBF lebih sering menentukan ranking akhir pada alpha=0.5."),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 8 - Sensitivitas batas kelompok (tertil & kuintil)
# ===========================================================================

def run_sensitivity_analysis(dfs: dict) -> pd.DataFrame:
    """
    Potong ulang jumlah interaksi training menjadi tertil dan kuintil.
    Batas dihitung dari distribusi user evaluable (bukan dari data test).
    Laporkan NDCG@10 tiap model dan DoD Hybrid-Popularity (top - bottom).
    """
    # Ambil activity counts dari salah satu df (sama semua model)
    act_path = BASE_DIR / "data/supporting/user_activity_train.csv"
    act = pd.read_csv(act_path, dtype={"user_id": str})

    # Filter ke evaluable users saja
    evaluable = set(dfs["popularity"].index)
    act = act[act["user_id"].isin(evaluable)].copy()
    counts = act.set_index("user_id")["train_interactions"].to_dict()

    rows = []
    for scheme_name, n_bins in [("tertile", 3), ("quintile", 5)]:
        # Hitung batas kuantil dari distribusi user evaluable
        vals   = np.array(list(counts.values()), dtype=float)
        quants = np.linspace(0, 100, n_bins + 1)
        edges  = np.unique(np.percentile(vals, quants))

        if len(edges) < 3:
            continue

        # Assign label (Q1 = terbawah, Q_n = teratas)
        labels = [f"Q{i+1}" for i in range(len(edges) - 1)]
        bin_edges = list(edges)
        bin_edges[0]  -= 0.001   # include_lowest workaround

        def assign_group(v):
            for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                if lo < v <= hi:
                    return labels[i]
            return labels[-1]

        user_grp = {uid: assign_group(c) for uid, c in counts.items()}

        for m in MODELS:
            df = dfs[m].reset_index()
            df["sens_group"] = df["user_id"].map(user_grp)

            for lbl in labels:
                sub = df[df["sens_group"] == lbl]
                rows.append({
                    "scheme":       scheme_name,
                    "group_label":  lbl,
                    "model":        m,
                    "label":        MODEL_LABELS[m],
                    "n_users":      len(sub),
                    "mean_ndcg":    round(sub["NDCG@10"].mean(), 6) if len(sub) > 0 else np.nan,
                    "edge_lo":      round(edges[labels.index(lbl)], 1),
                    "edge_hi":      round(edges[labels.index(lbl) + 1], 1),
                })

        # DoD Hybrid-Popularity: top group - bottom group, NDCG@10
        for metric in ["NDCG@10"]:
            df_h   = dfs["hybrid"].reset_index()
            df_p   = dfs["popularity"].reset_index()
            df_h["sens_group"] = df_h["user_id"].map(user_grp)
            df_p["sens_group"] = df_p["user_id"].map(user_grp)

            top_lbl = labels[-1]
            bot_lbl = labels[0]

            common = set(dfs["hybrid"].index) & set(dfs["popularity"].index)
            hi_idx = [u for u in common if user_grp.get(u) == top_lbl]
            lo_idx = [u for u in common if user_grp.get(u) == bot_lbl]

            d_hi_v = (dfs["hybrid"].loc[hi_idx, metric].fillna(0).to_numpy(np.float64) -
                      dfs["popularity"].loc[hi_idx, metric].fillna(0).to_numpy(np.float64))
            d_lo_v = (dfs["hybrid"].loc[lo_idx, metric].fillna(0).to_numpy(np.float64) -
                      dfs["popularity"].loc[lo_idx, metric].fillna(0).to_numpy(np.float64))

            obs_stat = d_hi_v.mean() - d_lo_v.mean()
            rng  = np.random.default_rng(RANDOM_SEED)
            boot = np.array([
                rng.choice(d_hi_v, size=len(d_hi_v), replace=True).mean() -
                rng.choice(d_lo_v, size=len(d_lo_v), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo_s, ci_hi_s = np.percentile(boot, [2.5, 97.5])

            if   ci_lo_s > 0: interp = "DoD > 0 (CI tidak memuat 0)"
            elif ci_hi_s < 0: interp = "DoD < 0 (CI tidak memuat 0)"
            else:              interp = "Tidak dapat disimpulkan (CI memuat 0)"

            rows.append({
                "scheme":       f"{scheme_name}_DoD_Hybrid-Pop_{metric}",
                "group_label":  f"{top_lbl}_minus_{bot_lbl}",
                "model":        "hybrid_minus_popularity",
                "label":        "DoD: Hybrid-Popularity",
                "n_users":      f"hi={len(d_hi_v)}, lo={len(d_lo_v)}",
                "mean_ndcg":    round(float(obs_stat), 6),
                "edge_lo":      round(edges[0], 1),
                "edge_hi":      round(edges[-1], 1),
                "ci_lo_95":     round(float(ci_lo_s), 6),
                "ci_hi_95":     round(float(ci_hi_s), 6),
                "interpretation": interp,
            })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 9 - Figure: NDCG@10 bar chart dengan error bar CI 95%
# ===========================================================================

def make_ndcg_bar_figure(dfs: dict) -> None:
    """
    Bar chart NDCG@10 per kelompok dengan error bar bootstrap 95% CI.
    Error bar = CI sekitar mean per user (paired bootstrap vs mean = 0 tidak tepat;
    kita pakai bootstrap CI of the mean sebagai ukuran presisi).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib tidak tersedia; figure dilewati.")
        return

    rng = np.random.default_rng(RANDOM_SEED)

    def _ci_mean(vals: np.ndarray) -> tuple:
        boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                         for _ in range(2000)])
        return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    # Kumpulkan mean + CI per model per grup
    group_order = ["Overall", "Low", "Medium", "High"]
    model_order = ["popularity", "mf", "tfidf_cbf", "hybrid"]
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    n_grp    = len(group_order)
    n_model  = len(model_order)
    x        = np.arange(n_grp)
    width    = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, m in enumerate(model_order):
        df    = dfs[m].reset_index()
        means, errs_lo, errs_hi = [], [], []

        for grp in group_order:
            sub = df if grp == "Overall" else df[df["activity_group"] == grp]
            vals = sub["NDCG@10"].fillna(0).to_numpy(np.float64)
            mean = vals.mean()
            ci_lo, ci_hi = _ci_mean(vals)
            means.append(mean)
            errs_lo.append(mean - ci_lo)
            errs_hi.append(ci_hi - mean)

        offset = (mi - (n_model - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, label=MODEL_LABELS[m],
                      color=colors[mi], alpha=0.85)
        ax.errorbar(x + offset, means,
                    yerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=11)
    ax.set_ylabel("NDCG@10 (mean per user)", fontsize=11)
    ax.set_title("NDCG@10 per Kelompok Aktivitas (error bar = 95% CI bootstrap)",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)

    out_path = RESULT_DIR / "track_a_ndcg_bar.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure disimpan: {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("Track A - Analisis Lengkap v3")
    print(f"n_bootstrap={N_BOOTSTRAP}, seed={RANDOM_SEED}")
    print(f"Konfirmatori: Bonferroni k={BONFERRONI_K}, "
          f"alpha/sel={ALPHA_CONFIRMATORY:.6f}, CI={CI_CONF_LEVEL:.2f}%")
    print("=" * 70)

    dfs = load_user_metrics()
    print(f"Loaded: {list(dfs.keys())}")

    # 1. Tabel lengkap
    print("\n[1] Tabel metrik lengkap (* = terbaik per kolom per kelompok, argmax)")
    table = build_track_a_table(dfs)
    table.to_csv(RESULT_DIR / "track_a_table.csv", index=False)
    print_table_full(table)

    # Distribusi kelompok
    n_total = table.loc[table["group"] == "Overall", "n_users"].iloc[0]
    print()
    for grp, defn in GROUP_DEFINITIONS.items():
        n   = table.loc[(table["model"] == "popularity") & (table["group"] == grp), "n_users"].iloc[0]
        pct = 100 * n / n_total
        print(f"  {grp:7s}: {n:5d} users ({pct:.1f}%)  [{defn}]")
    print(f"  {'Total':7s}: {n_total:5d} users")

    # 2. Paired CI
    print(f"\n[2] Bootstrap CI Paired (n={N_BOOTSTRAP}, seed={RANDOM_SEED})")
    ci_df = run_paired_bootstrap(dfs)
    ci_df.to_csv(RESULT_DIR / "track_a_bootstrap_ci.csv", index=False)
    dcols = ["comparison", "group", "metric",
             "mean_a", "mean_b", "diff", "ci_lo", "ci_hi",
             "ci_level", "analysis_type", "significant"]
    print(ci_df[dcols].to_string(index=False))

    # 3. CI Independen High-Low
    print(f"\n[3] Bootstrap CI Independen High-Low (n={N_BOOTSTRAP})")
    indep_df = run_independent_high_low(dfs)
    indep_df.to_csv(RESULT_DIR / "track_a_indep_ci.csv", index=False)
    print(indep_df[["label", "metric", "mean_high", "mean_low",
                    "diff_high_minus_low", "ci_lo_95", "ci_hi_95",
                    "interpretation"]].to_string(index=False))

    # 4. Selisih-dari-selisih
    print(f"\n[4] Selisih-dari-selisih High-Low (n={N_BOOTSTRAP})")
    dod_df = run_diff_of_diff(dfs)
    dod_df.to_csv(RESULT_DIR / "track_a_diff_of_diff.csv", index=False)
    print(dod_df[["comparison", "metric", "mean_diff_high", "mean_diff_low",
                  "diff_of_diff", "ci_lo", "ci_hi",
                  "ci_level", "analysis_type", "interpretation"]].to_string(index=False))

    # 5. Popularitas item Top-10
    print("\n[5] Rata-rata popularitas item Top-10")
    pop_df = compute_avg_popularity(dfs)
    pop_df.to_csv(RESULT_DIR / "track_a_popularity.csv", index=False)
    print(pop_df.to_string(index=False))

    # 6. Popularitas item relevan
    print("\n[6] Popularitas item relevan di validation per kelompok")
    rel_pop_df = compute_relevant_popularity(dfs)
    rel_pop_df.to_csv(RESULT_DIR / "track_a_relevant_pop.csv", index=False)
    print(rel_pop_df.to_string(index=False))

    # 7. Overlap
    print("\n[7] Overlap Top-10 Hybrid vs MF / TF-IDF")
    overlap_df = compute_overlap(dfs)
    overlap_df.to_csv(RESULT_DIR / "track_a_overlap.csv", index=False)
    print(overlap_df[["group", "n_users",
                       "avg_overlap_hybrid_mf",
                       "avg_overlap_hybrid_tfidf"]].to_string(index=False))

    # 8. Sensitivitas batas kelompok
    print("\n[8] Analisis sensitivitas batas kelompok (tertil & kuintil)")
    sens_df = run_sensitivity_analysis(dfs)
    sens_df.to_csv(RESULT_DIR / "track_a_sensitivity.csv", index=False)
    # Cetak ringkasan: mean NDCG per scheme per model per group + DoD
    for scheme in ["tertile", "quintile"]:
        sub = sens_df[sens_df["scheme"] == scheme]
        if len(sub) == 0:
            continue
        print(f"\n  Scheme: {scheme}")
        pivot = sub.pivot_table(index="group_label", columns="label",
                                values="mean_ndcg", aggfunc="first")
        print(pivot.to_string())
        # DoD baris
        dod_key = f"{scheme}_DoD_Hybrid-Pop_NDCG@10"
        dod_row = sens_df[sens_df["scheme"] == dod_key]
        if len(dod_row) > 0:
            r = dod_row.iloc[0]
            interp = r.get("interpretation", "")
            ci_lo  = r.get("ci_lo_95", np.nan)
            ci_hi  = r.get("ci_hi_95", np.nan)
            print(f"  DoD {scheme}: {r['mean_ndcg']:+.6f} "
                  f"[{ci_lo:.6f}, {ci_hi:.6f}] -- {interp}")

    # 9. Figure
    print("\n[9] Membuat figure NDCG bar chart dengan error bar CI...")
    make_ndcg_bar_figure(dfs)

    print("\n" + "=" * 70)
    print("Selesai. File disimpan di results/track_a_*.csv dan *.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
