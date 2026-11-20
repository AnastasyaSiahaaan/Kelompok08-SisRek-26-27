"""
Track A - Analisis Lengkap (Bagian 12 Reproducibility Package)

Menghasilkan:
  results/track_a_table.csv        - metrik 4 model x 4 kelompok; nilai terbaik
                                     per kolom ditandai kolom *_best (argmax otomatis)
  results/track_a_bootstrap_ci.csv - paired CI; Bonferroni k=6 untuk konfirmatori
  results/track_a_indep_ci.csv     - independent CI selisih High-Low per model
  results/track_a_diff_of_diff.csv - uji selisih-dari-selisih (High-Low) antar model
  results/track_a_popularity.csv   - rata-rata popularitas item Top-10 per model/grup
  results/track_a_relevant_pop.csv - popularitas & cakupan Top-100 item RELEVAN
  results/track_a_overlap.csv      - overlap Top-10 Hybrid vs MF/TF-IDF per grup

Daftar konfirmatori yang ditetapkan sebelum FinalTest (Bonferroni k=6):
  Hybrid vs Popularity, NDCG@10, Low/Medium/High
  Hybrid vs MF,         NDCG@10, Low/Medium/High
  alpha per sel = 0.05 / 6 = 0.00833 -> CI 99.17%

Semua perbandingan lain: eksploratori (CI 95%, alpha=0.05).

Istilah:
  alpha (Hybrid): bobot MF dalam skor gabungan.
    skor_hybrid = alpha * norm_MF + (1-alpha) * norm_CBF
    alpha=0.5 berarti bobot sama antara MF dan TF-IDF CBF.
  Normalisasi: min-max per-user -- setiap vektor skor dinormalkan ke [0,1]
    sebelum digabungkan; mencegah satu komponen mendominasi karena skala berbeda.
  Batas kelompok: ditetapkan di src/config.py (EVAL_CONFIG) sebelum melihat
    data validasi; diimplementasikan dengan pd.cut(bins=[0,5,8,inf]).

Catatan reprodusibilitas:
  HitRate@10 dan MRR@10 untuk model Hybrid dihitung dari
  results/hybrid_recommendations_top10.csv (lihat src/add_hitrate_mrr.py),
  karena baseline dijalankan sebelum metrik tersebut ditambahkan.
  Recall@10 dan NDCG@10 Hybrid diverifikasi identik bit-untuk-bit.

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

RANDOM_SEED = 42
N_BOOTSTRAP = 10_000

# ---------------------------------------------------------------------------
# Model inti
# ---------------------------------------------------------------------------
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
# Daftar konfirmatori (ditetapkan sebelum FinalTest, dicatat di FREEZE_CONFIG)
# Bonferroni k=6 -> alpha per sel = 0.05/6 = 0.008333 -> CI 99.17%
# ---------------------------------------------------------------------------
CONFIRMATORY = [
    ("hybrid", "popularity", "Low",    "NDCG@10"),
    ("hybrid", "popularity", "Medium", "NDCG@10"),
    ("hybrid", "popularity", "High",   "NDCG@10"),
    ("hybrid", "mf",         "Low",    "NDCG@10"),
    ("hybrid", "mf",         "Medium", "NDCG@10"),
    ("hybrid", "mf",         "High",   "NDCG@10"),
]
BONFERRONI_K       = len(CONFIRMATORY)          # 6
ALPHA_CONFIRMATORY = 0.05 / BONFERRONI_K        # 0.008333
CI_CONF_LEVEL      = 100 * (1 - ALPHA_CONFIRMATORY)   # 99.17 %

PAIRED_COMPARISONS = [
    ("mf",        "popularity", "MF vs Popularity"),
    ("tfidf_cbf", "popularity", "TF-IDF vs Popularity"),
    ("hybrid",    "popularity", "Hybrid vs Popularity"),
    ("hybrid",    "mf",         "Hybrid vs MF"),
    ("hybrid",    "tfidf_cbf",  "Hybrid vs TF-IDF"),
]


# ===========================================================================
# BAGIAN 0 - Muat user_metrics
# ===========================================================================

def load_user_metrics() -> dict:
    dfs = {}
    for m in MODELS:
        path = RESULT_DIR / f"{m}_user_metrics.csv"
        df = pd.read_csv(path, dtype={"user_id": str})
        for col in ["HitRate@10", "MRR@10"]:
            if col not in df.columns:
                df[col] = np.nan
        dfs[m] = df.set_index("user_id")
    return dfs


# ===========================================================================
# BAGIAN 1 - Tabel metrik lengkap dengan penanda terbaik otomatis
# ===========================================================================

def build_track_a_table(dfs: dict) -> pd.DataFrame:
    """
    Hitung mean setiap metrik per model per kelompok.
    Tambahkan kolom boolean <metric>_best = True bila nilai tertinggi
    di kelompok itu (argmax otomatis, bukan manual).
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
                r[metric] = sub[metric].mean() if (metric in sub.columns and len(sub) > 0) else np.nan
            rows.append(r)

    table = pd.DataFrame(rows)

    # Persentase pengguna
    n_total = table.loc[table["group"] == "Overall", "n_users"].iloc[0]
    table["pct_users"] = table.apply(
        lambda r: f"{100 * r['n_users'] / n_total:.1f}%"
        if r["group"] != "Overall" else "100.0%", axis=1)

    # Tandai nilai terbaik per kolom metrik per kelompok (argmax otomatis)
    for metric in METRICS:
        col_best = f"{metric}_best"
        table[col_best] = False
        for grp in GROUPS:
            mask = table["group"] == grp
            vals = table.loc[mask, metric]
            if vals.notna().any():
                best_idx = vals.idxmax()
                table.loc[best_idx, col_best] = True

    return table


def print_table_with_bold(table: pd.DataFrame) -> None:
    """Cetak tabel dengan tanda * pada nilai terbaik per kelompok."""
    print(f"\n{'Model':<16} {'Kelompok':<8} {'n_users':>7} {'%':>6} "
          f"{'Recall@10':>10} {'NDCG@10':>10} {'HitRate@10':>11} {'MRR@10':>9}")
    print("-" * 82)
    for _, row in table.iterrows():
        def fmt(metric):
            val = row[metric]
            if pd.isna(val):
                return f"{'N/A':>10}"
            star = "*" if row.get(f"{metric}_best", False) else " "
            return f"{val:9.6f}{star}"
        print(f"{row['label']:<16} {row['group']:<8} {row['n_users']:>7} "
              f"{row['pct_users']:>6} "
              f"{fmt('Recall@10')} {fmt('NDCG@10')} {fmt('HitRate@10')} {fmt('MRR@10')}")


# ===========================================================================
# BAGIAN 2 - Bootstrap CI Paired
# ===========================================================================

def _bootstrap_paired(a: np.ndarray, b: np.ndarray,
                      n: int, seed: int, alpha: float = 0.05):
    """
    Paired bootstrap untuk mean(a-b).
    Returns (mean_diff, ci_lo, ci_hi).
    Inferensi signifikansi: 0 di luar CI.
    Hati-hati memakai kata 'terbukti' bila batas CI sangat dekat 0.
    """
    rng  = np.random.default_rng(seed)
    diff = a - b
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean()
                     for _ in range(n)])
    q_lo = 100 * alpha / 2
    q_hi = 100 * (1 - alpha / 2)
    return diff.mean(), np.percentile(boot, q_lo), np.percentile(boot, q_hi)


def run_paired_bootstrap(dfs: dict) -> pd.DataFrame:
    rows = []
    for model_a, model_b, comp_label in PAIRED_COMPARISONS:
        df_a = dfs[model_a]
        df_b = dfs[model_b]
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

                is_conf   = (model_a, model_b, grp, metric) in CONFIRMATORY
                alpha_use = ALPHA_CONFIRMATORY if is_conf else 0.05
                ci_pct    = f"{100*(1-alpha_use):.2f}%"
                atype     = "confirmatory (Bonferroni k=6)" if is_conf else "exploratory"

                md, ci_lo, ci_hi = _bootstrap_paired(a_v, b_v, N_BOOTSTRAP,
                                                      RANDOM_SEED, alpha_use)

                # Kata "signifikan" hanya berarti 0 di luar CI di level tersebut.
                # Jangan pakai "terbukti" bila margin sangat tipis (< 0.0001 dari 0).
                if ci_lo > 0:
                    sig = "Ya (+)"
                elif ci_hi < 0:
                    sig = "Ya (-)"
                else:
                    sig = "Tidak"

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
# BAGIAN 3 - Bootstrap CI Independen High-Low
# ===========================================================================

def _bootstrap_independent(a: np.ndarray, b: np.ndarray, n: int, seed: int):
    """
    Independent bootstrap mean(a)-mean(b).
    Dua kelompok berbeda sehingga tidak bisa paired.
    """
    rng  = np.random.default_rng(seed)
    boot = np.array([
        rng.choice(a, size=len(a), replace=True).mean() -
        rng.choice(b, size=len(b), replace=True).mean()
        for _ in range(n)
    ])
    return a.mean() - b.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def run_independent_high_low(dfs: dict) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        df = dfs[m].reset_index()
        hi = df.loc[df["activity_group"] == "High", "NDCG@10"].fillna(0).to_numpy(np.float64)
        lo = df.loc[df["activity_group"] == "Low",  "NDCG@10"].fillna(0).to_numpy(np.float64)
        obs, ci_lo, ci_hi = _bootstrap_independent(hi, lo, N_BOOTSTRAP, RANDOM_SEED)

        if ci_lo > 0:
            interp = "High > Low (CI tidak memuat 0)"
        elif ci_hi < 0:
            interp = "High < Low (CI tidak memuat 0)"
        else:
            interp = "Tidak dapat disimpulkan (CI memuat 0)"

        rows.append({
            "model":               m,
            "label":               MODEL_LABELS[m],
            "metric":              "NDCG@10",
            "mean_high":           round(float(hi.mean()), 6),
            "mean_low":            round(float(lo.mean()), 6),
            "diff_high_minus_low": round(float(obs), 6),
            "ci_lo_95":            round(float(ci_lo), 6),
            "ci_hi_95":            round(float(ci_hi), 6),
            "n_high":              len(hi),
            "n_low":               len(lo),
            "interpretation":      interp,
            "analysis_type":       "exploratory",
            "note": ("Bootstrap independen (user High dan Low berbeda). "
                     "Hati-hati: confounding dari item pool berbeda antar kelompok."),
            "n_bootstrap":         N_BOOTSTRAP,
            "seed":                RANDOM_SEED,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 4 - Selisih-dari-selisih (High-Low): (A-B)_High - (A-B)_Low
# Contoh: (Hybrid-Popularity)_High - (Hybrid-Popularity)_Low
# Metode: paired dalam kelompok, independen antar kelompok.
# ===========================================================================

DIFF_OF_DIFF_PAIRS = [
    ("hybrid", "popularity", "Hybrid-Popularity"),
    ("hybrid", "mf",         "Hybrid-MF"),
]


def run_diff_of_diff(dfs: dict) -> pd.DataFrame:
    """
    Uji apakah keuntungan model_a atas model_b berbeda antara High dan Low.
    Langkah:
      1. Hitung d_High[u] = metric_a[u] - metric_b[u] untuk u in High.
      2. Hitung d_Low[u]  = metric_a[u] - metric_b[u] untuk u in Low.
      3. Statistic = mean(d_High) - mean(d_Low).
      4. Bootstrap: resample d_High dan d_Low secara INDEPENDEN (antar kelompok)
         karena user-nya berbeda.
    """
    rows = []
    for model_a, model_b, label in DIFF_OF_DIFF_PAIRS:
        df_a = dfs[model_a]
        df_b = dfs[model_b]
        common = df_a.index.intersection(df_b.index)

        for metric in ["NDCG@10", "Recall@10"]:
            # Selisih per-user di tiap kelompok
            hi_idx = common[df_a.loc[common, "activity_group"] == "High"]
            lo_idx = common[df_a.loc[common, "activity_group"] == "Low"]

            d_hi = (df_a.loc[hi_idx, metric].fillna(0).to_numpy(np.float64) -
                    df_b.loc[hi_idx, metric].fillna(0).to_numpy(np.float64))
            d_lo = (df_a.loc[lo_idx, metric].fillna(0).to_numpy(np.float64) -
                    df_b.loc[lo_idx, metric].fillna(0).to_numpy(np.float64))

            obs_stat = d_hi.mean() - d_lo.mean()

            # Bootstrap independen antar kelompok
            rng = np.random.default_rng(RANDOM_SEED)
            boot = np.array([
                rng.choice(d_hi, size=len(d_hi), replace=True).mean() -
                rng.choice(d_lo, size=len(d_lo), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

            if ci_lo > 0:
                interp = "Interaksi ada: keuntungan model_a lebih besar di High"
            elif ci_hi < 0:
                interp = "Interaksi ada: keuntungan model_a lebih besar di Low"
            else:
                interp = "Tidak dapat disimpulkan (CI memuat 0)"

            rows.append({
                "comparison":        label,
                "model_a":           model_a,
                "model_b":           model_b,
                "metric":            metric,
                "mean_diff_high":    round(float(d_hi.mean()), 6),
                "mean_diff_low":     round(float(d_lo.mean()), 6),
                "diff_of_diff":      round(float(obs_stat), 6),
                "ci_lo_95":          round(float(ci_lo), 6),
                "ci_hi_95":          round(float(ci_hi), 6),
                "n_high":            len(d_hi),
                "n_low":             len(d_lo),
                "interpretation":    interp,
                "analysis_type":     "exploratory",
                "method": ("Paired dalam kelompok (user sama untuk model_a dan model_b), "
                           "independen antar kelompok (user High != user Low)."),
                "n_bootstrap":       N_BOOTSTRAP,
                "seed":              RANDOM_SEED,
            })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 5 - Popularitas item Top-10 yang direkomendasikan
# ===========================================================================

def compute_avg_popularity(dfs: dict) -> pd.DataFrame:
    pop_rank    = pd.read_csv(RESULT_DIR / "popularity_ranking.csv", dtype={"item_id": str})
    pop_map     = pop_rank.set_index("item_id")["positive_interactions"].to_dict()
    train       = pd.read_csv(BASE_DIR / "data/raw/interactions_train.csv",
                              dtype={"user_id": str, "item_id": str})
    seen_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    top_items_pop = pop_rank["item_id"].tolist()

    rows = []
    for m in MODELS:
        df_metrics = dfs[m].reset_index()[["user_id", "activity_group"]]

        if m == "popularity":
            rec_rows = []
            for _, row in df_metrics.iterrows():
                uid  = row["user_id"]
                seen = seen_by_user.get(uid, set())
                for item in [i for i in top_items_pop if i not in seen][:10]:
                    rec_rows.append({"user_id": uid, "item_id": item})
            recs = pd.DataFrame(rec_rows)
        else:
            recs = pd.read_csv(RESULT_DIR / f"{m}_recommendations_top10.csv",
                               dtype={"user_id": str, "item_id": str})

        recs = recs.merge(df_metrics, on="user_id", how="left")
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
    Untuk setiap kelompok aktivitas:
      - Rata-rata popularitas (positive_interactions training) item relevan
        (rating validation >= 4, dalam candidate set, belum dilihat training).
      - Proporsi item relevan yang masuk Top-100 item paling populer di training.
    Dihitung sekali (tidak per model) karena relevansi tidak tergantung model.
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

    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    # Item relevan per user: rating>=4, dalam kandidat, belum di training
    rel_rows = val[val["user_id"].isin(evaluated_users) & (val["rating"] >= 4)].copy()
    rel_rows["item_id"] = rel_rows["item_id"].astype(str)

    act_grouped = make_user_groups(act)
    group_map   = act_grouped.set_index("user_id")["activity_group"].to_dict()

    # Kumpulkan item relevan unik per user
    records = []
    for uid, grp_df in rel_rows.groupby("user_id"):
        u_seen = seen.get(uid, set())
        u_grp  = group_map.get(uid, "Unknown")
        for item in grp_df["item_id"]:
            item = str(item)
            if item in candidate_items and item not in u_seen:
                records.append({
                    "user_id":        uid,
                    "item_id":        item,
                    "activity_group": u_grp,
                    "pop":            pop_map.get(item, 0),
                    "in_top100":      item in top100,
                })

    df_rel = pd.DataFrame(records)

    rows = []
    for grp in ["Overall", "Low", "Medium", "High"]:
        sub = df_rel if grp == "Overall" else df_rel[df_rel["activity_group"] == grp]
        if len(sub) == 0:
            rows.append({"group": grp, "n_relevant_items": 0,
                         "n_unique_users": 0, "avg_relevant_pop": np.nan,
                         "pct_in_top100": np.nan})
            continue
        rows.append({
            "group":            grp,
            "n_relevant_items": len(sub),
            "n_unique_users":   sub["user_id"].nunique(),
            "avg_relevant_pop": round(sub["pop"].mean(), 4),
            "pct_in_top100":    round(100 * sub["in_top100"].mean(), 2),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 7 - Overlap Top-10 Hybrid vs MF / TF-IDF
# ===========================================================================

def compute_overlap(dfs: dict) -> pd.DataFrame:
    def to_sets(df):
        return df.groupby("user_id")["item_id"].apply(set).to_dict()

    h_sets = to_sets(pd.read_csv(RESULT_DIR / "hybrid_recommendations_top10.csv",
                                 dtype={"user_id": str, "item_id": str}))
    m_sets = to_sets(pd.read_csv(RESULT_DIR / "mf_recommendations_top10.csv",
                                 dtype={"user_id": str, "item_id": str}))
    t_sets = to_sets(pd.read_csv(RESULT_DIR / "tfidf_cbf_recommendations_top10.csv",
                                 dtype={"user_id": str, "item_id": str}))

    df_act = dfs["hybrid"].reset_index()[["user_id", "activity_group"]]
    rows = []
    for grp in ["Overall", "Low", "Medium", "High"]:
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
            "group":                   grp,
            "n_users":                 len(users),
            "avg_overlap_hybrid_mf":   round(np.mean(ov_mf), 4) if ov_mf else np.nan,
            "avg_overlap_hybrid_tfidf": round(np.mean(ov_tf), 4) if ov_tf else np.nan,
            "note": ("Eksploratori. Hipotesis: overlap tinggi dengan TF-IDF "
                     "menunjukkan skor CBF lebih sering menentukan ranking akhir "
                     "pada alpha=0.5. Perlu uji lanjut dengan variasi alpha."),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("Track A - Analisis Lengkap")
    print(f"n_bootstrap={N_BOOTSTRAP}, seed={RANDOM_SEED}")
    print(f"Konfirmatori: Bonferroni k={BONFERRONI_K}, "
          f"alpha/sel={ALPHA_CONFIRMATORY:.5f}, CI={CI_CONF_LEVEL:.2f}%")
    print("=" * 70)

    # 0. Muat data
    print("\n[0] Memuat user_metrics...")
    dfs = load_user_metrics()

    # 1. Tabel lengkap
    print("\n[1] Tabel metrik lengkap (* = terbaik per kolom per kelompok)")
    table = build_track_a_table(dfs)
    table.to_csv(RESULT_DIR / "track_a_table.csv", index=False)
    print_table_with_bold(table)

    # Distribusi kelompok
    print()
    n_total = table.loc[table["group"] == "Overall", "n_users"].iloc[0]
    for grp, defn in GROUP_DEFINITIONS.items():
        n   = table.loc[(table["model"] == "popularity") & (table["group"] == grp), "n_users"].iloc[0]
        pct = 100 * n / n_total
        print(f"  {grp:7s}: {n:5d} users ({pct:.1f}%)  [{defn}]")
    print(f"  {'Total':7s}: {n_total:5d} users")

    # 2. Bootstrap CI Paired
    print(f"\n[2] Bootstrap CI Paired (n={N_BOOTSTRAP}, seed={RANDOM_SEED})")
    ci_df = run_paired_bootstrap(dfs)
    ci_df.to_csv(RESULT_DIR / "track_a_bootstrap_ci.csv", index=False)
    dcols = ["comparison", "group", "metric", "mean_a", "mean_b",
             "diff", "ci_lo", "ci_hi", "ci_level", "analysis_type", "significant"]
    print(ci_df[dcols].to_string(index=False))

    # 3. CI Independen High-Low
    print(f"\n[3] Bootstrap CI Independen High-Low (n={N_BOOTSTRAP})")
    indep_df = run_independent_high_low(dfs)
    indep_df.to_csv(RESULT_DIR / "track_a_indep_ci.csv", index=False)
    print(indep_df[["label", "metric", "mean_high", "mean_low",
                    "diff_high_minus_low", "ci_lo_95", "ci_hi_95",
                    "interpretation"]].to_string(index=False))

    # 4. Selisih-dari-selisih
    print(f"\n[4] Selisih-dari-selisih (High-Low) (n={N_BOOTSTRAP})")
    dod_df = run_diff_of_diff(dfs)
    dod_df.to_csv(RESULT_DIR / "track_a_diff_of_diff.csv", index=False)
    print(dod_df[["comparison", "metric", "mean_diff_high", "mean_diff_low",
                  "diff_of_diff", "ci_lo_95", "ci_hi_95",
                  "interpretation"]].to_string(index=False))

    # 5. Popularitas item Top-10
    print("\n[5] Rata-rata popularitas item Top-10 yang direkomendasikan")
    pop_df = compute_avg_popularity(dfs)
    pop_df.to_csv(RESULT_DIR / "track_a_popularity.csv", index=False)
    print(pop_df.to_string(index=False))

    # 6. Popularitas item RELEVAN
    print("\n[6] Popularitas item relevan di validation per kelompok")
    rel_pop_df = compute_relevant_popularity(dfs)
    rel_pop_df.to_csv(RESULT_DIR / "track_a_relevant_pop.csv", index=False)
    print(rel_pop_df.to_string(index=False))

    # 7. Overlap
    print("\n[7] Overlap Top-10 Hybrid vs MF / TF-IDF")
    overlap_df = compute_overlap(dfs)
    overlap_df.to_csv(RESULT_DIR / "track_a_overlap.csv", index=False)
    print(overlap_df[["group", "n_users", "avg_overlap_hybrid_mf",
                       "avg_overlap_hybrid_tfidf"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("Selesai. File disimpan di results/track_a_*.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
