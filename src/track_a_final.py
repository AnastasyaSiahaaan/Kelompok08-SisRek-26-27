"""
Track A - Analisis Final (tambahan atas track_a_analysis.py v3)

Menghasilkan:
  results/track_a_table_final.csv    - 16 baris lengkap + kolom *_best
  results/track_a_regression.csv     - regresi NDCG ~ log_e(interaksi) per model
                                       + CI bootstrap kemiringan (10.000 iterasi)
  results/track_a_quintile_ndcg.csv  - NDCG per kuintil x model (tabel 5 x 4)
  results/track_a_sensitivity.csv    - sensitivitas tertil & kuintil (nilai-based)
  results/track_a_dod_sensitivity.csv - DoD Hybrid-Popularity per skema
  results/track_a_relevant_pop.csv   - popularitas item relevan; CI per USER
  results/track_a_ndcg_bar.png       - bar chart NDCG@10 dengan error bar CI 95%

PERBAIKAN PEMBAGIAN KELOMPOK:
  Versi lama memakai np.percentile(np.unique(vals)) -- percentil dari NILAI UNIK,
  bukan dari distribusi user. Pada data diskrit skewed (median=5, max=270) ini
  menghasilkan batas yang sangat besar (mis. Q1 mencakup 97% user).

  Versi ini memakai np.percentile(counts.to_numpy()) -- percentil dari DISTRIBUSI
  SEMUA USER, lalu pd.cut dengan batas nilai integer. Semua user dengan jumlah
  interaksi yang sama dijamin masuk kelompok yang sama.

  Kuintil hasil: Q1[1-3]=2899, Q2[4]=1873, Q3[5-6]=2355, Q4[7-11]=2462, Q5[12-270]=2070

Regresi:
  Dijalankan terpisah untuk masing-masing model (bukan selisih).
  NDCG@10_per_user ~ b0 + b1 * log_e(train_interactions).
  Selisih slope antar model tidak independen (berbagi komponen Popularity).

Popularitas item:
  "Popularitas" = jumlah interaksi positif (rating >= 4) di training.
  Definisi sama untuk item relevan dan item Top-10 rekomendasi.
  Rata-rata per USER (bukan per item) untuk item relevan.
  Rasio avg_top10_popularity / avg_relevant_pop sekitar 17x (satu orde besaran).

Jalankan dari root proyek:
    python src/track_a_final.py
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
METRICS = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]
GROUPS  = ["Overall", "Low", "Medium", "High"]


# ===========================================================================
# Muat user_metrics
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
# BAGIAN 1 - Tabel final 16 baris
# ===========================================================================

def build_and_save_final_table(dfs: dict) -> pd.DataFrame:
    """
    Baca track_a_table.csv yang sudah ada (argmax sudah benar),
    simpan salinan final ke track_a_table_final.csv, cetak 16 baris.
    """
    table = pd.read_csv(RESULT_DIR / "track_a_table.csv")

    # Urutan tampilan
    model_order = ["popularity", "mf", "tfidf_cbf", "hybrid"]
    group_order = ["Overall", "Low", "Medium", "High"]
    table["_mo"] = table["model"].map({m: i for i, m in enumerate(model_order)})
    table["_go"] = table["group"].map({g: i for i, g in enumerate(group_order)})
    table = table.sort_values(["_mo", "_go"]).drop(columns=["_mo", "_go"]).reset_index(drop=True)

    out_cols = ["label", "group", "n_users", "pct_users",
                "Recall@10", "NDCG@10", "HitRate@10", "MRR@10",
                "Recall@10_best", "NDCG@10_best", "HitRate@10_best", "MRR@10_best"]
    table[out_cols].to_csv(RESULT_DIR / "track_a_table_final.csv", index=False)

    # Cetak dengan tanda * dari kolom _best (baca langsung, tanpa logika ulang)
    hdr = (f"\n{'Model':<16} {'Kelompok':<8} {'n':>6} {'%':>6}  "
           f"{'Recall@10':>11} {'NDCG@10':>11} {'HitRate@10':>11} {'MRR@10':>11}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    prev = None
    for _, row in table.iterrows():
        if prev is not None and row["model"] != prev:
            print()
        prev = row["model"]

        def fmt(m):
            v = row[m]
            if pd.isna(v):
                return f"{'N/A':>11}"
            star = "*" if bool(row[f"{m}_best"]) else " "
            return f"{v:10.6f}{star}"

        print(f"{row['label']:<16} {row['group']:<8} {row['n_users']:>6} "
              f"{row['pct_users']:>6}  "
              f"{fmt('Recall@10')} {fmt('NDCG@10')} {fmt('HitRate@10')} {fmt('MRR@10')}")

    return table


# ===========================================================================
# BAGIAN 2 - Regresi NDCG@10 ~ log_e(interaksi) per model
# ===========================================================================

def run_trend_regression(dfs: dict) -> pd.DataFrame:
    """
    Regresi OLS terpisah untuk masing-masing model:
        NDCG@10_per_user ~ b0 + b1 * log_e(train_interactions)

    Bootstrap 10.000 resampling per user untuk CI kemiringan b1.
    Basis log: natural (log_e).

    CATATAN: Selisih slope antar model (mis. Hybrid - Popularity) TIDAK
    independen karena user yang sama muncul di kedua regresi. Laporkan
    selisih slope sebagai deskriptif, bukan uji terpisah.
    """
    act     = pd.read_csv(BASE_DIR / "data/supporting/user_activity_train.csv",
                          dtype={"user_id": str})
    int_map = act.set_index("user_id")["train_interactions"].to_dict()
    rng     = np.random.default_rng(RANDOM_SEED)
    rows    = []

    for m in MODELS:
        df      = dfs[m]
        uids    = df.index.tolist()
        ndcg    = df["NDCG@10"].fillna(0).to_numpy(np.float64)
        log_int = np.array([np.log(max(int_map.get(u, 1), 1)) for u in uids],
                           dtype=np.float64)

        X       = np.column_stack([np.ones(len(ndcg)), log_int])
        b_obs   = np.linalg.lstsq(X, ndcg, rcond=None)[0]
        b0_obs, b1_obs = b_obs[0], b_obs[1]

        n       = len(ndcg)
        boot_b1 = []
        for _ in range(N_BOOTSTRAP):
            idx   = rng.choice(n, size=n, replace=True)
            b_b   = np.linalg.lstsq(X[idx], ndcg[idx], rcond=None)[0]
            boot_b1.append(b_b[1])
        boot_b1  = np.array(boot_b1)
        ci_lo, ci_hi = np.percentile(boot_b1, [2.5, 97.5])

        if   ci_lo > 0: interp = "b1 > 0: NDCG cenderung naik seiring log(interaksi)"
        elif ci_hi < 0: interp = "b1 < 0: NDCG cenderung turun seiring log(interaksi)"
        else:           interp = "Tidak dapat disimpulkan (CI memuat 0)"

        rows.append({
            "model":          m,
            "label":          MODEL_LABELS[m],
            "metric":         "NDCG@10",
            "log_base":       "e (natural)",
            "b0_intercept":   round(float(b0_obs), 7),
            "b1_slope":       round(float(b1_obs), 7),
            "b1_ci_lo_95":    round(float(ci_lo), 7),
            "b1_ci_hi_95":    round(float(ci_hi), 7),
            "n_users":        n,
            "interpretation": interp,
            "analysis_type":  "exploratory",
            "note": (
                "Regresi OLS per model (bukan selisih). "
                "Selisih slope antar model tidak independen karena user sama. "
                "Bootstrap resampling per user, 10.000 iterasi, seed=42."
            ),
            "n_bootstrap":    N_BOOTSTRAP,
            "seed":           RANDOM_SEED,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 3 - NDCG@10 per tertil dan kuintil (nilai-based, bukan rank)
# ===========================================================================

def _cut_by_value(counts: pd.Series, n_bins: int, label_prefix: str):
    """
    Potong berdasarkan NILAI interaksi menggunakan pd.cut.
    Semua user dengan nilai interaksi yang sama SELALU masuk kelompok yang sama.
    Ukuran kelompok boleh tidak persis sama.

    Metode:
      Langkah 1: Hitung kuantil dari DISTRIBUSI SEMUA USER (bukan dari nilai unik).
        Ini memastikan tiap persentil merepresentasikan jumlah user proporsional,
        bukan representasi nilai unik yang sangat skewed pada data diskrit.
        Bug lama memakai np.unique(vals) lalu percentile -- menghasilkan batas
        yang sangat besar (mis. Q1 mencakup 97% user).
      Langkah 2: Bulatkan ke integer dan hapus duplikat.
      Langkah 3: pd.cut dengan include_lowest=True -- semua user interaksi sama
        dijamin masuk kelompok yang sama karena batas berdasarkan nilai.

    Contoh kuintil (n=11659 user, median=5):
      Kuantil user P20/40/60/80: [3, 4, 6, 11]
      Q1:[1-3], Q2:[4], Q3:[5-6], Q4:[7-11], Q5:[12-270]

    Returns:
        assignment : Series {user_id -> label}  (dtype str)
        edges_desc : list of (label, lo_inclusive, hi_inclusive, n_users)
    """
    # Langkah 1: persentil dari distribusi SEMUA user
    quant_points = np.linspace(0, 100, n_bins + 1)
    raw_edges    = np.percentile(counts.to_numpy(), quant_points)

    # Langkah 2: bulatkan ke integer, hapus duplikat
    int_edges = sorted(set(int(np.round(e)) for e in raw_edges))
    if len(int_edges) < 2:
        int_edges = [int(counts.min()), int(counts.max())]

    # Langkah 3: bins untuk pd.cut -- offset -1 pada batas bawah
    bins_cut   = [int_edges[0] - 1] + int_edges[1:]
    bins_cut   = sorted(set(bins_cut))
    n_actual   = len(bins_cut) - 1
    labels_lbl = [f"{label_prefix}{i+1}" for i in range(n_actual)]

    cut = pd.cut(counts, bins=bins_cut, labels=labels_lbl, include_lowest=True)
    assignment = cut.astype(str).rename("group_label")

    edges_desc = []
    for lbl, lo_b, hi_b in zip(labels_lbl, bins_cut[:-1], bins_cut[1:]):
        members = counts[cut == lbl]
        lo_inc  = int(members.min()) if len(members) > 0 else lo_b + 1
        hi_inc  = int(members.max()) if len(members) > 0 else hi_b
        edges_desc.append((lbl, lo_inc, hi_inc, len(members)))

    return assignment, edges_desc


def build_quintile_and_tertile_ndcg(dfs: dict) -> tuple:
    """
    Hitung NDCG@10 per kuintil dan tertil (nilai-based, pd.cut).
    Juga hitung DoD Hybrid-Popularity (top - bottom) dengan CI bootstrap.
    Returns (sens_df, dod_df).
    """
    act = pd.read_csv(BASE_DIR / "data/supporting/user_activity_train.csv",
                      dtype={"user_id": str})
    evaluable = set(dfs["popularity"].index)
    act = act[act["user_id"].isin(evaluable)].copy()
    counts = act.set_index("user_id")["train_interactions"]

    rng = np.random.default_rng(RANDOM_SEED)

    def _dod_ci(group_map, top_lbl, bot_lbl, metric="NDCG@10"):
        common  = set(dfs["hybrid"].index) & set(dfs["popularity"].index)
        hi_idx  = [u for u in common if group_map.get(u) == top_lbl]
        lo_idx  = [u for u in common if group_map.get(u) == bot_lbl]
        if not hi_idx or not lo_idx:
            return np.nan, np.nan, np.nan, "kelompok kosong"
        d_hi = (dfs["hybrid"].loc[hi_idx, metric].fillna(0).to_numpy(np.float64) -
                dfs["popularity"].loc[hi_idx, metric].fillna(0).to_numpy(np.float64))
        d_lo = (dfs["hybrid"].loc[lo_idx, metric].fillna(0).to_numpy(np.float64) -
                dfs["popularity"].loc[lo_idx, metric].fillna(0).to_numpy(np.float64))
        obs  = d_hi.mean() - d_lo.mean()
        boot = np.array([
            rng.choice(d_hi, size=len(d_hi), replace=True).mean() -
            rng.choice(d_lo, size=len(d_lo), replace=True).mean()
            for _ in range(N_BOOTSTRAP)
        ])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        if   ci_lo > 0: interp = "DoD > 0 (CI tidak memuat 0)"
        elif ci_hi < 0: interp = "DoD < 0 (CI tidak memuat 0)"
        else:           interp = "Tidak dapat disimpulkan (CI memuat 0)"
        return obs, ci_lo, ci_hi, interp

    all_rows = []
    dod_rows = []

    for scheme_name, n_bins, prefix in [("tertile", 3, "T"), ("quintile", 5, "Q")]:
        assignment, edges_desc = _cut_by_value(counts, n_bins, prefix)
        group_map = assignment.to_dict()

        # Cetak batas aktual
        print(f"\n  Skema {scheme_name}: batas nilai interaksi (inklusif)")
        for lbl, lo, hi, n in edges_desc:
            pct = 100 * n / len(evaluable)
            print(f"    {lbl}: [{lo}, {hi}]  n={n} ({pct:.1f}%)")

        labels_lbl = [e[0] for e in edges_desc]
        for lbl, lo_inc, hi_inc, _ in edges_desc:
            row = {"scheme": scheme_name, "group_label": lbl,
                   "edge_lo": lo_inc, "edge_hi": hi_inc}
            users_grp = [u for u in evaluable if group_map.get(u) == lbl]
            row["n_users"] = len(users_grp)
            row["pct_users"] = round(100 * len(users_grp) / len(evaluable), 1)
            for m in MODELS:
                df = dfs[m]
                v  = df.loc[df.index.isin(users_grp), "NDCG@10"].fillna(0)
                row[MODEL_LABELS[m]] = round(float(v.mean()), 6) if len(v) > 0 else np.nan
            all_rows.append(row)

        obs, ci_lo, ci_hi, interp = _dod_ci(group_map, labels_lbl[-1], labels_lbl[0])
        dod_rows.append({
            "scheme": scheme_name, "top_label": labels_lbl[-1],
            "bot_label": labels_lbl[0],
            "diff_of_diff": round(float(obs), 6) if not np.isnan(obs) else np.nan,
            "ci_lo_95":     round(float(ci_lo), 6) if not np.isnan(ci_lo) else np.nan,
            "ci_hi_95":     round(float(ci_hi), 6) if not np.isnan(ci_hi) else np.nan,
            "interpretation": interp, "metric": "NDCG@10",
            "comparison": "Hybrid-Popularity", "analysis_type": "exploratory",
            "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED,
        })

    return pd.DataFrame(all_rows), pd.DataFrame(dod_rows)


# backward-compat wrapper for callers expecting (df, mono_notes)
def build_quintile_ndcg(dfs: dict):
    sens_df, _ = build_quintile_and_tertile_ndcg(dfs)
    q_df = sens_df[sens_df["scheme"] == "quintile"].copy()
    q_df = q_df.rename(columns={"group_label": "quintile"})
    # monoton check
    mono_notes = {}
    for m in MODELS:
        col_vals = q_df[MODEL_LABELS[m]].tolist()
        diffs = [col_vals[i+1] - col_vals[i] for i in range(len(col_vals)-1)]
        if all(d >= -1e-9 for d in diffs):   mono_notes[MODEL_LABELS[m]] = "monoton naik"
        elif all(d <= 1e-9 for d in diffs):  mono_notes[MODEL_LABELS[m]] = "monoton turun"
        else:                                 mono_notes[MODEL_LABELS[m]] = "tidak monoton"
    return q_df, mono_notes


# ===========================================================================
# BAGIAN 4 - Popularitas item relevan: CI bootstrap per USER
# ===========================================================================

def compute_relevant_pop_per_user(dfs: dict) -> pd.DataFrame:
    """
    Hitung rata-rata popularitas item relevan dengan bootstrap PER USER
    (bukan per item): rata-ratakan dulu per user, lalu bootstrap distribusi mean-of-means.

    Tambahkan: rata-rata popularitas item Top-10 Popularity per kelompok
    (dari track_a_popularity.csv yang sudah ada) untuk perbandingan deskriptif.
    Narasi: deskriptif saja, tidak menyebut masking sebagai penyebab.
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

    rel_rows = val[val["user_id"].isin(evaluated_users) & (val["rating"] >= 4)].copy()
    rel_rows["item_id"] = rel_rows["item_id"].astype(str)

    # Hitung rata-rata popularitas item relevan PER USER
    user_avg_pop  = {}
    user_pct_top  = {}
    user_grp_map  = {}
    for uid, grp_df in rel_rows.groupby("user_id"):
        u_seen = seen.get(uid, set())
        u_grp  = group_map.get(uid, "Unknown")
        items  = [i for i in grp_df["item_id"] if i in candidate_items and i not in u_seen]
        if not items:
            continue
        pops = [pop_map.get(i, 0) for i in items]
        user_avg_pop[uid]  = np.mean(pops)
        user_pct_top[uid]  = np.mean([i in top100 for i in items])
        user_grp_map[uid]  = u_grp

    # Bootstrap CI per USER
    rng = np.random.default_rng(RANDOM_SEED)

    def _ci_user(user_vals: np.ndarray):
        n = len(user_vals)
        if n == 0:
            return np.nan, np.nan
        boot = np.array([rng.choice(user_vals, size=n, replace=True).mean()
                         for _ in range(N_BOOTSTRAP)])
        return np.percentile(boot, [2.5, 97.5])

    # Rata-rata popularitas Top-10 Popularity dari track_a_popularity.csv
    pop_top10_df = pd.read_csv(RESULT_DIR / "track_a_popularity.csv")
    pop_top10_popularity = (
        pop_top10_df[pop_top10_df["model"] == "popularity"]
        .set_index("group")["avg_popularity"].to_dict()
    )

    rows = []
    for grp in ["Overall", "Low", "Medium", "High"]:
        if grp == "Overall":
            uids = list(user_avg_pop.keys())
        else:
            uids = [u for u, g in user_grp_map.items() if g == grp]

        avg_vals = np.array([user_avg_pop[u] for u in uids if u in user_avg_pop], dtype=np.float64)
        pct_vals = np.array([user_pct_top[u]  for u in uids if u in user_avg_pop], dtype=np.float64)

        ci_avg_lo, ci_avg_hi = _ci_user(avg_vals)

        rows.append({
            "group":                   grp,
            "n_users_with_relevant":   len(avg_vals),
            "avg_relevant_pop_per_user":  round(float(avg_vals.mean()), 4) if len(avg_vals) > 0 else np.nan,
            "median_relevant_pop_per_user": round(float(np.median(avg_vals)), 4) if len(avg_vals) > 0 else np.nan,
            "ci_lo_avg_pop_95":        round(float(ci_avg_lo), 4) if not np.isnan(ci_avg_lo) else np.nan,
            "ci_hi_avg_pop_95":        round(float(ci_avg_hi), 4) if not np.isnan(ci_avg_hi) else np.nan,
            "avg_pct_relevant_in_top100": round(float(pct_vals.mean() * 100), 2) if len(pct_vals) > 0 else np.nan,
            "avg_popularity_top10_popularity_model": pop_top10_popularity.get(grp, np.nan),
            "ci_note": "CI dihitung bootstrap 10000 resampling per USER (bukan per item).",
        })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 5 - Figure diperbarui: error bar CI bootstrap mean NDCG per kelompok
# ===========================================================================

def make_updated_figure(dfs: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib tidak tersedia.")
        return

    rng = np.random.default_rng(RANDOM_SEED)

    def _ci_mean(vals: np.ndarray):
        boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                         for _ in range(2000)])
        return np.percentile(boot, [2.5, 97.5])

    group_order = ["Overall", "Low", "Medium", "High"]
    model_order = ["popularity", "mf", "tfidf_cbf", "hybrid"]
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    n_grp, n_mod = len(group_order), len(model_order)
    x, width     = np.arange(n_grp), 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for mi, m in enumerate(model_order):
        df    = dfs[m].reset_index()
        means, errs_lo, errs_hi = [], [], []
        for grp in group_order:
            sub  = df if grp == "Overall" else df[df["activity_group"] == grp]
            vals = sub["NDCG@10"].fillna(0).to_numpy(np.float64)
            mean = vals.mean()
            ci   = _ci_mean(vals)
            means.append(mean)
            errs_lo.append(mean - ci[0])
            errs_hi.append(ci[1] - mean)

        offset = (mi - (n_mod - 1) / 2) * width
        ax.bar(x + offset, means, width, label=MODEL_LABELS[m],
               color=colors[mi], alpha=0.85)
        ax.errorbar(x + offset, means,
                    yerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=11)
    ax.set_ylabel("NDCG@10 (mean per user)", fontsize=11)
    ax.set_title("NDCG@10 per Kelompok Aktivitas  (error bar = 95% CI bootstrap mean)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = RESULT_DIR / "track_a_ndcg_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Figure: {out}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("Track A - Analisis Final (tambahan)")
    print(f"n_bootstrap={N_BOOTSTRAP}, seed={RANDOM_SEED}")
    print("=" * 70)

    dfs = load_user_metrics()

    # 1. Tabel final 16 baris
    print("\n[1] Tabel metrik final (16 baris, argmax dari CSV)")
    build_and_save_final_table(dfs)
    print("  -> results/track_a_table_final.csv")

    # 2. Regresi per model
    print(f"\n[2] Regresi NDCG@10 ~ log_e(interaksi) per model (n_boot={N_BOOTSTRAP})")
    reg_df = run_trend_regression(dfs)
    reg_df.to_csv(RESULT_DIR / "track_a_regression.csv", index=False)
    print(reg_df[["label", "b0_intercept", "b1_slope",
                  "b1_ci_lo_95", "b1_ci_hi_95", "interpretation"]].to_string(index=False))
    # Selisih slope (deskriptif, bukan uji independen)
    slopes = dict(zip(reg_df["model"], reg_df["b1_slope"]))
    print("  Selisih slope terhadap Popularity (deskriptif, bukan uji independen):")
    for m in ["mf", "tfidf_cbf", "hybrid"]:
        print(f"    {MODEL_LABELS[m]} - Popularity: {slopes[m] - slopes['popularity']:+.7f}")

    # 3. NDCG per kuintil (dan tertil dalam sensitivity)
    print("\n[3] NDCG@10 per tertil dan kuintil (nilai-based, user interaksi sama -> kelompok sama)")
    sens_df, dod_df = build_quintile_and_tertile_ndcg(dfs)
    sens_df.to_csv(RESULT_DIR / "track_a_sensitivity.csv", index=False)
    dod_df.to_csv(RESULT_DIR / "track_a_dod_sensitivity.csv", index=False)

    # Simpan kuintil saja ke track_a_quintile_ndcg.csv (kompatibilitas)
    q_df = sens_df[sens_df["scheme"] == "quintile"].copy()
    q_df = q_df.rename(columns={"group_label": "quintile"})
    q_df.to_csv(RESULT_DIR / "track_a_quintile_ndcg.csv", index=False)

    for scheme in ["tertile", "quintile"]:
        sub = sens_df[sens_df["scheme"] == scheme]
        dcols = ["group_label", "edge_lo", "edge_hi", "n_users", "pct_users"] + \
                [MODEL_LABELS[m] for m in MODELS]
        print(f"\n  === {scheme.upper()} ===")
        print(sub[dcols].to_string(index=False))

    print("\n  DoD Hybrid-Popularity (top - bottom) per skema:")
    print(dod_df[["scheme", "top_label", "bot_label", "diff_of_diff",
                  "ci_lo_95", "ci_hi_95", "interpretation"]].to_string(index=False))
    # 4. Popularitas item relevan: CI per user
    print("\n[4] Popularitas item relevan (CI bootstrap per USER)")
    rel_pop_df = compute_relevant_pop_per_user(dfs)
    rel_pop_df.to_csv(RESULT_DIR / "track_a_relevant_pop.csv", index=False)
    print(rel_pop_df[["group", "n_users_with_relevant",
                       "avg_relevant_pop_per_user", "median_relevant_pop_per_user",
                       "ci_lo_avg_pop_95", "ci_hi_avg_pop_95",
                       "avg_pct_relevant_in_top100",
                       "avg_popularity_top10_popularity_model"]].to_string(index=False))

    # 5. Figure
    print("\n[5] Memperbarui figure...")
    make_updated_figure(dfs)

    print("\n" + "=" * 70)
    print("Selesai.")
    print("=" * 70)


if __name__ == "__main__":
    main()
