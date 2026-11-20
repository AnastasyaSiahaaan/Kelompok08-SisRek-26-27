"""
Track A - Analisis Final (tambahan atas track_a_analysis.py v3)

Menghasilkan:
  results/track_a_table_final.csv    - 16 baris lengkap + kolom *_best
  results/track_a_regression.csv     - regresi delta(NDCG) ~ log(interactions)
                                       + CI bootstrap kemiringan per model
  results/track_a_quintile_ndcg.csv  - NDCG@10 per kuintil x model (tabel 5x4)
  results/track_a_relevant_pop.csv   - diperbarui: CI bootstrap per USER
  results/track_a_ndcg_bar.png       - diperbarui: error bar CI bootstrap mean

Poin FREEZE_CONFIG:
  C7 berstatus EKSPLORATORI di validation, baru menjadi konfirmatori di FinalTest.
  Skema utama: [1-5],[6-8],[9+]. Tertil/kuintil = analisis sensitivitas saja.

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
# BAGIAN 2 - Regresi delta(NDCG) ~ log(interactions)
# ===========================================================================

def run_trend_regression(dfs: dict) -> pd.DataFrame:
    """
    Per user: delta = NDCG@10(model) - NDCG@10(popularity).
    Regresi OLS: delta ~ b0 + b1 * log(train_interactions).
    Bootstrap 10.000 resampling per user untuk CI kemiringan b1.

    Lakukan untuk: MF, TF-IDF, Hybrid vs Popularity.
    """
    act = pd.read_csv(BASE_DIR / "data/supporting/user_activity_train.csv",
                      dtype={"user_id": str})
    int_map = act.set_index("user_id")["train_interactions"].to_dict()

    rng  = np.random.default_rng(RANDOM_SEED)
    rows = []

    for m in ["mf", "tfidf_cbf", "hybrid"]:
        df_m   = dfs[m]
        df_pop = dfs["popularity"]
        common = df_m.index.intersection(df_pop.index)

        ndcg_m   = df_m.loc[common, "NDCG@10"].fillna(0).to_numpy(np.float64)
        ndcg_pop = df_pop.loc[common, "NDCG@10"].fillna(0).to_numpy(np.float64)
        delta    = ndcg_m - ndcg_pop

        # log(interactions), skip users tanpa data
        log_int = np.array([np.log(max(int_map.get(u, 1), 1)) for u in common],
                           dtype=np.float64)

        # OLS observed
        X   = np.column_stack([np.ones(len(log_int)), log_int])
        b_obs = np.linalg.lstsq(X, delta, rcond=None)[0]  # [b0, b1]
        b1_obs = b_obs[1]

        # Bootstrap CI untuk b1
        n = len(delta)
        boot_b1 = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, size=n, replace=True)
            X_b   = X[idx]
            d_b   = delta[idx]
            b_b   = np.linalg.lstsq(X_b, d_b, rcond=None)[0]
            boot_b1.append(b_b[1])
        boot_b1 = np.array(boot_b1)
        ci_lo, ci_hi = np.percentile(boot_b1, [2.5, 97.5])

        if ci_lo > 0:
            interp = "b1 > 0: delta meningkat seiring interaksi (CI tidak memuat 0)"
        elif ci_hi < 0:
            interp = "b1 < 0: delta menurun seiring interaksi (CI tidak memuat 0)"
        else:
            interp = "Tidak dapat disimpulkan (CI memuat 0)"

        rows.append({
            "model":          m,
            "label":          MODEL_LABELS[m],
            "comparison":     f"{MODEL_LABELS[m]} - Popularity",
            "metric":         "NDCG@10",
            "b0_intercept":   round(float(b_obs[0]), 7),
            "b1_slope":       round(float(b1_obs), 7),
            "b1_ci_lo_95":    round(float(ci_lo), 7),
            "b1_ci_hi_95":    round(float(ci_hi), 7),
            "n_users":        n,
            "interpretation": interp,
            "analysis_type":  "exploratory",
            "note": ("delta = NDCG(model) - NDCG(Popularity) per user. "
                     "log = log naturel jumlah interaksi training. "
                     "Bootstrap per user, 10000 iterasi, seed=42."),
            "n_bootstrap":    N_BOOTSTRAP,
            "seed":           RANDOM_SEED,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# BAGIAN 3 - NDCG@10 per kuintil x model (tabel 5 x 4)
# ===========================================================================

def build_quintile_ndcg(dfs: dict) -> pd.DataFrame:
    """
    Hitung NDCG@10 rata-rata tiap model di setiap kuintil interaksi training.
    Kuintil dihitung dari distribusi user evaluable.
    Tampilkan apakah pola monoton.
    """
    act = pd.read_csv(BASE_DIR / "data/supporting/user_activity_train.csv",
                      dtype={"user_id": str})
    evaluable = set(dfs["popularity"].index)
    act = act[act["user_id"].isin(evaluable)].copy()

    vals   = act["train_interactions"].to_numpy(float)
    edges  = np.percentile(vals, [0, 20, 40, 60, 80, 100])
    # Pastikan edges unik
    edges  = np.unique(edges)
    n_bins = len(edges) - 1
    labels = [f"Q{i+1}" for i in range(n_bins)]

    def assign_q(v):
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            if i == 0 and v <= hi:
                return labels[i]
            elif lo < v <= hi:
                return labels[i]
        return labels[-1]

    act["quintile"] = act["train_interactions"].apply(assign_q)
    q_map = act.set_index("user_id")["quintile"].to_dict()

    # Edge ranges per kuintil
    q_edges = {}
    for lbl, lo, hi in zip(labels, edges[:-1], edges[1:]):
        q_edges[lbl] = (int(lo), int(hi))

    rows = []
    for lbl in labels:
        row = {"quintile": lbl,
               "edge_lo": q_edges[lbl][0],
               "edge_hi": q_edges[lbl][1]}
        users_q = [u for u in evaluable if q_map.get(u) == lbl]
        row["n_users"] = len(users_q)
        for m in MODELS:
            df = dfs[m]
            vals_q = df.loc[df.index.isin(users_q), "NDCG@10"].fillna(0)
            row[MODEL_LABELS[m]] = round(float(vals_q.mean()), 6) if len(vals_q) > 0 else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)

    # Tambahkan cek monotonisitas per model
    mono_notes = {}
    for m in MODELS:
        col_vals = result[MODEL_LABELS[m]].tolist()
        diffs = [col_vals[i+1] - col_vals[i] for i in range(len(col_vals)-1)]
        if all(d >= -1e-9 for d in diffs):
            mono_notes[MODEL_LABELS[m]] = "monoton naik"
        elif all(d <= 1e-9 for d in diffs):
            mono_notes[MODEL_LABELS[m]] = "monoton turun"
        else:
            mono_notes[MODEL_LABELS[m]] = "tidak monoton"

    result["_note"] = ""  # placeholder
    return result, mono_notes


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

    # 2. Regresi tren kontinu
    print(f"\n[2] Regresi delta(NDCG) ~ log(interactions) (n_boot={N_BOOTSTRAP})")
    reg_df = run_trend_regression(dfs)
    reg_df.to_csv(RESULT_DIR / "track_a_regression.csv", index=False)
    print(reg_df[["label", "b1_slope", "b1_ci_lo_95", "b1_ci_hi_95",
                  "interpretation"]].to_string(index=False))

    # 3. NDCG per kuintil
    print("\n[3] NDCG@10 per kuintil x model")
    q_df, mono_notes = build_quintile_ndcg(dfs)
    q_df.to_csv(RESULT_DIR / "track_a_quintile_ndcg.csv", index=False)
    display_cols = ["quintile", "edge_lo", "edge_hi", "n_users"] + \
                   [MODEL_LABELS[m] for m in MODELS]
    print(q_df[display_cols].to_string(index=False))
    print("  Monotonisitas per model (Q1->Q5):")
    for lbl, note in mono_notes.items():
        print(f"    {lbl}: {note}")

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
