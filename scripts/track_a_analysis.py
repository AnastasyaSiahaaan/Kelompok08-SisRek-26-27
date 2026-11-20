"""
Track A - Tabel lengkap 4 model inti:
  Overall + Low / Medium / High
  Metrik: Recall@10, NDCG@10, HitRate@10, MRR@10
  Bootstrap CI 95% paired user-level (1000 resample, seed=42)
  RQ metrics: selisih NDCG High-Low, rata-rata popularitas item rekomendasi per grup
  Figure: batang NDCG@10 per grup dengan error bar 95% CI

Output:
  results/track_a_table.csv
  results/track_a_bootstrap_ci.csv
  results/track_a_popularity.csv
  results/track_a_ndcg_bar.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_common import RESULT_DIR, RAW_DIR

SEED = 42
N_BOOT = 1000
rng = np.random.default_rng(SEED)

CORE_MODELS = ["popularity", "mf", "tfidf_cbf", "hybrid"]
GROUPS = ["Low", "Medium", "High"]
METRICS = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]

# ---------------------------------------------------------------------------
# 1. Load per-user metrics for 4 core models
# ---------------------------------------------------------------------------
print("=" * 70)
print("TRACK A: 4 Model Inti x Overall/Low/Medium/High")
print("=" * 70)

user_data = {}
for model in CORE_MODELS:
    um = pd.read_csv(RESULT_DIR / f"{model}_user_metrics.csv", dtype={"user_id": str})
    user_data[model] = um

# ---------------------------------------------------------------------------
# 2. Activity group definition (from training only)
# ---------------------------------------------------------------------------
from baseline_common import make_user_groups
from pathlib import Path as _Path

activity = pd.read_csv(
    _Path("data/supporting/user_activity_train.csv"),
    dtype={"user_id": str}
)
activity = make_user_groups(activity)
print("\nDefinisi kelompok (dari data training saja):")
print("  Low    : train_interactions in [1, 5]  (pd.cut, include_lowest=True)")
print("  Medium : train_interactions in (5, 8]")
print("  High   : train_interactions in (8, inf)")
print()
grp_counts = activity["activity_group"].value_counts().reindex(["Low","Medium","High"])
for grp in ["Low","Medium","High"]:
    sub = activity[activity["activity_group"] == grp]
    mn = sub["train_interactions"].min()
    mx = sub["train_interactions"].max()
    n  = len(sub)
    print(f"  {grp:6}: n={n:5d}, interactions [{mn:2d}, {mx:3d}]")

# ---------------------------------------------------------------------------
# 3. Main metric table (micro-average per group)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABEL METRIK UTAMA (micro-average per-user)")
print("=" * 70)

rows = []
for model in CORE_MODELS:
    um = user_data[model]
    for grp_label in (["Overall"] + GROUPS):
        if grp_label == "Overall":
            sub = um
        else:
            sub = um[um["activity_group"] == grp_label]
        row = {"model": model, "group": grp_label, "n_users": len(sub)}
        for m in METRICS:
            if m in sub.columns:
                row[m] = sub[m].mean()
            else:
                row[m] = np.nan
        rows.append(row)

track_a = pd.DataFrame(rows)
track_a.to_csv(RESULT_DIR / "track_a_table.csv", index=False)

print(f"\n{'Model':<12} {'Group':<8} {'N':>6} "
      f"{'Recall@10':>12} {'NDCG@10':>12} {'HitRate@10':>12} {'MRR@10':>10}")
print("-" * 76)
for _, r in track_a.iterrows():
    ndcg = f"{r['NDCG@10']:.6f}" if not np.isnan(r.get('NDCG@10', np.nan)) else "N/A"
    rec  = f"{r['Recall@10']:.6f}" if not np.isnan(r.get('Recall@10', np.nan)) else "N/A"
    hr   = f"{r.get('HitRate@10',np.nan):.6f}" if not np.isnan(r.get('HitRate@10',np.nan)) else "N/A"
    mrr  = f"{r.get('MRR@10',np.nan):.6f}" if not np.isnan(r.get('MRR@10',np.nan)) else "N/A"
    print(f"  {r['model']:<12} {r['group']:<8} {int(r['n_users']):>6} "
          f"{rec:>12} {ndcg:>12} {hr:>12} {mrr:>10}")
    if r['group'] == 'High':
        print()

# ---------------------------------------------------------------------------
# 4. RQ Metrics: selisih High - Low NDCG@10
# ---------------------------------------------------------------------------
print("=" * 70)
print("RQ METRICS: NDCG@10 High - Low (perbedaan antar kelompok aktivitas)")
print("=" * 70)
for model in CORE_MODELS:
    low_ndcg  = track_a[(track_a["model"]==model) & (track_a["group"]=="Low")]["NDCG@10"].values[0]
    high_ndcg = track_a[(track_a["model"]==model) & (track_a["group"]=="High")]["NDCG@10"].values[0]
    diff = high_ndcg - low_ndcg
    sign = "+" if diff > 0 else ""
    print(f"  {model:<12}: High={high_ndcg:.6f}, Low={low_ndcg:.6f}, "
          f"High-Low={sign}{diff:.6f}")

# ---------------------------------------------------------------------------
# 5. Popularitas item yang direkomendasikan (dari training count)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("POPULARITAS ITEM YANG DIREKOMENDASIKAN (rata-rata train_count per grup)")
print("=" * 70)

train = pd.read_csv("data/raw/interactions_train.csv", dtype={"user_id": str, "item_id": str})
item_pop = train.groupby("item_id").size().rename("train_count").reset_index()
item_pop_dict = item_pop.set_index("item_id")["train_count"].to_dict()

pop_rows = []
for model in CORE_MODELS:
    um = user_data[model]
    # Load rankings top-10 if available, else use user_metrics approach
    # We'll compute avg popularity of recommended items via rankings files if available
    recs_file = RESULT_DIR / f"{model}_recommendations_top10.csv"
    if recs_file.exists():
        recs_df = pd.read_csv(recs_file, dtype={"user_id": str, "item_id": str})
        recs_df = recs_df.merge(um[["user_id","activity_group"]], on="user_id", how="left")
        recs_df["pop"] = recs_df["item_id"].map(item_pop_dict).fillna(0)
        for grp_label in (["Overall"] + GROUPS):
            if grp_label == "Overall":
                sub = recs_df
            else:
                sub = recs_df[recs_df["activity_group"] == grp_label]
            pop_rows.append({"model": model, "group": grp_label,
                             "avg_pop": sub["pop"].mean(),
                             "n_recs": len(sub)})
    else:
        pop_rows.append({"model": model, "group": "N/A", "avg_pop": np.nan, "n_recs": 0,
                         "note": "no top10 file"})

if pop_rows:
    pop_df = pd.DataFrame(pop_rows)
    pop_df.to_csv(RESULT_DIR / "track_a_popularity.csv", index=False)
    if "note" not in pop_df.columns or pop_df["note"].isna().all():
        print(f"\n{'Model':<12} {'Group':<8} {'Avg Train Pop':>14} {'N recs':>8}")
        print("-" * 48)
        for _, r in pop_df.iterrows():
            if not np.isnan(r.get("avg_pop", np.nan)):
                print(f"  {r['model']:<12} {r['group']:<8} {r['avg_pop']:>14.2f} {int(r['n_recs']):>8}")
    else:
        print("  Top-10 recommendation files tidak tersedia untuk semua model.")
        print("  Popularitas item tidak bisa dihitung tanpa file rekomendasi.")

# ---------------------------------------------------------------------------
# 6. Bootstrap CI — paired user-level
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"BOOTSTRAP CI 95% (paired user-level, n={N_BOOT}, seed={SEED})")
print("=" * 70)

# Pairs: each model vs Popularity (baseline)
# Plus adjacent comparisons
comparison_pairs = [
    ("mf", "popularity"),
    ("tfidf_cbf", "popularity"),
    ("hybrid", "popularity"),
    ("hybrid", "mf"),
    ("hybrid", "tfidf_cbf"),
]

ci_rows = []
for metric in ["Recall@10", "NDCG@10"]:
    for (model_a, model_b) in comparison_pairs:
        for grp_label in (["Overall"] + GROUPS):
            ua = user_data[model_a]
            ub = user_data[model_b]
            if grp_label != "Overall":
                ua = ua[ua["activity_group"] == grp_label]
                ub = ub[ub["activity_group"] == grp_label]
            # Align on user_id
            merged = ua[["user_id", metric]].merge(
                ub[["user_id", metric]], on="user_id", suffixes=("_a","_b")
            )
            if len(merged) == 0:
                continue
            vals_a = merged[f"{metric}_a"].values
            vals_b = merged[f"{metric}_b"].values
            obs_diff = vals_a.mean() - vals_b.mean()

            # Bootstrap
            n = len(merged)
            boot_diffs = np.empty(N_BOOT)
            for i in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                boot_diffs[i] = vals_a[idx].mean() - vals_b[idx].mean()

            lo = np.percentile(boot_diffs, 2.5)
            hi = np.percentile(boot_diffs, 97.5)
            sig = "Ya (+)" if lo > 0 else ("Ya (-)" if hi < 0 else "Tidak")
            ci_rows.append({
                "comparison": f"{model_a} vs {model_b}",
                "model_a": model_a, "model_b": model_b,
                "group": grp_label, "metric": metric,
                "mean_a": vals_a.mean(), "mean_b": vals_b.mean(),
                "diff": obs_diff, "ci_lo": lo, "ci_hi": hi,
                "significant": sig, "n_users": n
            })
            print(f"  {model_a} vs {model_b:12} | {grp_label:7} | {metric:12} | "
                  f"diff={obs_diff:+.6f} CI=[{lo:+.6f},{hi:+.6f}] {sig}")

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(RESULT_DIR / "track_a_bootstrap_ci.csv", index=False)
print(f"\nBootstrap CI disimpan ke results/track_a_bootstrap_ci.csv")

# ---------------------------------------------------------------------------
# 7. Figure: batang NDCG@10 per grup dengan error bar 95% CI
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FIGURE: NDCG@10 per kelompok aktivitas (dengan 95% CI error bar)")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    "popularity": "#4e79a7",
    "mf":         "#f28e2b",
    "tfidf_cbf":  "#59a14f",
    "hybrid":     "#e15759",
}
labels = {
    "popularity": "Popularity",
    "mf":         "MF (n=32)",
    "tfidf_cbf":  "TF-IDF CBF",
    "hybrid":     "Hybrid (α=0.5)",
}
x_groups = GROUPS
x = np.arange(len(x_groups))
n_models = len(CORE_MODELS)
width = 0.18
offsets = np.linspace(-(n_models-1)/2*width, (n_models-1)/2*width, n_models)

for mi, model in enumerate(CORE_MODELS):
    um = user_data[model]
    means = []
    errs_lo = []
    errs_hi = []
    for grp in x_groups:
        sub = um[um["activity_group"] == grp]["NDCG@10"].values
        mean_val = sub.mean()
        means.append(mean_val)
        # 95% CI for this group mean (bootstrap)
        boot_means = np.array([sub[rng.integers(0, len(sub), len(sub))].mean()
                                for _ in range(N_BOOT)])
        lo_err = mean_val - np.percentile(boot_means, 2.5)
        hi_err = np.percentile(boot_means, 97.5) - mean_val
        errs_lo.append(lo_err)
        errs_hi.append(hi_err)

    bars = ax.bar(x + offsets[mi], means, width,
                  yerr=[errs_lo, errs_hi],
                  label=labels[model],
                  color=colors[model],
                  alpha=0.85,
                  error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"})

ax.set_xlabel("Kelompok Aktivitas User (berdasarkan training)", fontsize=12)
ax.set_ylabel("NDCG@10 (micro-average, validation)", fontsize=12)
ax.set_title("Track A: NDCG@10 per Kelompok Aktivitas User\n"
             "4 Model Inti — 95% Bootstrap CI (paired user-level, n=1000, seed=42)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{g}\n(n={grp_counts[g]:,})" for g in x_groups], fontsize=11)
ax.legend(loc="upper right", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Annotate group boundaries
ax.text(0.01, 0.97, "Low: 1–5 interaksi | Medium: 6–8 | High: ≥9",
        transform=ax.transAxes, fontsize=8.5, va="top",
        color="gray", style="italic")

plt.tight_layout()
fig_path = RESULT_DIR / "track_a_ndcg_bar.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure disimpan ke {fig_path}")

print("\n" + "=" * 70)
print("SELESAI. File output:")
print("  results/track_a_table.csv")
print("  results/track_a_bootstrap_ci.csv")
print("  results/track_a_popularity.csv")
print("  results/track_a_ndcg_bar.png")
print("=" * 70)
