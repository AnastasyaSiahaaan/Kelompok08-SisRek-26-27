"""
Langkah 1 (rekonsiliasi):
  - Pindahkan _overall_metrics.csv dan _by_activity_group.csv lama ke results/archive/
  - Regenerasi dari _user_metrics.csv via recompute_from_user_metrics()
  - Regenerasi comparison_overall.csv dan comparison_by_activity_group.csv

Jalankan: python scripts/regenerate_summaries.py
"""
import sys
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_common import RESULT_DIR, recompute_from_user_metrics

ARCHIVE = RESULT_DIR / "archive"
ARCHIVE.mkdir(exist_ok=True)

MODELS = [
    "popularity",
    "mf",
    "tfidf_cbf",
    "hybrid",
    "popularity_recency",
    "item_knn",
    "hybrid_new",
]

# ---------------------------------------------------------------------------
# 1. Arsipkan file lama yang akan ditimpa
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: Arsipkan file lama ke results/archive/")
print("=" * 70)
archived = []
for model in MODELS:
    for suffix in ["_overall_metrics.csv", "_by_activity_group.csv"]:
        src = RESULT_DIR / f"{model}{suffix}"
        if src.exists():
            dst = ARCHIVE / f"{model}{suffix}"
            shutil.copy2(str(src), str(dst))
            archived.append(str(dst))
            print(f"  Copied: {src.name} -> archive/")

print(f"\n  Total file diarsipkan: {len(archived)}")

# ---------------------------------------------------------------------------
# 2. Regenerasi dari per-user metrics
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Regenerasi _overall_metrics.csv dan _by_activity_group.csv")
print("=" * 70)
print(f"\n{'Model':<20} {'N':>6} {'Recall@10':>12} {'NDCG@10':>12}")
print("-" * 54)
recomp_rows = []
recomp_grp_rows = []
for model in MODELS:
    um_path = RESULT_DIR / f"{model}_user_metrics.csv"
    if not um_path.exists():
        print(f"  {model:<20} MISSING user_metrics")
        continue
    overall, group, n = recompute_from_user_metrics(model)
    r = overall.get("Recall@10", np.nan)
    nd = overall.get("NDCG@10", np.nan)
    print(f"  {model:<20} {n:>6} {r:>12.6f} {nd:>12.6f}")
    # For comparison tables
    row = {"model": model, **overall}
    recomp_rows.append(row)
    for grp, gvals in group.items():
        um = pd.read_csv(um_path, dtype={"user_id": str})
        n_grp = len(um[um["activity_group"] == grp])
        recomp_grp_rows.append({"model": model, "activity_group": grp,
                                 "users": n_grp, **gvals})

# ---------------------------------------------------------------------------
# 3. Regenerasi comparison files dari hasil recomputed
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: Regenerasi comparison_overall.csv dan comparison_by_activity_group.csv")
print("=" * 70)

# Arsipkan comparison lama
for fname in ["comparison_overall.csv", "comparison_by_activity_group.csv"]:
    src = RESULT_DIR / fname
    if src.exists():
        shutil.copy2(str(src), str(ARCHIVE / fname))
        print(f"  Archived: {fname}")

# comparison_overall: baca dari *_overall_metrics.csv yang baru ditulis
comp_overall_rows = []
for model in MODELS:
    ov_path = RESULT_DIR / f"{model}_overall_metrics.csv"
    if ov_path.exists():
        df = pd.read_csv(ov_path)
        comp_overall_rows.append(df)

comp_overall = pd.concat(comp_overall_rows, ignore_index=True)
# Pastikan kolom konsisten
metric_cols = [c for c in ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]
               if c in comp_overall.columns]
comp_overall = comp_overall[["model"] + metric_cols]
comp_overall.to_csv(RESULT_DIR / "comparison_overall.csv", index=False)
print("\ncomparison_overall.csv:")
print(comp_overall.to_string(index=False))

# comparison_by_activity_group: baca dari *_by_activity_group.csv yang baru
comp_grp_rows = []
for model in MODELS:
    grp_path = RESULT_DIR / f"{model}_by_activity_group.csv"
    if grp_path.exists():
        df = pd.read_csv(grp_path)
        comp_grp_rows.append(df)

comp_grp = pd.concat(comp_grp_rows, ignore_index=True)
grp_metric_cols = [c for c in ["Recall_at_10", "NDCG_at_10", "HitRate_at_10", "MRR_at_10"]
                   if c in comp_grp.columns]
comp_grp = comp_grp[["model", "activity_group", "users"] + grp_metric_cols]
comp_grp.to_csv(RESULT_DIR / "comparison_by_activity_group.csv", index=False)
print("\ncomparison_by_activity_group.csv (pivot NDCG_at_10):")
pivot = comp_grp.pivot(index="model", columns="activity_group", values="NDCG_at_10")
print(pivot.to_string())

# ---------------------------------------------------------------------------
# 4. Verifikasi final
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: Verifikasi angka kunci")
print("=" * 70)
checks = [
    ("hybrid",    "NDCG@10", 0.009165459, 1e-5),
    ("hybrid",    "Recall@10", 0.014567127, 1e-5),
    ("tfidf_cbf", "NDCG@10", 0.007304246, 1e-5),
    ("tfidf_cbf", "Recall@10", 0.012568174, 1e-5),
    ("popularity","NDCG@10", 0.009222041, 1e-5),
    ("mf",        "NDCG@10", 0.007917352, 1e-5),
]
all_ok = True
for model, metric, expected, tol in checks:
    row = comp_overall[comp_overall["model"] == model]
    if row.empty:
        print(f"  MISSING: {model}")
        continue
    got = row[metric].iloc[0]
    ok = abs(got - expected) < tol
    status = "OK" if ok else f"FAIL (got {got:.9f}, expected {expected:.9f})"
    print(f"  {model:<20} {metric:<12} {status}")
    if not ok:
        all_ok = False

print(f"\nSemua cek: {'PASS' if all_ok else 'FAIL'}")
print("\nSelesai. File lama ada di results/archive/")
