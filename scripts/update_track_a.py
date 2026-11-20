"""
Update track_a_table.csv dengan HitRate@10 dan MRR@10 dari results/rerun/
(file resmi tidak ditimpa; rerun hanya untuk verifikasi dan melengkapi metrik).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_common import RESULT_DIR

RERUN_DIR = RESULT_DIR / "rerun"
CORE_MODELS = ["popularity", "mf", "tfidf_cbf", "hybrid"]
GROUPS = ["Low", "Medium", "High"]
METRICS = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]
LABELS = {"popularity": "Popularity", "mf": "MF (n=32)",
          "tfidf_cbf": "TF-IDF CBF", "hybrid": "Hybrid (a=0.5)"}

# Load rerun data (memiliki HitRate dan MRR)
user_data = {}
for model in CORE_MODELS:
    um = pd.read_csv(RERUN_DIR / f"{model}_user_metrics.csv", dtype={"user_id": str})
    user_data[model] = um
    # Verifikasi identitas dengan official (Recall dan NDCG)
    official = pd.read_csv(RESULT_DIR / f"{model}_user_metrics.csv", dtype={"user_id": str})
    merged = um.merge(official, on="user_id", suffixes=("_rerun","_off"))
    for col in ["Recall@10","NDCG@10"]:
        c_r, c_o = f"{col}_rerun", f"{col}_off"
        if c_r in merged.columns and c_o in merged.columns:
            assert (merged[c_r] - merged[c_o]).abs().max() < 1e-9, f"FAIL {model} {col}"

rows = []
for model in CORE_MODELS:
    um = user_data[model]
    for grp_label in (["Overall"] + GROUPS):
        sub = um if grp_label == "Overall" else um[um["activity_group"] == grp_label]
        row = {"model": model, "label": LABELS[model],
               "group": grp_label, "n_users": len(sub)}
        for m in METRICS:
            row[m] = sub[m].mean() if m in sub.columns else np.nan
        rows.append(row)

track_a = pd.DataFrame(rows)
track_a.to_csv(RESULT_DIR / "track_a_table.csv", index=False)

# Cetak tabel final
print("=" * 90)
print("TRACK A — TABEL FINAL (4 Model Inti, Semua Metrik)")
print("=" * 90)
print(f"\n{'Model':<16} {'Group':<8} {'N':>6} "
      f"{'Recall@10':>12} {'NDCG@10':>12} {'HitRate@10':>12} {'MRR@10':>10}")
print("-" * 80)
for model in CORE_MODELS:
    sub = track_a[track_a["model"] == model]
    for _, r in sub.iterrows():
        print(f"  {r['label']:<16} {r['group']:<8} {int(r['n_users']):>6} "
              f"{r['Recall@10']:>12.6f} {r['NDCG@10']:>12.6f} "
              f"{r['HitRate@10']:>12.6f} {r['MRR@10']:>10.6f}")
    print()

# RQ: selisih High - Low
print("RQ — NDCG@10 High minus Low:")
for model in CORE_MODELS:
    low  = track_a[(track_a["model"]==model)&(track_a["group"]=="Low")]["NDCG@10"].values[0]
    high = track_a[(track_a["model"]==model)&(track_a["group"]=="High")]["NDCG@10"].values[0]
    diff = high - low
    sign = "+" if diff >= 0 else ""
    print(f"  {LABELS[model]:<16}: {sign}{diff:.6f}")

print("\ntrack_a_table.csv updated with HitRate@10 and MRR@10.")
