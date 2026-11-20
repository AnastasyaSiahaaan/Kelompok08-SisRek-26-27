"""Summary tables: MF multi-seed dan Hybrid alpha."""
import re
import pandas as pd
import numpy as np

log = pd.read_csv('results/experiment_log.csv')

# ---- Hybrid alpha ----
hyb = log[log['model'] == 'hybrid'].copy()
hyb['alpha'] = hyb['params'].apply(
    lambda s: float(re.search(r"'alpha': ([0-9.]+)", s).group(1))
)
hyb = hyb.sort_values('alpha').reset_index(drop=True)
print("=== TABEL HYBRID ALPHA (Recall@10, NDCG@10) ===")
print(hyb[['alpha', 'recall_at_10', 'ndcg_at_10']].to_string(index=False))
best_ndcg_a = hyb.loc[hyb['ndcg_at_10'].idxmax()]
best_rec_a  = hyb.loc[hyb['recall_at_10'].idxmax()]
print(f"\nBest NDCG@10: alpha={best_ndcg_a['alpha']}  NDCG={best_ndcg_a['ndcg_at_10']:.6f}  Recall={best_ndcg_a['recall_at_10']:.6f}")
print(f"Best Recall@10: alpha={best_rec_a['alpha']}  Recall={best_rec_a['recall_at_10']:.6f}  NDCG={best_rec_a['ndcg_at_10']:.6f}")

# ---- MF multi-seed ----
mf = log[log['model'] == 'mf'].copy()
mf['n_comp'] = mf['params'].apply(
    lambda s: int(re.search(r"'n_components': ([0-9]+)", s).group(1))
)
mf['seed'] = mf['run_id'].apply(
    lambda s: int(re.search(r'seed([0-9]+)', s).group(1))
)
mf = mf.sort_values(['n_comp', 'seed']).reset_index(drop=True)

print("\n=== TABEL MF PER SEED ===")
print(mf[['n_comp', 'seed', 'recall_at_10', 'ndcg_at_10']].to_string(index=False))

grp = mf.groupby('n_comp').agg(
    recall_mean=('recall_at_10', 'mean'),
    recall_std=('recall_at_10', 'std'),
    ndcg_mean=('ndcg_at_10', 'mean'),
    ndcg_std=('ndcg_at_10', 'std'),
    n_seeds=('seed', 'count')
).reset_index()

print("\n=== TABEL MF MEAN+/-STD (3 seeds: 42, 123, 456) ===")
for _, row in grp.iterrows():
    print(f"  n_comp={int(row['n_comp']):3d} | "
          f"Recall@10={row['recall_mean']:.6f}+/-{row['recall_std']:.6f} | "
          f"NDCG@10={row['ndcg_mean']:.6f}+/-{row['ndcg_std']:.6f} | "
          f"n_seeds={int(row['n_seeds'])}")

best = grp.loc[grp['ndcg_mean'].idxmax()]
print(f"\nBest n_comp (mean NDCG@10): {int(best['n_comp'])}  "
      f"NDCG@10={best['ndcg_mean']:.6f}+/-{best['ndcg_std']:.6f}  "
      f"Recall@10={best['recall_mean']:.6f}+/-{best['recall_std']:.6f}")

# ---- Rekonsiliasi hybrid angka lama ----
print("\n=== REKONSILIASI HYBRID alpha=0.5 ===")
# Angka lama: hybrid_overall_metrics.csv
old = pd.read_csv('results/hybrid_overall_metrics.csv')
print(f"hybrid_overall_metrics.csv (run awal hybrid_baseline.py):")
print(f"  Recall@10={old['Recall@10'].iloc[0]:.6f}  NDCG@10={old['NDCG@10'].iloc[0]:.6f}")

# Angka tune: hybrid_tune_a05_overall_metrics.csv
tune = pd.read_csv('results/hybrid_tune_a05_overall_metrics.csv')
print(f"hybrid_tune_a05_overall_metrics.csv (tune_hyperparameters.py):")
print(f"  Recall@10={tune['Recall@10'].iloc[0]:.6f}  NDCG@10={tune['NDCG@10'].iloc[0]:.6f}")

# Angka extended: hybrid_tune_a05 (sama file)
# Check user count
old_u  = pd.read_csv('results/hybrid_user_metrics.csv')
tune_u = pd.read_csv('results/hybrid_tune_a05_user_metrics.csv')
print(f"\nJumlah user dievaluasi:")
print(f"  hybrid_user_metrics.csv       : {len(old_u)}")
print(f"  hybrid_tune_a05_user_metrics  : {len(tune_u)}")

# Check common users
common = set(old_u['user_id']).intersection(set(tune_u['user_id']))
print(f"  User yg sama di keduanya     : {len(common)}")

# Compare per-user NDCG
merged = old_u.merge(tune_u, on='user_id', suffixes=('_old','_new'))
print(f"\nPer-user NDCG@10 diff (old vs new) untuk {len(merged)} user bersama:")
print(f"  Max diff: {(merged['NDCG@10_old']-merged['NDCG@10_new']).abs().max():.8f}")
print(f"  Mean diff: {(merged['NDCG@10_old']-merged['NDCG@10_new']).mean():.8f}")
print(f"  Identical: {((merged['NDCG@10_old']-merged['NDCG@10_new']).abs() < 1e-10).all()}")
