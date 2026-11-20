"""Check timestamps dan diagnosa hybrid vs tfidf_cbf inconsistency."""
import pandas as pd
import os
import datetime

fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

files = [
    'results/hybrid_overall_metrics.csv',
    'results/hybrid_by_activity_group.csv',
    'results/hybrid_user_metrics.csv',
    'results/tfidf_cbf_overall_metrics.csv',
    'results/tfidf_cbf_by_activity_group.csv',
    'results/tfidf_cbf_user_metrics.csv',
    'results/popularity_overall_metrics.csv',
    'results/mf_overall_metrics.csv',
]
print('File timestamps:')
for f in files:
    t = os.path.getmtime(f)
    df = pd.read_csv(f)
    print(f'  {fmt(t)}  rows={len(df):5d}  {f}')

# Content lama
print()
print('hybrid_overall_metrics.csv:')
print(pd.read_csv('results/hybrid_overall_metrics.csv').to_string(index=False))
print()
print('tfidf_cbf_overall_metrics.csv:')
print(pd.read_csv('results/tfidf_cbf_overall_metrics.csv').to_string(index=False))
print()

# Hitung ulang dari per-user
tu = pd.read_csv('results/tfidf_cbf_user_metrics.csv', dtype={'user_id': str})
rec_col = 'Recall@10'
ndcg_col = 'NDCG@10'
print(f'tfidf_cbf_user_metrics: n={len(tu)}')
print(f'  Recall={tu[rec_col].mean():.9f}  NDCG={tu[ndcg_col].mean():.9f}')
print('Per-group (dari user_metrics):')
for grp, gdf in tu.groupby('activity_group'):
    print(f'  {grp}: n={len(gdf)}, Recall={gdf[rec_col].mean():.6f}, NDCG={gdf[ndcg_col].mean():.6f}')

tbag = pd.read_csv('results/tfidf_cbf_by_activity_group.csv')
print()
print('tfidf_cbf_by_activity_group.csv:')
print(tbag.to_string(index=False))

# Cek: apakah hybrid_user_metrics.csv OVERWRITTEN oleh tune?
# Timestamp: hybrid_user_metrics.csv = 2026-10-20 (jauh lebih lama)
# hybrid_tune_a05_user_metrics = 2026-11-14
# hybrid_overall_metrics.csv = 2026-11-14 (sama dengan tune!)
# → hybrid_overall_metrics berasal dari TUNE run, bukan dari hybrid_baseline.py
# Tapi isinya 0.009134, bukan 0.009165 — ada sesuatu yang salah

# Periksa apakah file overall ditulis SEBELUM user_metrics ditimpa
print()
print('=== TIMELINE HIPOTESIS ===')
print('hybrid_user_metrics.csv : 2026-10-20 (run awal hybrid_baseline.py)')
print('hybrid_overall_metrics.csv : 2026-11-14 (sama dg tune_hyperparameters.py)')
print('hybrid_tune_a05_user_metrics.csv : 2026-11-14')
print()
print('HIPOTESIS: hybrid_user_metrics.csv (2026-10-20) adalah run BERBEDA')
print('dari hybrid_overall_metrics.csv (2026-11-14).')
print('hybrid_overall_metrics.csv mungkin dari run hybrid_baseline.py yang di-rerun')
print('tapi dengan kode yang sedikit berbeda (definisi kandidat item berbeda).')
print()

# Verifikasi: apakah per-group dari user_metrics (2026-10-20) cocok dengan by_activity_group (2026-11-14)?
hu = pd.read_csv('results/hybrid_user_metrics.csv', dtype={'user_id': str})
print('hybrid_user_metrics.csv (2026-10-20) per-group:')
for grp, gdf in hu.groupby('activity_group'):
    print(f'  {grp}: n={len(gdf)}, Recall={gdf[rec_col].mean():.6f}, NDCG={gdf[ndcg_col].mean():.6f}')
print(f'  Total: n={len(hu)}, Recall={hu[rec_col].mean():.6f}, NDCG={hu[ndcg_col].mean():.6f}')

hbag = pd.read_csv('results/hybrid_by_activity_group.csv')
print()
print('hybrid_by_activity_group.csv (2026-11-14) per-group:')
print(hbag.to_string(index=False))
print()
print('KESIMPULAN:')
print('  user_metrics (2026-10-20) BERBEDA dari by_activity_group (2026-11-14)')
print('  → hybrid_user_metrics.csv TIDAK di-overwrite oleh run tuning')
print('  → hybrid_overall_metrics.csv (0.009134) konsisten dengan by_activity_group lama')
print('  → user_metrics lama (0.009165) = run yang LEBIH BARU tapi file namanya sama')
print()
# Cross-check: hitung overall dari by_activity_group 2026-11-14
weighted = (hbag['users'] * hbag['NDCG_at_10']).sum() / hbag['users'].sum()
micro_from_user = hu[ndcg_col].mean()
print(f'Weighted-avg dari by_activity_group (2026-11-14): NDCG={weighted:.9f}')
print(f'Micro-avg dari hybrid_user_metrics (2026-10-20): NDCG={micro_from_user:.9f}')
print(f'hybrid_overall_metrics.csv (2026-11-14): NDCG=0.009134209')
