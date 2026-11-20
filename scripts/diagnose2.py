"""Three-way comparison of hybrid output files."""
import pandas as pd
import os
import datetime

fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

hu   = pd.read_csv('results/hybrid_user_metrics.csv', dtype={'user_id': str})
hbag = pd.read_csv('results/hybrid_by_activity_group.csv')
tbag = pd.read_csv('results/hybrid_tune_a05_by_activity_group.csv')
hov  = pd.read_csv('results/hybrid_overall_metrics.csv')
tov  = pd.read_csv('results/hybrid_tune_a05_overall_metrics.csv')
tfu  = pd.read_csv('results/hybrid_tune_a05_user_metrics.csv', dtype={'user_id': str})

print('=== TIMESTAMPS ===')
for name, f in [
    ('hybrid_user_metrics', 'results/hybrid_user_metrics.csv'),
    ('hybrid_by_activity_group', 'results/hybrid_by_activity_group.csv'),
    ('hybrid_overall_metrics', 'results/hybrid_overall_metrics.csv'),
    ('hybrid_tune_a05_user_metrics', 'results/hybrid_tune_a05_user_metrics.csv'),
    ('hybrid_tune_a05_by_activity_group', 'results/hybrid_tune_a05_by_activity_group.csv'),
    ('hybrid_tune_a05_overall_metrics', 'results/hybrid_tune_a05_overall_metrics.csv'),
]:
    t = os.path.getmtime(f)
    print(f'  {fmt(t)}  {name}')

print()
print('=== hybrid_user_metrics.csv (2026-10-20) per-group ===')
for grp in ['Low', 'Medium', 'High']:
    gdf = hu[hu['activity_group'] == grp]
    ndcg_col = 'NDCG@10'
    rec_col = 'Recall@10'
    print(f'  {grp}: n={len(gdf)}, Recall={gdf[rec_col].mean():.6f}, NDCG={gdf[ndcg_col].mean():.6f}')
print(f'  Total: n={len(hu)}, Recall={hu[rec_col].mean():.6f}, NDCG={hu[ndcg_col].mean():.6f}')

print()
print('=== hybrid_by_activity_group.csv (2026-11-14) ===')
print(hbag.to_string(index=False))
print(f'  weighted NDCG: {(hbag.users * hbag.NDCG_at_10).sum() / hbag.users.sum():.9f}')

print()
print('=== hybrid_tune_a05_by_activity_group.csv (2026-11-14) ===')
print(tbag.to_string(index=False))
print(f'  weighted NDCG: {(tbag.users * tbag.NDCG_at_10).sum() / tbag.users.sum():.9f}')

print()
print('=== hybrid_overall_metrics.csv (2026-11-14) ===')
print(hov.to_string(index=False))

print()
print('=== hybrid_tune_a05_overall_metrics.csv (2026-11-14) ===')
print(tov.to_string(index=False))

print()
print('=== DIAGNO: apakah hybrid_by_activity_group IS hybrid_tune_a05 by_activity_group? ===')
m = hbag.merge(tbag, on='activity_group', suffixes=('_hbag','_tbag'))
ndcg_match = (abs(m['NDCG_at_10_hbag'] - m['NDCG_at_10_tbag']) < 1e-9).all()
rec_match  = (abs(m['Recall_at_10_hbag'] - m['Recall_at_10_tbag']) < 1e-9).all()
print(f'  NDCG identical: {ndcg_match}')
print(f'  Recall identical: {rec_match}')
print(m[['activity_group','NDCG_at_10_hbag','NDCG_at_10_tbag','Recall_at_10_hbag','Recall_at_10_tbag']].to_string(index=False))

print()
print('=== DIAGNO: hybrid_user_metrics vs hybrid_tune_a05_user_metrics ===')
mug = hu.merge(tfu, on='user_id', suffixes=('_orig','_tune'))
ndcg_col_o = 'NDCG@10_orig'
ndcg_col_t = 'NDCG@10_tune'
rec_col_o = 'Recall@10_orig'
rec_col_t = 'Recall@10_tune'
diff_ndcg = (mug[ndcg_col_o] - mug[ndcg_col_t]).abs()
diff_rec  = (mug[rec_col_o]  - mug[rec_col_t]).abs()
print(f'  Users matched: {len(mug)}')
print(f'  Max NDCG diff: {diff_ndcg.max():.10f}')
print(f'  Max Recall diff: {diff_rec.max():.10f}')
print(f'  Users with NDCG diff > 1e-9: {(diff_ndcg > 1e-9).sum()}')

print()
print('=== KESIMPULAN: SUMBER hybrid_overall_metrics (0.009134) ===')
# Hitung weighted dari hbag
wt_hbag = (hbag['users'] * hbag['NDCG_at_10']).sum() / hbag['users'].sum()
wt_tbag = (tbag['users'] * tbag['NDCG_at_10']).sum() / tbag['users'].sum()
micro_hu = hu['NDCG@10'].mean()
micro_tfu = tfu['NDCG@10'].mean()
print(f'  weighted-avg dari hybrid_by_activity_group:       {wt_hbag:.9f}')
print(f'  weighted-avg dari hybrid_tune_a05_by_act_group:   {wt_tbag:.9f}')
print(f'  micro-avg dari hybrid_user_metrics (2026-10-20):  {micro_hu:.9f}')
print(f'  micro-avg dari hybrid_tune_a05_user_metrics:      {micro_tfu:.9f}')
print(f'  hybrid_overall_metrics.csv says:                   0.009134209')
