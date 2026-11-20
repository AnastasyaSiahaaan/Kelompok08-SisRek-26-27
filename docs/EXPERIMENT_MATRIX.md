# EXPERIMENT MATRIX

Semua run eksperimen yang telah dijalankan pada validation.
Detail lengkap (params, metrik, timestamp, git commit) ada di
`results/experiment_log.csv`.

---

## Model Inti (Track A)

| ID | Model | Hyperparameter | Recall@10 | NDCG@10 | Status |
|---|---|---|---|---|---|
| E1 | Popularity | positive_threshold=4 | 0.013499 | 0.009222 | **Final** |
| E2a | MF (n=8) | n_components=8, seed=42 | 0.011825 | 0.007602 | Tuning |
| E2b | MF (n=16) | n_components=16, seed=42 | 0.012090 | 0.007428 | Tuning |
| E2c | MF (n=32) | n_components=32, seed=42 | 0.013094 | 0.007917 | **Final** |
| E2d | MF (n=64) | n_components=64, seed=42 | 0.011903 | 0.007428 | Tuning |
| E2e | MF (n=128) | n_components=128, seed=42 | 0.010780 | 0.006753 | Tuning |
| E3a | TF-IDF CBF (50k) | max_features=50000 | 0.012184 | 0.007125 | Tuning |
| E3b | TF-IDF CBF (100k) | max_features=100000 | 0.012568 | 0.007304 | **Final** |
| E4a | Hybrid (a=0.1) | alpha=0.1 | -- | -- | Tuning |
| E4b | Hybrid (a=0.2) | alpha=0.2 | -- | -- | Tuning |
| E4c | Hybrid (a=0.3) | alpha=0.3 | 0.013861 | 0.008226 | Tuning |
| E4d | Hybrid (a=0.4) | alpha=0.4 | -- | -- | Tuning |
| E4e | Hybrid (a=0.5) | alpha=0.5 | 0.014567 | 0.009165 | **Final** |
| E4f | Hybrid (a=0.6) | alpha=0.6 | -- | -- | Tuning |
| E4g | Hybrid (a=0.7) | alpha=0.7 | 0.014421 | 0.008622 | Tuning |
| E4h | Hybrid (a=0.8) | alpha=0.8 | -- | -- | Tuning |
| E4i | Hybrid (a=0.9) | alpha=0.9 | -- | -- | Tuning |

Kriteria seleksi: NDCG@10 pada validation.
Run E4a-E4i dilakukan pada grid alpha {0.1,...,0.9}; nilai alpha=0.5
dipilih karena tertinggi. Detail lengkap di `results/experiment_log.csv`.

---

## Model Appendix (bukan model inti Track A)

| ID | Model | Hyperparameter | Recall@10 | NDCG@10 | Status |
|---|---|---|---|---|---|
| A1 | Pop Recency (90d) | window_days=90 | 0.020700 | 0.012222 | Tuning |
| A2 | Pop Recency (180d) | window_days=180 | 0.021692 | 0.012917 | Appendix final |
| A3 | Pop Recency (365d) | window_days=365 | 0.020443 | 0.012495 | Tuning |
| A4 | Item-kNN (s=100) | shrinkage=100 | 0.018812 | 0.012351 | Tuning |
| A5 | Item-kNN (s=300) | shrinkage=300 | 0.019881 | 0.012987 | Tuning |
| A6 | Item-kNN (s=500) | shrinkage=500 | 0.019885 | 0.013118 | Appendix final |
| A7 | Item-kNN (s=1000) | shrinkage=1000 | 0.019466 | 0.013081 | Tuning |
| A8 | Item-kNN (s=2000) | shrinkage=2000 | 0.019511 | 0.013173 | Tuning |
| A9 | Hybrid New (best) | w_knn=0.4, w_cbf=0.4, w_pop6m=0.2 | 0.020883 | 0.012311 | Appendix final |

Hybrid New diuji dengan 27 kombinasi bobot (grid search);
semua dicatat di `results/experiment_log.csv`.

---

## Analisis Track A

| ID | Analisis | Output | Status |
|---|---|---|---|
| T1 | Tabel metrik 4x4 | track_a_table_final.csv | Final |
| T2 | Bootstrap CI paired (k=7 Bonferroni) | track_a_bootstrap_ci.csv | Final |
| T3 | CI independen High-Low | track_a_indep_ci.csv | Final |
| T4 | Selisih-dari-selisih (DoD) | track_a_diff_of_diff.csv | Final |
| T5 | Regresi delta ~ log(interactions) | track_a_regression.csv | Final |
| T6 | NDCG per kuintil | track_a_quintile_ndcg.csv | Final |
| T7 | Sensitivitas batas kelompok | track_a_sensitivity.csv | Final |
| T8 | Popularitas item relevan per kelompok | track_a_relevant_pop.csv | Final |
| T9 | Overlap Top-10 Hybrid vs MF/TF-IDF | track_a_overlap.csv | Final |
| T10 | Figure NDCG bar chart | track_a_ndcg_bar.png | Final |

---

## Multi-Seed MF (robustness check)

MF dijalankan dengan seed {42, 123, 456} untuk n_components {8,16,32,64,128}.
Hasil di `results/mf_tune_*_s{123,456}_*.csv`.
Seed resmi untuk hasil final: 42.
