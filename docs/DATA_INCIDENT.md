# Data Incident — commit 3a79cf2

**Tanggal kejadian:** Sekitar 2026-11-13 (commit `3a79cf2`, "Add W10 validation evaluation results")
**Penulis commit:** Yolanda Septania Saragih

## Apa yang terjadi
Commit `3a79cf2` menimpa `hybrid_overall_metrics.csv`, `hybrid_by_activity_group.csv`,
`tfidf_cbf_overall_metrics.csv`, dan `tfidf_cbf_by_activity_group.csv` dengan output
dari run berbeda (konfigurasi atau lingkungan yang belum diverifikasi), sementara file
`*_user_metrics.csv` tidak tertimpa (tetap dari run pertama yang benar).

## Dampak
Angka ringkasan (overall + per grup) untuk Hybrid dan TF-IDF tidak konsisten dengan
per-user metrics — terlihat dari selisih NDCG@10 Hybrid: 0.009134 (dari file ringkasan)
vs 0.009165 (dari micro-average `hybrid_user_metrics.csv`).

## Bagaimana ditemukan
Perbandingan micro-average per-user dari `hybrid_tune_a05_user_metrics.csv` (alpha=0.5)
dengan `hybrid_overall_metrics.csv` menghasilkan selisih 3.1e-5 — tidak bisa dijelaskan
sebagai efek pembulatan atau pembobotan kelompok.

## Perbaikan
Semua `*_overall_metrics.csv` dan `*_by_activity_group.csv` untuk 7 model diregenerasi
dari `*_user_metrics.csv` menggunakan fungsi tunggal `recompute_from_user_metrics()` di
`src/baseline_common.py` (commit `ef392e5`, 2026-11-20). File lama dipindahkan ke
`results/archive/` (tidak dihapus). Angka resmi: Hybrid NDCG@10 = **0.009165**.
