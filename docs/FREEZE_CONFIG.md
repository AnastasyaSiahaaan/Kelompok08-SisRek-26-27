# FREEZE_CONFIG — Keputusan Pra-FinalTest

Dokumen ini mencatat keputusan yang ditetapkan **sebelum** data test diakses.
Semua nilai di sini tidak boleh diubah setelah FinalTest dijalankan.

---

## 1. Model Final dan Hyperparameter

| Model | Parameter final | Catatan |
|---|---|---|
| Popularity | positive_threshold = 4 | Tidak ada tuning |
| MF (TruncatedSVD) | n_components = 32, n_iter = 7, random_state = 42 | Dipilih via NDCG@10 pada validation; lihat experiment_log.csv |
| TF-IDF CBF | max_features = 100 000, ngram_range = (1,2), min_df = 2, stop_words = english | Dipilih via NDCG@10 pada validation |
| Hybrid | alpha = 0.5, normalisasi = minmax per-user | Dipilih via NDCG@10 pada validation |

**Random seed global:** 42 (dikodekan di `src/config.py`, `RANDOM_SEED`).

---

## 2. Definisi Istilah Model

### alpha (Hybrid)
`alpha` adalah bobot komponen MF dalam skor gabungan:

```
skor_hybrid[u, i] = alpha * norm_MF[u, i]  +  (1 - alpha) * norm_CBF[u, i]
```

`alpha = 0.5` berarti bobot sama antara MF dan TF-IDF CBF.

### Normalisasi min-max per-user
Sebelum digabungkan, vektor skor setiap komponen dinormalkan ke [0, 1]
**per pengguna secara terpisah**:

```
norm[i] = (skor[i] - min(skor)) / (max(skor) - min(skor))
```

Jika max == min (semua skor sama), norm = 0 untuk semua item.
Tujuan: mencegah satu komponen mendominasi karena perbedaan skala absolut.

---

## 3. Definisi Kelompok Aktivitas

Batas ditetapkan di `src/config.py` (`EVAL_CONFIG["activity_bins"]`)
dan diimplementasikan di `src/baseline_common.make_user_groups()`:

```python
pd.cut(train_interactions, bins=[0, 5, 8, inf], labels=["Low","Medium","High"],
       include_lowest=True)
```

| Kelompok | Rentang interaksi training | n_users | % dari total evaluable |
|---|---|---|---|
| Low | 1 – 5 | 6 105 | 52.4% |
| Medium | 6 – 8 | 2 364 | 20.3% |
| High | >= 9 | 3 190 | 27.4% |
| **Total** | | **11 659** | **100%** |

Batas ini ditetapkan berdasarkan pertimbangan desain di awal proyek
(bukan setelah melihat distribusi validasi).

---

## 4. Daftar Perbandingan Konfirmatori

Ditetapkan sebelum FinalTest. Hanya perbandingan berikut yang diperlakukan
sebagai konfirmatori dengan koreksi Bonferroni.

**k = 6** (enam sel konfirmatori)
**alpha per sel = 0.05 / 6 = 0.008333**
**Tingkat CI = 99.17%**

| # | Perbandingan | Kelompok | Metrik |
|---|---|---|---|
| C1 | Hybrid vs Popularity | Low | NDCG@10 |
| C2 | Hybrid vs Popularity | Medium | NDCG@10 |
| C3 | Hybrid vs Popularity | High | NDCG@10 |
| C4 | Hybrid vs MF | Low | NDCG@10 |
| C5 | Hybrid vs MF | Medium | NDCG@10 |
| C6 | Hybrid vs MF | High | NDCG@10 |

Semua perbandingan lain (termasuk Recall@10, HitRate@10, MRR@10,
dan perbandingan lain) berstatus **eksploratori** (CI 95%, tanpa Bonferroni).

Hasil konfirmatori saat ini (validation) tercatat di
`results/track_a_bootstrap_ci.csv` (kolom `analysis_type`).

---

## 5. Protokol Evaluasi

- Data split: temporal, disediakan oleh panitia, tidak di-reshuffle.
- Evaluable users: `data/supporting/evaluable_users_validation_warm.csv`.
- Candidate set: `data/raw/candidate_items_warm.csv` (sama untuk semua model).
- Item relevan: rating validation >= 4.
- Seen-item masking: item yang ada di training dikeluarkan sebelum Top-10 dipilih.
- Metrik utama: Recall@10 dan NDCG@10 (level pengguna, lalu rata-rata).

---

## 6. Catatan Reprodusibilitas

- **HitRate@10 dan MRR@10 untuk Hybrid** dihitung dari
  `results/hybrid_recommendations_top10.csv` via `src/add_hitrate_mrr.py`,
  bukan dari re-run penuh. Recall@10 dan NDCG@10 diverifikasi identik.
- Semua baseline dapat dijalankan ulang dari nol; lihat README.md
  bagian "Running the Experiments".
- File arsip user_metrics lama tersimpan di `results/archive/`.

---

*Dokumen ini dibuat sebelum akses test set. Tanggal commit pertama
dapat diverifikasi di git log.*
