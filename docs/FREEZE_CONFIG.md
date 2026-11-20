# FREEZE_CONFIG -- Keputusan Pra-FinalTest

Dokumen ini mencatat keputusan yang ditetapkan **sebelum** data test diakses.
Semua nilai di sini tidak boleh diubah setelah FinalTest dijalankan.
Keputusan statistik di Bagian 4 dicatat karena pola ditemukan di validation.

---

## 1. Model Final dan Hyperparameter

| Model | Parameter final | Catatan |
|---|---|---|
| Popularity | positive_threshold = 4 | Tidak ada tuning |
| MF (TruncatedSVD) | n_components = 32, n_iter = 7, random_state = 42 | Dipilih via NDCG@10 pada validation |
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
per pengguna secara terpisah:

```
norm[i] = (skor[i] - min(skor)) / (max(skor) - min(skor))
```

Jika max == min (semua skor sama untuk user itu), norm = 0 untuk semua item.
Tujuan: mencegah satu komponen mendominasi karena perbedaan skala absolut.

---

## 3. Definisi Kelompok Aktivitas

Batas ditetapkan di `src/config.py` (`EVAL_CONFIG["activity_bins"]`)
dan diimplementasikan di `src/baseline_common.make_user_groups()`:

```python
pd.cut(train_interactions, bins=[0, 5, 8, inf],
       labels=["Low", "Medium", "High"], include_lowest=True)
```

| Kelompok | Rentang interaksi training | n_users | % dari total evaluable |
|---|---|---|---|
| Low | 1 -- 5 | 6 105 | 52.4% |
| Medium | 6 -- 8 | 2 364 | 20.3% |
| High | >= 9 | 3 190 | 27.4% |
| **Total** | | **11 659** | **100%** |

Batas ini ditetapkan berdasarkan pertimbangan desain di awal proyek
sebelum melihat distribusi data validasi.

---

## 4. Daftar Uji Konfirmatori

Ditetapkan sebelum FinalTest. Hanya uji-uji berikut yang diperlakukan
sebagai konfirmatori dengan koreksi Bonferroni.

**k = 7** (tujuh uji)
**alpha per uji = 0.05 / 7 = 0.007143**
**Tingkat CI = 99.29%**

### C1 -- C6: Paired bootstrap (level user, paired)

| # | Perbandingan | Kelompok | Metrik |
|---|---|---|---|
| C1 | Hybrid vs Popularity | Low | NDCG@10 |
| C2 | Hybrid vs Popularity | Medium | NDCG@10 |
| C3 | Hybrid vs Popularity | High | NDCG@10 |
| C4 | Hybrid vs MF | Low | NDCG@10 |
| C5 | Hybrid vs MF | Medium | NDCG@10 |
| C6 | Hybrid vs MF | High | NDCG@10 |

Alasan C1--C6: pertanyaan riset Track A mempertanyakan apakah model
yang lebih kompleks (Hybrid) mengungguli baseline pada setiap kelompok aktivitas.

### C7: Selisih-dari-selisih (DoD)

| # | Statistik | Metrik |
|---|---|---|
| C7 | (Hybrid - Popularity)_High minus (Hybrid - Popularity)_Low | NDCG@10 |

**Alasan C7:** pola ditemukan di validation -- Hybrid relatif lebih unggul
atas Popularity di kelompok High dibanding Low. Pola ini diuji secara
konfirmatori di test untuk mendeteksi interaksi model x kelompok aktivitas.

**Metode C7:** paired dalam kelompok (user sama untuk Hybrid dan Popularity),
independen antar kelompok (user High != user Low).

**Hasil di validation (DoD NDCG@10):**
- Hybrid-Popularity High: mean diff = +0.002839
- Hybrid-Popularity Low:  mean diff = -0.002469
- DoD = +0.005308, CI 99.29% = [+0.000186, +0.010553]
- Status: CI tidak memuat 0 (konfirmatori terpenuhi di validation)

Semua perbandingan lain berstatus **eksploratori** (CI 95%, tanpa Bonferroni).
Hasil konfirmatori validation tercatat di `results/track_a_bootstrap_ci.csv`
dan `results/track_a_diff_of_diff.csv`.

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
  bukan dari re-run penuh. Recall@10 dan NDCG@10 diverifikasi bit-identik.
- Semua baseline dapat dijalankan ulang dari nol; lihat README.md.
- File user_metrics lama (sebelum penambahan HitRate/MRR) tersimpan
  di `results/archive/`.
- Kolom `*_best` di `results/track_a_table.csv` dihasilkan oleh `idxmax()`
  per kelompok di `src/track_a_analysis.py`; tidak ada nilai yang di-set manual.

---

*Dokumen ini dibuat sebelum akses test set.
Tanggal commit pertama dapat diverifikasi di git log.*
