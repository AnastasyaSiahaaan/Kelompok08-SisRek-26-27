# FREEZE_CONFIG -- Keputusan Pra-FinalTest

Dokumen ini mencatat keputusan yang ditetapkan **sebelum** data test diakses.
Semua nilai tidak boleh diubah setelah FinalTest dijalankan.

---

## 1. Model Final dan Hyperparameter

| Model | Parameter final | Catatan |
|---|---|---|
| Popularity | positive_threshold = 4 | Tidak ada tuning |
| MF (TruncatedSVD) | n_components = 32, n_iter = 7, random_state = 42 | Dipilih via NDCG@10 pada validation |
| TF-IDF CBF | max_features = 100 000, ngram_range = (1,2), min_df = 2, stop_words = english | Dipilih via NDCG@10 pada validation |
| Hybrid | alpha = 0.5, normalisasi = minmax per-user | Dipilih via NDCG@10 pada validation |

**Random seed global:** 42 (`src/config.py`, `RANDOM_SEED`).

---

## 2. Definisi Istilah Model

### alpha (Hybrid)

`alpha` adalah bobot komponen MF dalam skor gabungan:

    skor_hybrid[u, i] = alpha * norm_MF[u, i]  +  (1 - alpha) * norm_CBF[u, i]

`alpha = 0.5` berarti bobot MF dan TF-IDF CBF sama.

### Normalisasi min-max per-user

Sebelum digabungkan, vektor skor setiap komponen dinormalkan ke [0, 1]
per pengguna secara terpisah:

    norm[i] = (skor[i] - min(skor)) / (max(skor) - min(skor))

Jika max == min, norm = 0 untuk semua item user tersebut.
Tujuan: mencegah satu komponen mendominasi karena perbedaan skala absolut.

---

## 3. Skema Pengelompokan Aktivitas

**Skema utama** (dipakai untuk semua tabel dan uji konfirmatori):

    pd.cut(train_interactions, bins=[0, 5, 8, inf],
           labels=["Low", "Medium", "High"], include_lowest=True)

Dikodekan di `src/config.py` (`EVAL_CONFIG["activity_bins"]`) dan
`src/baseline_common.make_user_groups()`.

| Kelompok | Rentang | n_users | % |
|---|---|---|---|
| Low | 1 -- 5 | 6 105 | 52.4% |
| Medium | 6 -- 8 | 2 364 | 20.3% |
| High | >= 9 | 3 190 | 27.4% |
| Total | | 11 659 | 100% |

Batas ini ditetapkan sebelum melihat distribusi data validasi.

**Skema alternatif (sensitivitas saja, bukan untuk uji konfirmatori):**
- Tertil: batas Q33/Q67 dari distribusi user evaluable
- Kuintil: batas Q20/Q40/Q60/Q80

Hasil sensitivitas (validation, NDCG@10): tertil menunjukkan DoD
Hybrid-Popularity (top-bottom) signifikan (CI 95% tidak memuat 0),
kuintil tidak. Pola **tidak seragam** antar skema -- kesimpulan sensitif
terhadap pilihan batas. Semua analisis skema alternatif berstatus
**eksploratori**.

---

## 4. Daftar Uji Konfirmatori (k=7, Bonferroni, CI 99.29%)

Ditetapkan sebelum FinalTest. Hanya uji-uji berikut diperlakukan
sebagai konfirmatori.

**alpha per uji = 0.05 / 7 = 0.007143**

### C1 -- C6: Paired bootstrap, skema utama

| # | Perbandingan | Kelompok | Metrik | Status validation |
|---|---|---|---|---|
| C1 | Hybrid vs Popularity | Low | NDCG@10 | Tidak signifikan |
| C2 | Hybrid vs Popularity | Medium | NDCG@10 | Tidak signifikan |
| C3 | Hybrid vs Popularity | High | NDCG@10 | Tidak signifikan |
| C4 | Hybrid vs MF | Low | NDCG@10 | Tidak signifikan |
| C5 | Hybrid vs MF | Medium | NDCG@10 | Tidak signifikan |
| C6 | Hybrid vs MF | High | NDCG@10 | Tidak signifikan |

Alasan: pertanyaan riset Track A mempertanyakan apakah Hybrid mengungguli
baseline di setiap kelompok aktivitas.

### C7: Selisih-dari-selisih (DoD) -- ditetapkan pra-FinalTest

| # | Statistik | Metrik | CI di validation | Status |
|---|---|---|---|---|
| C7 | (Hybrid - Popularity)_High minus (Hybrid - Popularity)_Low | NDCG@10 | CI 99.29% = [+0.000186, +0.010553] | **Konfirmatori di FinalTest** |

**Penting:** C7 berstatus **EKSPLORATORI** di validation karena pola
ditemukan dari data yang sama. C7 baru menjadi konfirmatori ketika
diuji pada data test yang belum pernah dilihat sebelumnya. Hasil
validation C7 dicatat sebagai pre-registration, bukan sebagai bukti.

Alasan memasukkan C7: pola interaksi model x kelompok ditemukan di
validation dan relevan untuk pertanyaan riset Track A. Menetapkannya
sebelum FinalTest mencegah post-hoc selection.

Metode C7: paired dalam kelompok (user sama untuk Hybrid dan Popularity),
independen antar kelompok (user High berbeda dari user Low).

Semua perbandingan lain: **eksploratori** (CI 95%, tanpa Bonferroni).

Hasil konfirmatori validation: `results/track_a_bootstrap_ci.csv`
dan `results/track_a_diff_of_diff.csv`.

---

## 5. Protokol Evaluasi

- Data split: temporal, disediakan panitia, tidak di-reshuffle.
- Evaluable users: `data/supporting/evaluable_users_validation_warm.csv`.
- Candidate set: `data/raw/candidate_items_warm.csv` (sama semua model).
- Item relevan: rating validation >= 4.
- Seen-item masking: item di training dikeluarkan sebelum Top-10 dipilih.
- Metrik utama: Recall@10 dan NDCG@10 (level pengguna, lalu rata-rata).

---

## 6. Catatan Reprodusibilitas

- **HitRate@10 dan MRR@10 untuk Hybrid** dihitung dari
  `results/hybrid_recommendations_top10.csv` via `src/add_hitrate_mrr.py`.
  Recall@10 dan NDCG@10 diverifikasi bit-identik dengan run asli.
- Kolom `*_best` di `results/track_a_table.csv` dihasilkan `idxmax()` per
  kelompok di `src/track_a_analysis.py`; tidak ada nilai di-set manual.
- File user_metrics lama (sebelum HitRate/MRR) tersimpan di `results/archive/`.

---

*Dibuat sebelum akses test set. Verifikasi tanggal commit: `git log --oneline docs/FREEZE_CONFIG.md`*
