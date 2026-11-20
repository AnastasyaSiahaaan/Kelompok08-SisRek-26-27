# GABIN -- Recommender System Project (Track A)

**Judul:** The Impact of User Interaction History on Collaborative,
Content-Based, and Hybrid Recommendation Performance

**Mata kuliah:** 12S4054 Sistem Rekomendasi, Institut Teknologi Del
**Kelompok:** G08  |  **Semester:** Gasal 2026/2027

---

## 1. Pertanyaan Riset

Bagaimana jumlah interaksi training seorang user memengaruhi performa
rekomendasi berbasis collaborative filtering, content-based, dan hybrid?

**Track A: User History Sparsity.**

---

## 2. Dataset

Sumber: Amazon Reviews 2023 -- Musical Instruments, 5-core,
absolute-timestamp split (<https://amazon-reviews-2023.github.io/>).

| Subset | Jumlah interaksi |
|---|---:|
| Training | 427 957 |
| Validation (full) | 41 284 |
| Validation (warm) | 30 360 |
| Test | 42 595 (tidak digunakan sebelum FinalTest) |
| Candidate items (warm) | 22 753 |
| Evaluable users (warm validation) | 11 659 |

Split disediakan panitia dan digunakan apa adanya; tidak di-reshuffle.
Test set tidak ada di repositori ini.

---

## 3. Metode

| Model | Deskripsi singkat |
|---|---|
| **Popularity** | Item diranking berdasarkan jumlah interaksi positif (rating >= 4) di training |
| **MF** | TruncatedSVD pada matriks user-item training; skor = dot product faktor user dan item |
| **TF-IDF CBF** | Profil user dari rata-rata TF-IDF item positif di training; skor = cosine similarity ke kandidat |
| **Hybrid** | alpha * norm_MF + (1-alpha) * norm_CBF; normalisasi min-max per-user; alpha=0.5 |

Semua model mengeluarkan item yang ada di training user sebelum memilih Top-10.

---

## 4. Protokol Evaluasi

- Metrik utama: **Recall@10** dan **NDCG@10** (rata-rata per user).
- Metrik tambahan: HitRate@10, MRR@10.
- Item relevan: rating validation >= 4.
- Candidate set: `data/raw/candidate_items_warm.csv` (sama untuk semua model).
- Hyperparameter dan model selection: hanya menggunakan validation.
  Test set tidak digunakan sebelum FinalTest.

**Catatan khusus (reprodusibilitas):**
HitRate@10 dan MRR@10 untuk model Hybrid dihitung dari
`results/hybrid_recommendations_top10.csv` via `src/add_hitrate_mrr.py`,
bukan dari re-run penuh, karena metrik tersebut ditambahkan setelah
run asli. Recall@10 dan NDCG@10 Hybrid diverifikasi bit-identik.

---

## 5. Kelompok Aktivitas

Dibagi berdasarkan jumlah interaksi training per user (fixed bins):

| Kelompok | Interaksi training | n users | % |
|---|---|---|---|
| Low | 1 -- 5 | 6 105 | 52.4% |
| Medium | 6 -- 8 | 2 364 | 20.3% |
| High | >= 9 | 3 190 | 27.4% |

Implementasi: `src/baseline_common.make_user_groups()`.

---

## 6. Struktur Repositori

```
Kelompok08-SisRek-26-27/
|-- CP/                          # Checkpoint documents (PDF, Markdown)
|-- data/
|   |-- raw/                     # interactions_train, validation, items, candidates
|   |-- processed/               # README.txt
|   `-- supporting/              # warm validation, evaluable users, activity files
|-- docs/
|   |-- EXPERIMENT_MATRIX.md     # daftar semua run eksperimen
|   |-- FREEZE_CONFIG.md         # keputusan pra-FinalTest
|   `-- LEAKAGE_CHECKLIST.md     # checklist leakage
|-- results/                     # output eksperimen (jangan edit manual)
|   |-- archive/                 # user_metrics lama (sebelum HitRate/MRR ditambah)
|   |-- track_a_table_final.csv  # tabel utama Track A
|   |-- track_a_bootstrap_ci.csv # CI paired lengkap
|   |-- track_a_diff_of_diff.csv # selisih-dari-selisih
|   |-- track_a_regression.csv   # regresi tren kontinu
|   |-- track_a_sensitivity.csv  # sensitivitas batas kelompok
|   |-- track_a_ndcg_bar.png     # figure utama
|   `-- experiment_log.csv       # log semua run
|-- scripts/                     # utilitas diagnostik (bukan pipeline utama)
|   |-- check_*.py               # cek validitas output
|   |-- inspect_*.py             # inspeksi data
|   |-- evaluate_popularity*.py  # evaluasi popularity mandiri
|   `-- generate_progress_report.py
|-- src/                         # kode pipeline utama
|   |-- baseline_common.py       # load_data, make_user_groups, evaluate_rankings
|   |-- config.py                # semua hyperparameter dan search space
|   |-- experiment_log.py        # logging run ke experiment_log.csv
|   |-- popularity_baseline.py
|   |-- popularity_recency_baseline.py
|   |-- mf_baseline.py
|   |-- tfidf_cbf_baseline.py
|   |-- item_knn_baseline.py
|   |-- hybrid_baseline.py
|   |-- hybrid_new_baseline.py
|   |-- tune_hyperparameters.py  # tuning MF/TF-IDF/Hybrid via validation
|   |-- compare_results.py
|   |-- bootstrap_ci.py          # CI untuk model appendix (kNN, Pop Recency, Hybrid New)
|   |-- dataset_statistics.py
|   |-- track_a_analysis.py      # analisis Track A utama (bootstrap, CI, overlap)
|   |-- track_a_final.py         # analisis tambahan (regresi, kuintil, CI per user)
|   `-- add_hitrate_mrr.py       # tambahkan HitRate/MRR dari recs yang sudah ada
|-- tests/
|   `-- test_metrics.py          # 6 unit test (Recall, NDCG, HitRate, MRR, groups, pipeline)
|-- requirements.txt             # versi library di-pin
`-- README.md
```

---

## 7. Setup Environment

```powershell
python -m pip install -r requirements.txt
```

Versi yang diuji: Python 3.13.7 pada Windows.

Jalankan unit test untuk memverifikasi instalasi:

```powershell
python -m unittest discover tests -v
```

---

## 8. Urutan Eksekusi -- dari Nol sampai Tabel Utama

Semua perintah dijalankan dari root proyek.
**Random seed 42** digunakan di semua model yang membutuhkan randomness
(dikodekan di `src/config.py`, `RANDOM_SEED = 42`).

### 8a. Statistik dataset (opsional)

```powershell
python src/dataset_statistics.py
```

### 8b. Jalankan 4 model inti

```powershell
python src/popularity_baseline.py
python src/mf_baseline.py
python src/tfidf_cbf_baseline.py
python src/hybrid_baseline.py
```

### 8c. Tambahkan HitRate/MRR untuk Hybrid (dan model lain yang perlu)

```powershell
python src/add_hitrate_mrr.py
```

### 8d. Bandingkan hasil

```powershell
python src/compare_results.py
```

### 8e. Analisis Track A (tabel, CI, overlap, selisih-dari-selisih)

```powershell
python src/track_a_analysis.py
```

### 8f. Analisis tambahan (regresi, kuintil, CI per user, figure)

```powershell
python src/track_a_final.py
```

### 8g. Regenerasi tabel utama dan figure dalam satu perintah

```powershell
python src/track_a_analysis.py; python src/track_a_final.py
```

Output utama:
- `results/track_a_table_final.csv` -- tabel 4 model x 4 kelompok x 4 metrik
- `results/track_a_ndcg_bar.png`   -- figure NDCG@10 dengan error bar CI

### 8h. Model appendix (opsional)

```powershell
python src/popularity_recency_baseline.py
python src/item_knn_baseline.py
python src/hybrid_new_baseline.py
python src/bootstrap_ci.py
```

### 8i. Tuning hyperparameter (sudah dilakukan, hasilnya di experiment_log.csv)

```powershell
python src/tune_hyperparameters.py
```

---

## 9. Output Utama

| File | Isi |
|---|---|
| `results/track_a_table_final.csv` | Recall/NDCG/HitRate/MRR per model per kelompok; * = terbaik per kolom |
| `results/track_a_bootstrap_ci.csv` | CI paired 95%/99.29% semua perbandingan |
| `results/track_a_diff_of_diff.csv` | DoD Hybrid-Popularity (High-Low) + CI |
| `results/track_a_regression.csv` | Regresi delta(NDCG) ~ log(interactions) + CI kemiringan |
| `results/track_a_sensitivity.csv` | NDCG per tertil/kuintil + DoD |
| `results/track_a_relevant_pop.csv` | Popularitas item relevan per kelompok (CI per user) |
| `results/track_a_ndcg_bar.png` | Bar chart NDCG@10 dengan error bar CI 95% |
| `results/experiment_log.csv` | Log semua run (timestamp, git commit, params, metrik) |
| `results/comparison_overall.csv` | Perbandingan ringkasan semua model |

---

## 10. Validasi Output

```powershell
python scripts/check_recommendations.py   # duplikat, seen items, jumlah reks
python scripts/check_popularity.py        # cakupan kandidat di ranking
python scripts/check_evaluation.py        # protokol evaluasi
python scripts/check_user_groups.py       # distribusi kelompok
```

---

## 11. Unit Test

```powershell
python -m unittest discover tests -v
# Ran 6 tests in ~0.06s -- OK
```

Test mencakup: Recall@10, NDCG@10, HitRate@10, MRR@10, user group definitions,
seen-item exclusion dalam pipeline evaluasi.

---

## 12. Experiment Log

Setiap run model dicatat ke `results/experiment_log.csv` oleh `src/experiment_log.py`.
Format: run_id, timestamp, git_commit, model, params, recall_at_10, ndcg_at_10,
metrik per kelompok, n_evaluated_users, notes.

Semua percobaan tuning (MF n_components, TF-IDF max_features, Hybrid alpha,
Item-kNN shrinkage, Pop Recency window_days, Hybrid New 3-way grid) tercatat di file ini.

---

## 13. Keputusan Pra-FinalTest

Lihat `docs/FREEZE_CONFIG.md` untuk:
- Hyperparameter final yang di-freeze
- Definisi skema pengelompokan (skema utama vs sensitivitas)
- Daftar 7 uji konfirmatori dengan Bonferroni k=7
- Catatan bahwa C7 (DoD) adalah eksploratori di validation dan
  baru konfirmatori di FinalTest
