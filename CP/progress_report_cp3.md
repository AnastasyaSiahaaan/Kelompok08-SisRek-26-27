# Progress Report CP3 — Desain Eksperimen

**Kelompok 08 | 12S4054 Sistem Rekomendasi | Institut Teknologi Del**
**Semester Gasal 2026/2027**

---

## 1. Ringkasan

CP3 menetapkan desain eksperimen lengkap sebelum eksperimen validasi skala penuh dimulai.
Protokol evaluasi warm-start, candidate set, leakage rules, hyperparameter search space, dan
experiment matrix telah ditetapkan dan dikodekan dalam repositori.

---

## 2. Protokol Evaluasi (Warm-Start)

Evaluasi menggunakan kondisi **warm-start** agar semua baseline dibandingkan secara adil.

| Komponen | Nilai |
|---|---|
| Data split | Official temporal split dari paket proyek; tidak di-reshuffle |
| Ukuran training | 427.957 interaksi |
| Ukuran validation | 41.284 (warm: 30.360) |
| Ukuran test | 42.595 (tidak digunakan sebelum W13) |
| Evaluable users | 11.659 pengguna dari `evaluable_users_validation_warm.csv` |
| Candidate universe | 22.753 item dari `candidate_items_warm.csv` (sama untuk semua model) |
| Ranking per user | Top-10 |
| Metrik wajib | **Recall@10** dan **NDCG@10** |
| Item relevan | Rating validation ≥ 4 |
| Seen-item masking | Item yang muncul di training dikeluarkan dari kandidat tiap user |

Model selection dan hyperparameter tuning **hanya menggunakan validation**. Test set tidak diakses.

---

## 3. Candidate Set & Leakage Rules

### 3.1 Candidate Set
- Item kandidat berasal dari `candidate_items_warm.csv` — sama untuk semua model.
- Item yang pernah diinteraksikan user pada training **dikeluarkan** sebelum rekomendasi dibuat.
- Semua baseline menggunakan candidate universe yang identik.

### 3.2 Leakage Rules
1. Model hanya dilatih dari data training; validation dan test tidak masuk proses training.
2. Official temporal split tidak diacak ulang.
3. User profile dan parameter model dibangun dari training saja. Field agregat (`average_rating`,
   `rating_number`) tidak digunakan.
4. Candidate protocol sama untuk semua model; tidak ada candidate set yang menguntungkan model tertentu.
5. Hyperparameter tuning dan model selection hanya menggunakan validation.

---

## 4. Evaluable Users & Activity Groups

Pengguna yang masuk evaluasi harus memenuhi:
- Memiliki riwayat interaksi pada training (warm-start).
- Memiliki ground-truth positif (rating ≥ 4) pada validation.

Kelompok aktivitas (Track A) didefinisikan dari jumlah interaksi training menggunakan `pd.cut`
dengan batas tetap:

| Kelompok | Jumlah interaksi training |
|---|---|
| Low | 1 – 5 |
| Medium | 6 – 8 |
| High | ≥ 9 |

Definisi ini konsisten untuk semua model dan berada di `make_user_groups()` dalam `baseline_common.py`.

---

## 5. Hyperparameter Search Space

Tuning dilakukan via validation menggunakan skrip `src/tune_hyperparameters.py`.
Semua nilai search space didefinisikan di `src/config.py`.

| Model | Parameter | Search Space | Metode Seleksi |
|---|---|---|---|
| Popularity | — | Tidak ada tuning | Tetap |
| Matrix Factorization | `n_components` | [32, 64, 128] | Recall@10 pada validation |
| TF-IDF CBF | `max_features` | [50.000, 100.000] | Recall@10 pada validation |
| Hybrid | `alpha` (bobot MF) | [0.3, 0.5, 0.7] | Recall@10 pada validation |

Parameter tetap di semua run: `random_state=42`, `n_iter=7` (MF), `ngram_range=(1,2)` (TF-IDF),
`min_df=2` (TF-IDF), normalisasi min-max per user (Hybrid).

---

## 6. Experiment Matrix

| ID | Model/Variant | Tujuan | Metrik |
|---|---|---|---|
| E1 | Popularity | Baseline bersama | Recall@10, NDCG@10 |
| E2a–E2c | MF (n=32/64/128) | Tuning collaborative baseline | Recall@10, NDCG@10 |
| E3a–E3b | TF-IDF CBF (feat=50k/100k) | Tuning content-based baseline | Recall@10, NDCG@10 |
| E4a–E4c | Hybrid (alpha=0.3/0.5/0.7) | Tuning hybrid; MF & CBF config dari hasil terbaik E2/E3 | Recall@10, NDCG@10 |
| E5 | Track-A Analysis | Analisis per activity group (Low/Medium/High) — RQ Track A | Recall@10, NDCG@10 per group |

File lengkap: `results/experiment_matrix.csv`.

---

## 7. Reproducibility Settings

- **Random seed:** `42` — dikodekan di `src/config.py` (`RANDOM_SEED`).
- **Config terpusat:** `src/config.py` — semua hyperparameter dan search space.
- **Run ID format:** `<model>_<param_key><param_val>_seed<seed>` (contoh: `mf_components64_seed42`).
- **Experiment log:** setiap run dicatat ke `results/experiment_log.csv` oleh `src/experiment_log.py`
  beserta timestamp, git commit hash, parameter, dan metrik.

---

## 8. Hasil Awal (CP2)

Berikut performa baseline awal (belum dituning, n_components=32, alpha=0.5):

| Model | Recall@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0135 | 0.0092 |
| Matrix Factorization | 0.0131 | 0.0079 |
| TF-IDF CBF | 0.0126 | 0.0073 |
| Hybrid | **0.0146** | 0.0092 |

Hybrid mengungguli semua model. Tuning pada W10–W12 diharapkan meningkatkan
performa terutama pada pengguna kelompok Low.

---

## 9. Evidence GitHub

- `src/config.py` — konfigurasi terpusat
- `src/experiment_log.py` — logging per run
- `src/tune_hyperparameters.py` — skrip tuning otomatis
- `results/experiment_matrix.csv` — experiment matrix lengkap
- Update pada `src/mf_baseline.py`, `src/tfidf_cbf_baseline.py`, `src/hybrid_baseline.py`
  untuk membaca parameter dari `config.py`

---

*Deadline CP3: Sabtu, 7 November 2026, 23.59 WIB*
