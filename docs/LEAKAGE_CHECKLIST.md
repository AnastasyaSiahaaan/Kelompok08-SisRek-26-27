# LEAKAGE CHECKLIST

Checklist ini memverifikasi bahwa tidak ada informasi dari validation atau test
yang masuk ke proses training atau pemilihan hyperparameter.

Tandai [x] = terpenuhi, [ ] = perlu ditangani.

---

## A. Data Split

- [x] Split temporal disediakan panitia; tidak di-reshuffle atau dimodifikasi.
- [x] Test set tidak ada di repositori (`*test*.csv` di-gitignore).
- [x] Tidak ada script yang membaca file test sebelum FinalTest.
- [x] `interactions_validation.csv` (full) tidak digunakan untuk evaluasi;
      hanya `interactions_validation_warm.csv` (via `load_data()` di
      `baseline_common.py`) yang dipakai.

## B. Training Model

- [x] Semua model hanya dilatih dari `interactions_train.csv`.
- [x] Kolom `average_rating` dan `rating_number` di `items.csv` tidak digunakan.
- [x] User profile (TF-IDF CBF, Hybrid) dibangun dari item positif training saja.
- [x] Timestamp training digunakan untuk Pop Recency (mencari max_ts),
      bukan dari validation atau test.

## C. Hyperparameter Tuning

- [x] Semua tuning (MF n_components, TF-IDF max_features, Hybrid alpha,
      Item-kNN shrinkage, Pop Recency window_days) menggunakan validation.
- [x] Test set tidak diakses selama tuning.
- [x] Setiap run tuning dicatat ke `results/experiment_log.csv` dengan
      timestamp dan git commit.

## D. Evaluasi

- [x] `evaluable_users_validation_warm.csv` hanya menentukan siapa yang
      dievaluasi; tidak ada informasi label di dalamnya yang digunakan
      untuk melatih model.
- [x] Seen-item masking dilakukan per user dari training saja.
- [x] Item relevan (rating >= 4) hanya dibaca dari validation untuk evaluasi,
      tidak untuk training.
- [x] Candidate set identik untuk semua model.

## E. Analisis Track A

- [x] Bootstrap CI dan regresi dihitung dari user_metrics yang dihasilkan
      evaluasi pada validation.
- [x] Uji konfirmatori C1-C7 ditetapkan sebelum FinalTest (lihat FREEZE_CONFIG.md).
- [x] DoD (C7) berstatus eksploratori di validation; baru konfirmatori di FinalTest.
- [x] Sensitivitas batas kelompok (tertil, kuintil) menggunakan distribusi
      user evaluable, bukan distribusi test.

## F. Repositori

- [x] `.gitignore` mengecualikan `*test*.csv` dan `data/raw/*test*`.
- [x] Tidak ada `git push --force` yang dilakukan.
- [x] Semua commit memiliki pesan deskriptif; bisa diverifikasi via `git log`.
- [x] File hasil lama diarsipkan di `results/archive/`, bukan dihapus.

---

## Verifikasi Teknis (jalankan sebelum FinalTest)

```powershell
# 1. Pastikan tidak ada referensi ke test file dalam script
Select-String -Path "src/*.py","scripts/*.py" -Pattern "test" -CaseSensitive | Where-Object { $_.Line -notmatch "#" -and $_.Line -notmatch "unittest" }

# 2. Pastikan tidak ada test CSV di repositori
Get-ChildItem -Recurse -Filter "*test*.csv" | Where-Object { $_.FullName -notmatch "archive" }

# 3. Jalankan unit test
python -m unittest discover tests -v

# 4. Verifikasi angka Popularity identik
python -c "import pandas as pd; df=pd.read_csv('results/popularity_overall_metrics.csv'); print(df)"
```

---

*Checklist ini harus ditinjau ulang sebelum setiap FinalTest.*
