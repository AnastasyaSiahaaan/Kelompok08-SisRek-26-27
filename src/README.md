# Source Code — GABIN

Folder ini berisi kode implementasi dan evaluasi sistem rekomendasi pada proyek GABIN.

## File Utama

- `baseline_common.py`: fungsi bersama untuk memuat data, mengelompokkan pengguna berdasarkan aktivitas, dan mengevaluasi rekomendasi.
- `popularity_baseline.py`: implementasi baseline berdasarkan popularitas item.
- `mf_baseline.py`: implementasi Matrix Factorization menggunakan TruncatedSVD.
- `tfidf_cbf_baseline.py`: implementasi Content-Based Filtering menggunakan TF-IDF.
- `hybrid_baseline.py`: implementasi rekomendasi hybrid yang menggabungkan skor MF dan CBF.
- `compare_results.py`: membandingkan hasil evaluasi keempat metode.

## Skrip Pemeriksaan dan Pendukung

- `check_recommendations.py`: memeriksa validitas daftar rekomendasi.
- `check_popularity.py`: memeriksa cakupan kandidat item pada ranking popularitas.
- `check_evaluation.py`: skrip pemeriksaan evaluasi.
- `check_user_groups.py`: skrip pemeriksaan kelompok aktivitas pengguna.
- `evaluate_popularity.py`: evaluasi popularity secara terpisah.
- `evaluate_popularity_by_group.py`: skrip evaluasi popularity berdasarkan kelompok aktivitas.
- `inspect_data.py`: membantu memeriksa data utama.
- `inspect_supporting.py`: membantu memeriksa data pendukung.

## Cara Menjalankan

Jalankan perintah dari direktori utama proyek:

```powershell
python src/popularity_baseline.py
python src/mf_baseline.py
python src/tfidf_cbf_baseline.py
python src/hybrid_baseline.py
python src/compare_results.py
```

Hasil eksperimen disimpan dalam folder `results/`.