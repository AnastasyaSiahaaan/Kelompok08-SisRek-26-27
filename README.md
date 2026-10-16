# GABIN — Recommender System Project

**Working title:** The Impact of User Interaction History on Collaborative, Content-Based, and Hybrid Recommendation Performance

## Tujuan proyek
Menganalisis bagaimana jumlah interaksi user berkaitan dengan performa:
- Popularity baseline
- Collaborative Filtering (MF atau Item-CF)
- Content-Based Filtering (TF-IDF)
- Hybrid recommendation

Metrik yang direncanakan dalam CP1: Recall@10 dan NDCG@10.

## Struktur folder
- `data/raw/`: file dataset asli yang diberikan dosen/asisten.
- `data/processed/`: data hasil preprocessing.
- `src/`: kode program.
- `results/`: hasil evaluasi, tabel, dan grafik.

## Persiapan di Windows + VS Code
1. Buka folder proyek ini di VS Code.
2. Buka **Terminal → New Terminal**.
3. Periksa Python:
   ```powershell
   python --version
   py --version
   ```
4. Buat virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Aktifkan:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
6. Pasang library:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Jalankan pemeriksaan folder dan environment:
   ```powershell
   python check_environment.py
   ```

Jika PowerShell menolak aktivasi environment, salin pesan error yang muncul dan minta bantuan sebelum mengubah pengaturan keamanan.

## Penting tentang dataset
Jangan membuat split acak atau mengunduh dataset lain tanpa konfirmasi. Proyek kuliah menggunakan split temporal yang ditentukan di materi proyek. Periksa eCourse atau folder Google Drive yang dibagikan dosen/asisten terlebih dahulu. Simpan file asli di `data/raw/` dan jangan mengubah file aslinya.

Setelah nama file dan kolom dataset diketahui, sesuaikan kode pemuatan data sebelum membuat baseline. Dataset belum disertakan di starter project ini.
