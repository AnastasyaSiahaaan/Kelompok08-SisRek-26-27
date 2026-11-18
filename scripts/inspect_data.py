
from pathlib import Path
import pandas as pd

# Lokasi folder dataset
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SUPPORTING_DIR = DATA_DIR / "supporting"

# File utama yang akan diperiksa
main_files = [
    "interactions_train.csv",
    "interactions_validation.csv",
    "items.csv",
    "candidate_items_warm.csv",
]

print("=" * 60)
print("PEMERIKSAAN DATASET GABIN")
print("=" * 60)

for filename in main_files:
    file_path = RAW_DIR / filename

    print(f"\n--- {filename} ---")

    if not file_path.exists():
        print("File tidak ditemukan:", file_path)
        continue

    try:
        df = pd.read_csv(file_path)

        print("Jumlah baris :", len(df))
        print("Jumlah kolom :", len(df.columns))
        print("Nama kolom   :", df.columns.tolist())
        print("\nContoh 5 baris:")
        print(df.head().to_string(index=False))

    except Exception as error:
        print("Gagal membaca file:", error)

print("\n" + "=" * 60)
print("PEMERIKSAAN SELESAI")
print("=" * 60)
