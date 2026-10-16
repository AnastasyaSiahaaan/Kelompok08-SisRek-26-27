
from pathlib import Path
import pandas as pd

# 1. Tentukan lokasi dataset
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "raw" / "interactions_train.csv"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Baca data training
print("Membaca data training...")
train = pd.read_csv(TRAIN_PATH)

print("Jumlah interaksi training:", len(train))

# 3. Ambil interaksi positif (rating 4 atau 5)
positive = train[train["rating"] >= 4].copy()

print("Jumlah interaksi positif:", len(positive))

# 4. Hitung jumlah interaksi positif setiap item
popularity = (
    positive.groupby("item_id")
    .size()
    .reset_index(name="positive_interactions")
)

# 5. Urutkan item dari paling populer
popularity = popularity.sort_values(
    by=["positive_interactions", "item_id"],
    ascending=[False, True]
).reset_index(drop=True)

# 6. Simpan ranking baseline
output_path = OUTPUT_DIR / "popularity_ranking.csv"
popularity.to_csv(output_path, index=False)

# 7. Tampilkan 10 item teratas
print("\n=== TOP 10 POPULARITY BASELINE ===")
print(popularity.head(10).to_string(index=False))

print("\nTotal item dengan interaksi positif:", len(popularity))
print("Hasil disimpan di:", output_path)
