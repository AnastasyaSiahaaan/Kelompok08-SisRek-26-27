
import pandas as pd
from pathlib import Path

# Lokasi file
results_dir = Path("results")

# Baca data training dan daftar kandidat
train = pd.read_csv(
    "data/raw/interactions_train.csv",
    dtype={"user_id": str, "item_id": str}
)

candidates = set(
    pd.read_csv(
        "data/raw/candidate_items_warm.csv",
        dtype={"item_id": str}
    )["item_id"]
)

# Item yang sudah pernah dilihat setiap pengguna saat training
seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

files = [
    "mf_recommendations_top10.csv",
    "tfidf_cbf_recommendations_top10.csv",
    "hybrid_recommendations_top10.csv"
]

for filename in files:
    filepath = results_dir / filename

    df = pd.read_csv(
        filepath,
        dtype={"user_id": str, "item_id": str}
    )

    # Periksa apakah item termasuk kandidat
    outside_candidates = (~df["item_id"].isin(candidates)).sum()

    # Periksa item yang sudah muncul dalam training pengguna
    seen_count = sum(
        item in seen.get(user, set())
        for user, item in zip(df["user_id"], df["item_id"])
    )

    # Periksa duplikasi rekomendasi untuk pengguna yang sama
    duplicate_count = df.duplicated(
        subset=["user_id", "item_id"]
    ).sum()

    # Periksa jumlah rekomendasi setiap pengguna
    recommendation_counts = df.groupby("user_id").size()
    users_not_10 = (recommendation_counts != 10).sum()

    print(f"\n--- {filename} ---")
    print("Item di luar kandidat:", int(outside_candidates))
    print("Item yang sudah ada di training pengguna:", seen_count)
    print("Duplikat item per pengguna:", int(duplicate_count))
    print("Pengguna dengan bukan 10 rekomendasi:", int(users_not_10))
