
from pathlib import Path
import pandas as pd

from baseline_common import load_data, evaluate_rankings, RESULT_DIR


def main():
    # 1. Baca dataset
    print("Membaca data training dan validation...")

    train, validation, users, candidates, activity = load_data()

    print("Jumlah interaksi training:", len(train))

    # 2. Ambil interaksi positif (rating >= 4)
    positive = train[train["rating"] >= 4].copy()

    print("Jumlah interaksi positif:", len(positive))

    # 3. Hitung jumlah interaksi positif setiap item
    popularity = (
        positive.groupby("item_id")
        .size()
        .reset_index(name="positive_interactions")
    )

    # 4. Urutkan item dari paling populer
    popularity = popularity.sort_values(
        by=["positive_interactions", "item_id"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # 5. Simpan ranking popularitas
    ranking_path = RESULT_DIR / "popularity_ranking.csv"
    popularity.to_csv(ranking_path, index=False)

    print("\n=== TOP 10 POPULARITY BASELINE ===")
    print(popularity.head(10).to_string(index=False))
    print("\nTotal item dengan interaksi positif:", len(popularity))
    print("Ranking disimpan di:", ranking_path)

    # 6. Siapkan ranking rekomendasi untuk setiap pengguna
    top_items = popularity["item_id"].astype(str).tolist()

    rankings = {
        str(user_id): top_items
        for user_id in users["user_id"].astype(str)
    }

    # 7. Evaluasi menggunakan evaluator bersama
    evaluate_rankings(
        model_name="popularity",
        rankings=rankings,
        train=train,
        validation=validation,
        users=users,
        candidates=candidates,
        activity=activity
    )


if __name__ == "__main__":
    main()
