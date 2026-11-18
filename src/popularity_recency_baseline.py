"""
Popularity Recency Baseline — menggunakan interaksi positif (rating >= 4)
dalam jendela waktu terakhir dari data training saja.

Hyperparameter: window_days=180 (dipilih via validation; lihat POP_RECENCY_CONFIG).
Timestamp training dalam milidetik Unix; max timestamp dipakai sebagai titik acuan.
Test set TIDAK dipakai di mana pun.
"""

import pandas as pd

from baseline_common import load_data, evaluate_rankings, RAW_DIR, RESULT_DIR
from config import POP_RECENCY_CONFIG, make_run_id


def main():
    window_days = POP_RECENCY_CONFIG["window_days"]
    run_id = make_run_id("popularity_recency", {"window_days": window_days})
    print(f"Run ID: {run_id}")

    # 1. Baca dataset
    print("Membaca data training dan validation...")
    train, validation, users, candidates, activity = load_data()

    # 2. Ambil interaksi positif (rating >= 4) dari training
    positive = train[train["rating"] >= 4].copy()
    print(f"Jumlah interaksi positif (training): {len(positive)}")

    # 3. Hitung jendela waktu dari max timestamp training
    #    Kolom timestamp dalam milidetik Unix; convert ke detik.
    if "timestamp" not in positive.columns:
        raise ValueError("Kolom 'timestamp' tidak ditemukan di data training.")

    ts = positive["timestamp"].astype(float)
    # Deteksi apakah timestamp dalam ms (> 1e11) atau detik
    if ts.max() > 1e11:
        ts = ts / 1000.0  # konversi ms -> detik

    max_ts = ts.max()
    window_seconds = window_days * 24 * 3600
    cutoff_ts = max_ts - window_seconds

    positive = positive.copy()
    positive["_ts"] = ts
    recent = positive[positive["_ts"] >= cutoff_ts]

    print(
        f"Window: {window_days} hari dari max_ts "
        f"({pd.Timestamp(max_ts, unit='s').date()})"
    )
    print(f"Cutoff: {pd.Timestamp(cutoff_ts, unit='s').date()}")
    print(f"Interaksi positif dalam window: {len(recent)}")

    # 4. Hitung popularitas dalam window
    popularity = (
        recent.groupby("item_id")
        .size()
        .reindex(candidates["item_id"].astype(str), fill_value=0)
        .rename("positive_interactions")
        .reset_index()
    )

    # Tie-break: item_id ascending (konsisten dengan popularity global)
    popularity = popularity.sort_values(
        by=["positive_interactions", "item_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    print("\n=== TOP 10 POPULARITY RECENCY BASELINE ===")
    print(popularity.head(10).to_string(index=False))

    # 5. Simpan ranking
    ranking_path = RESULT_DIR / "popularity_recency_ranking.csv"
    popularity.to_csv(ranking_path, index=False)
    print(f"\nRanking disimpan di: {ranking_path}")

    # 6. Siapkan rankings dict untuk semua evaluable users
    top_items = popularity["item_id"].astype(str).tolist()
    rankings = {
        str(user_id): top_items
        for user_id in users["user_id"].astype(str)
    }

    # 7. Evaluasi dengan protokol bersama
    evaluate_rankings(
        model_name="popularity_recency",
        rankings=rankings,
        train=train,
        validation=validation,
        users=users,
        candidates=candidates,
        activity=activity,
    )


if __name__ == "__main__":
    main()
