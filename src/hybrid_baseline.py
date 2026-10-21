
"""Hybrid Recommendation: MF + TF-IDF CBF."""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from baseline_common import (
    load_data,
    evaluate_rankings,
    RAW_DIR,
    RESULT_DIR,
)


def main():
    # 1. Memuat data
    train, validation, users, candidates, activity = load_data()

    train = train.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    # Bobot awal Hybrid: MF 50%, CBF 50%
    # Nilai awal CP2, belum dituning; akan dipilih ulang memakai validation sebelum freeze.
    alpha = 0.5

    # 2. Membentuk model MF menggunakan data training
    user_ids = train["user_id"].drop_duplicates().tolist()
    item_ids = train["item_id"].drop_duplicates().tolist()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {item: i for i, item in enumerate(item_ids)}

    pairs = (
        train.groupby(["user_id", "item_id"], as_index=False)["rating"]
        .mean()
    )

    R = coo_matrix(
        (
            pairs["rating"].astype(float),
            (
                pairs["user_id"].map(user_to_idx),
                pairs["item_id"].map(item_to_idx),
            ),
        ),
        shape=(len(user_ids), len(item_ids)),
    ).tocsr()

    n_components = min(32, min(R.shape) - 1)
    if n_components < 2:
        raise ValueError("Data tidak cukup untuk MF/SVD.")

    print("Melatih model MF...")
    mf = TruncatedSVD(
        n_components=n_components,
        n_iter=7,
        random_state=42,
    )
    user_factors = mf.fit_transform(R)
    item_factors = mf.components_.T

    # 3. Membentuk model TF-IDF CBF
    print("Membangun model TF-IDF CBF...")
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})

    text_columns = [
        col for col in [
            "title", "main_category", "categories",
            "features", "description", "content_text_basic",
        ]
        if col in items.columns
    ]

    items["combined_text"] = (
        items[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    )
    items = items.drop_duplicates("item_id").reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=100000,
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(items["combined_text"])
    item_text_idx = {item: i for i, item in enumerate(items["item_id"])}

    # 4. Kandidat yang tersedia pada kedua model
    candidate_ids = [
        str(item) for item in candidates["item_id"]
        if str(item) in item_to_idx and str(item) in item_text_idx
    ]

    candidate_mf_idx = np.array(
        [item_to_idx[item] for item in candidate_ids], dtype=int
    )
    candidate_text_idx = np.array(
        [item_text_idx[item] for item in candidate_ids], dtype=int
    )

    candidate_factors = item_factors[candidate_mf_idx]
    candidate_matrix = X[candidate_text_idx]

    # 5. Riwayat training untuk menghindari rekomendasi item yang sudah dilihat
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    positive = train[train["rating"] >= 4]
    positive_history = positive.groupby("user_id")["item_id"].apply(list).to_dict()

    rankings = {}
    recommendation_rows = []

    print("Menghitung rekomendasi Hybrid...")
    for user_id in users["user_id"].astype(str):
        seen = train_seen.get(user_id, set())

        # Skor MF
        if user_id in user_to_idx:
            mf_scores = user_factors[user_to_idx[user_id]] @ candidate_factors.T
        else:
            mf_scores = np.zeros(len(candidate_ids), dtype=float)

        # Skor CBF dari profil item positif pengguna
        history_idx = [
            item_text_idx[item]
            for item in positive_history.get(user_id, [])
            if item in item_text_idx
        ]

        if history_idx:
            profile = normalize(np.asarray(X[history_idx].mean(axis=0)))
            cbf_scores = np.asarray(profile @ candidate_matrix.T).ravel()
        else:
            cbf_scores = np.zeros(len(candidate_ids), dtype=float)

        # Normalisasi min-max agar skala skor lebih sebanding
        def minmax(scores):
            low, high = scores.min(), scores.max()
            if high == low:
                return np.zeros_like(scores, dtype=float)
            return (scores - low) / (high - low)

        mf_norm = minmax(mf_scores)
        cbf_norm = minmax(cbf_scores)

        hybrid_scores = alpha * mf_norm + (1 - alpha) * cbf_norm

        # Jangan merekomendasikan item yang sudah ada dalam training
        for i, item in enumerate(candidate_ids):
            if item in seen:
                hybrid_scores[i] = -np.inf

        valid = np.flatnonzero(np.isfinite(hybrid_scores))
        k = min(10, len(valid))
        top = sorted(
            valid,
            key=lambda i: (-hybrid_scores[i], candidate_ids[i]),
        )[:k]

        rankings[user_id] = [candidate_ids[i] for i in top]

    # 6. Simpan rekomendasi Top-10
    for user_id, items_top in rankings.items():
        for rank, item_id in enumerate(items_top, start=1):
            recommendation_rows.append({
                "user_id": user_id,
                "rank": rank,
                "item_id": item_id,
            })

    pd.DataFrame(
        recommendation_rows,
        columns=["user_id", "rank", "item_id"],
    ).to_csv(RESULT_DIR / "hybrid_recommendations_top10.csv", index=False)

    # 7. Evaluasi menggunakan evaluator yang sama
    evaluate_rankings(
        "hybrid",
        rankings,
        train,
        validation,
        users,
        candidates,
        activity,
    )


if __name__ == "__main__":
    main()
