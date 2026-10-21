"""Matrix Factorization baseline using TruncatedSVD."""

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

from baseline_common import (
    load_data,
    evaluate_rankings,
    RESULT_DIR,
)


# 1. Memuat data
train, validation, users, candidates, activity = load_data()

train = train.copy()
train["user_id"] = train["user_id"].astype(str)
train["item_id"] = train["item_id"].astype(str)


# 2. Membuat pemetaan ID berdasarkan data training
user_ids = train["user_id"].drop_duplicates().tolist()
item_ids = train["item_id"].drop_duplicates().tolist()

user_to_idx = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

item_to_idx = {
    item_id: index
    for index, item_id in enumerate(item_ids)
}


# 3. Membentuk matriks interaksi sparse
# Rating berulang untuk pasangan pengguna-item dirata-ratakan.
pairs = (
    train.groupby(["user_id", "item_id"], as_index=False)["rating"]
    .mean()
)

rows = pairs["user_id"].map(user_to_idx).to_numpy()
cols = pairs["item_id"].map(item_to_idx).to_numpy()
values = pairs["rating"].astype(float).to_numpy()

R = coo_matrix(
    (values, (rows, cols)),
    shape=(len(user_ids), len(item_ids)),
).tocsr()


# 4. Melatih Matrix Factorization dengan TruncatedSVD
# Nilai awal CP2, belum dituning; akan dipilih ulang memakai validation sebelum freeze.
n_components = min(32, min(R.shape) - 1)

if n_components < 2:
    raise ValueError(
        "Data tidak cukup untuk menjalankan MF/SVD."
    )

print("Melatih Matrix Factorization (TruncatedSVD)...")

model = TruncatedSVD(
    n_components=n_components,
    n_iter=7,
    random_state=42,
)

user_factors = model.fit_transform(R)
item_factors = model.components_.T


# 5. Memilih kandidat yang dikenal model
candidate_item_ids = [
    item_id
    for item_id in candidates["item_id"].astype(str).tolist()
    if item_id in item_to_idx
]

candidate_indices = np.array(
    [item_to_idx[item_id] for item_id in candidate_item_ids],
    dtype=int,
)

candidate_factors = item_factors[candidate_indices]


# 6. Menyiapkan riwayat interaksi pengguna
train_seen = (
    train.groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)

evaluable_users = users["user_id"].astype(str).tolist()

candidate_position = {
    item_id: position
    for position, item_id in enumerate(candidate_item_ids)
}

rankings = {}

print("Menghitung skor dan menyusun rekomendasi Top-10...")


# 7. Menghasilkan rekomendasi untuk setiap pengguna evaluasi
for user_id in evaluable_users:
    if user_id not in user_to_idx:
        rankings[user_id] = []
        continue

    user_index = user_to_idx[user_id]

    # Menghitung skor hanya untuk kandidat.
    scores = user_factors[user_index] @ candidate_factors.T

    # Mengeluarkan item yang sudah muncul di training.
    seen_items = train_seen.get(user_id, set())

    seen_positions = [
        candidate_position[item_id]
        for item_id in seen_items
        if item_id in candidate_position
    ]

    scores[seen_positions] = -np.inf

    # Mengambil maksimal 10 kandidat yang belum dilihat.
    valid_positions = np.flatnonzero(np.isfinite(scores))
    k = min(10, len(valid_positions))

    if k == 0:
        rankings[user_id] = []
        continue

    # Memilih kandidat dengan skor tertinggi tanpa mengurutkan
    # seluruh daftar kandidat.
    top_positions = valid_positions[
        np.argpartition(scores[valid_positions], -k)[-k:]
    ]

    # Urutan akhir: skor tertinggi, lalu item_id untuk memecahkan seri.
    top_positions = sorted(
        top_positions,
        key=lambda position: (
            -scores[position],
            candidate_item_ids[position],
        ),
    )

    rankings[user_id] = [
        candidate_item_ids[position]
        for position in top_positions
    ]


# 8. Menyimpan rekomendasi Top-10
recommendation_rows = [
    {
        "user_id": user_id,
        "rank": rank,
        "item_id": item_id,
    }
    for user_id, items in rankings.items()
    for rank, item_id in enumerate(items, start=1)
]

pd.DataFrame(
    recommendation_rows,
    columns=["user_id", "rank", "item_id"],
).to_csv(
    RESULT_DIR / "mf_recommendations_top10.csv",
    index=False,
)


# 9. Mengevaluasi hasil dengan fungsi bersama
evaluate_rankings(
    "mf",
    rankings,
    train,
    validation,
    users,
    candidates,
    activity,
)