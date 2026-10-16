"""TF-IDF Content-Based Filtering baseline."""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from baseline_common import (
    load_data,
    evaluate_rankings,
    RAW_DIR,
    RESULT_DIR,
)


# 1. Memuat data
train, validation, users, candidates, activity = load_data()

train = train.copy()
train["user_id"] = train["user_id"].astype(str)
train["item_id"] = train["item_id"].astype(str)


# 2. Memuat metadata item
items = pd.read_csv(
    RAW_DIR / "items.csv",
    dtype={"item_id": str},
)

items["item_id"] = items["item_id"].astype(str)

# Kolom teks yang tersedia pada dataset.
text_columns = [
    column
    for column in [
        "title",
        "main_category",
        "categories",
        "features",
        "description",
        "content_text_basic",
    ]
    if column in items.columns
]

if not text_columns:
    raise ValueError(
        "Tidak ditemukan kolom teks pada items.csv."
    )

# Menggabungkan informasi teks item.
items["combined_text"] = (
    items[text_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

# Memastikan setiap item hanya memiliki satu baris metadata.
items = (
    items.drop_duplicates("item_id")
    .reset_index(drop=True)
)


# 3. Membentuk representasi TF-IDF
print("Membangun representasi TF-IDF item...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    stop_words="english",
    min_df=2,
    ngram_range=(1, 2),
    dtype=np.float32,
)

X = vectorizer.fit_transform(items["combined_text"])

item_to_idx = {
    item_id: index
    for index, item_id in enumerate(items["item_id"])
}


# 4. Membentuk riwayat positif pengguna dari data training
# Rating 4 atau 5 dianggap sebagai interaksi positif.
positive_train = train[train["rating"] >= 4]

user_items = (
    positive_train.groupby("user_id")["item_id"]
    .apply(list)
    .to_dict()
)

# Seluruh interaksi training digunakan untuk menyaring item yang sudah dilihat.
train_seen = (
    train.groupby("user_id")["item_id"]
    .apply(set)
    .to_dict()
)


# 5. Menentukan kandidat yang memiliki metadata
candidate_ids = [
    item_id
    for item_id in candidates["item_id"].astype(str).tolist()
    if item_id in item_to_idx
]

candidate_indices = np.array(
    [item_to_idx[item_id] for item_id in candidate_ids],
    dtype=int,
)

candidate_matrix = X[candidate_indices]


# 6. Membuat rekomendasi untuk setiap pengguna evaluasi
rankings = {}

print("Membuat rekomendasi TF-IDF untuk pengguna evaluasi...")

for user_id in users["user_id"].astype(str):
    # Mengambil item positif dalam riwayat training pengguna.
    history = [
        item_to_idx[item_id]
        for item_id in user_items.get(user_id, [])
        if item_id in item_to_idx
    ]

    seen_items = train_seen.get(user_id, set())

    # Pengguna tanpa riwayat positif yang memiliki metadata
    # tidak dapat dibuatkan profil TF-IDF.
    if not history:
        rankings[user_id] = []
        continue

    # Profil pengguna adalah rata-rata vektor TF-IDF item
    # positif dalam riwayat training, lalu dinormalisasi.
    profile = normalize(
        np.asarray(X[history].mean(axis=0))
    )

    # Menghitung kemiripan kosinus antara profil pengguna
    # dan setiap kandidat.
    scores = np.asarray(
        profile @ candidate_matrix.T
    ).ravel()

    # Urutkan berdasarkan skor tertinggi.
    # item_id digunakan sebagai pemecah seri.
    order = np.lexsort(
        (
            np.array(candidate_ids, dtype=object),
            -scores,
        )
    )

    recommendations = []

    for position in order:
        item_id = candidate_ids[position]

        # Mengeluarkan item yang sudah muncul dalam data training.
        if item_id not in seen_items:
            recommendations.append(item_id)

        if len(recommendations) == 10:
            break

    rankings[user_id] = recommendations


# 7. Menyimpan rekomendasi Top-10
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
    RESULT_DIR / "tfidf_cbf_recommendations_top10.csv",
    index=False,
)


# 8. Mengevaluasi hasil dengan fungsi bersama
evaluate_rankings(
    "tfidf_cbf",
    rankings,
    train,
    validation,
    users,
    candidates,
    activity,
)