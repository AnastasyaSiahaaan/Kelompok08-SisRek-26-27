
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Lokasi folder
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SUPPORT_DIR = BASE_DIR / "data" / "supporting"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Baca data
print("Membaca data...")

train = pd.read_csv(RAW_DIR / "interactions_train.csv")
validation = pd.read_csv(
    SUPPORT_DIR / "interactions_validation_warm.csv"
)
evaluable_users = pd.read_csv(
    SUPPORT_DIR / "evaluable_users_validation_warm.csv"
)
candidates = pd.read_csv(RAW_DIR / "candidate_items_warm.csv")

# 3. Siapkan daftar kandidat item
candidate_items = set(candidates["item_id"].astype(str))

# 4. Buat ranking popularitas dari training saja
positive_train = train[train["rating"] >= 4]

popularity = (
    positive_train.groupby("item_id")
    .size()
    .sort_values(ascending=False)
)

# Urutan item berdasarkan popularitas.
# Tie-break berdasarkan item_id agar hasil konsisten.
ranking = (
    popularity.rename("positive_interactions")
    .reset_index()
    .sort_values(
        ["positive_interactions", "item_id"],
        ascending=[False, True]
    )
)

# Hanya item yang masuk daftar kandidat resmi
ranking = ranking[
    ranking["item_id"].astype(str).isin(candidate_items)
]

ranked_items = ranking["item_id"].astype(str).tolist()

print("Jumlah pengguna evaluasi:", len(evaluable_users))
print("Jumlah kandidat:", len(candidate_items))
print("Jumlah item dalam ranking:", len(ranked_items))

# 5. Item yang sudah pernah diinteraksikan pengguna saat training
seen_by_user = (
    train.groupby("user_id")["item_id"]
    .apply(lambda x: set(x.astype(str)))
    .to_dict()
)

# 6. Item relevan pada validasi: rating 4 atau 5
validation = validation[
    validation["user_id"].isin(evaluable_users["user_id"])
    & (validation["rating"] >= 4)
].copy()

relevant_by_user = (
    validation.groupby("user_id")["item_id"]
    .apply(lambda x: set(x.astype(str)) & candidate_items)
    .to_dict()
)

# 7. Fungsi evaluasi Recall@10 dan NDCG@10
def evaluate_user(user_id, relevant_items):
    seen_items = seen_by_user.get(user_id, set())

    # Jangan rekomendasikan item yang sudah dilihat saat training
    recommendations = [
        item for item in ranked_items
        if item not in seen_items
    ][:10]

    if not relevant_items:
        return None

    hits = [
        1 if item in relevant_items else 0
        for item in recommendations
    ]

    recall = sum(hits) / len(relevant_items)

    dcg = sum(
        hit / np.log2(position + 2)
        for position, hit in enumerate(hits)
    )

    ideal_hits = min(len(relevant_items), 10)
    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg

# 8. Hitung metrik untuk pengguna yang memiliki item relevan
results = []

for user_id in evaluable_users["user_id"]:
    relevant_items = relevant_by_user.get(user_id, set())

    if not relevant_items:
        continue

    scores = evaluate_user(user_id, relevant_items)

    if scores is not None:
        results.append({
            "user_id": user_id,
            "Recall@10": scores[0],
            "NDCG@10": scores[1]
        })

result_df = pd.DataFrame(results)

# 9. Simpan hasil per pengguna
output_path = RESULT_DIR / "popularity_validation_user_metrics.csv"
result_df.to_csv(output_path, index=False)

# Simpan metrik keseluruhan dengan format yang sama seperti baseline lain.
overall_metrics = pd.DataFrame([{
    "model": "popularity",
    "Recall@10": result_df["Recall@10"].mean(),
    "NDCG@10": result_df["NDCG@10"].mean(),
}])
overall_output_path = RESULT_DIR / "popularity_overall_metrics.csv"
overall_metrics.to_csv(overall_output_path, index=False)

print("\n=== HASIL EVALUASI POPULARITY ===")
print("Pengguna yang berhasil dievaluasi:", len(result_df))

if not result_df.empty:
    print("Mean Recall@10:", round(result_df["Recall@10"].mean(), 4))
    print("Mean NDCG@10:", round(result_df["NDCG@10"].mean(), 4))

print("Hasil per pengguna disimpan di:", output_path)
print("Metrik keseluruhan disimpan di:", overall_output_path)
