
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SUPPORT_DIR = BASE_DIR / "data" / "supporting"

train = pd.read_csv(RAW_DIR / "interactions_train.csv")
validation = pd.read_csv(
    SUPPORT_DIR / "interactions_validation_warm.csv"
)
users = pd.read_csv(
    SUPPORT_DIR / "evaluable_users_validation_warm.csv"
)
candidates = pd.read_csv(RAW_DIR / "candidate_items_warm.csv")

candidate_items = set(candidates["item_id"].astype(str))
evaluated_users = set(users["user_id"])

# Item yang pernah berinteraksi dengan pengguna saat training
seen_by_user = (
    train.groupby("user_id")["item_id"]
    .apply(lambda x: set(x.astype(str)))
    .to_dict()
)

# Hanya rating 4 atau 5 yang dianggap relevan
relevant = validation[
    validation["user_id"].isin(evaluated_users)
    & (validation["rating"] >= 4)
].copy()

relevant_by_user = (
    relevant.groupby("user_id")["item_id"]
    .apply(lambda x: set(x.astype(str)) & candidate_items)
    .to_dict()
)

users_with_relevant = 0
users_with_unseen_relevant = 0
total_relevant = 0
total_unseen_relevant = 0

for user_id in evaluated_users:
    items = relevant_by_user.get(user_id, set())
    unseen_items = items - seen_by_user.get(user_id, set())

    if items:
        users_with_relevant += 1
        total_relevant += len(items)

    if unseen_items:
        users_with_unseen_relevant += 1
        total_unseen_relevant += len(unseen_items)

print("=== PEMERIKSAAN EVALUASI ===")
print("Total pengguna evaluasi:", len(evaluated_users))
print("Pengguna dengan item relevan dalam kandidat:", users_with_relevant)
print("Total item relevan dalam kandidat:", total_relevant)
print("Pengguna dengan item relevan yang belum dilihat:", users_with_unseen_relevant)
print("Total item relevan yang belum dilihat:", total_unseen_relevant)
