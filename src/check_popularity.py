import pandas as pd

c = pd.read_csv("data/raw/candidate_items_warm.csv")
r = pd.read_csv("results/popularity_ranking.csv")

print("Jumlah kandidat:", c["item_id"].nunique())
print("Jumlah item ranking:", r["item_id"].nunique())
print("Kandidat yang belum masuk ranking:",
      len(set(c["item_id"]) - set(r["item_id"])))