"""
Langkah 3: Rerun 4 model inti (seed=42) ke results/rerun/.
Bandingkan per-user dengan file resmi (assert max diff < 1e-9).
Test set TIDAK disentuh.

Jalankan: python scripts/rerun_core_models.py
"""
import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from baseline_common import (
    load_data, evaluate_rankings, make_user_groups,
    RAW_DIR, RESULT_DIR
)
from config import (
    POPULARITY_CONFIG, MF_CONFIG, TFIDF_CONFIG, HYBRID_CONFIG,
    RANDOM_SEED
)

RERUN_DIR = RESULT_DIR / "rerun"
RERUN_DIR.mkdir(exist_ok=True)

def compare_to_official(model_name, official_dir=RESULT_DIR, rerun_dir=RERUN_DIR):
    """Compare per-user metrics; assert max diff < 1e-9."""
    off = pd.read_csv(official_dir / f"{model_name}_user_metrics.csv", dtype={"user_id": str})
    new = pd.read_csv(rerun_dir / f"{model_name}_user_metrics.csv", dtype={"user_id": str})
    merged = off.merge(new, on="user_id", suffixes=("_off", "_new"))
    metric_cols = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]
    max_diffs = {}
    for col in metric_cols:
        c_off = f"{col}_off"
        c_new = f"{col}_new"
        if c_off in merged.columns and c_new in merged.columns:
            d = (merged[c_off] - merged[c_new]).abs().max()
            max_diffs[col] = d
    rows_match = len(merged) == len(off) == len(new)
    return rows_match, max_diffs

print("=" * 70)
print("RERUN 4 MODEL INTI -> results/rerun/")
print("=" * 70)
print("Memuat data...")
train, validation, users, candidates, activity = load_data()
train = train.copy()
train["user_id"] = train["user_id"].astype(str)
train["item_id"] = train["item_id"].astype(str)

# ----- Model 1: Popularity -----
print("\n--- Model 1: Popularity ---")
from collections import Counter
positive = train[train["rating"] >= POPULARITY_CONFIG["positive_threshold"]]
item_counts = Counter(positive["item_id"].astype(str))
candidate_items = candidates["item_id"].astype(str).tolist()
sorted_items = sorted(candidate_items, key=lambda x: (-item_counts.get(x, 0), x))
train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
rankings_pop = {}
for uid in users["user_id"].astype(str):
    seen = train_seen.get(uid, set())
    recs = [i for i in sorted_items if i not in seen][:10]
    rankings_pop[uid] = recs
# Save to rerun dir by monkey-patching RESULT_DIR temporarily
import baseline_common
_orig = baseline_common.RESULT_DIR
baseline_common.RESULT_DIR = RERUN_DIR
evaluate_rankings("popularity", rankings_pop, train, validation, users, candidates, activity)
baseline_common.RESULT_DIR = _orig
print("Comparing to official...")
ok, diffs = compare_to_official("popularity")
print(f"  rows_match={ok}, max_diffs={diffs}")
assert ok and all(v < 1e-9 for v in diffs.values()), f"FAIL: {diffs}"
print("  PASS: identik dengan official")

# ----- Model 2: MF -----
print("\n--- Model 2: Matrix Factorization (n=32, seed=42) ---")
user_ids = train["user_id"].drop_duplicates().tolist()
item_ids = train["item_id"].drop_duplicates().tolist()
user_to_idx = {u: i for i, u in enumerate(user_ids)}
item_to_idx = {item: i for i, item in enumerate(item_ids)}
pairs = train.groupby(["user_id","item_id"], as_index=False)["rating"].mean()
R = coo_matrix(
    (pairs["rating"].astype(float).to_numpy(),
     (pairs["user_id"].map(user_to_idx).to_numpy(),
      pairs["item_id"].map(item_to_idx).to_numpy())),
    shape=(len(user_ids), len(item_ids))
).tocsr()
n_comp = min(MF_CONFIG["n_components"], min(R.shape)-1)
mf_model = TruncatedSVD(n_components=n_comp, n_iter=MF_CONFIG["n_iter"],
                         random_state=RANDOM_SEED)
user_factors = mf_model.fit_transform(R)
item_factors = mf_model.components_.T
cand_ids = [str(i) for i in candidates["item_id"] if str(i) in item_to_idx]
cand_idx = np.array([item_to_idx[c] for c in cand_ids], dtype=int)
cand_factors = item_factors[cand_idx]
cand_pos = {item: pos for pos, item in enumerate(cand_ids)}
train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
rankings_mf = {}
for uid in users["user_id"].astype(str):
    if uid not in user_to_idx:
        rankings_mf[uid] = []; continue
    scores = user_factors[user_to_idx[uid]] @ cand_factors.T
    seen = train_seen.get(uid, set())
    for i in seen:
        if i in cand_pos: scores[cand_pos[i]] = -np.inf
    valid = np.flatnonzero(np.isfinite(scores))
    k = min(10, len(valid))
    if k == 0: rankings_mf[uid] = []; continue
    top = valid[np.argpartition(scores[valid], -k)[-k:]]
    top = sorted(top, key=lambda p: (-scores[p], cand_ids[p]))
    rankings_mf[uid] = [cand_ids[p] for p in top]
baseline_common.RESULT_DIR = RERUN_DIR
evaluate_rankings("mf", rankings_mf, train, validation, users, candidates, activity)
baseline_common.RESULT_DIR = _orig
print("Comparing to official...")
ok, diffs = compare_to_official("mf")
print(f"  rows_match={ok}, max_diffs={diffs}")
assert ok and all(v < 1e-9 for v in diffs.values()), f"FAIL: {diffs}"
print("  PASS: identik dengan official")

# ----- Model 3: TF-IDF CBF -----
print("\n--- Model 3: TF-IDF CBF (max_features=100000) ---")
items_df = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})
text_cols = [c for c in ["title","main_category","categories","features",
                           "description","content_text_basic"] if c in items_df.columns]
items_df["combined_text"] = items_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
items_df = items_df.drop_duplicates("item_id").reset_index(drop=True)
vect = TfidfVectorizer(
    max_features=TFIDF_CONFIG["max_features"],
    stop_words=TFIDF_CONFIG["stop_words"],
    min_df=TFIDF_CONFIG["min_df"],
    ngram_range=TFIDF_CONFIG["ngram_range"],
    dtype=np.float32
)
X = vect.fit_transform(items_df["combined_text"])
item_text_idx = {item: i for i, item in enumerate(items_df["item_id"])}
cand_ids_cbf = [str(i) for i in candidates["item_id"] if str(i) in item_text_idx]
cand_mat = X[np.array([item_text_idx[c] for c in cand_ids_cbf], dtype=int)]
positive_train = train[train["rating"] >= TFIDF_CONFIG["positive_threshold"]]
user_items = positive_train.groupby("user_id")["item_id"].apply(list).to_dict()
train_seen_s = train.groupby("user_id")["item_id"].apply(set).to_dict()
rankings_cbf = {}
for uid in users["user_id"].astype(str):
    hist = [item_text_idx[i] for i in user_items.get(uid,[]) if i in item_text_idx]
    seen = train_seen_s.get(uid, set())
    if not hist: rankings_cbf[uid] = []; continue
    profile = normalize(np.asarray(X[hist].mean(axis=0)))
    scores = np.asarray(profile @ cand_mat.T).ravel()
    order = np.lexsort((np.array(cand_ids_cbf, dtype=object), -scores))
    recs = []
    for pos in order:
        item = cand_ids_cbf[pos]
        if item not in seen: recs.append(item)
        if len(recs) == 10: break
    rankings_cbf[uid] = recs
baseline_common.RESULT_DIR = RERUN_DIR
evaluate_rankings("tfidf_cbf", rankings_cbf, train, validation, users, candidates, activity)
baseline_common.RESULT_DIR = _orig
print("Comparing to official...")
ok, diffs = compare_to_official("tfidf_cbf")
print(f"  rows_match={ok}, max_diffs={diffs}")
assert ok and all(v < 1e-9 for v in diffs.values()), f"FAIL: {diffs}"
print("  PASS: identik dengan official")

# ----- Model 4: Hybrid (MF + CBF, alpha=0.5) -----
print("\n--- Model 4: Hybrid (alpha=0.5) ---")
alpha = HYBRID_CONFIG["alpha"]  # 0.5
cand_hybrid = [str(i) for i in candidates["item_id"]
               if str(i) in item_to_idx and str(i) in item_text_idx]
cmf = item_factors[np.array([item_to_idx[c] for c in cand_hybrid], dtype=int)]
ctxt = X[np.array([item_text_idx[c] for c in cand_hybrid], dtype=int)]
positive_hist = train[train["rating"] >= 4].groupby("user_id")["item_id"].apply(list).to_dict()
def minmax(arr):
    lo, hi = arr.min(), arr.max()
    if hi == lo: return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)
rankings_hyb = {}
for uid in users["user_id"].astype(str):
    seen = train_seen.get(uid, set())
    mf_sc = user_factors[user_to_idx[uid]] @ cmf.T if uid in user_to_idx else np.zeros(len(cand_hybrid))
    hist = [item_text_idx[i] for i in positive_hist.get(uid,[]) if i in item_text_idx]
    cbf_sc = np.asarray(normalize(np.asarray(X[hist].mean(axis=0))) @ ctxt.T).ravel() if hist else np.zeros(len(cand_hybrid))
    hyb_sc = alpha * minmax(mf_sc) + (1 - alpha) * minmax(cbf_sc)
    for i, item in enumerate(cand_hybrid):
        if item in seen: hyb_sc[i] = -np.inf
    valid = np.flatnonzero(np.isfinite(hyb_sc))
    top = sorted(valid, key=lambda i: (-hyb_sc[i], cand_hybrid[i]))[:min(10, len(valid))]
    rankings_hyb[uid] = [cand_hybrid[i] for i in top]
baseline_common.RESULT_DIR = RERUN_DIR
evaluate_rankings("hybrid", rankings_hyb, train, validation, users, candidates, activity)
baseline_common.RESULT_DIR = _orig
print("Comparing to official...")
ok, diffs = compare_to_official("hybrid")
print(f"  rows_match={ok}, max_diffs={diffs}")
assert ok and all(v < 1e-9 for v in diffs.values()), f"FAIL: {diffs}"
print("  PASS: identik dengan official")

print("\n" + "=" * 70)
print("SEMUA 4 MODEL RERUN PASS: identik dengan official (max diff < 1e-9)")
print("=" * 70)
