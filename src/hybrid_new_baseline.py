"""
Hybrid Baru — 3-Way Fusion: Item-kNN + TF-IDF CBF + Popularity Recency (6M).

Kombinasi bobot (w_knn, w_cbf, w_pop6m) dipilih dari validasi:
    terbaik: w_knn=0.4, w_cbf=0.4, w_pop6m=0.2  → NDCG@10=0.016582
Normalisasi: minmax per-user untuk kNN dan CBF; global minmax untuk Pop6M.

Implementasi hemat memori untuk kNN: tidak pernah materialize S penuh.
Scoring kNN tervektor per user: C_sub = R_hist @ R_cand.T (sparse).

Lihat HYBRID_NEW_CONFIG di config.py untuk nilai terpilih.
Test set TIDAK dipakai di mana pun.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from baseline_common import load_data, evaluate_rankings, RAW_DIR, RESULT_DIR
from config import (
    ITEM_KNN_CONFIG,
    TFIDF_CONFIG,
    POP_RECENCY_CONFIG,
    HYBRID_NEW_CONFIG,
    POPULARITY_CONFIG,
    make_run_id,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_popularity_6m(train, window_days, pos_threshold):
    """Return array skor pop-recency per kandidat (global minmax)."""
    positive = train[train["rating"] >= pos_threshold].copy()
    ts = positive["timestamp"].astype(float)
    if ts.max() > 1e11:
        ts = ts / 1000.0
    positive = positive.copy()
    positive["_ts"] = ts
    max_ts = ts.max()
    cutoff = max_ts - window_days * 24 * 3600
    recent = positive[positive["_ts"] >= cutoff]
    pop = recent.groupby("item_id").size().to_dict()
    return pop


def build_binary_matrix(train, pos_threshold):
    """
    Bangun R_bin (CSR biner positif).
    Returns R_bin, R_bin_T, norms, user_to_idx, item_to_idx
    """
    positive = train[train["rating"] >= pos_threshold].copy()
    positive["user_id"] = positive["user_id"].astype(str)
    positive["item_id"] = positive["item_id"].astype(str)
    positive = positive.drop_duplicates(["user_id", "item_id"])

    user_ids = positive["user_id"].unique().tolist()
    item_ids = positive["item_id"].unique().tolist()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {item: i for i, item in enumerate(item_ids)}

    rows = positive["user_id"].map(user_to_idx).values
    cols = positive["item_id"].map(item_to_idx).values
    data = np.ones(len(positive), dtype=np.float32)

    R_bin = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    R_bin_T = R_bin.T.tocsr()
    norms = np.asarray(np.sqrt(R_bin.power(2).sum(axis=0))).ravel().astype(np.float32)

    print(f"[kNN] R_bin: {len(user_ids)} users x {len(item_ids)} items, nnz={R_bin.nnz}")
    return R_bin, R_bin_T, norms, user_to_idx, item_to_idx


def knn_score_user(u_hist_indices, R_bin_T, norms, cand_cols, shrinkage):
    """
    Tervektor: score(c) = sum_{j in H} C[j,c] / (norms[j]*norms[c] + shrinkage).
    C_sub = R_hist @ R_cand.T (sparse matmul, bukan Python loop).
    """
    if len(u_hist_indices) == 0:
        return np.zeros(len(cand_cols), dtype=np.float32)

    R_hist = R_bin_T[u_hist_indices, :]        # (|H|, n_users)
    R_cand = R_bin_T[cand_cols, :]             # (n_cand, n_users)
    C_sub  = R_hist.dot(R_cand.T).toarray().astype(np.float64)  # (|H|, n_cand)

    norms_hist = norms[u_hist_indices][:, np.newaxis]  # (|H|, 1)
    norms_cand = norms[cand_cols][np.newaxis, :]       # (1, n_cand)
    denom = norms_hist * norms_cand + shrinkage         # (|H|, n_cand)

    return (C_sub / denom).sum(axis=0).astype(np.float32)  # (n_cand,)


def build_tfidf(train, pos_threshold, tfidf_config):
    """
    Bangun X TF-IDF, item_to_idx, dan riwayat positif per user (sebagai list indeks).
    """
    items_df = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})
    items_df["item_id"] = items_df["item_id"].astype(str)

    text_cols = [
        c for c in [
            "title", "main_category", "categories",
            "features", "description", "content_text_basic",
        ]
        if c in items_df.columns
    ]
    items_df["combined_text"] = (
        items_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    )
    items_df = items_df.drop_duplicates("item_id").reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        stop_words=tfidf_config["stop_words"],
        min_df=tfidf_config["min_df"],
        ngram_range=tfidf_config["ngram_range"],
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(items_df["combined_text"])
    item_to_idx = {item: i for i, item in enumerate(items_df["item_id"])}

    positive = train[train["rating"] >= pos_threshold].copy()
    positive["user_id"] = positive["user_id"].astype(str)
    positive["item_id"] = positive["item_id"].astype(str)

    user_items = (
        positive.groupby("user_id")["item_id"]
        .apply(lambda s: [item_to_idx[i] for i in s if i in item_to_idx])
        .to_dict()
    )

    print(f"[CBF] TF-IDF matrix: {X.shape[0]} items x {X.shape[1]} features")
    return X, item_to_idx, user_items


def minmax(scores):
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    w_knn   = HYBRID_NEW_CONFIG["w_knn"]
    w_cbf   = HYBRID_NEW_CONFIG["w_cbf"]
    w_pop6m = HYBRID_NEW_CONFIG["w_pop6m"]
    shrinkage   = ITEM_KNN_CONFIG["shrinkage"]
    window_days = POP_RECENCY_CONFIG["window_days"]
    pos_threshold = POPULARITY_CONFIG["positive_threshold"]

    run_id = make_run_id(
        "hybrid_new",
        {"wknn": w_knn, "wcbf": w_cbf, "wpop": w_pop6m},
    )
    print(f"Run ID: {run_id}")
    print(f"Bobot: kNN={w_knn}, CBF={w_cbf}, Pop6M={w_pop6m}")

    # 1. Load data
    print("\nMembaca data training dan validation...")
    train, validation, users, candidates, activity = load_data()
    train = train.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    candidate_ids_all = candidates["item_id"].astype(str).tolist()
    cand_global_map = {item: i for i, item in enumerate(candidate_ids_all)}
    n_cand_all = len(candidate_ids_all)

    # 2. Build Popularity 6M (global scores, normalized once)
    print("\nMembangun Popularity Recency (6M)...")
    pop_counts = build_popularity_6m(train, window_days, pos_threshold)
    pop_values = np.array(
        [pop_counts.get(item, 0) for item in candidate_ids_all],
        dtype=np.float64,
    )
    lo, hi = pop_values.min(), pop_values.max()
    pop_norm_global = (pop_values - lo) / (hi - lo) if hi > lo else np.zeros(n_cand_all)
    print(f"  Items dengan popularitas dalam 6M: {sum(v > 0 for v in pop_values)}")

    # 3. Build Item-kNN (memory-efficient)
    print("\nMembangun Item-kNN ...")
    R_bin, R_bin_T, knn_norms, knn_user_to_idx, knn_item_to_idx = build_binary_matrix(
        train, pos_threshold
    )
    # Kandidat yang ada di kNN model
    candidate_knn_ids  = [item for item in candidate_ids_all if item in knn_item_to_idx]
    candidate_knn_cols = np.array([knn_item_to_idx[item] for item in candidate_knn_ids], dtype=np.int32)
    # Posisi global tiap kandidat kNN
    candidate_knn_global = np.array([cand_global_map[item] for item in candidate_knn_ids], dtype=np.int32)
    print(f"  kNN candidates in model: {len(candidate_knn_ids)}/{n_cand_all}")

    # 4. Build TF-IDF CBF
    print("\nMembangun TF-IDF CBF ...")
    X_tfidf, cbf_item_to_idx, cbf_user_items = build_tfidf(
        train, pos_threshold, TFIDF_CONFIG
    )
    candidate_cbf_ids  = [item for item in candidate_ids_all if item in cbf_item_to_idx]
    candidate_cbf_idx  = np.array([cbf_item_to_idx[item] for item in candidate_cbf_ids], dtype=np.int32)
    candidate_cbf_global = np.array([cand_global_map[item] for item in candidate_cbf_ids], dtype=np.int32)
    X_cand_cbf = X_tfidf[candidate_cbf_idx]
    print(f"  CBF candidates in model: {len(candidate_cbf_ids)}/{n_cand_all}")

    # 5. Training seen items
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    # 6. Hitung rekomendasi untuk setiap user
    print("\nMenghitung rekomendasi Hybrid (3-way) untuk setiap user...")
    rankings = {}
    n_users = len(users)

    for i, user_id in enumerate(users["user_id"].astype(str)):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_users}", flush=True)

        seen = train_seen.get(user_id, set())
        scores = np.zeros(n_cand_all, dtype=np.float64)

        # --- kNN scores (vectorized) ---
        if user_id in knn_user_to_idx:
            u_idx = knn_user_to_idx[user_id]
            u_hist = R_bin[u_idx].indices
            knn_raw = knn_score_user(u_hist, R_bin_T, knn_norms, candidate_knn_cols, shrinkage)
            knn_norm = minmax(knn_raw)
            scores[candidate_knn_global] += w_knn * knn_norm

        # --- CBF scores ---
        history_idx = cbf_user_items.get(user_id, [])
        if history_idx:
            profile = normalize(np.asarray(X_tfidf[history_idx].mean(axis=0)))
            cbf_raw = np.asarray(profile @ X_cand_cbf.T).ravel().astype(np.float64)
            cbf_norm = minmax(cbf_raw)
            scores[candidate_cbf_global] += w_cbf * cbf_norm

        # --- Pop6M scores (global, pre-computed) ---
        scores += w_pop6m * pop_norm_global

        # Masking seen items
        seen_indices = [cand_global_map[item] for item in seen if item in cand_global_map]
        if seen_indices:
            scores[seen_indices] = -np.inf

        valid = np.flatnonzero(np.isfinite(scores))
        k = min(10, len(valid))
        top = sorted(valid, key=lambda j: (-scores[j], candidate_ids_all[j]))[:k]
        rankings[user_id] = [candidate_ids_all[j] for j in top]

    # 7. Evaluasi
    evaluate_rankings(
        model_name="hybrid_new",
        rankings=rankings,
        train=train,
        validation=validation,
        users=users,
        candidates=candidates,
        activity=activity,
    )


if __name__ == "__main__":
    main()
