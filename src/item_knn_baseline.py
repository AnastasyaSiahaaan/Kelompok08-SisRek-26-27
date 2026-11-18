"""
Item-kNN Baseline — Item-Item Cosine dengan Shrinkage.

Rumus similarity:
    sim(i, j) = C[i,j] / (norms[i] * norms[j] + shrinkage)
di mana C = R_bin.T @ R_bin (item co-occurrence), R_bin adalah matriks biner
dari interaksi positif (rating >= 4) di training.

Implementasi hemat memori: skor per-user dihitung langsung sebagai
    u_scores[cand] = sum_j( R_bin[u,j] * sim(j, cand) )
tanpa materialize S penuh. Hanya norm tiap item yang disimpan.

Hyperparameter terpilih: shrinkage=500 (dipilih via validation; lihat ITEM_KNN_CONFIG).
Test set TIDAK dipakai di mana pun.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from baseline_common import load_data, evaluate_rankings, RESULT_DIR
from config import ITEM_KNN_CONFIG, POPULARITY_CONFIG, make_run_id


def build_binary_matrix(train, pos_threshold=4):
    """
    Bangun R_bin (CSR, biner positif) dari training.
    Returns R_bin, user_to_idx, item_to_idx, item_ids_list
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
    return R_bin, user_to_idx, item_to_idx, item_ids


def compute_item_norms(R_bin):
    """
    Hitung L2-norm tiap item (kolom dari R_bin).
    norm[j] = sqrt( sum_u R_bin[u,j]^2 ) = sqrt( nnz di kolom j )
    karena R_bin biner.
    """
    # sum of squares per column = number of users who rated item j positively
    norms = np.asarray(np.sqrt(R_bin.power(2).sum(axis=0))).ravel()
    return norms.astype(np.float32)


def score_user(u_hist_indices, R_bin_T, norms, cand_cols, shrinkage):
    """
    Hitung skor Item-kNN yang benar dan tervektor untuk satu user:

        score(c) = sum_{j in H} C[j,c] / (norms[j] * norms[c] + shrinkage)

    Implementasi tervektor (tanpa Python loop):
        C_sub = R_bin_T[u_hist, :] @ R_cand.T  → (|H|, n_cand)
        denom = norms[u_hist][:,None] * norms[cand_cols][None,:] + shrinkage
        scores = (C_sub / denom).sum(axis=0)

    u_hist_indices : ndarray int — indeks item positif user di R_bin
    R_bin_T        : CSR (n_items, n_users)
    norms          : (n_items,) sqrt jumlah interaksi positif per item
    cand_cols      : ndarray int — indeks kandidat di R_bin
    shrinkage      : float
    """
    if len(u_hist_indices) == 0:
        return np.zeros(len(cand_cols), dtype=np.float32)

    # R_cand: (n_cand, n_users) sparse
    R_cand = R_bin_T[cand_cols, :]

    # R_hist: (|H|, n_users) sparse
    R_hist = R_bin_T[u_hist_indices, :]

    # C_sub[j, c] = co-occurrence count item j dan kandidat c
    # = R_hist @ R_cand.T  → dense (|H|, n_cand)
    C_sub = R_hist.dot(R_cand.T).toarray().astype(np.float64)  # (|H|, n_cand)

    # denom[j, c] = norms[j] * norms[c] + shrinkage
    norms_hist = norms[u_hist_indices][:, np.newaxis]   # (|H|, 1)
    norms_cand = norms[cand_cols][np.newaxis, :]        # (1, n_cand)
    denom = norms_hist * norms_cand + shrinkage          # (|H|, n_cand)

    scores = (C_sub / denom).sum(axis=0)                # (n_cand,)
    return scores.astype(np.float32)


def main():
    shrinkage = ITEM_KNN_CONFIG["shrinkage"]
    pos_threshold = POPULARITY_CONFIG["positive_threshold"]
    run_id = make_run_id("item_knn", {"shrinkage": shrinkage})
    print(f"Run ID: {run_id}")

    # 1. Baca dataset
    print("Membaca data training dan validation...")
    train, validation, users, candidates, activity = load_data()

    train = train.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    # 2. Bangun matriks biner
    print("Membangun matriks biner R_bin ...")
    R_bin, user_to_idx, item_to_idx, item_ids_list = build_binary_matrix(
        train, pos_threshold
    )
    print(f"R_bin: {R_bin.shape[0]} users x {R_bin.shape[1]} items, nnz={R_bin.nnz}")

    # Transpose (akan dipakai untuk lookup baris-item)
    R_bin_T = R_bin.T.tocsr()

    # Norm tiap item
    norms = compute_item_norms(R_bin)

    # 3. Filter kandidat ke item yang ada di model
    candidate_ids_all = candidates["item_id"].astype(str).tolist()
    candidate_ids = [item for item in candidate_ids_all if item in item_to_idx]
    candidate_cols = np.array([item_to_idx[item] for item in candidate_ids], dtype=np.int32)
    print(f"Kandidat dalam model: {len(candidate_ids)} / {len(candidate_ids_all)}")

    # 4. Training seen items (semua interaksi)
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    # 5. Hitung skor per user
    print("Menghitung rekomendasi Item-kNN untuk setiap user...")
    rankings = {}
    n_users = len(users)

    for i, user_id in enumerate(users["user_id"].astype(str)):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1}/{n_users}", flush=True)

        seen = train_seen.get(user_id, set())

        if user_id in user_to_idx:
            u_idx = user_to_idx[user_id]
            u_hist_indices = R_bin[u_idx].indices  # indeks item positif user
            scores = score_user(u_hist_indices, R_bin_T, norms, candidate_cols, shrinkage)
        else:
            scores = np.zeros(len(candidate_ids), dtype=np.float32)

        # Masking item yang sudah dilihat
        for j, item in enumerate(candidate_ids):
            if item in seen:
                scores[j] = -np.inf

        valid = np.flatnonzero(np.isfinite(scores))
        k = min(10, len(valid))
        top = sorted(valid, key=lambda j: (-scores[j], candidate_ids[j]))[:k]
        rankings[user_id] = [candidate_ids[j] for j in top]

    # 6. Evaluasi
    evaluate_rankings(
        model_name="item_knn",
        rankings=rankings,
        train=train,
        validation=validation,
        users=users,
        candidates=candidates,
        activity=activity,
    )


if __name__ == "__main__":
    main()
