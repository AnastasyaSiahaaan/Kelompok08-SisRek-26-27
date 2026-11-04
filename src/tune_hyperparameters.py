"""
Hyperparameter tuning via validation set.
Menjalankan semua kombinasi search space dari config.py,
mencatat setiap run ke experiment_log.csv, dan mencetak
konfigurasi terbaik berdasarkan Recall@10 pada validation.

PERHATIAN: tuning HANYA menggunakan validation. Test set tidak disentuh.

Cara menjalankan:
    python src/tune_hyperparameters.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Pastikan src/ ada di path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_common import load_data, evaluate_rankings, RAW_DIR, RESULT_DIR
from config import (
    MF_SEARCH_SPACE,
    TFIDF_SEARCH_SPACE,
    HYBRID_SEARCH_SPACE,
    RANDOM_SEED,
    make_run_id,
)
from experiment_log import log_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_interaction_matrix(train):
    user_ids = train["user_id"].drop_duplicates().tolist()
    item_ids = train["item_id"].drop_duplicates().tolist()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {item: i for i, item in enumerate(item_ids)}

    pairs = train.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
    R = coo_matrix(
        (
            pairs["rating"].astype(float).to_numpy(),
            (
                pairs["user_id"].map(user_to_idx).to_numpy(),
                pairs["item_id"].map(item_to_idx).to_numpy(),
            ),
        ),
        shape=(len(user_ids), len(item_ids)),
    ).tocsr()
    return R, user_ids, item_ids, user_to_idx, item_to_idx


def _extract_metrics(model_name, result_dir=RESULT_DIR):
    """Membaca metrik dari file hasil yang baru ditulis."""
    overall = pd.read_csv(result_dir / f"{model_name}_overall_metrics.csv")
    by_group = pd.read_csv(result_dir / f"{model_name}_by_activity_group.csv")
    user_metrics = pd.read_csv(result_dir / f"{model_name}_user_metrics.csv")

    overall_dict = {
        "Recall@10": overall["Recall@10"].iloc[0],
        "NDCG@10": overall["NDCG@10"].iloc[0],
    }

    group_dict = {}
    for _, row in by_group.iterrows():
        group_dict[row["activity_group"]] = {
            "Recall@10": row["Recall_at_10"],
            "NDCG@10": row["NDCG_at_10"],
        }

    return overall_dict, group_dict, len(user_metrics)


# ---------------------------------------------------------------------------
# Tuning MF
# ---------------------------------------------------------------------------

def tune_mf(train, validation, users, candidates, activity):
    print("\n" + "=" * 60)
    print("TUNING: Matrix Factorization")
    print("=" * 60)

    results = []
    R, user_ids, item_ids, user_to_idx, item_to_idx = _build_interaction_matrix(train)

    candidate_ids = [
        str(item) for item in candidates["item_id"]
        if str(item) in item_to_idx
    ]
    candidate_indices = np.array([item_to_idx[c] for c in candidate_ids], dtype=int)
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    evaluable_users = users["user_id"].astype(str).tolist()

    for n_comp in MF_SEARCH_SPACE["n_components"]:
        run_id = make_run_id("mf", {"components": n_comp})
        print(f"\n  Run: {run_id}")

        actual_n = min(n_comp, min(R.shape) - 1)
        model = TruncatedSVD(
            n_components=actual_n,
            n_iter=7,
            random_state=RANDOM_SEED,
        )
        user_factors = model.fit_transform(R)
        item_factors = model.components_.T
        candidate_factors = item_factors[candidate_indices]
        candidate_position = {item: pos for pos, item in enumerate(candidate_ids)}

        rankings = {}
        for user_id in evaluable_users:
            if user_id not in user_to_idx:
                rankings[user_id] = []
                continue
            scores = user_factors[user_to_idx[user_id]] @ candidate_factors.T
            seen = train_seen.get(user_id, set())
            seen_pos = [candidate_position[i] for i in seen if i in candidate_position]
            scores[seen_pos] = -np.inf
            valid = np.flatnonzero(np.isfinite(scores))
            k = min(10, len(valid))
            if k == 0:
                rankings[user_id] = []
                continue
            top = valid[np.argpartition(scores[valid], -k)[-k:]]
            top = sorted(top, key=lambda p: (-scores[p], candidate_ids[p]))
            rankings[user_id] = [candidate_ids[p] for p in top]

        # Tulis sementara ke result dir dengan suffix run_id agar tidak menimpa
        _model_key = f"mf_tune_{n_comp}"
        evaluate_rankings(_model_key, rankings, train, validation, users, candidates, activity)
        overall, group, n_users = _extract_metrics(_model_key)

        log_run(run_id, "mf", {"n_components": n_comp}, overall, group, n_users)
        results.append((n_comp, overall["Recall@10"], overall["NDCG@10"]))
        print(f"  Recall@10={overall['Recall@10']:.4f}  NDCG@10={overall['NDCG@10']:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  >> MF terbaik: n_components={best[0]}  Recall@10={best[1]:.4f}")
    return best[0]


# ---------------------------------------------------------------------------
# Tuning TF-IDF CBF
# ---------------------------------------------------------------------------

def tune_tfidf(train, validation, users, candidates, activity):
    print("\n" + "=" * 60)
    print("TUNING: TF-IDF CBF")
    print("=" * 60)

    results = []
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})
    text_cols = [c for c in ["title", "main_category", "categories", "features",
                              "description", "content_text_basic"] if c in items.columns]
    items["combined_text"] = items[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    items = items.drop_duplicates("item_id").reset_index(drop=True)

    positive_train = train[train["rating"] >= 4]
    user_items = positive_train.groupby("user_id")["item_id"].apply(list).to_dict()
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    for max_feat in TFIDF_SEARCH_SPACE["max_features"]:
        run_id = make_run_id("tfidf_cbf", {"maxfeat": max_feat})
        print(f"\n  Run: {run_id}")

        vectorizer = TfidfVectorizer(
            max_features=max_feat,
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        X = vectorizer.fit_transform(items["combined_text"])
        item_to_idx = {item: i for i, item in enumerate(items["item_id"])}

        candidate_ids = [str(i) for i in candidates["item_id"] if str(i) in item_to_idx]
        candidate_matrix = X[np.array([item_to_idx[c] for c in candidate_ids], dtype=int)]

        rankings = {}
        for user_id in users["user_id"].astype(str):
            history = [item_to_idx[i] for i in user_items.get(user_id, []) if i in item_to_idx]
            seen = train_seen.get(user_id, set())
            if not history:
                rankings[user_id] = []
                continue
            profile = normalize(np.asarray(X[history].mean(axis=0)))
            scores = np.asarray(profile @ candidate_matrix.T).ravel()
            order = np.lexsort((np.array(candidate_ids, dtype=object), -scores))
            recs = []
            for pos in order:
                item = candidate_ids[pos]
                if item not in seen:
                    recs.append(item)
                if len(recs) == 10:
                    break
            rankings[user_id] = recs

        _model_key = f"tfidf_tune_{max_feat}"
        evaluate_rankings(_model_key, rankings, train, validation, users, candidates, activity)
        overall, group, n_users = _extract_metrics(_model_key)

        log_run(run_id, "tfidf_cbf", {"max_features": max_feat}, overall, group, n_users)
        results.append((max_feat, overall["Recall@10"], overall["NDCG@10"]))
        print(f"  Recall@10={overall['Recall@10']:.4f}  NDCG@10={overall['NDCG@10']:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  >> TF-IDF terbaik: max_features={best[0]}  Recall@10={best[1]:.4f}")
    return best[0]


# ---------------------------------------------------------------------------
# Tuning Hybrid
# ---------------------------------------------------------------------------

def tune_hybrid(train, validation, users, candidates, activity, best_mf_n, best_tfidf_feat):
    print("\n" + "=" * 60)
    print("TUNING: Hybrid (MF + CBF)")
    print("=" * 60)

    # Bangun MF dengan n_components terbaik
    R, user_ids, item_ids, user_to_idx, item_to_idx = _build_interaction_matrix(train)
    actual_n = min(best_mf_n, min(R.shape) - 1)
    mf_model = TruncatedSVD(n_components=actual_n, n_iter=7, random_state=RANDOM_SEED)
    user_factors = mf_model.fit_transform(R)
    item_factors = mf_model.components_.T

    # Bangun TF-IDF dengan max_features terbaik
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})
    text_cols = [c for c in ["title", "main_category", "categories", "features",
                              "description", "content_text_basic"] if c in items.columns]
    items["combined_text"] = items[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    items = items.drop_duplicates("item_id").reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=best_tfidf_feat,
        stop_words="english",
        min_df=2,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    X = vectorizer.fit_transform(items["combined_text"])
    item_text_idx = {item: i for i, item in enumerate(items["item_id"])}

    candidate_ids = [
        str(i) for i in candidates["item_id"]
        if str(i) in item_to_idx and str(i) in item_text_idx
    ]
    cand_mf = item_factors[np.array([item_to_idx[c] for c in candidate_ids], dtype=int)]
    cand_text = X[np.array([item_text_idx[c] for c in candidate_ids], dtype=int)]

    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    positive_hist = train[train["rating"] >= 4].groupby("user_id")["item_id"].apply(list).to_dict()

    def minmax(scores):
        lo, hi = scores.min(), scores.max()
        if hi == lo:
            return np.zeros_like(scores, dtype=float)
        return (scores - lo) / (hi - lo)

    results = []
    for alpha in HYBRID_SEARCH_SPACE["alpha"]:
        run_id = make_run_id("hybrid", {"alpha": alpha})
        print(f"\n  Run: {run_id}")

        rankings = {}
        for user_id in users["user_id"].astype(str):
            seen = train_seen.get(user_id, set())
            mf_scores = (
                user_factors[user_to_idx[user_id]] @ cand_mf.T
                if user_id in user_to_idx else np.zeros(len(candidate_ids))
            )
            hist_idx = [item_text_idx[i] for i in positive_hist.get(user_id, []) if i in item_text_idx]
            if hist_idx:
                profile = normalize(np.asarray(X[hist_idx].mean(axis=0)))
                cbf_scores = np.asarray(profile @ cand_text.T).ravel()
            else:
                cbf_scores = np.zeros(len(candidate_ids))

            hybrid = alpha * minmax(mf_scores) + (1 - alpha) * minmax(cbf_scores)
            for i, item in enumerate(candidate_ids):
                if item in seen:
                    hybrid[i] = -np.inf

            valid = np.flatnonzero(np.isfinite(hybrid))
            top = sorted(valid, key=lambda i: (-hybrid[i], candidate_ids[i]))[:min(10, len(valid))]
            rankings[user_id] = [candidate_ids[i] for i in top]

        _model_key = f"hybrid_tune_a{str(alpha).replace('.','')}"
        evaluate_rankings(_model_key, rankings, train, validation, users, candidates, activity)
        overall, group, n_users = _extract_metrics(_model_key)

        log_run(run_id, "hybrid", {"alpha": alpha, "mf_n": best_mf_n, "tfidf_feat": best_tfidf_feat},
                overall, group, n_users)
        results.append((alpha, overall["Recall@10"], overall["NDCG@10"]))
        print(f"  alpha={alpha}  Recall@10={overall['Recall@10']:.4f}  NDCG@10={overall['NDCG@10']:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  >> Hybrid terbaik: alpha={best[0]}  Recall@10={best[1]:.4f}")
    return best[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Memuat data...")
    train, validation, users, candidates, activity = load_data()

    train = train.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    best_mf_n = tune_mf(train, validation, users, candidates, activity)
    best_tfidf_feat = tune_tfidf(train, validation, users, candidates, activity)
    best_alpha = tune_hybrid(train, validation, users, candidates, activity, best_mf_n, best_tfidf_feat)

    print("\n" + "=" * 60)
    print("HASIL TUNING — UPDATE config.py dengan nilai berikut:")
    print("=" * 60)
    print(f'  MF_CONFIG["n_components"]     = {best_mf_n}')
    print(f'  TFIDF_CONFIG["max_features"]  = {best_tfidf_feat}')
    print(f'  HYBRID_CONFIG["alpha"]        = {best_alpha}')
    print("\nLog lengkap: results/experiment_log.csv")
