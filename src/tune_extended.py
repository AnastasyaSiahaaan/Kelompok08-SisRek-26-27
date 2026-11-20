"""
Eksperimen lanjutan (W11):
  1. Perpanjang grid alpha Hybrid Klasik (MF + TF-IDF) ke {0.1, ..., 0.9}.
     Alpha yang sudah ada (0.3, 0.5, 0.7) dilewati (duplikasi dicek di experiment_log).
  2. Jalankan MF dengan seed 123 dan 456 untuk n_components {8, 16, 32, 64, 128}
     (seed 42 sudah dilakukan sebelumnya).

Hasil dicatat ke results/experiment_log.csv.
Test set TIDAK disentuh.

Jalankan: python src/tune_extended.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_common import load_data, evaluate_rankings, RAW_DIR, RESULT_DIR
from config import MF_SEARCH_SPACE, RANDOM_SEED, make_run_id
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


def _existing_run_ids():
    log_path = RESULT_DIR / "experiment_log.csv"
    if not log_path.exists():
        return set()
    df = pd.read_csv(log_path)
    return set(df["run_id"].tolist())


def minmax(scores):
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Part 1: Extended Hybrid Alpha Grid
# ---------------------------------------------------------------------------

HYBRID_ALPHA_FULL = [round(a * 0.1, 1) for a in range(1, 10)]  # 0.1..0.9
MF_N_BEST = 32          # dari tuning sebelumnya
TFIDF_FEAT_BEST = 100_000


def tune_hybrid_extended(train, validation, users, candidates, activity):
    print("\n" + "=" * 60)
    print("TUNING LANJUTAN: Hybrid Alpha {0.1 ... 0.9}")
    print("=" * 60)

    existing = _existing_run_ids()

    # Bangun MF
    R, user_ids, item_ids, user_to_idx, item_to_idx = _build_interaction_matrix(train)
    actual_n = min(MF_N_BEST, min(R.shape) - 1)
    mf_model = TruncatedSVD(n_components=actual_n, n_iter=7, random_state=RANDOM_SEED)
    user_factors = mf_model.fit_transform(R)
    item_factors = mf_model.components_.T

    # Bangun TF-IDF
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_id": str})
    text_cols = [c for c in ["title", "main_category", "categories", "features",
                              "description", "content_text_basic"] if c in items.columns]
    items["combined_text"] = items[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
    items = items.drop_duplicates("item_id").reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_FEAT_BEST,
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

    results = []
    for alpha in HYBRID_ALPHA_FULL:
        run_id = make_run_id("hybrid", {"alpha": alpha})
        if run_id in existing:
            print(f"  Skip (sudah ada): {run_id}")
            # Baca dari log untuk tampilkan ringkasan
            log_df = pd.read_csv(RESULT_DIR / "experiment_log.csv")
            row = log_df[log_df["run_id"] == run_id].iloc[0]
            results.append((alpha, row["recall_at_10"], row["ndcg_at_10"]))
            continue

        print(f"\n  Run: {run_id}  (alpha={alpha})")

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

        # Format key: hybrid_tune_a01, a02, ..., a09
        alpha_str = str(alpha).replace(".", "")  # e.g. "0.1" -> "01"
        if len(alpha_str) == 1:
            alpha_str = "0" + alpha_str
        _model_key = f"hybrid_tune_a{alpha_str}"
        evaluate_rankings(_model_key, rankings, train, validation, users, candidates, activity)
        overall, group, n_users = _extract_metrics(_model_key)

        log_run(run_id, "hybrid",
                {"alpha": alpha, "mf_n": MF_N_BEST, "tfidf_feat": TFIDF_FEAT_BEST},
                overall, group, n_users)
        results.append((alpha, overall["Recall@10"], overall["NDCG@10"]))
        print(f"  alpha={alpha}  Recall@10={overall['Recall@10']:.6f}  NDCG@10={overall['NDCG@10']:.6f}")

    print("\n  Semua alpha:")
    for alpha, rec, ndcg in sorted(results, key=lambda x: x[0]):
        print(f"    alpha={alpha:.1f}  Recall@10={rec:.6f}  NDCG@10={ndcg:.6f}")

    best_ndcg = max(results, key=lambda x: x[2])
    print(f"\n  >> Hybrid terbaik (NDCG): alpha={best_ndcg[0]}  NDCG@10={best_ndcg[2]:.6f}")
    best_recall = max(results, key=lambda x: x[1])
    print(f"  >> Hybrid terbaik (Recall): alpha={best_recall[0]}  Recall@10={best_recall[1]:.6f}")

    return best_ndcg[0]


# ---------------------------------------------------------------------------
# Part 2: MF Multi-Seed
# ---------------------------------------------------------------------------

MF_SEEDS = [42, 123, 456]
MF_N_COMPONENTS_LIST = MF_SEARCH_SPACE["n_components"]  # [8, 16, 32, 64]


def tune_mf_multiseed(train, validation, users, candidates, activity):
    print("\n" + "=" * 60)
    print("TUNING MF MULTI-SEED: seed={42, 123, 456}, n_components={8,16,32,64,128}")
    print("=" * 60)

    existing = _existing_run_ids()
    R, user_ids, item_ids, user_to_idx, item_to_idx = _build_interaction_matrix(train)

    candidate_ids = [
        str(item) for item in candidates["item_id"]
        if str(item) in item_to_idx
    ]
    candidate_indices = np.array([item_to_idx[c] for c in candidate_ids], dtype=int)
    train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    evaluable_users = users["user_id"].astype(str).tolist()

    # n_components termasuk 128 yang sudah ada di log tapi belum di SEARCH_SPACE
    n_components_all = sorted(set(MF_N_COMPONENTS_LIST + [128]))

    results_by_n = {n: [] for n in n_components_all}

    for seed in MF_SEEDS:
        for n_comp in n_components_all:
            run_id = f"mf_components{n_comp}_seed{seed}"
            if run_id in existing:
                print(f"  Skip (sudah ada): {run_id}")
                log_df = pd.read_csv(RESULT_DIR / "experiment_log.csv")
                row = log_df[log_df["run_id"] == run_id].iloc[0]
                results_by_n[n_comp].append((seed, row["recall_at_10"], row["ndcg_at_10"]))
                continue

            print(f"\n  Run: {run_id}")
            actual_n = min(n_comp, min(R.shape) - 1)
            model = TruncatedSVD(n_components=actual_n, n_iter=7, random_state=seed)
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

            _model_key = f"mf_tune_{n_comp}_s{seed}"
            evaluate_rankings(_model_key, rankings, train, validation, users, candidates, activity)
            overall, group, n_users = _extract_metrics(_model_key)

            # log dengan run_id yang mencakup seed
            log_run(run_id, "mf", {"n_components": n_comp, "seed": seed}, overall, group, n_users)
            results_by_n[n_comp].append((seed, overall["Recall@10"], overall["NDCG@10"]))
            print(f"  Recall@10={overall['Recall@10']:.6f}  NDCG@10={overall['NDCG@10']:.6f}")

    # Ringkasan mean ± std per n_components
    print("\n  Ringkasan MF multi-seed (mean ± std NDCG@10):")
    summary = []
    for n_comp in n_components_all:
        vals = results_by_n[n_comp]
        if len(vals) == 0:
            continue
        recalls = [v[1] for v in vals]
        ndcgs = [v[2] for v in vals]
        mean_ndcg = np.mean(ndcgs)
        std_ndcg = np.std(ndcgs)
        mean_rec = np.mean(recalls)
        std_rec = np.std(recalls)
        print(f"    n_comp={n_comp:3d}  Recall@10={mean_rec:.6f}±{std_rec:.6f}"
              f"  NDCG@10={mean_ndcg:.6f}±{std_ndcg:.6f}")
        summary.append((n_comp, mean_rec, std_rec, mean_ndcg, std_ndcg))

    if summary:
        best = max(summary, key=lambda x: x[3])
        print(f"\n  >> MF terbaik (mean NDCG): n_components={best[0]}  NDCG@10={best[3]:.6f}±{best[4]:.6f}")
        return best[0]
    return MF_N_BEST


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Memuat data...")
    train, validation, users, candidates, activity = load_data()

    train = train.copy()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    print("\n--- Part 1: Hybrid Alpha Extended ---")
    best_alpha = tune_hybrid_extended(train, validation, users, candidates, activity)

    print("\n--- Part 2: MF Multi-Seed ---")
    best_mf_n = tune_mf_multiseed(train, validation, users, candidates, activity)

    print("\n" + "=" * 60)
    print("SELESAI. Update config.py jika perlu:")
    print("=" * 60)
    print(f'  HYBRID_CONFIG["alpha"]      = {best_alpha}  # dari extended grid')
    print(f'  MF_CONFIG["n_components"]   = {best_mf_n}  # dari multi-seed')
    print("Log lengkap: results/experiment_log.csv")
