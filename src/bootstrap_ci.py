"""
Bootstrap Confidence Interval (95%) untuk perbedaan metrik berpasangan antar model (paired user-level).

Metode: Paired user-level bootstrap, 1000 iterasi, seed=42.
Metrik: Recall@10, NDCG@10 (serta HitRate@10 dan MRR@10 bila tersedia di kedua model).
Tingkat analisis: Overall dan per user activity group (Low, Medium, High).

Perbandingan utama sesuai instruksi:
  1. Item-kNN vs Pop Recency
  2. Hybrid New vs Item-kNN
  3. Hybrid New vs Pop Recency
  4. Model Terbaik (Item-kNN untuk NDCG, Pop Recency untuk Recall) vs Popularity Global
  5. Hybrid New vs Popularity Global

Input:  results/{model}_user_metrics.csv
Output: results/bootstrap_ci.csv
"""

import numpy as np
import pandas as pd

from baseline_common import RESULT_DIR

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
METRICS = ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]

COMPARISONS = [
    ("item_knn", "popularity_recency", "Item-kNN vs Pop Recency"),
    ("hybrid_new", "item_knn", "Hybrid New vs Item-kNN"),
    ("hybrid_new", "popularity_recency", "Hybrid New vs Pop Recency"),
    ("item_knn", "popularity", "Item-kNN (Best NDCG) vs Popularity Global"),
    ("popularity_recency", "popularity", "Pop Recency (Best Recall) vs Popularity Global"),
    ("hybrid_new", "popularity", "Hybrid New vs Popularity Global"),
]

GROUPS = ["Overall", "Low", "Medium", "High"]


def load_user_metrics(model_name: str) -> pd.DataFrame:
    path = RESULT_DIR / f"{model_name}_user_metrics.csv"
    if not path.exists():
        # Fallback check for popularity_validation_user_metrics if needed
        if model_name == "popularity":
            alt_path = RESULT_DIR / "popularity_validation_user_metrics.csv"
            if alt_path.exists():
                path = alt_path
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path, dtype={"user_id": str})
    return df.set_index("user_id")


def bootstrap_paired_diff(a: np.ndarray, b: np.ndarray, n: int, seed: int):
    """
    Paired bootstrap: hitung distribusi mean(a) - mean(b) dengan resampling berpasangan.
    Returns: mean_diff, ci_low (2.5%), ci_high (97.5%)
    """
    rng = np.random.default_rng(seed)
    diffs = a - b
    n_obs = len(diffs)
    boot_means = np.array(
        [rng.choice(diffs, size=n_obs, replace=True).mean() for _ in range(n)]
    )
    return diffs.mean(), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def run_bootstrap_ci():
    rows = []

    for comp_entry in COMPARISONS:
        if len(comp_entry) == 3:
            model_a, model_b, comp_label = comp_entry
        else:
            model_a, model_b = comp_entry
            comp_label = f"{model_a} vs {model_b}"

        print(f"\n=======================================================")
        print(f"Comparison: {comp_label} ({model_a} vs {model_b})")
        print(f"=======================================================")

        try:
            df_a = load_user_metrics(model_a)
            df_b = load_user_metrics(model_b)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Temukan user yang ada di kedua file
        common_users = df_a.index.intersection(df_b.index)
        print(f"  User bersama: {len(common_users)}")

        sub_a = df_a.loc[common_users]
        sub_b = df_b.loc[common_users]

        for group in GROUPS:
            if group == "Overall":
                grp_a = sub_a
                grp_b = sub_b
            else:
                # Filter berdasarkan activity_group
                mask_a = grp_a_mask = (sub_a["activity_group"] == group)
                grp_a = sub_a[mask_a]
                grp_b = sub_b[mask_a]

            n_group_users = len(grp_a)
            if n_group_users == 0:
                continue

            for metric in METRICS:
                if metric not in grp_a.columns or metric not in grp_b.columns:
                    continue

                a_vals = grp_a[metric].fillna(0).values.astype(np.float64)
                b_vals = grp_b[metric].fillna(0).values.astype(np.float64)

                mean_diff, ci_low, ci_high = bootstrap_paired_diff(
                    a_vals, b_vals, n=N_BOOTSTRAP, seed=RANDOM_SEED
                )

                mean_a = a_vals.mean()
                mean_b = b_vals.mean()

                # Signifikan jika 0 berada di luar interval 95% CI
                if ci_low > 0:
                    sig = "Ya (+)"
                elif ci_high < 0:
                    sig = "Ya (-)"
                else:
                    sig = "Tidak"

                row = {
                    "comparison": comp_label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "group": group,
                    "metric": metric,
                    "mean_a": round(float(mean_a), 6),
                    "mean_b": round(float(mean_b), 6),
                    "diff_mean": round(float(mean_diff), 6),
                    "ci_low_95": round(float(ci_low), 6),
                    "ci_high_95": round(float(ci_high), 6),
                    "significant": sig,
                    "n_users": n_group_users,
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": RANDOM_SEED,
                }
                rows.append(row)

                print(
                    f"  [{group:<7}] {metric:<10}: {model_a}={mean_a:.6f}, {model_b}={mean_b:.6f}, "
                    f"diff={mean_diff:+.6f} [95% CI: {ci_low:+.6f}, {ci_high:+.6f}] -> {sig}"
                )

    if rows:
        result_df = pd.DataFrame(rows)
        out_path = RESULT_DIR / "bootstrap_ci.csv"
        result_df.to_csv(out_path, index=False)
        print(f"\nHasil bootstrap CI disimpan di: {out_path}")
        return result_df
    else:
        print("\nTidak ada perbandingan yang dapat dihitung.")
        return pd.DataFrame()


if __name__ == "__main__":
    run_bootstrap_ci()
