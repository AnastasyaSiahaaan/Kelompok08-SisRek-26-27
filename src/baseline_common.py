from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
SUPPORT_DIR = BASE_DIR / "data" / "supporting"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Memuat data train, validation warm, kandidat, dan aktivitas pengguna."""
    train = pd.read_csv(
        RAW_DIR / "interactions_train.csv",
        dtype={"user_id": str, "item_id": str},
    )

    validation = pd.read_csv(
        SUPPORT_DIR / "interactions_validation_warm.csv",
        dtype={"user_id": str, "item_id": str},
    )

    users = pd.read_csv(
        SUPPORT_DIR / "evaluable_users_validation_warm.csv",
        dtype={"user_id": str},
    )

    candidates = pd.read_csv(
        RAW_DIR / "candidate_items_warm.csv",
        dtype={"item_id": str},
    )

    activity = pd.read_csv(
        SUPPORT_DIR / "user_activity_train.csv",
        dtype={"user_id": str},
    )

    return train, validation, users, candidates, activity


def make_user_groups(activity):
    """
    Membagi pengguna menjadi Low, Medium, dan High berdasarkan
    jumlah interaksi di data training, dengan batas tetap (pd.cut):
    Low = 1-5, Medium = 6-8, High >= 9.

    Pengguna dengan jumlah interaksi sama selalu berada pada
    kelompok yang sama.
    """
    activity = activity.copy()

    activity["activity_group"] = pd.cut(
        activity["train_interactions"],
        bins=[0, 5, 8, float("inf")],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    return activity


def recompute_from_user_metrics(model_name, result_dir=None):
    """
    Hitung ulang _overall_metrics.csv dan _by_activity_group.csv dari
    _user_metrics.csv sebagai satu-satunya sumber kebenaran.

    Gunakan fungsi ini untuk memastikan konsistensi antara file ringkasan
    dan data per-user.

    Parameters
    ----------
    model_name : str
        Nama model; akan membaca ``{model_name}_user_metrics.csv``.
    result_dir : Path, optional
        Direktori hasil; default RESULT_DIR.

    Returns
    -------
    overall_dict : dict
        {"Recall@10": float, "NDCG@10": float, ...} (micro-average)
    group_dict : dict
        {"Low": {...}, "Medium": {...}, "High": {...}}
    n_users : int
    """
    if result_dir is None:
        result_dir = RESULT_DIR
    result_dir = Path(result_dir)

    um = pd.read_csv(
        result_dir / f"{model_name}_user_metrics.csv",
        dtype={"user_id": str},
    )

    metric_cols = [c for c in ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"]
                   if c in um.columns]

    # Overall: micro-average
    overall_vals = um[metric_cols].mean()
    overall_dict = overall_vals.to_dict()

    overall_df = overall_vals.to_frame().T
    overall_df.insert(0, "model", model_name)
    overall_df.to_csv(
        result_dir / f"{model_name}_overall_metrics.csv", index=False
    )

    # Per-group
    group_dict = {}
    if "activity_group" in um.columns:
        agg = {"users": ("user_id", "nunique")}
        for col in metric_cols:
            safe = col.replace("@", "_at_")
            agg[safe] = (col, "mean")
        grouped = (
            um.groupby("activity_group", as_index=False)
            .agg(**agg)
        )
        grouped.insert(0, "model", model_name)
        grouped.to_csv(
            result_dir / f"{model_name}_by_activity_group.csv", index=False
        )
        for _, row in grouped.iterrows():
            grp = row["activity_group"]
            group_dict[grp] = {col: row[col.replace("@", "_at_")] for col in metric_cols
                                if col.replace("@", "_at_") in row}

    return overall_dict, group_dict, len(um)


def evaluate_rankings(
    model_name,
    rankings,
    train,
    validation,
    users,
    candidates,
    activity,
):
    """
    Mengevaluasi rekomendasi menggunakan Recall@10 dan NDCG@10.

    Item relevan: rating validation >= 4.
    Kandidat: candidate_items_warm.csv.
    Item yang sudah muncul di train dikeluarkan dari rekomendasi
    dan dari himpunan item relevan.
    """
    candidate_items = set(candidates["item_id"].astype(str))
    evaluated_users = set(users["user_id"].astype(str))

    train = train.copy()
    validation = validation.copy()

    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)

    validation["user_id"] = validation["user_id"].astype(str)
    validation["item_id"] = validation["item_id"].astype(str)

    # Item yang telah berinteraksi dengan pengguna pada data train.
    seen = (
        train.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # Item relevan dari data validation.
    relevant_rows = validation[
        validation["user_id"].isin(evaluated_users)
        & (validation["rating"] >= 4)
    ]

    relevant = (
        relevant_rows.groupby("user_id")["item_id"]
        .apply(lambda values: set(values.astype(str)) & candidate_items)
        .to_dict()
    )

    # Kelompok aktivitas ditentukan dari data training.
    grouped_activity = make_user_groups(activity)

    group_map = (
        grouped_activity
        .set_index("user_id")["activity_group"]
        .to_dict()
    )

    rows = []

    for user_id in sorted(evaluated_users):
        # Item relevan harus berada dalam kandidat dan belum muncul di train.
        relevant_items = (
            relevant.get(user_id, set())
            - seen.get(user_id, set())
        )

        # Pengguna tanpa item relevan tidak dihitung dalam metrik.
        if not relevant_items:
            continue

        # Pastikan rekomendasi hanya berisi kandidat yang belum dilihat,
        # tanpa duplikasi, dan maksimal 10 item.
        recommendations = []

        for item_id in rankings.get(user_id, []):
            item_id = str(item_id)

            if (
                item_id in candidate_items
                and item_id not in seen.get(user_id, set())
                and item_id not in recommendations
            ):
                recommendations.append(item_id)

            if len(recommendations) == 10:
                break

        hits = [
            1 if item_id in relevant_items else 0
            for item_id in recommendations
        ]

        # Recall@10
        recall = sum(hits) / len(relevant_items)

        # DCG@10
        dcg = sum(
            hit / np.log2(position + 2)
            for position, hit in enumerate(hits)
        )

        # Ideal DCG@10
        ideal_length = min(len(relevant_items), 10)

        idcg = sum(
            1 / np.log2(position + 2)
            for position in range(ideal_length)
        )

        # NDCG@10
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # HitRate@10
        hit_rate = 1.0 if sum(hits) > 0 else 0.0

        # MRR@10
        first_hit = next((i + 1 for i, h in enumerate(hits) if h == 1), 0)
        mrr = 1.0 / first_hit if first_hit > 0 else 0.0

        rows.append({
            "user_id": user_id,
            "activity_group": str(
                group_map.get(user_id, "Unknown")
            ),
            "Recall@10": recall,
            "NDCG@10": ndcg,
            "HitRate@10": hit_rate,
            "MRR@10": mrr,
        })

    user_metrics = pd.DataFrame(rows)

    # Simpan metrik masing-masing pengguna.
    user_path = RESULT_DIR / f"{model_name}_user_metrics.csv"
    user_metrics.to_csv(user_path, index=False)

    if user_metrics.empty:
        print(
            "Tidak ada pengguna yang dapat dievaluasi. "
            "Periksa data dan candidate protocol."
        )
        return

    # Metrik keseluruhan: rata-rata per pengguna.
    metric_cols = [c for c in ["Recall@10", "NDCG@10", "HitRate@10", "MRR@10"] if c in user_metrics.columns]
    overall = (
        user_metrics[metric_cols]
        .mean()
        .to_frame()
        .T
    )

    overall.insert(0, "model", model_name)

    # Metrik berdasarkan kelompok aktivitas pengguna.
    grouped = (
        user_metrics.groupby("activity_group", as_index=False)
        .agg(
            users=("user_id", "nunique"),
            Recall_at_10=("Recall@10", "mean"),
            NDCG_at_10=("NDCG@10", "mean"),
            HitRate_at_10=("HitRate@10", "mean"),
            MRR_at_10=("MRR@10", "mean"),
        )
    )

    grouped.insert(0, "model", model_name)

    # Simpan ringkasan hasil.
    overall.to_csv(
        RESULT_DIR / f"{model_name}_overall_metrics.csv",
        index=False,
    )

    grouped.to_csv(
        RESULT_DIR / f"{model_name}_by_activity_group.csv",
        index=False,
    )

    print(f"\n=== {model_name.upper()} — OVERALL ===")
    print(overall.to_string(index=False))

    print(f"\n=== {model_name.upper()} — BY ACTIVITY GROUP ===")
    print(grouped.to_string(index=False))

    print("Jumlah pengguna dievaluasi:", len(user_metrics))
    print("File hasil:", user_path)