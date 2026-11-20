"""
Hitung HitRate@10 dan MRR@10 dari recommendations_top10.csv yang sudah ada,
untuk model yang user_metrics-nya belum punya kolom tersebut.

Script ini deterministik — menggunakan rekomendasi yang sudah di-freeze,
sehingga Recall@10 dan NDCG@10 yang dihasilkan harus identik bit-untuk-bit
dengan yang ada di user_metrics.csv saat ini.

Jalankan: python src/add_hitrate_mrr.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from baseline_common import load_data, make_user_groups, RESULT_DIR


def compute_full_metrics_from_recs(model_name: str, recs_path: Path):
    """
    Hitung semua 4 metrik dari file recommendations_top10.csv yang sudah ada.
    Verifikasi bahwa Recall@10 dan NDCG@10 identik dengan user_metrics.csv lama.
    """
    train, validation, users, candidates, activity = load_data()
    train["user_id"] = train["user_id"].astype(str)
    train["item_id"] = train["item_id"].astype(str)
    validation["user_id"] = validation["user_id"].astype(str)
    validation["item_id"] = validation["item_id"].astype(str)

    candidate_items = set(candidates["item_id"].astype(str))
    evaluated_users = set(users["user_id"].astype(str))

    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

    relevant_rows = validation[
        validation["user_id"].isin(evaluated_users) & (validation["rating"] >= 4)
    ]
    relevant = (
        relevant_rows.groupby("user_id")["item_id"]
        .apply(lambda v: set(v.astype(str)) & candidate_items)
        .to_dict()
    )

    grouped = make_user_groups(activity)
    group_map = grouped.set_index("user_id")["activity_group"].to_dict()

    # Baca rekomendasi
    recs_df = pd.read_csv(recs_path, dtype={"user_id": str, "item_id": str})
    recs_by_user = (
        recs_df.sort_values("rank")
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    rows = []
    for user_id in sorted(evaluated_users):
        rel = relevant.get(user_id, set()) - seen.get(user_id, set())
        if not rel:
            continue

        recs = recs_by_user.get(user_id, [])
        hits = [1 if item in rel else 0 for item in recs]

        recall = sum(hits) / len(rel)
        dcg = sum(h / np.log2(p + 2) for p, h in enumerate(hits))
        idcg = sum(1 / np.log2(p + 2) for p in range(min(len(rel), 10)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        hit_rate = 1.0 if sum(hits) > 0 else 0.0
        first_hit = next((i + 1 for i, h in enumerate(hits) if h == 1), 0)
        mrr = 1.0 / first_hit if first_hit > 0 else 0.0

        rows.append({
            "user_id": user_id,
            "activity_group": str(group_map.get(user_id, "Unknown")),
            "Recall@10": recall,
            "NDCG@10": ndcg,
            "HitRate@10": hit_rate,
            "MRR@10": mrr,
        })

    df_new = pd.DataFrame(rows)

    # Verifikasi bit-untuk-bit vs file lama (Recall/NDCG harus identik)
    old_path = RESULT_DIR / f"{model_name}_user_metrics.csv"
    if old_path.exists():
        df_old = pd.read_csv(old_path, dtype={"user_id": str}).sort_values("user_id").reset_index(drop=True)
        df_check = df_new.sort_values("user_id").reset_index(drop=True)
        # Bandingkan Recall@10 dan NDCG@10
        for metric in ["Recall@10", "NDCG@10"]:
            if metric in df_old.columns:
                diff = (df_old[metric].fillna(0) - df_check[metric].fillna(0)).abs().max()
                if diff > 1e-12:
                    raise ValueError(
                        f"VERIFIKASI GAGAL: {model_name} {metric} max diff = {diff:.2e}. "
                        "Rekomendasi mungkin berbeda dari yang digunakan sebelumnya."
                    )
        print(f"  [OK] {model_name}: Recall@10 dan NDCG@10 identik dengan file lama.")

    return df_new


def main():
    # Model yang recs_top10-nya tersedia tapi user_metrics belum punya HitRate/MRR
    targets = [
        ("hybrid", RESULT_DIR / "hybrid_recommendations_top10.csv"),
    ]

    # Cek model lain juga kalau perlu
    for model_name in ["mf", "tfidf_cbf"]:
        um = pd.read_csv(RESULT_DIR / f"{model_name}_user_metrics.csv", nrows=1)
        if "HitRate@10" not in um.columns:
            recs_path = RESULT_DIR / f"{model_name}_recommendations_top10.csv"
            if recs_path.exists():
                targets.append((model_name, recs_path))

    for model_name, recs_path in targets:
        print(f"\nMemproses: {model_name} ...")
        df = compute_full_metrics_from_recs(model_name, recs_path)
        print(f"  Users evaluated: {len(df)}")
        print(f"  Recall@10:  {df['Recall@10'].mean():.10f}")
        print(f"  NDCG@10:    {df['NDCG@10'].mean():.10f}")
        print(f"  HitRate@10: {df['HitRate@10'].mean():.10f}")
        print(f"  MRR@10:     {df['MRR@10'].mean():.10f}")

        out_path = RESULT_DIR / f"{model_name}_user_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"  Disimpan: {out_path}")

    print("\nSelesai. Semua user_metrics sudah punya HitRate@10 dan MRR@10.")


if __name__ == "__main__":
    main()
