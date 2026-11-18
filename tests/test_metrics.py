"""
Unit test untuk validasi metrik evaluasi rekomendasi (Recall@10, NDCG@10, HitRate@10, MRR@10),
candidate filtering, seen-item masking, dan pembagian user activity group.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Tambahkan src ke path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baseline_common import make_user_groups, evaluate_rankings


class TestRecommenderMetrics(unittest.TestCase):
    """Pengujian unit test terhadap perhitungan metrik evaluasi."""

    def test_recall_at_10_calculation(self):
        """Memvalidasi formula Recall@10 = sum(hits) / len(relevant_items)."""
        relevant_items = {"item_A", "item_B", "item_C", "item_D"}

        # Kasus 1: Sempurna (semua 4 item ada di top-10)
        recs_perfect = ["item_A", "item_B", "item_C", "item_D", "item_E", "item_F"]
        hits_perfect = [1 if i in relevant_items else 0 for i in recs_perfect]
        recall_perfect = sum(hits_perfect) / len(relevant_items)
        self.assertAlmostEqual(recall_perfect, 1.0)

        # Kasus 2: Parsial (2 item dari 4 ada di top-10)
        recs_partial = ["item_A", "item_X", "item_B", "item_Y", "item_Z"]
        hits_partial = [1 if i in relevant_items else 0 for i in recs_partial]
        recall_partial = sum(hits_partial) / len(relevant_items)
        self.assertAlmostEqual(recall_partial, 0.5)

        # Kasus 3: Zero hit
        recs_zero = ["item_X", "item_Y", "item_Z"]
        hits_zero = [1 if i in relevant_items else 0 for i in recs_zero]
        recall_zero = sum(hits_zero) / len(relevant_items)
        self.assertAlmostEqual(recall_zero, 0.0)

    def test_ndcg_at_10_calculation(self):
        """
        Memvalidasi formula DCG, IDCG, dan NDCG@10 dengan diskon posisi log2(rank + 1).
        rank 1 -> position 0 -> log2(0 + 2) = log2(2) = 1.0
        rank 2 -> position 1 -> log2(1 + 2) = log2(3) = 1.5849625
        rank 3 -> position 2 -> log2(2 + 2) = log2(4) = 2.0
        """
        relevant_items = {"item_1", "item_2"}

        # Kasus 1: Ideal ranking (item relevan di rank 1 dan rank 2)
        recs_ideal = ["item_1", "item_2", "item_3", "item_4"]
        hits_ideal = [1 if i in relevant_items else 0 for i in recs_ideal]
        dcg_ideal = sum(hit / np.log2(pos + 2) for pos, hit in enumerate(hits_ideal))
        idcg = sum(1 / np.log2(pos + 2) for pos in range(min(len(relevant_items), 10)))
        ndcg_ideal = dcg_ideal / idcg
        self.assertAlmostEqual(ndcg_ideal, 1.0)

        # Kasus 2: Sub-optimal ranking (item relevan di rank 1 dan rank 3)
        # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        # NDCG = 1.5 / 1.63092975 = 0.91972
        recs_sub = ["item_1", "item_X", "item_2", "item_Y"]
        hits_sub = [1 if i in relevant_items else 0 for i in recs_sub]
        dcg_sub = sum(hit / np.log2(pos + 2) for pos, hit in enumerate(hits_sub))
        ndcg_sub = dcg_sub / idcg
        expected_ndcg = 1.5 / (1.0 + 1.0 / np.log2(3))
        self.assertAlmostEqual(ndcg_sub, expected_ndcg, places=5)

    def test_more_than_ten_relevant_items_idcg(self):
        """Memvalidasi kasus pengguna dengan >10 item relevan (misal 15 item)."""
        relevant_items = {f"item_{i}" for i in range(15)} # 15 item relevan
        recs = [f"item_{i}" for i in range(10)] # 10 hit sempurna di top-10

        hits = [1 if i in relevant_items else 0 for i in recs]
        # Recall@10 harus 10 / 15 = 2/3
        recall = sum(hits) / len(relevant_items)
        self.assertAlmostEqual(recall, 10.0 / 15.0)

        # IDCG dipotong pada min(len(relevant_items), 10) = 10
        dcg = sum(h / np.log2(p + 2) for p, h in enumerate(hits))
        ideal_length = min(len(relevant_items), 10)
        self.assertEqual(ideal_length, 10)
        idcg = sum(1 / np.log2(p + 2) for p in range(ideal_length))
        ndcg = dcg / idcg
        self.assertAlmostEqual(ndcg, 1.0)

    def test_hitrate_and_mrr_calculation(self):
        """Memvalidasi perhitungan HitRate@10 dan MRR@10."""
        relevant_items = {"item_A", "item_B"}

        # Kasus 1: Hit pertama di rank 1
        recs1 = ["item_A", "item_X", "item_B"]
        hits1 = [1 if i in relevant_items else 0 for i in recs1]
        hr1 = 1.0 if sum(hits1) > 0 else 0.0
        first_hit1 = next((i + 1 for i, h in enumerate(hits1) if h == 1), 0)
        mrr1 = 1.0 / first_hit1 if first_hit1 > 0 else 0.0
        self.assertEqual(hr1, 1.0)
        self.assertEqual(mrr1, 1.0)

        # Kasus 2: Hit pertama di rank 3
        recs2 = ["item_X", "item_Y", "item_B"]
        hits2 = [1 if i in relevant_items else 0 for i in recs2]
        hr2 = 1.0 if sum(hits2) > 0 else 0.0
        first_hit2 = next((i + 1 for i, h in enumerate(hits2) if h == 1), 0)
        mrr2 = 1.0 / first_hit2 if first_hit2 > 0 else 0.0
        self.assertEqual(hr2, 1.0)
        self.assertAlmostEqual(mrr2, 1.0 / 3.0)

        # Kasus 3: Zero hit
        recs3 = ["item_X", "item_Y", "item_Z"]
        hits3 = [1 if i in relevant_items else 0 for i in recs3]
        hr3 = 1.0 if sum(hits3) > 0 else 0.0
        first_hit3 = next((i + 1 for i, h in enumerate(hits3) if h == 1), 0)
        mrr3 = 1.0 / first_hit3 if first_hit3 > 0 else 0.0
        self.assertEqual(hr3, 0.0)
        self.assertEqual(mrr3, 0.0)

    def test_user_groups_definition(self):
        """Memvalidasi batas pengelompokan aktivitas: Low (1-5), Medium (6-8), High (>=9)."""
        activity_df = pd.DataFrame({
            "user_id": ["u1", "u2", "u3", "u4", "u5", "u6"],
            "train_interactions": [1, 5, 6, 8, 9, 50],
        })
        grouped = make_user_groups(activity_df)

        groups = dict(zip(grouped["user_id"], grouped["activity_group"].astype(str)))
        self.assertEqual(groups["u1"], "Low")
        self.assertEqual(groups["u2"], "Low")
        self.assertEqual(groups["u3"], "Medium")
        self.assertEqual(groups["u4"], "Medium")
        self.assertEqual(groups["u5"], "High")
        self.assertEqual(groups["u6"], "High")

    def test_seen_item_exclusion_in_pipeline(self):
        """Memvalidasi bahwa item di train dikeluarkan dari rekomendasi dan relevansi."""
        train = pd.DataFrame({
            "user_id": ["u1"],
            "item_id": ["item_seen"],
            "rating": [5.0],
        })
        validation = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "item_id": ["item_seen", "item_new"],
            "rating": [5.0, 5.0],
        })
        users = pd.DataFrame({"user_id": ["u1"]})
        candidates = pd.DataFrame({"item_id": ["item_seen", "item_new", "item_other"]})
        activity = pd.DataFrame({"user_id": ["u1"], "train_interactions": [1]})

        rankings = {"u1": ["item_seen", "item_new", "item_other"]}

        import tempfile
        import baseline_common
        orig_dir = baseline_common.RESULT_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_common.RESULT_DIR = Path(tmpdir)
            try:
                evaluate_rankings(
                    "test_model",
                    rankings,
                    train,
                    validation,
                    users,
                    candidates,
                    activity,
                )
                user_res = pd.read_csv(Path(tmpdir) / "test_model_user_metrics.csv")
                self.assertAlmostEqual(user_res["Recall@10"].iloc[0], 1.0)
                self.assertAlmostEqual(user_res["NDCG@10"].iloc[0], 1.0)
                self.assertAlmostEqual(user_res["HitRate@10"].iloc[0], 1.0)
                self.assertAlmostEqual(user_res["MRR@10"].iloc[0], 1.0)
            finally:
                baseline_common.RESULT_DIR = orig_dir


if __name__ == "__main__":
    unittest.main()
