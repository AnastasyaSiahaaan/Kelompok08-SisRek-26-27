"""
Utilitas logging eksperimen untuk reproducibility.
Setiap run model wajib memanggil log_run() agar hasilnya
dapat ditelusuri kembali ke config dan commit yang menghasilkannya.

Hasil disimpan di results/experiment_log.csv.
"""

import csv
import datetime
import subprocess
from pathlib import Path

from baseline_common import RESULT_DIR


LOG_PATH = RESULT_DIR / "experiment_log.csv"

LOG_COLUMNS = [
    "run_id",
    "timestamp",
    "git_commit",
    "model",
    "params",
    "recall_at_10",
    "ndcg_at_10",
    "recall_low",
    "ndcg_low",
    "recall_medium",
    "ndcg_medium",
    "recall_high",
    "ndcg_high",
    "n_evaluated_users",
    "notes",
]


def _get_git_commit() -> str:
    """Mengembalikan hash commit git saat ini (7 karakter)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def log_run(
    run_id: str,
    model: str,
    params: dict,
    overall_metrics: dict,
    group_metrics: dict | None = None,
    n_evaluated_users: int = 0,
    notes: str = "",
) -> None:
    """
    Mencatat satu run eksperimen ke experiment_log.csv.

    Parameters
    ----------
    run_id : str
        ID unik run, dibuat dari config.make_run_id().
    model : str
        Nama model (popularity / mf / tfidf_cbf / hybrid).
    params : dict
        Hyperparameter yang dipakai pada run ini.
    overall_metrics : dict
        {"Recall@10": float, "NDCG@10": float}
    group_metrics : dict, optional
        {"Low": {"Recall@10": float, "NDCG@10": float}, "Medium": ..., "High": ...}
    n_evaluated_users : int
        Jumlah pengguna yang masuk evaluasi.
    notes : str
        Catatan tambahan (opsional).
    """
    # Inisialisasi file log jika belum ada
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()

    group_metrics = group_metrics or {}
    row = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _get_git_commit(),
        "model": model,
        "params": str(params),
        "recall_at_10": overall_metrics.get("Recall@10", ""),
        "ndcg_at_10": overall_metrics.get("NDCG@10", ""),
        "recall_low": group_metrics.get("Low", {}).get("Recall@10", ""),
        "ndcg_low": group_metrics.get("Low", {}).get("NDCG@10", ""),
        "recall_medium": group_metrics.get("Medium", {}).get("Recall@10", ""),
        "ndcg_medium": group_metrics.get("Medium", {}).get("NDCG@10", ""),
        "recall_high": group_metrics.get("High", {}).get("Recall@10", ""),
        "ndcg_high": group_metrics.get("High", {}).get("NDCG@10", ""),
        "n_evaluated_users": n_evaluated_users,
        "notes": notes,
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)

    print(f"[experiment_log] Run dicatat: {run_id}")
