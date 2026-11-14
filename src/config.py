"""
Konfigurasi terpusat untuk semua eksperimen.
Ubah nilai di sini sebelum menjalankan eksperimen baru.
Setiap run wajib dipanggil dengan config ini agar reproducible.
"""

# ---------------------------------------------------------------------------
# Random seed global — dipakai oleh semua model yang membutuhkan randomness
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Popularity baseline
# Tidak ada hyperparameter yang perlu di-tuning.
# Definisi skor: jumlah interaksi positif (rating >= 4) pada data training.
# ---------------------------------------------------------------------------
POPULARITY_CONFIG = {
    "positive_threshold": 4,   # rating minimum untuk dianggap positif
}

# ---------------------------------------------------------------------------
# Matrix Factorization (TruncatedSVD)
# n_components yang dipakai saat ini (default run):
#   dipilih dari SEARCH SPACE_MF_FACTORS via validation sebelum freeze.
# ---------------------------------------------------------------------------
MF_CONFIG = {
    "n_components": 32,
    "n_iter": 7,
    "random_state": RANDOM_SEED,
    "positive_threshold": 4,
}

# ---------------------------------------------------------------------------
# TF-IDF Content-Based Filtering
# max_features yang dipakai saat ini dipilih dari SEARCH SPACE via validation.
# ---------------------------------------------------------------------------
TFIDF_CONFIG = {
    "max_features": 100_000,   # nilai terpilih; lihat TFIDF_SEARCH_SPACE
    "ngram_range": (1, 2),
    "stop_words": "english",
    "min_df": 2,
    "positive_threshold": 4,
}

# ---------------------------------------------------------------------------
# Hybrid (MF + TF-IDF CBF)
# alpha: bobot MF; (1 - alpha): bobot CBF
# ---------------------------------------------------------------------------
HYBRID_CONFIG = {
    "alpha": 0.5,              # nilai terpilih; lihat HYBRID_SEARCH_SPACE
    "normalization": "minmax", # normalisasi per-user sebelum digabung
}

# ---------------------------------------------------------------------------
# Search space — dipakai untuk tuning via validation (BUKAN test set)
# ---------------------------------------------------------------------------
MF_SEARCH_SPACE = {
    "n_components": [32, 64, 128],
}

TFIDF_SEARCH_SPACE = {
    "max_features": [50_000, 100_000],
}

HYBRID_SEARCH_SPACE = {
    "alpha": [0.3, 0.5, 0.7],
}

# ---------------------------------------------------------------------------
# Evaluasi & protokol
# ---------------------------------------------------------------------------
EVAL_CONFIG = {
    "top_k": 10,
    "relevant_threshold": 4,   # rating validation >= nilai ini = relevan
    "activity_bins": [0, 5, 8, float("inf")],
    "activity_labels": ["Low", "Medium", "High"],
    "metrics": ["Recall@10", "NDCG@10"],
}

# ---------------------------------------------------------------------------
# Penamaan run — format: <model>_<param_key><param_val>_seed<seed>
# Contoh: mf_components64_seed42
# ---------------------------------------------------------------------------
def make_run_id(model: str, params: dict, seed: int = RANDOM_SEED) -> str:
    """Membuat ID run yang unik dan dapat dilacak."""
    param_str = "_".join(
        f"{k}{v}" for k, v in sorted(params.items())
    )
    return f"{model}_{param_str}_seed{seed}"
