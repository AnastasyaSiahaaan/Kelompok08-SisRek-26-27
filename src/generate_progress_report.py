"""
Generate Progress Report CP3 sebagai PDF.
Jalankan: python src/generate_progress_report.py
Output: CP/progress_report_cp3.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "CP" / "progress_report_cp3.pdf"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()

BLUE = colors.HexColor("#00467F")
LIGHT_BLUE = colors.HexColor("#E8F0F7")
WHITE = colors.white
GRAY = colors.HexColor("#F0F0F0")

title_style = ParagraphStyle(
    "Title", parent=styles["Normal"],
    fontSize=16, fontName="Helvetica-Bold",
    textColor=BLUE, alignment=TA_CENTER, spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=10, alignment=TA_CENTER, spaceAfter=2,
)
section_style = ParagraphStyle(
    "Section", parent=styles["Normal"],
    fontSize=12, fontName="Helvetica-Bold",
    textColor=BLUE, spaceBefore=12, spaceAfter=4,
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=9, spaceAfter=4, leading=13,
)
bullet_style = ParagraphStyle(
    "Bullet", parent=styles["Normal"],
    fontSize=9, spaceAfter=2, leading=13,
    leftIndent=12, bulletIndent=0,
)
bold_style = ParagraphStyle(
    "Bold", parent=styles["Normal"],
    fontSize=9, fontName="Helvetica-Bold", spaceAfter=2,
)

TABLE_HEADER = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GRAY]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
])

def section(title):
    return [
        Paragraph(title, section_style),
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=4),
    ]

def body(text):
    return Paragraph(text, body_style)

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def bold(text):
    return Paragraph(text, bold_style)

def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TABLE_HEADER)
    return t

def sp(h=4):
    return Spacer(1, h)

# ---------------------------------------------------------------------------
# Konten
# ---------------------------------------------------------------------------
story = []

# Header
story += [
    Paragraph("Progress Report CP3 — Desain Eksperimen", title_style),
    Paragraph("Kelompok 08 &nbsp;|&nbsp; 12S4054 Sistem Rekomendasi &nbsp;|&nbsp; Institut Teknologi Del", subtitle_style),
    Paragraph("Semester Gasal 2026/2027 &nbsp;|&nbsp; Deadline: 7 November 2026", subtitle_style),
    HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=8),
]

# 1. Ringkasan
story += section("1. Ringkasan")
story += [
    body("CP3 menetapkan desain eksperimen lengkap sebelum eksperimen validasi skala penuh dimulai pada W10–W12. "
         "Protokol evaluasi warm-start, candidate set, leakage rules, hyperparameter search space, dan experiment "
         "matrix telah ditetapkan dan dikodekan dalam repositori GitHub Kelompok 08."),
    body("<b>Track:</b> A — Sparsitas Riwayat User."),
    body("<b>Research Question:</b> Bagaimana jumlah interaksi yang dimiliki user memengaruhi performa "
         "collaborative, content-based, dan hybrid recommendation?"),
    sp(),
]

# 2. Protokol Evaluasi
story += section("2. Protokol Evaluasi (Warm-Start)")
story += [
    body("Evaluasi menggunakan kondisi warm-start agar semua baseline dibandingkan secara adil. "
         "Model selection dan hyperparameter tuning <b>hanya menggunakan validation</b>. Test set tidak diakses."),
    sp(4),
    table(
        [
            ["Komponen", "Nilai"],
            ["Data split", "Official temporal split; tidak di-reshuffle"],
            ["Ukuran training", "427.957 interaksi"],
            ["Ukuran validation", "41.284 (warm: 30.360)"],
            ["Ukuran test", "42.595 (tidak digunakan sebelum W13)"],
            ["Evaluable users", "11.659 dari evaluable_users_validation_warm.csv"],
            ["Candidate universe", "22.753 item dari candidate_items_warm.csv (sama semua model)"],
            ["Ranking per user", "Top-10"],
            ["Metrik wajib", "Recall@10 dan NDCG@10"],
            ["Item relevan", "Rating validation ≥ 4"],
            ["Seen-item masking", "Item training dikeluarkan dari kandidat tiap user"],
        ],
        col_widths=[5.5*cm, 11*cm],
    ),
    sp(),
]

# 3. Candidate Set & Leakage Rules
story += section("3. Candidate Set & Leakage Rules")
story += [
    bold("Candidate Set:"),
    bullet("Item kandidat berasal dari candidate_items_warm.csv — sama untuk semua model."),
    bullet("Item yang pernah diinteraksikan user pada training dikeluarkan sebelum rekomendasi dibuat."),
    sp(4),
    bold("Leakage Rules:"),
    bullet("Model hanya dilatih dari data training; validation dan test tidak masuk proses training."),
    bullet("Official temporal split tidak diacak ulang."),
    bullet("User profile dibangun dari training saja. Field average_rating dan rating_number tidak digunakan."),
    bullet("Candidate protocol sama untuk semua model; tidak ada candidate set yang menguntungkan model tertentu."),
    bullet("Hyperparameter tuning hanya menggunakan validation. Test set tidak digunakan sebelum W13."),
    sp(),
]

# 4. Evaluable Users & Activity Groups
story += section("4. Evaluable Users & Activity Groups")
story += [
    body("Pengguna yang masuk evaluasi harus: (a) memiliki riwayat interaksi pada training, "
         "dan (b) memiliki ground-truth positif (rating ≥ 4) pada validation."),
    body("Kelompok aktivitas (Track A) didefinisikan dengan pd.cut, batas tetap, "
         "konsisten untuk semua model (make_user_groups() di baseline_common.py):"),
    sp(4),
    table(
        [
            ["Kelompok", "Jumlah interaksi training"],
            ["Low", "1 – 5"],
            ["Medium", "6 – 8"],
            ["High", "≥ 9"],
        ],
        col_widths=[5*cm, 11.5*cm],
    ),
    sp(),
]

# 5. Hyperparameter Search Space
story += section("5. Hyperparameter Search Space")
story += [
    body("Tuning dilakukan via validation menggunakan src/tune_hyperparameters.py. "
         "Semua nilai search space didefinisikan di src/config.py."),
    sp(4),
    table(
        [
            ["Model", "Parameter", "Search Space", "Metode Seleksi"],
            ["Popularity", "—", "Tidak ada tuning", "Tetap"],
            ["Matrix Factorization", "n_components", "{32, 64, 128}", "Recall@10 pada validation"],
            ["TF-IDF CBF", "max_features", "{50.000, 100.000}", "Recall@10 pada validation"],
            ["Hybrid", "alpha (bobot MF)", "{0.3, 0.5, 0.7}", "Recall@10 pada validation"],
        ],
        col_widths=[4*cm, 3.5*cm, 4*cm, 5*cm],
    ),
    sp(4),
    body("Parameter tetap: random_state=42, n_iter=7 (MF), ngram_range=(1,2), min_df=2 (TF-IDF), "
         "normalisasi min-max per user (Hybrid)."),
    sp(),
]

# 6. Experiment Matrix
story += section("6. Experiment Matrix")
story += [
    table(
        [
            ["ID", "Model/Variant", "Tujuan", "Metrik"],
            ["E1", "Popularity", "Baseline bersama", "Recall@10, NDCG@10"],
            ["E2a–c", "MF (n=32/64/128)", "Tuning collaborative baseline", "Recall@10, NDCG@10"],
            ["E3a–b", "TF-IDF CBF (50k/100k)", "Tuning content-based baseline", "Recall@10, NDCG@10"],
            ["E4a–c", "Hybrid (α=0.3/0.5/0.7)", "Tuning hybrid; config dari E2/E3 terbaik", "Recall@10, NDCG@10"],
            ["E5", "Track-A Analysis", "Analisis per activity group (Low/Med/High)", "Recall@10, NDCG@10 per group"],
        ],
        col_widths=[1.5*cm, 3.5*cm, 7*cm, 4.5*cm],
    ),
    sp(),
]

# 7. Reproducibility
story += section("7. Reproducibility Settings")
story += [
    bullet("<b>Random seed:</b> 42 — dikodekan di src/config.py (RANDOM_SEED)."),
    bullet("<b>Config terpusat:</b> src/config.py — semua hyperparameter dan search space."),
    bullet("<b>Run ID format:</b> &lt;model&gt;_&lt;param&gt;&lt;val&gt;_seed&lt;seed&gt; "
           "(contoh: mf_components64_seed42)."),
    bullet("<b>Experiment log:</b> setiap run dicatat ke results/experiment_log.csv "
           "beserta timestamp, git commit hash, parameter, dan metrik."),
    sp(),
]

# 8. Hasil Awal
story += section("8. Hasil Awal (CP3 — Config Baru, Belum Dituning)")
story += [
    table(
        [
            ["Model", "Run ID", "Recall@10", "NDCG@10"],
            ["Popularity", "—", "0.0135", "0.0092"],
            ["Matrix Factorization", "mf_components64_seed42", "0.0119", "0.0074"],
            ["TF-IDF CBF", "tfidf_cbf_maxfeat100000_seed42", "0.0126", "0.0073"],
            ["Hybrid (best)", "hybrid_alpha0.5_seed42", "0.0147", "0.0087"],
        ],
        col_widths=[3.5*cm, 6.5*cm, 2.5*cm, 2.5*cm],
    ),
    sp(4),
    body("Hybrid mengungguli semua model. Tuning pada W10–W12 difokuskan pada peningkatan "
         "terutama untuk pengguna kelompok Low yang memiliki riwayat interaksi terbatas."),
    sp(),
]

# 9. Evidence GitHub
story += section("9. Evidence GitHub")
story += [
    bullet("src/config.py — konfigurasi dan search space terpusat"),
    bullet("src/experiment_log.py — logging per run (run ID, timestamp, git commit, metrik)"),
    bullet("src/tune_hyperparameters.py — skrip tuning otomatis via validation set"),
    bullet("results/experiment_matrix.csv — experiment matrix lengkap (E1–E5)"),
    bullet("src/mf_baseline.py, src/tfidf_cbf_baseline.py, src/hybrid_baseline.py "
           "— diperbarui untuk membaca parameter dari config.py"),
    bullet("CP/progress_report_cp3.md — progress report"),
]

# ---------------------------------------------------------------------------
# Build PDF
# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(
    str(OUTPUT_PATH),
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)
doc.build(story)
print(f"PDF berhasil dibuat: {OUTPUT_PATH}")
