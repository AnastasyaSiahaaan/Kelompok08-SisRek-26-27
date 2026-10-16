
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SUPPORT_DIR = BASE_DIR / "data" / "supporting"

for file_path in sorted(SUPPORT_DIR.glob("*.csv")):
    print("\n" + "=" * 60)
    print("FILE:", file_path.name)

    try:
        df = pd.read_csv(file_path)

        print("Ukuran:", df.shape)
        print("Kolom:", df.columns.tolist())
        print("\nContoh 5 baris:")
        print(df.head().to_string(index=False))

    except Exception as e:
        print("Gagal membaca file:", e)
