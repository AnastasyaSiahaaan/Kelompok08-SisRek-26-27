import sys
from pathlib import Path

print("=== GABIN: Environment Check ===")
print("Python version   :", sys.version.split()[0])
print("Project folder   :", Path.cwd())
print("Python executable:", sys.executable)

folders = [
    Path("data/raw"),
    Path("data/processed"),
    Path("src"),
    Path("results"),
]

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Folder tersedia: {folder}")

print("\nEnvironment check selesai")
