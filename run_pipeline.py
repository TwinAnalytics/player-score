from pathlib import Path
from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    season = "2024-2025"

    root = Path(__file__).resolve().parent
    raw_dir = root / "Data" / "Raw"
    processed_dir = root / "Data" / "Processed"

    out = run_full_pipeline(season, raw_dir, processed_dir)
    print(f"Fertiger Score-Export: {out}")