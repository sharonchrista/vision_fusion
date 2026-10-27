"""
scripts/preprocess.py

Run ONCE to prepare the dataset for training:
  1. Extract all four .zip archives
  2. Inspect extracted directory structure
  3. Write train/val/test split manifests
  4. (Extend with shard writing once actual file layout is known)

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --raw_dir data/raw --extracted_dir data/extracted
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make src importable from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import extract_all, inspect_extracted
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="LiquidFusionNet data preprocessing")
    parser.add_argument("--raw_dir",       default="data/raw",       type=Path)
    parser.add_argument("--extracted_dir", default="data/extracted",  type=Path)
    parser.add_argument("--processed_dir", default="data/processed",  type=Path)
    parser.add_argument("--skip_extract",  action="store_true",
                        help="Skip extraction (zips already extracted)")
    args = parser.parse_args()

    console.rule("[bold blue]LiquidFusionNet — Data Preprocessing[/bold blue]")

    # Stage 1: Extract
    if not args.skip_extract:
        console.print("\n[bold]Stage 1:[/bold] Extracting archives …")
        extracted = extract_all(args.raw_dir, args.extracted_dir)
    else:
        extracted = {
            k: args.extracted_dir / k
            for k in ["multimodal", "lidar", "radar", "camera"]
            if (args.extracted_dir / k).exists()
        }
        console.print(f"[yellow]Skipping extraction. Found:[/yellow] {list(extracted.keys())}")

    # Stage 2: Inspect
    if extracted:
        console.print("\n[bold]Stage 2:[/bold] Inspecting extracted structure …")
        schema = inspect_extracted(extracted)

        # Save schema for reference
        import json
        schema_path = args.extracted_dir / "schema.json"
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)
        console.print(f"[green]Schema saved[/green] → {schema_path}")

    console.print(
        "\n[bold yellow]Next steps:[/bold yellow]\n"
        "  1. Review data/extracted/schema.json to understand file layout\n"
        "  2. Adapt src/data/preprocessing.py shard writing to actual layout\n"
        "  3. Run: python scripts/preprocess.py --skip_extract  (to re-run from Stage 2)\n"
        "  4. Then run: python scripts/train.py\n"
    )
    console.rule()


if __name__ == "__main__":
    main()
