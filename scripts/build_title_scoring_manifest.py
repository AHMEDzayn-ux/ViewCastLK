"""Build the deterministic input manifest for Gemini title scoring."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd


MANIFEST_COLUMNS = ["video_id", "title", "title_sha256"]


def title_sha256(title: str) -> str:
    """Hash the exact UTF-8 title so later merges can detect title changes."""
    return hashlib.sha256(title.encode("utf-8")).hexdigest()


def read_source_titles(source_path: Path) -> pd.DataFrame:
    """Read IDs and titles without treating strings such as 'NA' as missing."""
    source = pd.read_csv(
        source_path,
        usecols=["video_id", "title"],
        dtype={"video_id": "string", "title": "string"},
        keep_default_na=False,
    )

    empty_video_id = source["video_id"].isna() | source["video_id"].str.strip().eq("")
    empty_title = source["title"].isna() | source["title"].str.strip().eq("")
    duplicate_video_id = source["video_id"].duplicated(keep=False)

    if empty_video_id.any():
        raise ValueError(f"Source contains {int(empty_video_id.sum())} empty video_id values")
    if empty_title.any():
        raise ValueError(f"Source contains {int(empty_title.sum())} empty titles")
    if duplicate_video_id.any():
        examples = source.loc[duplicate_video_id, "video_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Source contains duplicate video_id values, for example: {examples}")

    return source


def build_manifest(source_path: Path, output_path: Path) -> pd.DataFrame:
    """Create a stable, one-row-per-video UTF-8 CSV and return it."""
    source = read_source_titles(source_path)
    manifest = source.copy()
    manifest["title_sha256"] = manifest["title"].map(title_sha256)
    manifest = manifest[MANIFEST_COLUMNS].sort_values("video_id", kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    manifest.to_csv(temporary_path, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary_path, output_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Dataset/viewcastlk_training_table.csv"),
        help="Master training-table CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Dataset/title_scoring_manifest.csv"),
        help="Output title-scoring manifest CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.source, args.out)
    print(f"Wrote {len(manifest):,} titles to {args.out}")


if __name__ == "__main__":
    main()
