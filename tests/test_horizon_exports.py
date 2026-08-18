from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dataset"
HORIZONS = (7, 14, 21, 30)
REMOVED_COLUMNS = {
    "definition",
    "caption",
    "made_for_kids",
    "description_length",
}


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


@unittest.skipUnless(
    (DATASET_ROOT / "viewcastlk_training_table.csv").exists(),
    "source training table is not available",
)
class HorizonExportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        columns = [
            "video_id",
            "eligible",
            "is_live_broadcast",
            "title_changed",
        ]
        for horizon in HORIZONS:
            columns.extend(
                [f"d{horizon}_usable", f"d{horizon}_views"]
            )
        cls.source = pd.read_csv(
            DATASET_ROOT / "viewcastlk_training_table.csv",
            usecols=columns,
            low_memory=False,
        )

    def test_exports_remove_content_columns_and_ineligible_rows(self) -> None:
        for horizon in HORIZONS:
            target = f"d{horizon}_views"
            usable = f"d{horizon}_usable"
            exported = pd.read_csv(
                DATASET_ROOT
                / "model_horizon_datasets"
                / f"viewcastlk_day_{horizon}.csv",
                low_memory=False,
            )
            assignments = pd.read_csv(
                DATASET_ROOT
                / "model_split_metadata"
                / f"viewcastlk_day_{horizon}_split_assignments.csv",
                low_memory=False,
            )
            expected = (
                as_bool(self.source["eligible"])
                & ~as_bool(self.source["is_live_broadcast"])
                & ~as_bool(self.source["title_changed"])
                & as_bool(self.source[usable])
                & self.source[target].notna()
            )

            self.assertTrue(REMOVED_COLUMNS.isdisjoint(exported.columns))
            self.assertEqual(len(exported), int(expected.sum()))
            self.assertEqual(len(assignments), len(exported))

            live_video_ids = set(
                self.source.loc[
                    as_bool(self.source["is_live_broadcast"]), "video_id"
                ].astype(str)
            )
            exported_video_ids = set(assignments["video_id"].astype(str))
            self.assertTrue(live_video_ids.isdisjoint(exported_video_ids))
            changed_title_video_ids = set(
                self.source.loc[
                    as_bool(self.source["title_changed"]), "video_id"
                ].astype(str)
            )
            self.assertTrue(
                changed_title_video_ids.isdisjoint(exported_video_ids)
            )


if __name__ == "__main__":
    unittest.main()
