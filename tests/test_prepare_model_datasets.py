from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.prepare_model_datasets import add_channel_history_features


class ChannelHistoryFeatureTests(unittest.TestCase):
    def test_targets_appear_only_after_their_observation_time(self) -> None:
        frame = pd.DataFrame(
            {
                "video_id": ["a", "b", "c", "d"],
                "channel_id": ["channel"] * 4,
                "category_name": ["Music", "News", "Music", "Music"],
                "is_short": [True, False, True, True],
                "published_at": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                    "2026-01-08T11:00:00Z",
                    "2026-01-08T13:00:00Z",
                ],
                "d7_views": [100.0, 200.0, 300.0, 400.0],
                "d7_hours_off": [12.0, 0.0, 0.0, 0.0],
                "d7_usable": [True] * 4,
                "d30_views": [1000.0, 2000.0, 3000.0, 4000.0],
                "d30_hours_off": [0.0] * 4,
                "d30_usable": [True] * 4,
            }
        )
        result = add_channel_history_features(frame)

        self.assertEqual(result.loc[1, "prior_channel_video_count"], 1)
        self.assertEqual(result.loc[1, "prior_d7_view_count"], 0)
        self.assertEqual(result.loc[2, "prior_d7_view_count"], 0)
        self.assertEqual(result.loc[3, "prior_d7_view_count"], 1)
        self.assertEqual(result.loc[3, "prior_d7_median_views"], 100)
        self.assertEqual(result.loc[3, "prior_same_category_d7_median_views"], 100)
        self.assertEqual(result.loc[3, "prior_same_format_d7_count"], 1)
        self.assertEqual(result.loc[3, "prior_same_format_d7_median_views"], 100)
        self.assertEqual(result.loc[3, "prior_d30_view_count"], 0)

    def test_simultaneous_uploads_do_not_count_each_other_as_history(self) -> None:
        frame = pd.DataFrame(
            {
                "video_id": ["a", "b"],
                "channel_id": ["channel", "channel"],
                "category_name": ["Music", "Music"],
                "is_short": [True, True],
                "published_at": ["2026-01-01T00:00:00Z"] * 2,
                "d7_views": [100.0, 200.0],
                "d7_hours_off": [0.0, 0.0],
                "d7_usable": [True, True],
                "d30_views": [1000.0, 2000.0],
                "d30_hours_off": [0.0, 0.0],
                "d30_usable": [True, True],
            }
        )
        result = add_channel_history_features(frame)
        np.testing.assert_array_equal(result["prior_channel_video_count"], [0, 0])
        np.testing.assert_array_equal(result["uploads_previous_7d"], [0, 0])

    def test_recent_trend_uses_only_already_observed_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "video_id": ["a", "b", "c"],
                "channel_id": ["channel"] * 3,
                "category_name": ["Music"] * 3,
                "is_short": [True] * 3,
                "published_at": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-09T00:00:00Z",
                    "2026-01-17T00:00:00Z",
                ],
                "d7_views": [100.0, 200.0, 300.0],
                "d7_hours_off": [0.0] * 3,
                "d7_usable": [True] * 3,
                "d30_views": [1000.0, 2000.0, 3000.0],
                "d30_hours_off": [0.0] * 3,
                "d30_usable": [True] * 3,
            }
        )
        result = add_channel_history_features(frame)
        self.assertTrue(np.isnan(result.loc[1, "prior_d7_recent_log_trend"]))
        self.assertGreater(result.loc[2, "prior_d7_recent_log_trend"], 0)


if __name__ == "__main__":
    unittest.main()
