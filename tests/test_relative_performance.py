from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.train_relative_performance_trajectory import (
    reconstruct_views,
    relative_log_target,
    strict_reconcile,
)


class RelativePerformanceTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = pd.DataFrame(
            {
                "prior_d7_view_count": [10, 1, 8],
                "prior_d7_median_views": [1_000, 10_000, 5_000],
                "prior_d30_view_count": [10, 1, 8],
                "prior_d30_median_views": [2_000, 20_000, 9_000],
                "ch_avg_views_per_video_at_publish": [1_500, 3_000, 4_500],
            }
        )

    def test_relative_target_round_trips_to_absolute_views(self) -> None:
        actual = np.asarray([500.0, 4_000.0, 15_000.0])
        target = relative_log_target(
            self.features,
            actual,
            horizon_days=21,
            strategy="interpolated_d7_d30",
        )

        reconstructed = reconstruct_views(
            self.features,
            target,
            horizon_days=21,
            strategy="interpolated_d7_d30",
        )

        np.testing.assert_allclose(reconstructed, actual)

    def test_strict_reconciliation_does_not_feed_models_into_each_other(self) -> None:
        independent = np.asarray(
            [[100.0, 90.0, 120.0, 110.0], [5.0, 6.0, 7.0, 8.0]]
        )

        reconciled = strict_reconcile(independent)

        np.testing.assert_array_equal(
            reconciled,
            [[100.0, 101.0, 120.0, 121.0], [5.0, 6.0, 7.0, 8.0]],
        )
        np.testing.assert_array_equal(
            independent,
            [[100.0, 90.0, 120.0, 110.0], [5.0, 6.0, 7.0, 8.0]],
        )


if __name__ == "__main__":
    unittest.main()
