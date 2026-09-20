from __future__ import annotations

import unittest

import numpy as np

from scripts.train_monotonic_trajectory import HORIZONS
from scripts.train_title_residual_correction import (
    corrected_trajectory,
    normalize_title,
    residual_targets,
    select_configuration,
)


class TitleResidualCorrectionTests(unittest.TestCase):
    def test_title_normalization_is_unicode_and_whitespace_stable(self) -> None:
        self.assertEqual(normalize_title("  A\t title\n"), "A title")
        self.assertEqual(normalize_title("ＡＢＣ"), "ABC")

    def test_perfect_semantic_correction_beats_constant_control(self) -> None:
        base = np.full((4, 4), 100.0)
        pattern = np.asarray([200.0, 50.0, 200.0, 50.0])
        targets = {horizon: pattern.copy() for horizon in HORIZONS}
        residuals = residual_targets(targets, base)
        text_oof = {(32, 10.0): residuals.copy()}
        constant_oof = np.broadcast_to(residuals.mean(axis=0), residuals.shape)

        comparison, selected_text, selected_constant = select_configuration(
            targets=targets,
            base_oof=base,
            text_oof=text_oof,
            constant_oof=constant_oof,
        )

        self.assertFalse(comparison.empty)
        self.assertEqual(selected_text["strength"], 1.0)
        self.assertLess(
            selected_text["mean_horizon_rmsle"],
            selected_constant["mean_horizon_rmsle"],
        )

    def test_corrected_trajectory_remains_strictly_increasing(self) -> None:
        base = np.asarray([[100.0, 110.0, 120.0, 130.0]])
        correction = np.asarray([[1.0, -1.0, -1.0, -1.0]])

        corrected = corrected_trajectory(base, correction, strength=1.0)

        self.assertTrue((np.diff(corrected, axis=1) > 0).all())


if __name__ == "__main__":
    unittest.main()
