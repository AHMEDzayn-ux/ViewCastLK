from __future__ import annotations

import unittest

import numpy as np

from scripts.train_breakout_ensemble import (
    CANDIDATES,
    ProbabilityBlendRecipe,
    blend_probability_mapping,
    fit_calibrator,
    probability_blend_recipes,
    recipe_prediction_matrix,
    select_recipe_index,
)


class BreakoutEnsembleSelectionTests(unittest.TestCase):
    def test_recipes_are_unique_convex_and_cover_both_blend_methods(self) -> None:
        recipes = probability_blend_recipes()
        identities = [(recipe.blend_method, recipe.weights) for recipe in recipes]

        self.assertEqual(len(recipes), 207)
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual({recipe.blend_method for recipe in recipes}, {"arithmetic", "logit"})
        for recipe in recipes:
            self.assertEqual(len(recipe.weights), len(CANDIDATES))
            self.assertTrue(all(weight >= 0 for weight in recipe.weights))
            self.assertAlmostEqual(sum(recipe.weights), 1.0)

    def test_selector_can_choose_exact_single_candidate(self) -> None:
        recipes = probability_blend_recipes()
        labels = np.asarray([False, False, True, True])
        candidate_probabilities = np.column_stack(
            [
                [0.01, 0.05, 0.95, 0.99],
                [0.99, 0.95, 0.05, 0.01],
                [0.5, 0.5, 0.5, 0.5],
                [0.6, 0.6, 0.6, 0.6],
                [0.4, 0.4, 0.4, 0.4],
            ]
        )

        selected = select_recipe_index(
            labels,
            recipe_prediction_matrix(candidate_probabilities, recipes),
            recipes,
        )

        self.assertEqual(recipes[selected].weights[0], 1.0)
        self.assertEqual(recipes[selected].component_count, 1)

    def test_logit_blend_and_probability_calibrators_are_bounded(self) -> None:
        recipe = ProbabilityBlendRecipe(
            name="test", weights=(0.25, 0.75, 0.0, 0.0, 0.0), blend_method="logit"
        )
        raw = blend_probability_mapping(
            {
                "xgb_current": np.asarray([0.1, 0.7]),
                "xgb_shallow": np.asarray([0.3, 0.9]),
            },
            recipe,
        )
        labels = np.asarray([False, False, False, True, True, True])
        training_probability = np.asarray([0.05, 0.1, 0.3, 0.6, 0.8, 0.95])

        self.assertTrue(((raw > 0) & (raw < 1)).all())
        for method in ("identity", "platt", "beta", "isotonic"):
            calibrated = fit_calibrator(
                method, training_probability, labels
            ).predict_probability(raw)
            self.assertTrue(np.isfinite(calibrated).all())
            self.assertTrue(((calibrated >= 0) & (calibrated <= 1)).all())


if __name__ == "__main__":
    unittest.main()
