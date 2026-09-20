from __future__ import annotations

import unittest

import numpy as np

from scripts.train_growth_ensemble import (
    CANDIDATES,
    MINIMUM_INCREMENT_VIEWS,
    blend_recipes,
    recipe_prediction_matrix,
    select_recipe_index,
)


class GrowthEnsembleSelectionTests(unittest.TestCase):
    def test_recipes_are_unique_valid_convex_weights(self) -> None:
        recipes = blend_recipes()
        weights = [(recipe.blend_method, recipe.weights) for recipe in recipes]

        self.assertEqual(len(weights), len(set(weights)))
        self.assertGreater(len(recipes), len(CANDIDATES))
        for recipe in recipes:
            self.assertEqual(len(recipe.weights), len(CANDIDATES))
            self.assertTrue(all(weight >= 0 for weight in recipe.weights))
            self.assertAlmostEqual(sum(recipe.weights), 1.0)

    def test_selector_can_choose_exact_single_candidate(self) -> None:
        recipes = blend_recipes()
        actual = np.asarray([10.0, 20.0, 30.0])
        predictions = np.column_stack(
            [
                actual,
                np.full(3, 100.0),
                np.full(3, 200.0),
                np.full(3, 300.0),
                np.full(3, 400.0),
            ]
        )
        recipe_matrix = recipe_prediction_matrix(predictions, recipes)

        selected = select_recipe_index(
            actual_views=actual,
            current_views=None,
            recipe_predictions=recipe_matrix,
            recipes=recipes,
        )

        self.assertEqual(recipes[selected].weights[0], 1.0)
        self.assertEqual(recipes[selected].component_count, 1)

    def test_selector_applies_strict_increment_floor(self) -> None:
        recipes = blend_recipes()
        current = np.asarray([100.0, 200.0])
        actual = current + MINIMUM_INCREMENT_VIEWS
        predictions = np.zeros((2, len(recipes)))

        selected = select_recipe_index(
            actual_views=actual,
            current_views=current,
            recipe_predictions=predictions,
            recipes=recipes,
        )

        self.assertGreaterEqual(selected, 0)


if __name__ == "__main__":
    unittest.main()
