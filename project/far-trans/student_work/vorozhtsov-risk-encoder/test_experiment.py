"""Validation regressions: malformed model data must not silently become a profile."""
import json
import unittest

from run_experiment import make_prompt, validate_profile


class ModelOutputTests(unittest.TestCase):
    def setUp(self):
        self.purchases = [{"age_days": 3, "bands": [0, 1, 2, None, 0, 1, 2],
                           "isin": "SECRET_ASSET", "transaction_id": "SECRET_TRANSACTION", "date": "2021-05-29"}]

    def test_prompt_redacts_identity_and_dates(self):
        prompt = make_prompt(self.purchases)
        payload = json.loads(prompt.split("HISTORY=", 1)[1])
        self.assertEqual(payload, [{"age": 3, "bands": [0, 1, 2, None, 0, 1, 2]}])
        self.assertNotIn("SECRET", prompt)
        self.assertNotIn("2021-05-29", prompt)

    def test_unknown_metric_is_shared_uniform_control(self):
        profile = validate_profile(json.dumps({"profile": [[1, 0, 0]] * 7}), self.purchases)
        self.assertEqual(profile[3], [1 / 3] * 3)
        self.assertEqual(profile[0], [1, 0, 0])

    def test_small_rounding_error_normalizes(self):
        profile = validate_profile(json.dumps({"profile": [[.33, .33, .33]] * 7}), self.purchases)
        self.assertAlmostEqual(sum(profile[0]), 1)

    def test_invalid_shape_boolean_nan_or_sum_raises(self):
        invalid = [[], [[1, 0, 0]] * 6, [[True, 0, 0]] * 7,
                   [[float("nan"), .5, .5]] * 7, [[.1, .1, .1]] * 7,
                   [[-.1, .6, .5]] * 7, [[1.1, 0, -.1]] * 7]
        for profile in invalid:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                validate_profile(json.dumps({"profile": profile}), self.purchases)


if __name__ == "__main__":
    unittest.main()
