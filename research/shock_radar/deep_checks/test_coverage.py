import math
import unittest
from .coverage import classify, drawdown, quantile, summary


class CoverageTests(unittest.TestCase):
    def test_interval_is_inclusive(self):
        self.assertEqual(classify(.1, .1, .2), 'covered')
        self.assertEqual(classify(.2, .1, .2), 'covered')
        self.assertEqual(classify(.099, .1, .2), 'below')
        self.assertEqual(classify(.201, .1, .2), 'above')

    def test_invalid_never_dropped(self):
        self.assertEqual(classify(math.nan, .1, .2), 'invalid')
        self.assertEqual(classify(.1, .2, .1), 'invalid')
        values = summary([{'position':'covered'}, {'position':'invalid'}])
        self.assertEqual(values['coverage_all_expected'], .5)
        self.assertEqual(values['invalid'], 1)
        self.assertIsNone(summary([])['coverage_valid'])

    def test_spot_counts_as_peak(self):
        self.assertAlmostEqual(drawdown([100, 90, 95]), .1)
        self.assertAlmostEqual(drawdown([100, 120, 96]), .2)
        self.assertAlmostEqual(drawdown([100, 101, 102]), 0)

    def test_finite_sample_linear_quantile(self):
        self.assertAlmostEqual(quantile(list(range(10)), .05), .45)
        self.assertAlmostEqual(quantile(list(range(10)), .95), 8.55)


if __name__ == '__main__':
    unittest.main()
