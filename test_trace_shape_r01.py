import math
import unittest

from trace_shape_r01 import summarize_trace


class TemporalFeatures(unittest.TestCase):
    def test_constant(self):
        result = summarize_trace([2.0]*9, [0.5]*9, [-1.0]*9, [True]*9)
        self.assertEqual(result['valid_token_count'], 9)
        self.assertTrue(all(value == 0 for key, value in result.items() if key != 'valid_token_count'))

    def test_normalized_linear_slope(self):
        signals = [i/8 for i in range(9)]
        result = summarize_trace(signals, signals, signals, [True]*9)
        self.assertAlmostEqual(result['entropy_position_slope'], 1.0)
        self.assertEqual(result['entropy_excess_variation'], 0.0)

    def test_temporal_reversal_distinguishable_not_mean(self):
        rising = [0., 0., 1., 1., 2., 2.]
        falling = rising[::-1]
        a = summarize_trace(rising, rising, rising, [True]*6)
        b = summarize_trace(falling, falling, falling, [True]*6)
        self.assertEqual(sum(rising), sum(falling))
        self.assertEqual(a['entropy_front_tail_delta'], -b['entropy_front_tail_delta'])
        self.assertGreater(a['entropy_front_tail_delta'], 0)

    def test_fluctuation(self):
        result = summarize_trace([0., 2., 0., 2., 0.], [1.]*5, [-1.]*5, [True]*5)
        self.assertEqual(result['entropy_excess_variation'], 8.)

    def test_padding_and_eos_removed_before_segmentation(self):
        a = summarize_trace([0., 1., 2.], [0., 1., 2.], [-3., -2., -1.], [True]*3)
        b = summarize_trace([0., math.nan, 1., 2., math.inf], [0., 9., 1., 2., 9.],
                            [-3., 9., -2., -1., 9.], [True, False, True, True, False])
        self.assertEqual(a, b)

    def test_empty_and_singleton(self):
        for values in ([], [0.]):
            result = summarize_trace(values, values, values, [True]*len(values))
            self.assertEqual(len(result), 13)
            self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_lengths_rejected(self):
        with self.assertRaises(ValueError):
            summarize_trace([1.], [], [1.], [True])

    def test_nonfinite_valid_rejected(self):
        with self.assertRaises(ValueError):
            summarize_trace([math.nan], [0.], [-1.], [True])

    def test_nonbool_mask_rejected(self):
        with self.assertRaises(ValueError):
            summarize_trace([1.], [0.], [-1.], [1])


if __name__ == '__main__':
    unittest.main()
