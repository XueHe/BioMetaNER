import csv
import statistics
import unittest

from biometaner.statistics import summarize


class ResultTests(unittest.TestCase):
    def test_all_figure4_statistics_from_integer_counts(self):
        with open('results/figure4/per_seed_per_type.csv') as handle:
            calculated = summarize(list(csv.DictReader(handle)))
        with open('results/figure4/summary.csv') as handle:
            expected = list(csv.DictReader(handle))
        self.assertEqual(len(calculated), 225)
        by_key = {(r['experiment'], r['backbone'], r['type']): r for r in calculated}
        for row in expected:
            r = by_key[(row['experiment'], row['backbone'], row['type'])]
            self.assertAlmostEqual(r['mean'], float(row['mean']), places=14)
            self.assertAlmostEqual(r['sample_sd'], float(row['sample_sd']), places=14)
            self.assertEqual(r['n_seeds'], int(row['n_seeds']))

    def test_not_pooling_seeds_and_single_seed_sd_unknown(self):
        rows = [dict(experiment='test', backbone='test', seed=s, type='MIC', TP=tp, FP=fp, FN=fn)
                for s, tp, fp, fn in [(42, 1, 0, 9), (1, 5, 20, 5)]]
        result = summarize(rows, check_golden=False)[0]
        expected = statistics.mean([2 / 11, 10 / 35])
        self.assertAlmostEqual(result['mean'], expected)
        self.assertNotAlmostEqual(result['mean'], 12 / 46)
        self.assertEqual(summarize(rows[:1], check_golden=False)[0]['sample_sd'], '')

    def test_duplicate_count_row_rejected(self):
        row = dict(experiment='test', backbone='test', seed=42, type='MIC', TP=1, FP=0, FN=496)
        with self.assertRaises(ValueError):
            summarize([row, row])


if __name__ == '__main__':
    unittest.main()
