from copy import deepcopy
import unittest

from biometaner.evaluation import f1, score, strict_counts, load_golden
from biometaner.matcher import partial_counts


def span(start, end, typ='MIC', **kwargs):
    return dict(start=start, end=end, type=typ, **kwargs)


class MatchingTests(unittest.TestCase):
    def test_partial_not_strict(self):
        self.assertEqual(partial_counts([span(0, 5)], [span(1, 4)], 'MIC'), (1, 0, 0))
        self.assertEqual(strict_counts([span(0, 5)], [span(1, 4)], 'MIC'), (0, 1, 1))

    def test_type_aware_and_touching_boundaries(self):
        self.assertEqual(partial_counts([span(0, 5)], [span(0, 5, 'DIS')]), (0, 1, 1))
        self.assertEqual(partial_counts([span(0, 5)], [span(5, 7)]), (0, 1, 1))

    def test_one_to_one_duplicates(self):
        for matcher in [partial_counts, strict_counts]:
            self.assertEqual(matcher([span(0, 5)], [span(0, 5), span(0, 5)], 'MIC'), (1, 1, 0))

    def test_prediction_order_not_maximum_matching_or_confidence_sort(self):
        gold = [span(0, 5), span(5, 10)]
        broad, narrow = span(0, 10, score=.1), span(0, 5, score=.9)
        self.assertEqual(partial_counts(gold, [broad, narrow]), (1, 1, 1))
        self.assertEqual(partial_counts(gold, [narrow, broad]), (2, 0, 0))

    def test_zero_denominator(self):
        self.assertEqual(f1(0, 0, 0), 0)

    def test_gold_hash_fails_for_synthetic(self):
        with self.assertRaises(ValueError):
            load_golden('examples/gold.jsonl')

    def test_document_and_text_validation(self):
        gold = [dict(pmid='1', text='Alpha', entities=[span(0, 5)])]
        predictions = [dict(pmid='1', text='Alpha', pred_entities=[span(0, 5)])]
        self.assertEqual(score(gold, predictions)['partial']['macro_F1'], .25)
        bad = deepcopy(predictions)
        bad[0]['text'] = 'alpha'
        with self.assertRaises(ValueError):
            score(gold, bad)
        with self.assertRaises(ValueError):
            score(gold, predictions + predictions)


if __name__ == '__main__':
    unittest.main()
