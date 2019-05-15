import unittest
import yaml
import os
import numpy as np

from pythainlp.benchmarks import word_tokenisation

with open("./tests/data/sentences.yml", 'r') as stream:
    TEST_DATA = yaml.safe_load(stream)

def _print(text):
    if "TEST_VERBOSE" in os.environ and os.environ["TEST_VERBOSE"]:
        print(text)

class TestSegmentationBenchmark(unittest.TestCase):
    def test_binary_representation(self):
        sentence = "อากาศ|ร้อน|มาก|ครับ"
        rept, _ = word_tokenisation._binary_representation(sentence)

        self.assertEqual(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            rept.tolist()
        )

    def test_compute_stats(self):
        _print('')
        for pair in TEST_DATA['sentences']:
            exp, act = pair['expected'], pair['actual']

            _print('Expected: %s\n  Actual: %s' % (exp, act))
            result = word_tokenisation._compute_stats(
                exp,
                act
            ) 

            _print(result)
            self.assertIsNotNone(result)

    def test_benchmark(self):
        expected = []
        actual = []
        for pair in TEST_DATA['sentences']:
            expected.append(pair['expected'])
            actual.append(pair['actual'])

        df = word_tokenisation.benchmark(expected, actual)
        print(df)

        _print(df.describe())

        self.assertIsNotNone(df)

    def test_count_correctly_tokenised_words(self):
        for d in TEST_DATA['binary_sentences']:
            sample = np.array(list(d['actual'])).astype(int)
            ref_sample = np.array(list(d['expected'])).astype(int)

            wb = list(word_tokenisation._find_word_boudaries(ref_sample))

            self.assertEqual(
                word_tokenisation._count_correctly_tokenised_words(sample, wb),
                d['expected_count']
            )


if __name__ == '__main__':
    unittest.main()