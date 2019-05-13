import unittest
import yaml
import os

from pythainlp.benchmarks import word_tokenisation

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
        with open("./tests/data/sentences.yml", 'r') as stream:
            test_sentences = yaml.safe_load(stream)

        _print('')
        for pair in test_sentences:
            exp, act = pair['expected'], pair['actual']

            _print('Expected: %s\n  Actual: %s' % (exp, act))
            result = word_tokenisation._compute_stats(
                exp,
                act
            ) 

            _print(result)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()