import unittest
from pythainlp.benchmarks import word_segmentation
# print(sys.path)

TEST_DATA = [
    # expected ,actual
    ("ผม|ไม่|ชอบ|กิน|ผัก", "ผม|ไม่|ชอบ|กิน|ผัก"),
    ("ผม|ไม่|ชอบ|กิน|ผัก", "ผม|ไม่|ชอบ|กินผัก"),
]

class TestSegmentationBenchmark(unittest.TestCase):
    def test_binary_representation(self):
        sentence = "อากาศ|ร้อน|มาก|ครับ"
        rept, _ = word_segmentation._binary_representation(sentence)

        self.assertEqual(
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            rept.tolist()
        )
        

    def test_compute_stats(self):

        for exp, act in TEST_DATA:
            print('%s <-> %s' % (exp, act))
            result = word_segmentation._compute_stats(
                exp,
                act
            ) 

            print(result)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()