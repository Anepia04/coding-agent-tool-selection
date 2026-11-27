"""stats 的单元测试。"""

import unittest

from stats import median, summarize


class TestStats(unittest.TestCase):
    def test_summarize(self) -> None:
        self.assertEqual(summarize([1.0, 2.0, 3.0]), (1.0, 3.0, 2.0))

    def test_median(self) -> None:
        self.assertEqual(median([3, 1, 2]), 2)


if __name__ == "__main__":
    unittest.main()
