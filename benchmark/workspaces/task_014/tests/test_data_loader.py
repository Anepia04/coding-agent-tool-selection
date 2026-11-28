"""data_loader 的单元测试。"""

import unittest

from data_loader import load_records, total_score


class TestDataLoader(unittest.TestCase):
    def test_load_records(self) -> None:
        self.assertEqual(len(load_records("scores.txt")), 2)

    def test_total_score(self) -> None:
        self.assertEqual(total_score("scores.txt"), 170)


if __name__ == "__main__":
    unittest.main()
