"""data_loader 的单元测试。"""

import unittest

from data_loader import load_records, load_scores


class TestDataLoader(unittest.TestCase):
    def test_load_records(self) -> None:
        rows = load_records("sample.txt")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["city"], "beijing")

    def test_load_scores(self) -> None:
        rows = load_scores("sample.txt")
        self.assertEqual(rows[1]["city"], "")


if __name__ == "__main__":
    unittest.main()
