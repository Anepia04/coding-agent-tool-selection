"""report 的单元测试。"""

import unittest

from data_loader import load_names


class TestReport(unittest.TestCase):
    def test_load_names(self) -> None:
        self.assertEqual(load_names("names.txt"), ["alice", "bob"])


if __name__ == "__main__":
    unittest.main()
