"""全部模块的单元测试。"""

import unittest

from calculator import add
from data_loader import load_records
from text_utils import normalize


class TestAll(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(1, 2), 3)

    def test_load_records(self) -> None:
        self.assertEqual(len(load_records("sample.txt")), 2)

    def test_normalize(self) -> None:
        self.assertEqual(normalize(" a  b "), "a b")


if __name__ == "__main__":
    unittest.main()
