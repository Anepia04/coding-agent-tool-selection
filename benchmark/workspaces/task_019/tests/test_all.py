"""全部模块的单元测试。"""

import unittest

from calculator import add, divide
from parsers import parse_int
from text_utils import normalize, truncate


class TestAll(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(1, 2), 3)

    def test_divide(self) -> None:
        self.assertEqual(divide(6, 3), 2)

    def test_parse_int(self) -> None:
        self.assertEqual(parse_int("x", 7), 7)

    def test_normalize(self) -> None:
        self.assertEqual(normalize(" a  b "), "a b")

    def test_truncate(self) -> None:
        self.assertEqual(truncate("abcdef", 3), "abc...")


if __name__ == "__main__":
    unittest.main()
