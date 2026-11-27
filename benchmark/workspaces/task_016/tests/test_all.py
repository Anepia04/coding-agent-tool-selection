"""calculator 与 text_utils 的单元测试。"""

import unittest

from calculator import divide, percentage
from text_utils import normalize, truncate


class TestAll(unittest.TestCase):
    def test_divide(self) -> None:
        self.assertEqual(divide(6, 3), 2)

    def test_percentage(self) -> None:
        self.assertEqual(percentage(1, 0), 0.0)

    def test_normalize(self) -> None:
        self.assertEqual(normalize(" a  b "), "a b")

    def test_truncate(self) -> None:
        self.assertEqual(truncate("abcdef", 3), "abc...")


if __name__ == "__main__":
    unittest.main()
