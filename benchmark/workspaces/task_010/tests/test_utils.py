"""utils 的单元测试。"""

import unittest

from utils import average, clamp, normalize, percent, truncate


class TestUtils(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize(" a  b "), "a b")

    def test_truncate(self) -> None:
        self.assertEqual(truncate("abcdef", 3), "abc...")

    def test_clamp(self) -> None:
        self.assertEqual(clamp(15, 0, 10), 10)

    def test_percent(self) -> None:
        self.assertEqual(percent(1, 0), 0.0)

    def test_average(self) -> None:
        self.assertEqual(average([1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
