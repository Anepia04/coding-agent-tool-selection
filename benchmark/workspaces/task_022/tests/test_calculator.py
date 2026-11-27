"""calculator 的单元测试。"""

import unittest

from calculator import add, divide, subtract


class TestCalculator(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(1, 2), 3)

    def test_subtract(self) -> None:
        self.assertEqual(subtract(5, 3), 2)

    def test_divide(self) -> None:
        self.assertEqual(divide(6, 3), 2)


if __name__ == "__main__":
    unittest.main()
