"""calculator 的单元测试（位于项目根目录）。"""

import unittest

from calculator import add


class TestCalculator(unittest.TestCase):
    def test_add(self) -> None:
        self.assertEqual(add(1, 2), 3)


if __name__ == "__main__":
    unittest.main()
