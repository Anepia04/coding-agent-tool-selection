"""calculator 模块的单元测试。"""

from __future__ import annotations

import unittest

import calculator


class TestAdd(unittest.TestCase):
    def test_positive_operands(self) -> None:
        self.assertEqual(calculator.add(2, 3), 5)

    def test_negative_operand(self) -> None:
        self.assertEqual(calculator.add(-2, 3), 1)


class TestSubtract(unittest.TestCase):
    def test_positive_result(self) -> None:
        self.assertEqual(calculator.subtract(5, 3), 2)

    def test_negative_result(self) -> None:
        self.assertEqual(calculator.subtract(3, 5), -2)


class TestMultiply(unittest.TestCase):
    def test_positive_operands(self) -> None:
        self.assertEqual(calculator.multiply(4, 3), 12)

    def test_zero_operand(self) -> None:
        self.assertEqual(calculator.multiply(7, 0), 0)


class TestDivide(unittest.TestCase):
    def test_exact_division(self) -> None:
        self.assertEqual(calculator.divide(6, 3), 2)

    def test_fraction_result(self) -> None:
        self.assertEqual(calculator.divide(1, 4), 0.25)

    def test_negative_operand(self) -> None:
        self.assertEqual(calculator.divide(-6, 3), -2)


if __name__ == "__main__":
    unittest.main()
