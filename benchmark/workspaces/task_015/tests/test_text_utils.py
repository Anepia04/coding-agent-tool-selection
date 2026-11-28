"""text_utils 的单元测试（只覆盖了 ASCII 输入）。"""

import unittest

from text_utils import to_bytes, to_display


class TestTextUtils(unittest.TestCase):
    def test_to_bytes_ascii(self) -> None:
        self.assertEqual(to_bytes("ab"), b"ab")

    def test_to_display_ascii(self) -> None:
        self.assertEqual(to_display("ab"), "ab")


if __name__ == "__main__":
    unittest.main()
