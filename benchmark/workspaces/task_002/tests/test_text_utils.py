"""text_utils 的单元测试。"""

import unittest

from text_utils import normalize, truncate


class TestTextUtils(unittest.TestCase):
    def test_normalize_collapses_spaces(self) -> None:
        self.assertEqual(normalize("  a   b  "), "a b")

    def test_truncate_short_text_unchanged(self) -> None:
        self.assertEqual(truncate("abc", 5), "abc")

    def test_truncate_long_text(self) -> None:
        self.assertEqual(truncate("abcdef", 3), "abc...")


if __name__ == "__main__":
    unittest.main()
