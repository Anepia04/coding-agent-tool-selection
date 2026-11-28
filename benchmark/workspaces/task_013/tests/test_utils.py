"""utils 的单元测试。"""

import unittest

from utils import format_name, parse_pair


class TestUtils(unittest.TestCase):
    def test_format_name(self) -> None:
        self.assertEqual(format_name(" alice "), "Alice")

    def test_parse_pair(self) -> None:
        self.assertEqual(parse_pair("a = 1"), ("a", "1"))


if __name__ == "__main__":
    unittest.main()
