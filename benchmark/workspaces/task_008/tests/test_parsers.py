"""parsers 的单元测试。"""

import unittest

from parsers import parse_port, parse_retries, parse_timeout


class TestParsers(unittest.TestCase):
    def test_parse_port(self) -> None:
        self.assertEqual(parse_port(" 9000 "), 9000)
        self.assertEqual(parse_port(""), 8000)

    def test_parse_timeout(self) -> None:
        self.assertEqual(parse_timeout("bad"), 30)

    def test_parse_retries(self) -> None:
        self.assertEqual(parse_retries("5"), 5)


if __name__ == "__main__":
    unittest.main()
