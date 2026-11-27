"""text_utils 的单元测试（位于项目根目录）。"""

import unittest

from text_utils import normalize


class TestTextUtils(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize(" a  b "), "a b")


if __name__ == "__main__":
    unittest.main()
