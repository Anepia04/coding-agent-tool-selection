"""report 的单元测试。"""

import unittest

from report import build_summary


class TestReport(unittest.TestCase):
    def test_build_summary(self) -> None:
        self.assertEqual(build_summary("sample.txt"), "sample.txt: 3 行")


if __name__ == "__main__":
    unittest.main()
