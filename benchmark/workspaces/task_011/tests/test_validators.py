"""validators 的单元测试（异常分支尚未覆盖）。"""

import unittest

from validators import validate_name


class TestValidators(unittest.TestCase):
    def test_validate_name(self) -> None:
        self.assertEqual(validate_name(" alice "), "alice")


if __name__ == "__main__":
    unittest.main()
