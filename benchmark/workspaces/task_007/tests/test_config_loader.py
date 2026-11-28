"""config_loader 的单元测试。"""

import unittest

from config_loader import get_port, load_config


class TestConfigLoader(unittest.TestCase):
    def test_load_config(self) -> None:
        config = load_config("config.json")
        self.assertEqual(config["host"], "localhost")

    def test_get_port(self) -> None:
        self.assertEqual(get_port(load_config("config.json")), 8000)


if __name__ == "__main__":
    unittest.main()
