"""命令行入口（调用 load 时参数名写错了）。"""

from __future__ import annotations

from data_loader import load


def main() -> None:
    """打印样本文件的行数。"""
    lines = load(file_path="sample.txt")
    print(len(lines))


if __name__ == "__main__":
    main()
