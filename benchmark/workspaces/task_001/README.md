# sample_project

一个用于教学与实验的最小 Python 计算器项目，仅使用标准库。

## 目录结构

```
sample_project/
├── calculator.py          # 计算器实现
├── tests/
│   ├── __init__.py
│   └── test_calculator.py # 单元测试
├── requirements.txt       # 无第三方依赖
└── README.md
```

## 运行测试

在项目根目录执行：

```bash
python -m unittest discover -v
```

也可以只跑单个测试文件：

```bash
python -m unittest tests.test_calculator -v
```

## 已知问题

`calculator.divide(a, b)` 在 `b == 0` 时会直接抛出 `ZeroDivisionError`，
调用方拿到的是解释器层面的异常，无法获得可读的错误说明。

## 验收标准

1. 全部单元测试通过；
2. 除零场景返回明确的错误提示，而不是直接崩溃。
