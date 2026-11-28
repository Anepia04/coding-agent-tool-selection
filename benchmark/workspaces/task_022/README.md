# sample_project (task_022)

极简示例项目，用于 Coding Agent 评测任务。

若 .git 文件存在，该task会被识别为仓库，出现嵌套仓库的情况。
因此将 .git 改名为 .gitdata 避免这个问题。
实验前请将名字改回 .git

## 运行测试

```bash
python -m unittest discover -s tests -v
```
