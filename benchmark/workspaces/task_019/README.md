# sample_project (task_019)

极简示例项目，用于 Coding Agent 评测任务。

## 待处理

根目录的 `fix.patch` 是上游维护者提供的统一 diff 补丁，**尚未应用**。
补丁内容为移除源码中残留的调试输出。

## 运行测试

```bash
python -m unittest discover -s tests -v
```
