"""Qwen4-Max 真实 API 联调脚本（手动执行，不属于 unittest 流程）。

用法：
    python scripts/test_qwen_api.py
    python scripts/test_qwen_api.py "请用一句话介绍 Python"

配置来源（优先级从高到低）：
    1. 系统环境变量
    2. 项目根目录 .env（由 python-dotenv 自动加载）

必需变量：
    LLM_API_KEY   Qwen API Key（禁止硬编码，脚本不会打印其内容）
可选变量：
    LLM_MODEL     默认 qwen4-max
    LLM_BASE_URL  默认使用阿里云 MaaS 兼容接口地址

退出码：0 成功；2 缺少 API Key；3 调用失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.llm_client import LLMClient, LLMClientError  # noqa: E402
from config import get_settings  # noqa: E402

DEFAULT_MODEL = "qwen4-max"
DEFAULT_BASE_URL = (
    "https://llm-ly6g49klymb0kpyw.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_PROMPT = "请回复：API连接成功"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    prompt = args[0] if args else DEFAULT_PROMPT

    settings = get_settings()
    model = settings.model.model or DEFAULT_MODEL
    base_url = settings.model.base_url or DEFAULT_BASE_URL
    api_key = settings.model.api_key

    print("==== Qwen API 联调 ====")
    print(f"模型: {model}")
    print(f"Base URL: {base_url}")

    if not api_key:
        print("API Key: 未配置", file=sys.stderr)
        print(
            "错误：缺少 LLM_API_KEY。请在项目根目录 .env 中设置 "
            "LLM_API_KEY=<你的 Key>，或通过系统环境变量提供。",
            file=sys.stderr,
        )
        return 2
    print("API Key: 已配置（内容不打印）")
    print(f"请求: {prompt}")
    print()

    client = LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=settings.model.temperature,
    )

    try:
        response = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=min(settings.model.max_tokens, 256),
        )
    except ValueError as exc:  # 参数 / Key 校验失败
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except LLMClientError as exc:  # 网络异常、超时、API 错误或响应结构无效
        print(f"调用失败：{exc}", file=sys.stderr)
        print("请检查网络连通性、Base URL、模型名称与 API Key 是否有效。", file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001 - 脚本入口兜底，避免泄露堆栈中的敏感信息
        print(f"未预期错误（{type(exc).__name__}）：{exc}", file=sys.stderr)
        return 3

    print("---- 模型回复 ----")
    print(response.content if response.content else "<空回复>")

    if response.tool_calls:
        print("---- 工具调用（非预期）----")
        for call in response.tool_calls:
            print(f"- {call.name}({call.arguments}) id={call.id}")

    print("---- Token usage ----")
    print(f"input_tokens : {response.input_tokens}")
    print(f"output_tokens: {response.output_tokens}")
    print(f"total_tokens : {response.total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
