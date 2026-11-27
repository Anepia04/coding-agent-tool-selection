"""LLMClient 测试。

全部使用 Mock 模拟 OpenAI SDK 响应，不发起真实网络请求、不消耗 API 费用。
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.llm_client import LLMClient, LLMClientError, LLMResponse, ToolCall


def build_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    """构造与 OpenAI SDK 结构一致的工具调用对象。"""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def build_response(
    content=None,
    tool_calls=None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens=None,
    with_usage: bool = True,
):
    """构造一个最小化的 ChatCompletion 响应对象。"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = None
    if with_usage:
        total = total_tokens
        if total is None:
            total = prompt_tokens + completion_tokens
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
        usage=usage,
    )


class LLMClientTestBase(unittest.TestCase):
    """注入 Mock 客户端，避免任何真实网络调用。"""

    def make_client(self, **kwargs) -> tuple[LLMClient, MagicMock]:
        kwargs.setdefault("api_key", "test-key")
        client = LLMClient(model="test-model", **kwargs)
        fake_client = MagicMock()
        client._client = fake_client
        return client, fake_client.chat.completions.create


class TestTextResponse(LLMClientTestBase):
    """普通文本回复。"""

    def test_parse_text_response(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            content="已修复除零异常。",
            prompt_tokens=12,
            completion_tokens=8,
        )

        response = client.chat([{"role": "user", "content": "修复除零异常"}])

        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, "已修复除零异常。")
        self.assertEqual(response.tool_calls, [])
        self.assertFalse(response.has_tool_calls)

        call_kwargs = create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "test-model")
        self.assertEqual(call_kwargs["messages"], [{"role": "user", "content": "修复除零异常"}])
        self.assertNotIn("tools", call_kwargs)  # 未传工具时不应携带 tools 字段

    def test_empty_messages_rejected(self) -> None:
        client, create = self.make_client()
        with self.assertRaises(ValueError):
            client.chat([])
        create.assert_not_called()


class TestToolCalls(LLMClientTestBase):
    """Function Calling 解析。"""

    def test_single_tool_call(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            content=None,
            tool_calls=[build_tool_call("call_1", "read_file", '{"path": "src/main.py"}')],
        )

        response = client.chat(
            [{"role": "user", "content": "查看 main.py"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )

        self.assertTrue(response.has_tool_calls)
        self.assertIsNone(response.content)
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertIsInstance(call, ToolCall)
        self.assertEqual(call.id, "call_1")
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.arguments, {"path": "src/main.py"})
        self.assertIn("tools", create.call_args.kwargs)

    def test_multiple_tool_calls(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            tool_calls=[
                build_tool_call("call_1", "read_file", '{"path": "a.py"}'),
                build_tool_call("call_2", "run_tests", '{"timeout": 60}'),
                build_tool_call("call_3", "git_diff", "{}"),
            ],
        )

        response = client.chat([{"role": "user", "content": "修复并验证"}])

        self.assertEqual([call.name for call in response.tool_calls],
                         ["read_file", "run_tests", "git_diff"])
        self.assertEqual([call.id for call in response.tool_calls],
                         ["call_1", "call_2", "call_3"])
        self.assertEqual(response.tool_calls[1].arguments, {"timeout": 60})
        self.assertEqual(response.tool_calls[2].arguments, {})

    def test_arguments_already_dict(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            tool_calls=[build_tool_call("call_1", "write_file", '{"path": "b.py", "content": ""}')],
        )
        response = client.chat([{"role": "user", "content": "写文件"}])
        self.assertEqual(response.tool_calls[0].arguments, {"path": "b.py", "content": ""})

    def test_invalid_arguments_json_raises(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            tool_calls=[build_tool_call("call_1", "read_file", "{not-json")],
        )
        with self.assertRaises(LLMClientError):
            client.chat([{"role": "user", "content": "读取"}])

    def test_missing_function_field_raises(self) -> None:
        client, create = self.make_client()
        broken_call = SimpleNamespace(id="call_1", type="function", function=None)
        create.return_value = build_response(tool_calls=[broken_call])
        with self.assertRaises(LLMClientError):
            client.chat([{"role": "user", "content": "读取"}])


class TestTokenUsage(LLMClientTestBase):
    """Token 统计。"""

    def test_usage_is_parsed(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            content="ok", prompt_tokens=120, completion_tokens=45, total_tokens=165
        )

        response = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(response.input_tokens, 120)
        self.assertEqual(response.output_tokens, 45)
        self.assertEqual(response.total_tokens, 165)
        self.assertEqual(response.to_dict()["total_tokens"], 165)

    def test_total_tokens_fallback(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(
            content="ok", prompt_tokens=10, completion_tokens=5, total_tokens=0
        )
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(response.total_tokens, 15)

    def test_missing_usage_defaults_to_zero(self) -> None:
        client, create = self.make_client()
        create.return_value = build_response(content="ok", with_usage=False)
        response = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual((response.input_tokens, response.output_tokens, response.total_tokens),
                         (0, 0, 0))


class TestErrors(LLMClientTestBase):
    """API 异常、超时与无效响应。"""

    def test_api_exception_wrapped(self) -> None:
        client, create = self.make_client()
        create.side_effect = RuntimeError("connection reset")
        with self.assertRaises(LLMClientError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("connection reset", str(ctx.exception))

    def test_timeout_wrapped(self) -> None:
        client, create = self.make_client()
        create.side_effect = TimeoutError("request timed out")
        with self.assertRaises(LLMClientError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_empty_choices_raises(self) -> None:
        client, create = self.make_client()
        create.return_value = SimpleNamespace(choices=[], usage=None)
        with self.assertRaises(LLMClientError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_missing_message_raises(self) -> None:
        client, create = self.make_client()
        create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=None)], usage=None)
        with self.assertRaises(LLMClientError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_missing_api_key_raises(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "", "OPENAI_API_KEY": ""}, clear=False):
            client = LLMClient(model="test-model")
            self.assertIsNone(client.api_key)
            with self.assertRaises(ValueError):
                client.chat([{"role": "user", "content": "hi"}])

    def test_api_key_from_environment(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "env-key"}, clear=False):
            client = LLMClient(model="test-model")
            self.assertEqual(client.api_key, "env-key")


class TestRequestOptions(LLMClientTestBase):
    """初始化参数与单次调用参数。"""

    def test_default_options_and_overrides(self) -> None:
        client, create = self.make_client(temperature=0.1, max_tokens=256)
        create.return_value = build_response(content="ok")

        client.chat([{"role": "user", "content": "hi"}], temperature=0.9, timeout=30)

        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0.9)   # 单次调用覆盖默认值
        self.assertEqual(kwargs["max_tokens"], 256)    # 未覆盖时沿用初始化配置
        self.assertEqual(kwargs["timeout"], 30)

    def test_base_url_is_stored(self) -> None:
        client = LLMClient(model="qwen-plus", api_key="k",
                           base_url="https://example.com/v1")
        self.assertEqual(client.base_url, "https://example.com/v1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
