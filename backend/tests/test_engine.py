import asyncio
import json
import unittest

import httpx

from app.engine import TaskError, chat, completion_text
from app.providers import build_chat_payload


class CompletionNormalizationTests(unittest.TestCase):
    def test_chat_completions_text(self):
        text, reasoning_only = completion_text({"choices": [{"message": {"content": "hello"}}]})
        self.assertEqual(text, "hello")
        self.assertFalse(reasoning_only)

    def test_chat_completions_content_parts(self):
        text, _ = completion_text({"choices": [{"message": {"content": [{"type": "text", "text": "hello"}]}}]})
        self.assertEqual(text, "hello")

    def test_responses_api_output_text(self):
        text, _ = completion_text({"output_text": "hello"})
        self.assertEqual(text, "hello")

    def test_responses_api_output_items(self):
        text, _ = completion_text({"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]})
        self.assertEqual(text, "hello")

    def test_reasoning_is_not_returned_as_user_facing_content(self):
        text, reasoning_only = completion_text({"choices": [{"message": {"content": None, "reasoning_content": "thinking"}}]})
        self.assertEqual(text, "")
        self.assertTrue(reasoning_only)


class ProviderPayloadTests(unittest.TestCase):
    def test_connection_probe_can_disable_deepseek_thinking(self):
        url, payload = build_chat_payload("deepseek", "deepseek-flash", None, [{"role": "user", "content": "ping"}], {"max_tokens": 64, "thinking": "disabled"})
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_unsupported_top_k_is_not_sent(self):
        for provider, region in (("deepseek", None), ("moonshot", "cn"), ("openai", None)):
            with self.subTest(provider=provider):
                _, payload = build_chat_payload(provider, "test-model", region, [{"role": "user", "content": "ping"}], {"top_k": 40})
                self.assertNotIn("top_k", payload)
        _, qwen = build_chat_payload("qwen", "test-model", "cn", [{"role": "user", "content": "ping"}], {"top_k": 40})
        self.assertEqual(qwen["top_k"], 40)


class ChatRequestTests(unittest.TestCase):
    def test_chat_normalizes_response_and_usage(self):
        async def run():
            def handler(request: httpx.Request) -> httpx.Response:
                self.assertEqual(request.url.path, "/chat/completions")
                return httpx.Response(200, json={"choices": [{"message": {"content": "connected"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await chat(client, "deepseek", "deepseek-flash", None, "test-key", [{"role": "user", "content": "ping"}], {"thinking": "disabled"})

        text, usage = asyncio.run(run())
        self.assertEqual(text, "connected")
        self.assertEqual(usage, {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4})

    def test_reasoning_only_has_actionable_error(self):
        async def run():
            def handler(_: httpx.Request) -> httpx.Response:
                return httpx.Response(200, json={"choices": [{"message": {"content": None, "reasoning_content": "thinking"}}]})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await chat(client, "deepseek", "deepseek-flash", None, "test-key", [{"role": "user", "content": "ping"}], {})

        with self.assertRaises(TaskError) as caught:
            asyncio.run(run())
        self.assertEqual(caught.exception.kind, "reasoning_only")
        self.assertIn("增加最大输出 Token", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
