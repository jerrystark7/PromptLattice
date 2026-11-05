import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.engine import TaskError, chat, completion_text, evaluate, improve, jev_assess
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
    def test_provider_region_endpoints_are_selected(self):
        messages = [{"role": "user", "content": "ping"}]
        self.assertEqual(build_chat_payload("qwen", "qwen-plus", "cn", messages, {})[0], "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(build_chat_payload("qwen", "qwen-plus", "international", messages, {})[0], "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(build_chat_payload("moonshot", "kimi-k3", "cn", messages, {})[0], "https://api.moonshot.cn/v1/chat/completions")
        self.assertEqual(build_chat_payload("moonshot", "kimi-k3", "international", messages, {})[0], "https://api.moonshot.ai/v1/chat/completions")
        self.assertEqual(build_chat_payload("openai", "gpt-4.1", None, messages, {})[0], "https://api.openai.com/v1/chat/completions")

    def test_connection_probe_can_disable_deepseek_thinking(self):
        url, payload = build_chat_payload("deepseek", "deepseek-flash", None, [{"role": "user", "content": "ping"}], {"max_tokens": 64, "thinking": "disabled"})
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_deepseek_thinking_mode_omits_temperature_and_checks_top_p_range(self):
        _, payload = build_chat_payload("deepseek", "deepseek-flash", None, [{"role": "user", "content": "ping"}], {"temperature": .7, "top_p": .96, "thinking": "enabled"})
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["top_p"], .96)
        with self.assertRaisesRegex(ValueError, "0.95–1"):
            build_chat_payload("deepseek", "deepseek-flash", None, [{"role": "user", "content": "ping"}], {"top_p": .8, "thinking": "enabled"})

    def test_unsupported_top_k_is_not_sent(self):
        for provider, region in (("deepseek", None), ("moonshot", "cn"), ("openai", None)):
            with self.subTest(provider=provider):
                _, payload = build_chat_payload(provider, "test-model", region, [{"role": "user", "content": "ping"}], {"top_k": 40})
                self.assertNotIn("top_k", payload)
        _, qwen = build_chat_payload("qwen", "qwen-plus", "cn", [{"role": "user", "content": "ping"}], {"top_k": 40})
        self.assertEqual(qwen["top_k"], 40)

    def test_unknown_qwen_id_does_not_guess_top_k_support(self):
        _, payload = build_chat_payload("qwen", "qwen-custom-model", "cn", [{"role": "user", "content": "ping"}], {"top_k": 40})
        self.assertNotIn("top_k", payload)

    def test_openai_compatible_request_fields_are_adapted_by_provider(self):
        messages = [{"role": "user", "content": "ping"}]
        _, qwen = build_chat_payload("qwen", "qwen-plus", "cn", messages, {"temperature": .6, "top_p": .8, "top_k": 20, "max_tokens": 512})
        self.assertEqual({key for key in qwen if key not in {"model", "messages"}}, {"temperature", "top_p", "top_k", "max_tokens"})
        _, moonshot = build_chat_payload("moonshot", "kimi-k3", "international", messages, {"temperature": .6, "top_p": .8, "top_k": 20, "max_tokens": 512})
        self.assertEqual({key for key in moonshot if key not in {"model", "messages"}}, {"temperature", "top_p", "max_completion_tokens"})
        _, openai = build_chat_payload("openai", "gpt-4.1", None, messages, {"temperature": .6, "top_p": .8, "top_k": 20, "max_tokens": 512})
        self.assertEqual({key for key in openai if key not in {"model", "messages"}}, {"temperature", "top_p", "max_completion_tokens"})


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


class JevRequestTests(unittest.TestCase):
    def test_noul_probabilities_are_read_from_named_answers_and_usage(self):
        async def run():
            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                self.assertEqual(set(body["questions"]), {"is_satisfactory", "requirements_met", "format_and_length_met", "grounded_in_input", "style_and_tone_met"})
                self.assertTrue(all(question["type"] == "noul" for question in body["questions"].values()))
                self.assertIn('"current_prompt": "current"', body["state"])
                answers = {name: {"noul": probability} for name, probability in zip(body["questions"], (0.73, 0.91, 0.4, 0.88, 0.7))}
                return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14}})

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                return await jev_assess(client, "jev-test-key", "original", "current", "system", "input", "expected", "reference", "output")

        probability, checks, usage = asyncio.run(run())
        self.assertEqual(probability, 0.73)
        self.assertEqual(checks, {"requirements_met": 0.91, "format_and_length_met": 0.4, "grounded_in_input": 0.88, "style_and_tone_met": 0.7})
        self.assertEqual(usage, {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14})

    def test_missing_noul_probability_fails_without_inventing_score(self):
        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": {}}))) as client:
                return await jev_assess(client, "jev-test-key", "original", "current", "user", "", "expected", "", "output")

        with self.assertRaises(TaskError) as caught:
            asyncio.run(run())
        self.assertEqual(caught.exception.kind, "jev_invalid_response")


class IndependentEvaluationTests(unittest.TestCase):
    def test_model_evaluation_uses_active_prompt_and_server_computes_weighted_score(self):
        async def run(hard_failures):
            response = {
                "criterion_scores": {"requirements": 80, "factuality": 80, "format": 50, "style": 100},
                "hard_constraint_failures": hard_failures,
                "strengths": ["clear"], "issues": ["too short"], "missing_requirements": ["length"],
            }
            async_mock = AsyncMock(return_value=(json.dumps(response), {"total_tokens": 7}))
            async with httpx.AsyncClient() as client:
                with patch("app.engine.chat", new=async_mock):
                    result, usage = await evaluate(
                        client,
                        {"provider": "deepseek", "model": "deepseek-flash", "region": None, "target_api_key": "secret", "params": {}},
                        "strict 110-130 character prompt", "system", "sample input", "about 120 chars", "", "short output",
                    )
            messages = async_mock.await_args.args[5]
            evaluated = json.loads(messages[1]["content"])
            self.assertEqual(evaluated["current_prompt"], "strict 110-130 character prompt")
            self.assertEqual(evaluated["prompt_mode"], "system")
            self.assertNotIn("jev_probability", evaluated)
            return result, usage

        result, usage = asyncio.run(run([]))
        self.assertEqual(result["score"], 77)
        self.assertEqual(result["criterion_weights"], {"requirements": 35, "factuality": 30, "format": 20, "style": 15})
        self.assertEqual(usage, [{"total_tokens": 7}])

        result, _ = asyncio.run(run(["The output is shorter than the required range."]))
        self.assertEqual(result["score"], 59)


class ParameterRecommendationTests(unittest.TestCase):
    def test_unsupported_and_out_of_range_suggestions_are_sanitized(self):
        async def run():
            response = {
                "advice": "state the output length more clearly",
                "reason": "the sample is shorter than requested",
                "new_prompt": "a complete revised prompt",
                "parameter_advice": "lower temperature slightly",
                "parameter_reason": "reduce variation for strict length constraints",
                "next_parameters": {"temperature": 3, "top_p": 0.7, "top_k": 80, "max_tokens": 40000, "thinking": "disabled"},
            }
            current = {"temperature": 1, "top_p": 0.99, "top_k": 40, "max_tokens": 2048, "thinking": "enabled"}
            parameter_capabilities = {"parameters": {
                "temperature": {"supported": True, "minimum": 0, "maximum": 2},
                "top_p": {"supported": True, "minimum": 0.95, "maximum": 1},
                "top_k": {"supported": False, "reason": "Top-k unsupported"},
                "max_tokens": {"supported": True},
            }, "thinking": {"available": True, "selected": "enabled", "choices": ["enabled", "disabled"]}}
            async_mock = AsyncMock(return_value=(json.dumps(response), {"total_tokens": 11}))
            config = {"provider": "deepseek", "model": "deepseek-flash", "region": None, "target_api_key": "secret", "params": current}
            async with httpx.AsyncClient() as client:
                with patch("app.engine.chat", new=async_mock):
                    recommendation, usage = await improve(
                        client, config, "prompt", "user", "sample", "expected", "", "output", 0.3,
                        {"score": 60}, parameter_capabilities,
                        [{"round_no": 1, "prompt": "previous full prompt", "output": "previous generated output", "parameters": {"temperature": 1.2}, "evaluation": {"score": 48, "issues": ["too long"]}}],
                    )
            sent = json.loads(async_mock.await_args.args[5][1]["content"])
            self.assertEqual(sent["current_parameters"], current)
            self.assertEqual(sent["previous_rounds"][0]["prompt"], "previous full prompt")
            self.assertEqual(sent["previous_rounds"][0]["parameters"]["temperature"], 1.2)
            self.assertEqual(sent["previous_rounds"][0]["evaluation"]["score"], 48)
            self.assertFalse(any("api_key" in key.lower() for key in sent))
            return recommendation, usage

        recommendation, usage = asyncio.run(run())
        self.assertEqual(recommendation["next_parameters"], {"temperature": 2, "top_p": 0.99, "top_k": 40, "max_tokens": 32768, "thinking": "disabled"})
        self.assertEqual(len(recommendation["parameter_warnings"]), 4)
        self.assertEqual(usage, [{"total_tokens": 11}])


if __name__ == "__main__":
    unittest.main()
