import json
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app import main, storage
from app.engine import TaskError


def request(auto=False, max_rounds=1, optimization_history=None):
    return main.RunRequest(
        prompt="original prompt",
        prompt_mode="system",
        test_input="sample input",
        expected="expected result",
        reference="reference only for evaluation",
        model=main.ModelSettings(provider="deepseek", model="deepseek-flash", thinking="disabled"),
        target_api_key="target-secret-test-value",
        jev_api_key="jev-secret-test-value",
        auto_optimize=auto,
        threshold=85,
        max_rounds=max_rounds,
        optimization_history=optimization_history or [],
    )


class RunWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "runs.db"
        self.db_patch = patch.object(storage, "DB_PATH", self.db_path)
        self.db_patch.start()
        storage.initialize()

    async def asyncTearDown(self):
        main.STOP_FLAGS.clear()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    async def run_stream(self, payload):
        response = await main.stream_run(payload)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    async def test_auto_optimization_passes_exact_new_prompt_and_stops_on_first_dual_threshold(self):
        generated_prompts = []
        generated_parameters = []

        async def generate(_client, _provider, _model, _region, _key, messages, params):
            generated_prompts.append(messages)
            generated_parameters.append(dict(params))
            return f"output-{len(generated_prompts)}", {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}

        with patch.object(main, "chat", new=AsyncMock(side_effect=generate)), \
             patch.object(main, "jev_assess", new=AsyncMock(side_effect=[(0.6, {"requirements_met": 0.45}, {}), (0.9, {"requirements_met": 0.96}, {})])), \
             patch.object(main, "evaluate", new=AsyncMock(side_effect=[({"score": 70, "strengths": [], "issues": [], "missing_requirements": ["length"]}, []), ({"score": 91, "strengths": ["clear"], "issues": [], "missing_requirements": []}, [])])), \
             patch.object(main, "improve", new=AsyncMock(return_value=({"advice": "clarify", "reason": "missing length", "new_prompt": "complete revised prompt", "parameter_advice": "降低随机性", "parameter_reason": "提高约束遵循", "next_parameters": {"temperature": 0.2, "top_p": 1, "top_k": 40, "max_tokens": 2048, "thinking": "disabled"}, "parameter_warnings": []}, []))) as improve:
            stream = await self.run_stream(request(auto=True, max_rounds=4))

        run_id = json.loads(next(line[6:] for line in stream.splitlines() if line.startswith("data: ") and '"run_id"' in line and '"mode"' in line)) ["run_id"]
        result = storage.get_run(run_id)
        self.assertEqual([round_item["prompt"] for round_item in result["rounds"]], ["original prompt", "complete revised prompt"])
        self.assertEqual([round_item["status"] for round_item in result["rounds"]], ["completed", "target_reached"])
        self.assertEqual(result["rounds"][1]["output"], "output-2")
        self.assertEqual(result["rounds"][0]["jev_checks"]["requirements_met"], 0.45)
        self.assertEqual(result["rounds"][0]["next_parameters"]["temperature"], 0.2)
        self.assertEqual(result["rounds"][1]["parameters"]["temperature"], 0.2)
        self.assertFalse(result["rounds"][1].get("next_prompt"))
        self.assertEqual(len(generated_prompts), 2)
        self.assertEqual(generated_prompts[1][0], {"role": "system", "content": "complete revised prompt"})
        self.assertEqual(generated_prompts[1][1], {"role": "user", "content": "sample input"})
        self.assertEqual(generated_parameters[1]["temperature"], 0.2)
        self.assertEqual(improve.await_count, 1)
        self.assertEqual(result["status"], "target_reached")
        raw_db = self.db_path.read_bytes()
        self.assertNotIn(b"target-secret-test-value", raw_db)
        self.assertNotIn(b"jev-secret-test-value", raw_db)

    async def test_manual_run_sends_history_only_to_suggestion_call_and_saves_it_without_keys(self):
        prior = main.OptimizationHistoryRound(
            source_run_id="prior-run", round_no=1, prompt="prior prompt", prompt_mode="system",
            test_input="sample input", expected="expected result", output="prior output",
            parameters={"temperature": 1.1, "top_p": 0.9, "target_api_key": "history-secret-value"}, jev_probability=0.4,
            evaluation={"score": 45, "issues": ["did not follow structure"]},
        )
        generated_messages = []

        async def generate(_client, _provider, _model, _region, _key, messages, _params):
            generated_messages.append(messages)
            return "current output", {"total_tokens": 3}

        async def improve(*args):
            self.assertEqual(args[-1][0]["prompt"], "prior prompt")
            self.assertEqual(args[-1][0]["parameters"]["temperature"], 1.1)
            return ({"advice": "clarify", "reason": "previous output missed structure", "new_prompt": "new prompt", "parameter_advice": "keep", "parameter_reason": "stable", "next_parameters": {"temperature": 0.7, "top_p": 1, "top_k": 40, "max_tokens": 2048, "thinking": "disabled"}, "parameter_warnings": []}, {})

        payload = request(optimization_history=[prior])
        with patch.object(main, "chat", new=AsyncMock(side_effect=generate)), \
             patch.object(main, "jev_assess", new=AsyncMock(return_value=(0.4, {}, {}))), \
             patch.object(main, "evaluate", new=AsyncMock(return_value=({"score": 45, "strengths": [], "issues": [], "missing_requirements": []}, {}))), \
             patch.object(main, "improve", new=AsyncMock(side_effect=improve)):
            await self.run_stream(payload)

        run = storage.get_run(next(iter(storage.list_runs()))["id"])
        self.assertEqual(run["config"]["optimization_history"][0]["output"], "prior output")
        self.assertNotIn("prior output", generated_messages[0][0]["content"])
        raw_db = self.db_path.read_bytes()
        self.assertNotIn(b"target-secret-test-value", raw_db)
        self.assertNotIn(b"jev-secret-test-value", raw_db)
        self.assertNotIn(b"history-secret-value", raw_db)

    async def test_auto_suggestion_context_accumulates_previous_rounds_but_generation_does_not(self):
        generated_messages = []

        async def generate(_client, _provider, _model, _region, _key, messages, _params):
            generated_messages.append(messages)
            return f"output-{len(generated_messages)}", {"total_tokens": 2}

        recommendation = lambda prompt: ({"advice": "improve", "reason": "prior issue", "new_prompt": prompt, "parameter_advice": "keep", "parameter_reason": "stable", "next_parameters": {"temperature": 0.7, "top_p": 1, "top_k": 40, "max_tokens": 2048, "thinking": "disabled"}, "parameter_warnings": []}, {})
        with patch.object(main, "chat", new=AsyncMock(side_effect=generate)), \
             patch.object(main, "jev_assess", new=AsyncMock(return_value=(0.3, {"requirements_met": 0.25}, {}))), \
             patch.object(main, "evaluate", new=AsyncMock(return_value=({"score": 40, "strengths": [], "issues": ["missed requirement"], "missing_requirements": ["required section"]}, {}))), \
             patch.object(main, "improve", new=AsyncMock(side_effect=[recommendation("second prompt"), recommendation("third prompt")])) as improve:
            await self.run_stream(request(auto=True, max_rounds=3))

        run = storage.get_run(next(iter(storage.list_runs()))["id"])
        self.assertEqual([item["prompt"] for item in run["rounds"]], ["original prompt", "second prompt", "third prompt"])
        self.assertEqual(improve.await_args_list[0].args[-1], [])
        second_context = improve.await_args_list[1].args[-1]
        self.assertEqual(len(second_context), 1)
        self.assertEqual(second_context[0]["prompt"], "original prompt")
        self.assertEqual(second_context[0]["output"], "output-1")
        self.assertEqual(second_context[0]["parameters"]["temperature"], 0.7)
        self.assertEqual(second_context[0]["evaluation"]["issues"], ["missed requirement"])
        self.assertNotIn("output-1", generated_messages[1][0]["content"])
        self.assertEqual(run["rounds"][1]["optimization_history_rounds"], 1)

    async def test_jev_failure_preserves_generated_output_and_marks_run_failed(self):
        with patch.object(main, "chat", new=AsyncMock(return_value=("kept output", {"total_tokens": 2}))), \
             patch.object(main, "jev_assess", new=AsyncMock(side_effect=TaskError("Jev unavailable", "jev_timeout"))), \
             patch.object(main, "evaluate", new=AsyncMock()) as evaluate:
            stream = await self.run_stream(request())

        run_id = next(iter(storage.list_runs()))["id"]
        result = storage.get_run(run_id)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["rounds"][0]["output"], "kept output")
        self.assertEqual(result["rounds"][0]["error"]["kind"], "jev_timeout")
        self.assertIn('"status": "failed"', stream)
        evaluate.assert_not_awaited()

    async def test_stop_during_generation_keeps_output_and_skips_jev(self):
        async def generate(*_args):
            next(iter(main.STOP_FLAGS.values())).set()
            return "partial output", {"total_tokens": 1}

        with patch.object(main, "chat", new=AsyncMock(side_effect=generate)), \
             patch.object(main, "jev_assess", new=AsyncMock()) as jev:
            stream = await self.run_stream(request(auto=True, max_rounds=15))

        result = storage.get_run(next(iter(storage.list_runs()))["id"])
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["rounds"][0]["status"], "stopped")
        self.assertEqual(result["rounds"][0]["output"], "partial output")
        self.assertIn('"status": "stopped"', stream)
        jev.assert_not_awaited()

    async def test_max_rounds_includes_the_original_prompt_round(self):
        with patch.object(main, "chat", new=AsyncMock(side_effect=[("first", {}), ("second", {})])), \
             patch.object(main, "jev_assess", new=AsyncMock(side_effect=[(0.4, {}, {}), (0.5, {}, {})])), \
             patch.object(main, "evaluate", new=AsyncMock(side_effect=[({"score": 50, "strengths": [], "issues": [], "missing_requirements": []}, []), ({"score": 55, "strengths": [], "issues": [], "missing_requirements": []}, [])])), \
             patch.object(main, "improve", new=AsyncMock(return_value=({"advice": "change", "reason": "low score", "new_prompt": "second prompt", "parameter_advice": "keep", "parameter_reason": "stable", "next_parameters": {"temperature": 0.7, "top_p": 1, "top_k": 40, "max_tokens": 2048, "thinking": "disabled"}, "parameter_warnings": []}, []))) as improve:
            await self.run_stream(request(auto=True, max_rounds=2))

        result = storage.get_run(next(iter(storage.list_runs()))["id"])
        self.assertEqual([item["prompt"] for item in result["rounds"]], ["original prompt", "second prompt"])
        self.assertEqual([item["status"] for item in result["rounds"]], ["completed", "max_rounds"])
        self.assertEqual(result["rounds"][1]["output"], "second")
        self.assertFalse(result["rounds"][1].get("next_prompt"))
        self.assertEqual(improve.await_count, 1)
        self.assertEqual(result["status"], "max_rounds")

    async def test_reconcile_marks_active_request_as_stopping_and_signals_it(self):
        storage.create_run("active-refresh", "auto", "p", {})
        flag = asyncio.Event()
        main.STOP_FLAGS["active-refresh"] = flag
        result = await main.reconcile_runs()
        self.assertEqual(result["reconciled"], ["active-refresh"])
        self.assertTrue(flag.is_set())
        self.assertEqual(storage.get_run("active-refresh")["status"], "stopping")

    async def test_initialization_marks_unrecoverable_running_tasks_interrupted(self):
        storage.create_run("stale-run", "auto", "p", {})
        storage.save_round("stale-run", 1, {"round_no": 1, "status": "running", "prompt": "p", "output": "partial"})
        storage.initialize()
        result = storage.get_run("stale-run")
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["rounds"][0]["status"], "interrupted")
        self.assertEqual(result["rounds"][0]["output"], "partial")


if __name__ == "__main__":
    unittest.main()
