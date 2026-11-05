import test from "node:test";
import assert from "node:assert/strict";
import { adoptRoundPrompt, buildMessages, demoReachedFirstThreshold, demoRun, demoRounds, hasValidNextPrompt, optimizationHistoryForRun, removeSecrets, replacePromptOnly, restoreRoundConfig, supportedParameterPatch } from "./workflow.mjs";

test("system prompt messages retain the system role and test input", () => {
  assert.deepEqual(buildMessages("system rules", "user sample", "system"), [
    { role: "system", content: "system rules" },
    { role: "user", content: "user sample" },
  ]);
});

test("adopting a prompt changes only the prompt text", () => {
  const draft = { prompt: "old", promptMode: "system", testInput: "case", provider: "qwen", expected: "target" };
  assert.deepEqual(replacePromptOnly(draft, "new"), { ...draft, prompt: "new" });
});

test("adopting current system prompt keeps its role and every test setting", () => {
  const draft = { prompt: "old", promptMode: "user", testInput: "current case", provider: "openai", region: "default", model: "gpt-4.1", params: { temperature: 0.4 }, expected: "target", reference: "reference" };
  const adopted = adoptRoundPrompt(draft, "revised", { prompt_mode: "system" }, "current");
  assert.deepEqual(adopted, { ...draft, prompt: "revised", promptMode: "system" });
});

test("adopting a historical round changes only the prompt until full restore is requested", () => {
  const draft = { prompt: "old", promptMode: "user", testInput: "current case", provider: "openai", model: "gpt-4.1" };
  assert.deepEqual(adoptRoundPrompt(draft, "historical", { prompt_mode: "system" }, "history"), { ...draft, prompt: "historical" });
});

test("restoring a saved round restores its complete scenario without credentials", () => {
  const restored = restoreRoundConfig({ prompt: "p", prompt_mode: "system", test_input: "i", expected: "e", reference: "r", provider: "qwen", region: "cn", model: "qwen-plus", parameters: { top_k: 20 }, threshold: 90, max_rounds: 6, auto_optimize: true });
  assert.equal(restored.promptMode, "system");
  assert.equal(restored.testInput, "i");
  assert.deepEqual(restored.params, { temperature: 0.7, top_p: 1, top_k: 20, max_tokens: 2048, thinking: "default" });
  assert.equal("apiKey" in restored, false);
});

test("parameter adoption filters out unsupported, missing, and nonnumeric values", () => {
  assert.deepEqual(supportedParameterPatch(
    { temperature: 0.2, top_p: 0.8, top_k: 20, max_tokens: "4096" },
    { parameters: { temperature: { supported: true }, top_p: { supported: false }, top_k: { supported: true }, max_tokens: { supported: true } } },
  ), { temperature: 0.2, top_k: 20 });
  assert.deepEqual(supportedParameterPatch(
    { thinking: "disabled" },
    { thinking: { available: true, choices: ["enabled", "disabled"] } },
  ), { thinking: "disabled" });
});

test("suggestion context includes prior outputs, prompts, scores and parameters for the same scenario", () => {
  const scenario = { provider: "deepseek", region: "default", model: "deepseek-chat", prompt_mode: "system", test_input: "sample", expected: "expected", reference: "" };
  const run = {
    id: "run-2",
    config: { ...scenario, optimization_history: [{ source_run_id: "run-1", round_no: 1, prompt: "older prompt", output: "older output", parameters: { temperature: 0.9, api_key: "must-not-pass" }, evaluation: { score: 44 } }] },
    rounds: [{ round_no: 1, status: "completed", prompt: "current prompt", prompt_mode: "system", test_input: "sample", expected: "expected", reference: "", parameters: { temperature: 0.2, top_p: 0.95 }, output: "current output", jev_probability: 0.7, evaluation: { score: 71, issues: ["too brief"] }, next_prompt: "next prompt", next_parameters: { temperature: 0.1 } }],
  };
  const history = optimizationHistoryForRun(run, scenario);
  assert.equal(history.length, 2);
  assert.equal(history[0].prompt, "older prompt");
  assert.equal(history[0].parameters.api_key, undefined);
  assert.equal(history[1].prompt, "current prompt");
  assert.equal(history[1].output, "current output");
  assert.equal(history[1].parameters.temperature, 0.2);
  assert.equal(history[1].evaluation.score, 71);
  assert.equal(history[1].next_prompt, "next prompt");
});

test("suggestion context is omitted for a different scenario and is bounded to 15 rounds", () => {
  const scenario = { provider: "qwen", region: "cn", model: "qwen-plus", prompt_mode: "user", test_input: "x", expected: "y", reference: "" };
  const rounds = Array.from({ length: 20 }, (_, index) => ({ round_no: index + 1, status: "completed", prompt: `p${index}`, output: `o${index}`, evaluation: { score: index } }));
  const matchingRun = { id: "run", config: scenario, rounds };
  assert.equal(optimizationHistoryForRun(matchingRun, { ...scenario, expected: "different" }).length, 0);
  const history = optimizationHistoryForRun(matchingRun, scenario);
  assert.equal(history.length, 15);
  assert.equal(history[0].round_no, 6);
  assert.deepEqual(optimizationHistoryForRun(demoRun, scenario), []);
});

test("demo stops on the first round where both scores reach threshold and does not invent a next prompt", () => {
  assert.equal(demoReachedFirstThreshold(demoRounds, 85), 1);
  assert.equal(demoRounds[0].next_prompt, demoRounds[1].prompt);
  assert.equal(demoRounds[1].status, "target_reached");
  assert.equal(demoRounds[1].next_prompt, "");
  assert.equal(hasValidNextPrompt(demoRounds[1]), false);
  assert.equal(demoRun.rounds.length, 2);
});

test("terminal rounds do not expose empty or invalid recommendation actions", () => {
  for (const status of ["target_reached", "max_rounds", "stopped", "repeated_prompt", "failed", "interrupted"]) {
    assert.equal(hasValidNextPrompt({ status, next_prompt: "" }), false);
  }
  assert.equal(hasValidNextPrompt({ status: "completed", next_prompt: "full prompt" }), true);
});

test("export sanitization removes credentials recursively", () => {
  const safe = removeSecrets({ run: { api_key: "secret", round: { jev_api_key: "secret2", output: "ok" } } });
  assert.deepEqual(safe, { run: { round: { output: "ok" } } });
});
