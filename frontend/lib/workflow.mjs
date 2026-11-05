/** @typedef {"user" | "system"} PromptMode */

export const demoRounds = [
  {
    round_no: 1, status: "completed", prompt_mode: "user",
    prompt: "你是一名专业的产品文案。介绍以下产品。",
    test_input: "轻量通勤背包，18L，防泼水，适合日常通勤。",
    expected: "用专业亲切的中文介绍，约 120 字；说明容量、防泼水与通勤场景，不夸大功能。",
    reference: "", provider: "qwen", region: "cn", model: "qwen-plus",
    parameters: { temperature: 0.7, top_p: 0.8, top_k: 20, max_tokens: 1024, thinking: "default" },
    output: "这款轻量通勤背包拥有 18L 容量，适合日常通勤与短途出行。面料具备防泼水能力，可应对日常溅水。简洁设计方便搭配，让通勤收纳更轻松。",
    jev_probability: 0.62,
    evaluation: { score: 68, strengths: ["语气自然，提到 18L 与通勤场景。"], issues: ["内容偏短，场景表达略泛。"], missing_requirements: ["未说明防泼水能力边界。"] },
    advice: "明确篇幅和功能边界，并要求逐项覆盖已知信息。",
    reason: "初始提示词未说明长度、口吻或禁止推断的内容。",
    next_prompt: "你是一名专业且亲切的中文产品文案撰写者。根据用户提供的事实写一段约 120 字的产品介绍。必须覆盖产品类型、容量、已知功能和适用场景；不得补充未提供的材质、性能或保证。对“防泼水”等功能只按输入描述表达，不夸大效果。保持简洁自然。",
    usage: {}, duration_ms: 1840,
  },
  {
    round_no: 2, status: "target_reached", prompt_mode: "user",
    prompt: "你是一名专业且亲切的中文产品文案撰写者。根据用户提供的事实写一段约 120 字的产品介绍。必须覆盖产品类型、容量、已知功能和适用场景；不得补充未提供的材质、性能或保证。对“防泼水”等功能只按输入描述表达，不夸大效果。保持简洁自然。",
    test_input: "轻量通勤背包，18L，防泼水，适合日常通勤。",
    expected: "用专业亲切的中文介绍，约 120 字；说明容量、防泼水与通勤场景，不夸大功能。",
    reference: "", provider: "qwen", region: "cn", model: "qwen-plus",
    parameters: { temperature: 0.7, top_p: 0.8, top_k: 20, max_tokens: 1024, thinking: "default" },
    output: "这款轻量通勤背包拥有 18L 容量，为日常通勤提供轻便实用的收纳选择。它具备防泼水功能，可应对日常少量水滴。简洁设计便于搭配，也适用于短途出行。",
    jev_probability: 0.88,
    evaluation: { score: 91, strengths: ["覆盖容量、防泼水和通勤场景，语气亲切。", "未将防泼水夸大为防水。"], issues: ["字数略短于约 120 字。"], missing_requirements: [] },
    advice: "把字数要求设成范围，可减少对近似字数的过度惩罚。",
    reason: "主要约束已达到，放宽篇幅表述会更稳定。",
    next_prompt: "",
    stop_reason: "Jev 与模型评分均达到 85% 演示阈值。",
    usage: {}, duration_ms: 1710,
  },
];

export const demoRun = {
  id: "demo-promptlattice",
  status: "target_reached",
  mode: "auto",
  initial_prompt: demoRounds[0].prompt,
  latest_prompt: demoRounds[1].prompt,
  stop_reason: "演示数据：第 2 轮 Jev 88%、模型评分 91%，首次同时达到 85% 阈值。",
  rounds: demoRounds,
  config: { threshold: 85, max_rounds: 15, auto_optimize: true },
};

/** @param {string} prompt @param {string} testInput @param {PromptMode} mode */
export function buildMessages(prompt, testInput, mode) {
  if (mode === "system") return [{ role: "system", content: prompt }, { role: "user", content: testInput }];
  const content = testInput.trim() ? `${prompt}\n\n---\n测试输入：\n${testInput}` : prompt;
  return [{ role: "user", content }];
}

/** Only replace the editor prompt. All other draft fields remain untouched. */
export function replacePromptOnly(draft, nextPrompt) {
  return { ...draft, prompt: nextPrompt };
}

/** Adopt a prompt without changing test settings; current-run adoption preserves that round's role. */
export function adoptRoundPrompt(draft, nextPrompt, round, runOrigin) {
  const next = replacePromptOnly(draft, nextPrompt);
  if (runOrigin === "current" || runOrigin === "demo") {
    next.promptMode = round?.prompt_mode || draft.promptMode;
  }
  return next;
}

/** Restore a complete test case from a saved round; secrets are intentionally absent. */
export function restoreRoundConfig(round) {
  return {
    prompt: round.prompt,
    promptMode: round.prompt_mode || "user",
    testInput: round.test_input || "",
    expected: round.expected || "",
    reference: round.reference || "",
    provider: round.provider,
    region: round.region || "default",
    model: round.model,
    params: { temperature: 0.7, top_p: 1, top_k: 40, max_tokens: 2048, thinking: "default", ...(round.parameters || {}) },
    threshold: round.threshold ?? 85,
    maxRounds: round.max_rounds ?? 15,
    mode: round.auto_optimize ? "auto" : "manual",
  };
}

/** Apply only provider parameters explicitly marked supported by the active capability table. */
export function supportedParameterPatch(suggestion, capabilities) {
  const patch = {};
  for (const key of ["temperature", "top_p", "top_k", "max_tokens"]) {
    const value = suggestion?.[key];
    if (capabilities?.parameters?.[key]?.supported && typeof value === "number" && Number.isFinite(value)) patch[key] = value;
  }
  if (capabilities?.thinking?.available && capabilities.thinking.choices.includes(suggestion?.thinking)) patch.thinking = suggestion.thinking;
  return patch;
}

function historyText(value, limit) {
  if (typeof value !== "string") return "";
  if (value.length <= limit) return value;
  const half = Math.floor((limit - 20) / 2);
  return `${value.slice(0, half)}\n…历史文本中间已省略…\n${value.slice(-half)}`;
}

function historyRound(round, sourceRunId) {
  const parameters = round?.parameters || {};
  const nextParameters = round?.next_parameters || {};
  const evaluation = round?.evaluation || {};
  const takeParameters = (source) => Object.fromEntries(
    ["temperature", "top_p", "top_k", "max_tokens", "thinking"]
      .filter((key) => source[key] == null || ["number", "string"].includes(typeof source[key]))
      .filter((key) => source[key] != null)
      .map((key) => [key, source[key]]),
  );
  const takeList = (value) => Array.isArray(value) ? value.slice(0, 8).map((entry) => historyText(entry, 240)) : [];
  return {
    source_run_id: round?.source_run_id || sourceRunId || "inherited",
    round_no: Number.isInteger(round?.round_no) && round.round_no > 0 ? round.round_no : 1,
    status: historyText(round?.status || "completed", 40),
    prompt: historyText(round?.prompt, 1800),
    prompt_mode: round?.prompt_mode === "system" ? "system" : "user",
    test_input: historyText(round?.test_input, 1200),
    expected: historyText(round?.expected, 1200),
    reference: historyText(round?.reference, 600),
    parameters: takeParameters(parameters),
    output: historyText(round?.output, 1400),
    jev_probability: typeof round?.jev_probability === "number" ? round.jev_probability : null,
    jev_checks: Object.fromEntries(["requirements_met", "format_and_length_met", "grounded_in_input", "style_and_tone_met"].map((key) => [key, typeof round?.jev_checks?.[key] === "number" ? round.jev_checks[key] : null])),
    evaluation: {
      score: Number.isFinite(evaluation.score) ? evaluation.score : null,
      criterion_scores: evaluation.criterion_scores && typeof evaluation.criterion_scores === "object" ? evaluation.criterion_scores : {},
      hard_constraint_failures: takeList(evaluation.hard_constraint_failures),
      strengths: takeList(evaluation.strengths),
      issues: takeList(evaluation.issues),
      missing_requirements: takeList(evaluation.missing_requirements),
    },
    advice: historyText(round?.advice, 500),
    reason: historyText(round?.reason, 500),
    next_prompt: historyText(round?.next_prompt, 1800),
    parameter_advice: historyText(round?.parameter_advice, 500),
    parameter_reason: historyText(round?.parameter_reason, 500),
    next_parameters: takeParameters(nextParameters),
  };
}

function sameScenario(actual, expected) {
  if (!actual || !expected) return false;
  return actual.provider === expected.provider
    && actual.model === expected.model
    && (actual.region || "default") === (expected.region || "default")
    && (actual.prompt_mode || actual.promptMode || "user") === (expected.prompt_mode || expected.promptMode || "user")
    && (actual.test_input ?? actual.testInput ?? "") === (expected.test_input ?? expected.testInput ?? "")
    && (actual.expected ?? "") === (expected.expected ?? "")
    && (actual.reference ?? "") === (expected.reference ?? "");
}

/**
 * Build bounded context for suggestion generation only. A prior run is reused
 * only when its test scenario/model match; the latest prompt and parameters may
 * differ because those are the variables being iterated.
 */
export function optimizationHistoryForRun(run, scenario) {
  if (!run || run.id === demoRun.id || !Array.isArray(run.rounds)) return [];
  const config = run.config || {};
  const identity = ["provider", "model", "region", "prompt_mode", "test_input", "expected", "reference"];
  const hasConfig = identity.some((key) => config[key] != null);
  const runScenario = hasConfig ? config : run.rounds[0];
  if (!sameScenario(runScenario, scenario)) return [];

  const collected = [];
  const seen = new Set();
  const add = (item, fallbackId) => {
    const compact = historyRound(item, fallbackId);
    const key = `${compact.source_run_id}:${compact.round_no}`;
    if (seen.has(key)) return;
    seen.add(key);
    if (compact.output || compact.jev_probability != null || compact.evaluation.score != null) collected.push(compact);
  };
  for (const item of Array.isArray(config.optimization_history) ? config.optimization_history : []) add(item, "inherited");
  for (const item of run.rounds) add(item, run.id);
  return collected.slice(-15);
}

const terminalStatuses = new Set(["target_reached", "max_rounds", "stopped", "repeated_prompt", "failed", "interrupted"]);
export function hasValidNextPrompt(round) {
  return Boolean(round?.next_prompt?.trim()) && !terminalStatuses.has(round?.status);
}

export function demoReachedFirstThreshold(rounds, threshold) {
  for (let i = 0; i < rounds.length; i++) {
    const round = rounds[i];
    if (round.jev_probability * 100 >= threshold && (round.evaluation?.score ?? -1) >= threshold) return i;
  }
  return -1;
}

/** Return a copy safe for export even if a future backend accidentally adds credentials. */
export function removeSecrets(value) {
  if (Array.isArray(value)) return value.map(removeSecrets);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !/api.?key|authorization|headers|target.?key|jev.?key/i.test(key))
    .map(([key, entry]) => [key, removeSecrets(entry)]));
}
