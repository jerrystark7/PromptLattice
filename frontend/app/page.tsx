"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
type Provider = "deepseek" | "qwen" | "moonshot" | "openai";
type Params = { temperature: number; top_p: number; top_k: number; max_tokens: number };
type Evaluation = { score: number; strengths: string[]; issues: string[]; missing_requirements: string[] };
type Usage = { input_tokens?: number | null; output_tokens?: number | null; total_tokens?: number | null };
type Round = { round_no: number; created_at?: string; status: string; prompt: string; test_input: string; expected: string; reference: string; provider: string; model: string; parameters?: Record<string, unknown>; output?: string; jev_probability?: number; evaluation?: Evaluation; advice?: string; reason?: string; next_prompt?: string; usage?: Record<string, Usage | Usage[] | null>; duration_ms?: number; stop_reason?: string; error?: { kind: string; message: string } };
type Run = { id: string; created_at?: string; status: string; mode?: string; initial_prompt: string; latest_prompt: string; stop_reason: string; rounds: Round[]; round_count?: number };
type Capabilities = { parameters: Record<string, { supported: boolean; reason: string }>; warnings: string[] };

const providers: Record<Provider, { label: string; regions: { id: string; name: string }[]; models: string[]; topK: boolean }> = {
  deepseek: { label: "DeepSeek", regions: [{ id: "default", name: "默认入口" }], models: ["deepseek-flash", "deepseek-v4-pro"], topK: false },
  qwen: { label: "通义千问 Qwen", regions: [{ id: "cn", name: "中国内地" }, { id: "international", name: "国际站" }], models: ["qwen-plus", "qwen-max", "qwen-turbo", "qwen3.5-plus"], topK: true },
  moonshot: { label: "Moonshot / Kimi", regions: [{ id: "cn", name: "中国内地" }, { id: "international", name: "国际站" }], models: ["kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k"], topK: false },
  openai: { label: "OpenAI GPT", regions: [{ id: "default", name: "默认入口" }], models: ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"], topK: false },
};
const demoRounds: Round[] = [
  { round_no: 1, status: "completed", prompt: "你是一名专业的产品文案。介绍以下产品。", test_input: "轻量通勤背包，18L，防泼水，适合日常通勤。", expected: "用专业亲切的中文介绍，约 120 字；说明容量、防泼水与通勤场景，不夸大功能。", reference: "", provider: "qwen", model: "qwen-plus", output: "这款轻量通勤背包拥有 18L 容量，适合日常通勤与短途出行。面料具备防泼水能力，可应对小雨和日常溅水。简洁设计方便搭配，让通勤收纳更轻松。", jev_probability: .62, evaluation: { score: 68, strengths: ["语气自然，提到 18L 与通勤场景。"], issues: ["内容偏短，场景表达略泛。"], missing_requirements: ["未说明防泼水能力边界。"] }, advice: "明确篇幅和功能边界，并要求逐项覆盖已知信息。", reason: "初始提示词未说明长度、口吻或禁止推断的内容。", next_prompt: "你是一名专业且亲切的中文产品文案撰写者。根据用户提供的事实写一段约 120 字的产品介绍。必须覆盖产品类型、容量、已知功能和适用场景；不得补充未提供的材质、性能或保证。对“防泼水”等功能只按输入描述表达，不夸大效果。保持简洁自然。" },
  { round_no: 2, status: "completed", prompt: "你是一名专业且亲切的中文产品文案撰写者。根据用户提供的事实写一段约 120 字的产品介绍。必须覆盖产品类型、容量、已知功能和适用场景；不得补充未提供的材质、性能或保证。对“防泼水”等功能只按输入描述表达，不夸大效果。保持简洁自然。", test_input: "轻量通勤背包，18L，防泼水，适合日常通勤。", expected: "用专业亲切的中文介绍，约 120 字；说明容量、防泼水与通勤场景，不夸大功能。", reference: "", provider: "qwen", model: "qwen-plus", output: "这款轻量通勤背包拥有 18L 容量，为日常通勤提供轻便实用的收纳选择。它具备防泼水功能，适合应对日常使用中的少量水滴。简洁的设计便于搭配，也适用于短途出行。产品信息以已知参数为准，让每次通勤都更从容。", jev_probability: .88, evaluation: { score: 91, strengths: ["覆盖容量、防泼水和通勤场景，语气亲切。", "未将防泼水夸大为防水。"], issues: ["字数略短于约 120 字。"], missing_requirements: [] }, advice: "把字数要求设成范围，可减少对近似字数的过度惩罚。", reason: "主要约束已达到，放宽篇幅表述会更稳定。", next_prompt: "你是一名专业、亲切的中文产品文案撰写者。根据用户输入的已知事实，写一段 100–130 字的产品介绍。必须自然涵盖产品类型、容量、已知功能及适用场景。不得添加未提供的材质、性能、认证或保证；“防泼水”不得写成“防水”。语言简洁可信，避免空泛口号。" },
  { round_no: 3, status: "target_reached", prompt: "你是一名专业、亲切的中文产品文案撰写者。根据用户输入的已知事实，写一段 100–130 字的产品介绍。必须自然涵盖产品类型、容量、已知功能及适用场景。不得添加未提供的材质、性能、认证或保证；“防泼水”不得写成“防水”。语言简洁可信，避免空泛口号。", test_input: "轻量通勤背包，18L，防泼水，适合日常通勤。", expected: "用专业亲切的中文介绍，约 120 字；说明容量、防泼水与通勤场景，不夸大功能。", reference: "", provider: "qwen", model: "qwen-plus", output: "这款轻量通勤背包拥有 18L 容量，为日常通勤带来轻便的收纳选择。它具备防泼水功能，可应对日常少量水滴，让通勤途中多一份安心。简洁利落的设计适合日常搭配，也能满足短途出行的收纳需要。每一处介绍都来自已知产品信息，不夸大功能，让选购判断更加清晰。", jev_probability: .94, evaluation: { score: 96, strengths: ["长度与语气符合要求，关键信息完整。", "说明防泼水边界，未添加额外规格。"], issues: [], missing_requirements: [] }, advice: "本轮两项评估均达到目标阈值。", reason: "已达到自动优化阈值。", next_prompt: "" },
];
const examplePrompt = "你是一名专业的产品文案。介绍以下产品。";
const exampleExpected = "用专业但亲切的中文介绍产品，约 120 字；说明已知卖点，不夸大未提供的功能。";
const exampleInput = "轻量通勤背包，18L，防泼水，适合日常通勤。";

function Icon({ children }: { children: React.ReactNode }) { return <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>; }
function tokenCount(round: Round) {
  const calls = Object.values(round.usage || {}).flatMap(value => Array.isArray(value) ? value : value ? [value] : []);
  if (!calls.length) return "未提供";
  let count = 0; let available = 0;
  for (const call of calls) { if (call.total_tokens != null) { count += call.total_tokens; available++; } else if (call.input_tokens != null || call.output_tokens != null) { count += (call.input_tokens || 0) + (call.output_tokens || 0); available++; } }
  if (!available) return "未提供";
  return `${count.toLocaleString()} tokens${available < calls.length ? " · 部分未提供" : ""}`;
}
function usageLabel(round: Round, provider: "model" | "jev") {
  const entries = Object.entries(round.usage || {}).filter(([key]) => provider === "jev" ? key === "jev" : key !== "jev").flatMap(([, value]) => Array.isArray(value) ? value : value ? [value] : []);
  let count = 0, available = 0;
  for (const item of entries) { if (item.total_tokens != null) { count += item.total_tokens; available++; } else if (item.input_tokens != null || item.output_tokens != null) { count += (item.input_tokens || 0) + (item.output_tokens || 0); available++; } }
  return available ? `${count} tokens${available < entries.length ? "*" : ""}` : "未提供";
}
function sentMessages(prompt: string, testInput: string, mode: string) { return mode === "system" ? [{ role: "system", content: prompt }, { role: "user", content: testInput }] : [{ role: "user", content: testInput.trim() ? `${prompt}\n\n---\n测试输入：\n${testInput}` : prompt }]; }
function roundTime(round: Round, demo: boolean) { return demo ? "演示数据" : round.created_at ? new Date(round.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "时间未提供"; }

export default function Home() {
  const [prompt, setPrompt] = useState(examplePrompt), [expected, setExpected] = useState(exampleExpected), [testInput, setTestInput] = useState(exampleInput), [reference, setReference] = useState("");
  const [promptMode, setPromptMode] = useState("user"), [provider, setProvider] = useState<Provider>("deepseek"), [region, setRegion] = useState("default"), [model, setModel] = useState("deepseek-flash");
  const [apiKey, setApiKey] = useState(""), [jevKey, setJevKey] = useState(""), [showKey, setShowKey] = useState(false), [showJevKey, setShowJevKey] = useState(false);
  const [params, setParams] = useState<Params>({ temperature: .7, top_p: 1, top_k: 40, max_tokens: 2048 }), [caps, setCaps] = useState<Capabilities | null>(null);
  const [threshold, setThreshold] = useState(85), [maxRounds, setMaxRounds] = useState(15), [mode, setMode] = useState<"manual" | "auto">("manual");
  const [status, setStatus] = useState("idle"), [stage, setStage] = useState(""), [error, setError] = useState(""), [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [run, setRun] = useState<Run | null>(null), [activeRound, setActiveRound] = useState(0), [history, setHistory] = useState<Run[]>([]), [demo, setDemo] = useState(false);
  const [showPreview, setShowPreview] = useState(true), [showCompare, setShowCompare] = useState(false), [editPrompt, setEditPrompt] = useState(false), [draftPrompt, setDraftPrompt] = useState("");
  const [keyExpanded, setKeyExpanded] = useState(true), [connectionMessage, setConnectionMessage] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const busy = status === "running", activeRoundData = run?.rounds[activeRound] || null, meta = providers[provider];
  const messages = useMemo(() => sentMessages(prompt, testInput, promptMode), [prompt, testInput, promptMode]);

  useEffect(() => { setRegion(providers[provider].regions[0].id); setModel(providers[provider].models[0]); }, [provider]);
  useEffect(() => { fetch(`${API}/api/capabilities?provider=${provider}&model=${encodeURIComponent(model)}&region=${region}`).then(r => r.ok ? r.json() : null).then(d => d && setCaps(d)).catch(() => setCaps(null)); }, [provider, model, region]);
  async function refreshHistory() { try { const res = await fetch(`${API}/api/runs`); if (res.ok) setHistory(await res.json()); } catch { /* backend may be offline */ } }
  useEffect(() => { refreshHistory(); }, []);

  function validate() {
    const next: Record<string, string> = {};
    if (!prompt.trim()) next.prompt = "请输入当前提示词。";
    if (!expected.trim()) next.expected = "请描述期望效果，作为本次评测依据。";
    if (promptMode === "system" && !testInput.trim()) next.testInput = "系统提示词模式需要填写测试输入。";
    if (!apiKey.trim()) next.apiKey = "请填写模型服务商 API Key。";
    if (!jevKey.trim()) next.jevKey = "请填写 Jev API Key。";
    setFieldErrors(next); return Object.keys(next).length === 0;
  }

  async function startRun(e?: FormEvent, automatic = mode === "auto") {
    e?.preventDefault(); if (busy || !validate()) return;
    setDemo(false); setError(""); setStage("正在连接服务"); setStatus("running"); setRun(null); setActiveRound(0); setFieldErrors({});
    const controller = new AbortController(); abortRef.current = controller;
    const payload = { prompt, prompt_mode: promptMode, expected, test_input: testInput, reference, model: { provider, model, region, ...params }, target_api_key: apiKey, jev_api_key: jevKey, auto_optimize: automatic, threshold, max_rounds: automatic ? maxRounds : 1 };
    let id = "", finalStatus = "";
    try {
      const response = await fetch(`${API}/api/runs/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
      if (!response.ok || !response.body) { const data = await response.json().catch(() => null); throw new Error(data?.detail?.message || data?.detail || "无法启动评测，请检查后端服务。"); }
      const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buffer += decoder.decode(value, { stream: true }); const blocks = buffer.split("\n\n"); buffer = blocks.pop() || "";
        for (const block of blocks) {
          const eventName = block.match(/^event: (.+)$/m)?.[1], raw = block.match(/^data: (.+)$/m)?.[1]; if (!eventName || !raw) continue;
          const data = JSON.parse(raw);
          if (eventName === "started") { id = data.run_id; setRun({ id, status: "running", mode: data.mode, initial_prompt: prompt, latest_prompt: prompt, stop_reason: "", rounds: [] }); }
          else if (eventName === "stage") setStage(data.message);
          else if (eventName === "round") setRun(prev => prev ? { ...prev, rounds: [...prev.rounds.filter(x => x.round_no !== data.round.round_no), data.round].sort((a,b) => a.round_no-b.round_no), latest_prompt: data.round.next_prompt || data.round.prompt } : prev);
          else if (eventName === "error") setError(data.message);
          else if (eventName === "finished") { finalStatus = data.status; setRun(prev => prev ? { ...prev, status: data.status, stop_reason: data.stop_reason } : prev); setStatus(data.status); setStage(data.stop_reason || "运行完成"); }
        }
      }
      if (id) { await refreshHistory(); const latest = await fetch(`${API}/api/runs/${id}`).then(r => r.ok ? r.json() : null).catch(() => null); if (latest) setRun(latest); }
      if (!finalStatus) setStatus("completed");
    } catch (err) { if ((err as Error).name === "AbortError") { setStatus("stopped"); setStage("已停止"); } else { setStatus("failed"); setError((err as Error).message || "请求失败"); } }
    finally { abortRef.current = null; }
  }

  async function stopRun() {
    if (!run?.id) { abortRef.current?.abort(); return; }
    setStage("正在停止；当前 API 请求结束后不会启动下一轮");
    try { await fetch(`${API}/api/runs/${run.id}/stop`, { method: "DELETE" }); } catch { /* backend checks cancellation between calls */ }
  }
  async function testConnection() {
    if (!apiKey.trim()) { setConnectionMessage("请先填写模型 API Key。"); return; }
    setConnectionMessage("正在发送一条简短测试请求…");
    try { const res = await fetch(`${API}/api/test-connection`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: { provider, model, region }, target_api_key: apiKey }) }); const data = await res.json(); if (!res.ok) throw new Error(data?.detail?.message || "连接失败，请检查 Key 和模型 ID。"); setConnectionMessage(`连接成功 · 用量 ${data.usage?.total_tokens ?? "未提供"} tokens`); }
    catch (err) { setConnectionMessage((err as Error).message); }
  }
  function loadDemo() {
    setDemo(true); setStatus("target_reached"); setError(""); setStage("演示数据 · 自动优化已完成"); setRun({ id: "demo-promptlattice", status: "target_reached", mode: "auto", initial_prompt: demoRounds[0].prompt, latest_prompt: demoRounds[2].prompt, stop_reason: "Jev 与模型评分均达到目标阈值（演示）", rounds: demoRounds });
    setPrompt(demoRounds[0].prompt); setExpected(demoRounds[0].expected); setTestInput(demoRounds[0].test_input); setProvider("qwen"); setRegion("cn"); setModel("qwen-plus"); setActiveRound(2); setMode("auto"); window.scrollTo({ top: 0, behavior: "smooth" });
  }
  async function openHistory(id: string) { try { const value = await fetch(`${API}/api/runs/${id}`).then(r => r.ok ? r.json() : null); if (value) { setDemo(false); setRun(value); setActiveRound(Math.max(0, value.rounds.length - 1)); setStatus(value.status); } } catch { /* no-op */ } }
  async function copy(value: string) { try { await navigator.clipboard.writeText(value); setStage("已复制到剪贴板"); setTimeout(() => setStage(""), 1500); } catch { setError("无法访问剪贴板，请手动选择并复制。"); } }
  function adopt(value: string) { if (!value) return; setPrompt(value); setPromptMode("user"); setEditPrompt(false); setShowCompare(false); window.scrollTo({ top: 0, behavior: "smooth" }); }
  function exportRun(format: "json" | "markdown") {
    if (!run) return;
    const body = format === "json" ? JSON.stringify(run, null, 2) : [`# PromptLattice 运行记录`, ``, `- 模式：${demo ? "演示数据" : run.mode === "auto" ? "自动优化" : "单轮评测"}`, `- 状态：${run.status}`, `- 停止原因：${run.stop_reason || "—"}`, ``, ...run.rounds.flatMap(r => [`## 第 ${r.round_no} 轮 · ${r.status}`, ``, `模型：${r.provider} / ${r.model}`, `Jev 达标判断概率：${r.jev_probability == null ? "未提供" : `${Math.round(r.jev_probability*100)}%`}`, `模型符合度：${r.evaluation?.score ?? "未提供"}/100`, `耗时：${r.duration_ms == null ? "未提供" : `${(r.duration_ms/1000).toFixed(1)} 秒`}`, `Token：${tokenCount(r)}`, ``, `### 提示词`, "```", r.prompt, "```", ``, `### 输出`, "```", r.output || "未生成", "```", ``, `### 建议`, r.advice || "—", ``, `### 新提示词`, "```", r.next_prompt || "—", "```", ``])].join("\n");
    const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([body], { type: format === "json" ? "application/json" : "text/markdown" })); link.download = `promptlattice-${run.id}.${format === "json" ? "json" : "md"}`; link.click(); URL.revokeObjectURL(link.href);
  }
  const current = activeRoundData, previous = activeRound > 0 ? run?.rounds[activeRound - 1] : null;
  const jevPct = current?.jev_probability == null ? null : Math.round(current.jev_probability * 100);

  return <main className="app-shell">
    <header className="topbar"><a className="brand" href="#top"><span className="brand-mark"><i/><i/><i/><i/></span><span>Prompt<span className="brand-light">Lattice</span></span></a><div className="top-context"><span className="context-dot"/>提示词评测工作台<span className="context-divider"/>本地会话</div><div className="top-actions"><button className="quiet-button demo-button" onClick={loadDemo}><Icon><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z"/></Icon>体验演示</button><span className="top-separator"/><span className="avatar">PL</span></div></header>
    <div className="page-wrap" id="top">
      <div className="page-heading"><div><div className="eyebrow">PROMPT ENGINEERING WORKSPACE</div><h1>评测与迭代</h1><p>验证提示词在真实模型上的表现，逐轮优化并保留每个版本。</p></div><div className="heading-meta"><span className="secure-chip"><Icon><path d="M5 10h14v10H5zM8 10V7a4 4 0 0 1 8 0v3"/></Icon>Key 仅用于当前请求</span><span>不会保存到浏览器或历史记录</span></div></div>

      <form className="workspace-grid" onSubmit={e => startRun(e)}>
        <section className="editor-column">
          <div className="panel editor-panel">
            <div className="panel-heading"><div className="section-title"><span className="step-number">01</span><div><h2>定义评测任务</h2><p>明确提示词、实际输入与预期效果</p></div></div><span className="required-note">* 必填</span></div>
            <label className="field-label" htmlFor="prompt">当前提示词 <span className="label-hint">{promptMode === "system" ? "SYSTEM ROLE" : "USER MESSAGE"}</span></label>
            <div className="mode-selector"><button type="button" className={`mode-option ${promptMode === "user" ? "selected" : ""}`} onClick={() => setPromptMode("user")}>作为用户消息 <span>默认</span></button><button type="button" className={`mode-option ${promptMode === "system" ? "selected" : ""}`} onClick={() => setPromptMode("system")}>作为系统提示词</button></div>
            <textarea id="prompt" className="code-input prompt-input" value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="写下要测试的提示词…" aria-invalid={!!fieldErrors.prompt}/>{fieldErrors.prompt && <div className="field-error">{fieldErrors.prompt}</div>}
            <label className="field-label spaced" htmlFor="expected">期望效果 <span className="required-star">*</span><span className="label-hint">仅用于评测，不会发送给目标模型生成</span></label>
            <textarea id="expected" className="text-input expected-input" value={expected} onChange={e => setExpected(e.target.value)} placeholder="描述语气、结构、长度和明确约束。" aria-invalid={!!fieldErrors.expected}/>{fieldErrors.expected && <div className="field-error">{fieldErrors.expected}</div>}
            <div className="two-fields"><div className="sub-field"><label className="field-label" htmlFor="test-input">测试输入 <span className="label-hint">可选</span></label><textarea id="test-input" className="text-input small-textarea" value={testInput} onChange={e => setTestInput(e.target.value)} placeholder="给提示词的真实输入…" aria-invalid={!!fieldErrors.testInput}/>{fieldErrors.testInput && <div className="field-error">{fieldErrors.testInput}</div>}</div><div className="sub-field"><label className="field-label" htmlFor="reference">参考答案 <span className="label-hint">评测用</span></label><textarea id="reference" className="text-input small-textarea" value={reference} onChange={e => setReference(e.target.value)} placeholder="可选，只作为评测依据；不发给生成模型。"/></div></div>
            {promptMode === "user" && testInput.trim() && <div className="message-preview"><div className="preview-title"><span className="preview-led"/>实际发送给模型 <span>user</span><button type="button" onClick={() => setShowPreview(!showPreview)}>{showPreview ? "收起" : "查看"}<Icon><path d="m6 9 6 6 6-6"/></Icon></button></div>{showPreview && <pre>{JSON.stringify(messages, null, 2)}</pre>}</div>}
            {promptMode === "system" && <div className="info-callout"><span className="info-icon">i</span><span>提示词会作为 <code>system</code> 消息发送，测试输入作为 <code>user</code> 消息。此模式需要填写测试输入。</span></div>}
          </div>

          <div className="panel config-panel">
            <div className="panel-heading"><div className="section-title"><span className="step-number">02</span><div><h2>选择真实模型</h2><p>生成、独立评分和提示词改写均使用此模型</p></div></div><span className="provider-count">4 家服务商</span></div>
            <div className="model-select-grid"><div className="sub-field"><label className="field-label" htmlFor="provider">模型服务商</label><select id="provider" className="select-input" value={provider} onChange={e => setProvider(e.target.value as Provider)}>{Object.entries(providers).map(([id, value]) => <option key={id} value={id}>{value.label}</option>)}</select></div>{meta.regions.length > 1 && <div className="sub-field"><label className="field-label" htmlFor="region">API 区域</label><select id="region" className="select-input" value={region} onChange={e => setRegion(e.target.value)}>{meta.regions.map(r => <option key={r.id} value={r.id}>{r.name}</option>)}</select></div>}<div className="sub-field model-id-field"><label className="field-label" htmlFor="model">模型 ID <span className="label-hint">可手动输入</span></label><input id="model" className="text-input single-line" list="model-list" value={model} onChange={e => setModel(e.target.value)}/><datalist id="model-list">{meta.models.map(m => <option key={m}>{m}</option>)}</datalist></div></div>
            <div className="accordion-title"><div><span>模型参数</span><small>不支持的参数不会发送</small></div><span className="param-availability">按服务商和模型 ID 判断</span></div>
            <div className="param-grid">{([ ["temperature","Temperature",0,2,.1], ["top_p","Top-p",0,1,.05], ["top_k","Top-k",1,100,1], ["max_tokens","最大输出 Token",1,32768,1] ] as [keyof Params,string,number,number,number][]).map(([key,label,min,max,step]) => { const supported = caps?.parameters[key]?.supported ?? (key === "top_k" ? meta.topK : true); return <label key={key} className={`param-control ${supported ? "" : "disabled"}`} title={!supported ? caps?.parameters[key]?.reason : undefined}><span className="param-name">{label}{!supported && <span className="unsupported-mark">不支持</span>}</span><div className="param-input-wrap"><input type="range" min={min} max={max} step={step} disabled={!supported} value={params[key]} onChange={e => setParams(p => ({...p,[key]:Number(e.target.value)}))}/><input className="param-number" type="number" min={min} max={max} step={step} disabled={!supported} value={params[key]} onChange={e => setParams(p => ({...p,[key]:Number(e.target.value)}))}/></div>{!supported && <small className="param-reason">{caps?.parameters[key]?.reason || "服务商接口不支持"}</small>}</label>; })}</div>
            {!!caps?.warnings.length && <div className="warning-note">{caps.warnings.join(" ")}</div>}
            <div className="key-accordion-head"><div><h3>API Key</h3><p>仅在当前页面内存中使用，服务重启后需重新输入。</p></div><button type="button" className="text-button" onClick={() => setKeyExpanded(!keyExpanded)}>{keyExpanded ? "收起" : "展开"}<Icon><path d="m6 9 6 6 6-6"/></Icon></button></div>
            {keyExpanded && <div className="key-fields"><div className="sub-field"><label className="field-label" htmlFor="model-key">{meta.label} API Key <span className="required-star">*</span></label><div className="secret-wrap"><input id="model-key" className="text-input secret-input" type={showKey ? "text" : "password"} autoComplete="off" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="粘贴 API Key" aria-invalid={!!fieldErrors.apiKey}/><button type="button" onClick={() => setShowKey(!showKey)}>{showKey ? "隐藏" : "显示"}</button></div>{fieldErrors.apiKey && <div className="field-error">{fieldErrors.apiKey}</div>}</div><div className="sub-field"><label className="field-label" htmlFor="jev-key">Jev API Key <span className="required-star">*</span></label><div className="secret-wrap"><input id="jev-key" className="text-input secret-input" type={showJevKey ? "text" : "password"} autoComplete="off" value={jevKey} onChange={e => setJevKey(e.target.value)} placeholder="TypeSafe Jev API Key" aria-invalid={!!fieldErrors.jevKey}/><button type="button" onClick={() => setShowJevKey(!showJevKey)}>{showJevKey ? "隐藏" : "显示"}</button></div>{fieldErrors.jevKey && <div className="field-error">{fieldErrors.jevKey}</div>}</div><div className="key-note"><Icon><path d="M5 10h14v10H5zM8 10V7a4 4 0 0 1 8 0v3"/></Icon><span>Key 通过 POST 传给本地后端，不进入 URL、SQLite、日志或导出。运行期间，相关内容会按流程发送给所选模型服务商和 Jev。</span></div><button type="button" className="outline-button test-connection" disabled={busy} onClick={testConnection}>测试模型连接 <span>· 可能产生少量 API 费用</span></button>{connectionMessage && <div className="connection-message">{connectionMessage}</div>}</div>}
          </div>
        </section>

        <aside className="run-column"><div className="panel run-panel"><div className="panel-heading"><div className="section-title"><span className="step-number">03</span><div><h2>运行设置</h2><p>选择手动评测或自动优化</p></div></div></div>
          <div className="mode-tabs"><button type="button" className={mode === "manual" ? "selected" : ""} onClick={() => setMode("manual")}><Icon><path d="m8 5 11 7-11 7z" fill="currentColor"/></Icon>单轮评测</button><button type="button" className={mode === "auto" ? "selected" : ""} onClick={() => setMode("auto")}><Icon><path d="M4 4h7v7H4zM14 4h6v6h-6zM4 14h6v6H4zM14 13h6v7h-6z"/></Icon>自动优化</button></div>
          {mode === "auto" && <div className="auto-settings"><div className="auto-description">自动生成、评估并改写提示词，达到目标后停止。</div><div className="auto-inputs"><label>达标阈值 <span><input type="number" min="0" max="100" value={threshold} onChange={e => setThreshold(Math.max(0,Math.min(100,Number(e.target.value))))}/> %</span></label><label>最大轮数 <span><input type="number" min="1" max="15" value={maxRounds} onChange={e => setMaxRounds(Math.max(1,Math.min(15,Number(e.target.value))))}/> 轮</span></label></div><div className="cost-warning"><span>!</span><div><strong>自动优化会发起多次 API 调用</strong><p>最大轮数包含第 1 轮。每轮至少调用目标模型与 Jev，实际费用由对应服务商收取。</p></div></div></div>}
          <div className="run-summary"><div className="summary-row"><span>模型</span><strong>{meta.label} <i>/</i> {model || "输入模型 ID"}</strong></div><div className="summary-row"><span>提示词角色</span><strong>{promptMode === "system" ? "System + User" : "User message"}</strong></div><div className="summary-row"><span>期望效果</span><strong>仅用于评测</strong></div></div>
          <div className="run-buttons">{busy ? <button type="button" className="stop-button" onClick={stopRun}><Icon><rect x="6" y="6" width="12" height="12" fill="currentColor"/></Icon>停止运行</button> : <button type="submit" className="primary-button"><Icon><path d="m8 5 11 7-11 7z" fill="currentColor"/></Icon>{mode === "auto" ? "开始自动优化" : "运行评测"}</button>}</div>
          <div className="run-stage" aria-live="polite">{busy && <span className="spinner"/>}<span>{stage || (status === "idle" ? "准备就绪" : status === "failed" ? "运行失败" : status === "stopped" ? "已停止" : "运行完成")}</span>{status === "target_reached" && <span className="stage-success">已达到目标</span>}</div>
          {error && <div className="error-banner"><strong>本轮未完成</strong><span>{error}</span></div>}{!error && Object.keys(fieldErrors).length > 0 && <div className="error-banner"><strong>请完善必填项</strong><span>根据输入框下方提示补全后再运行。</span></div>}
          <div className="run-privacy">API Key 仅随请求传递。刷新页面后需要重新输入。</div>
        </div>
        <div className="panel history-panel"><div className="history-title"><div><span className="history-icon"><Icon><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></Icon></span><div><h3>运行记录</h3><p>最近的真实评测</p></div></div><span className="history-count">{history.length}</span></div>{!history.length ? <div className="history-empty">完成评测后，记录会保存到本地 SQLite。</div> : <div className="history-list">{history.slice(0,5).map(item => <button key={item.id} className={`history-item ${run?.id === item.id ? "active" : ""}`} onClick={() => openHistory(item.id)}><span className={`history-status ${item.status === "failed" ? "failed" : item.status === "target_reached" ? "success" : ""}`}/><span className="history-item-main"><strong>{item.initial_prompt.slice(0,42) || "提示词评测"}</strong><small>{new Date(item.created_at || "").toLocaleString()} · {item.round_count} 轮</small></span><span className="history-open">›</span></button>)}</div>}</div></aside>
      </form>

      <section className="results-section"><div className="results-heading"><div><div className="results-title-row"><span className="step-number">04</span><h2>评测结果</h2>{demo && <span className="demo-tag">演示数据</span>}{run && !demo && <span className="result-run-id">RUN / {run.id.slice(0,8).toUpperCase()}</span>}</div><p>{run ? `共 ${run.rounds.length} 轮 · ${run.status === "target_reached" ? "已达到目标" : run.status === "failed" ? "运行失败" : run.status === "stopped" ? "已停止" : run.status === "max_rounds" ? "达到最大轮数" : "评测完成"}${run.stop_reason ? ` · ${run.stop_reason}` : ""}` : "运行后在此查看生成内容、评估结果与提示词建议。"}</p></div>{run && <div className="result-actions"><button className="outline-button" onClick={() => exportRun("json")}>导出 JSON</button><button className="outline-button" onClick={() => exportRun("markdown")}>导出 Markdown</button></div>}</div>
        {run?.rounds.length ? <div className="result-layout"><aside className="timeline panel"><div className="timeline-heading"><div><h3>迭代时间线</h3><p>选择版本查看本轮详情</p></div><span>{run.rounds.length} 轮</span></div><div className="timeline-items">{run.rounds.map((r,i) => <button key={r.round_no} className={`timeline-item ${activeRound === i ? "active" : ""}`} onClick={() => { setActiveRound(i); setShowCompare(false); }}><span className={`timeline-marker ${r.status === "target_reached" ? "reached" : r.status === "failed" ? "failed" : ""}`}>{r.status === "target_reached" ? "✓" : r.round_no}</span><span className="timeline-content"><strong>第 {r.round_no} 轮 {r.status === "target_reached" && <em>已达标</em>}{r.status === "failed" && <em className="bad">失败</em>}</strong><small>{roundTime(r,demo)} · {r.model} · {r.duration_ms == null ? "耗时未提供" : `${(r.duration_ms/1000).toFixed(1)}s`}</small></span>{r.evaluation && <span className="timeline-score">{r.evaluation.score}</span>}</button>)}</div><div className="timeline-bottom"><span className="timeline-line"/>初始提示词<button onClick={() => copy(run.initial_prompt)}>复制</button></div></aside>
          {current && <div className="result-detail"><div className="round-toolbar"><div><span className="round-index">ROUND {String(current.round_no).padStart(2,"0")}</span><span className="round-model">{providers[current.provider as Provider]?.label || current.provider} · {current.model}</span><button className="text-button" onClick={() => copy(current.prompt)}>复制本轮提示词</button></div><div><span>{current.duration_ms == null ? "耗时未提供" : `${(current.duration_ms/1000).toFixed(1)} 秒`}</span><span className="toolbar-separator">·</span><span>模型 {usageLabel(current,"model")} · Jev {usageLabel(current,"jev")}</span></div></div>
            <div className="score-grid"><div className="score-card jev-score"><div className="score-card-title"><span>Jev 达标判断概率</span><span className="score-source">JEV NOUL</span></div><div className="score-number">{jevPct == null ? "—" : <>{jevPct}<small>%</small></>}</div><div className="score-meter"><i style={{width:`${jevPct ?? 0}%`}}/></div><p>Jev 对本次输出的判断，不是未来成功的统计概率。</p></div><div className="score-card model-score"><div className="score-card-title"><span>模型符合度评分</span><span className="score-source">{current.evaluation ? "MODEL REVIEW" : "未完成"}</span></div><div className="score-number">{current.evaluation?.score ?? "—"}<small>{current.evaluation ? "/ 100" : ""}</small></div><div className="score-meter"><i style={{width:`${current.evaluation?.score ?? 0}%`}}/></div><p>{previous?.evaluation ? `较上一轮 ${current.evaluation!.score - previous.evaluation.score > 0 ? "+" : ""}${current.evaluation!.score - previous.evaluation.score} 分` : "模型独立评分，不以 Jev 分数为依据。"}</p></div></div>
            <div className="result-tabs"><button className="active">生成结果</button><button onClick={() => document.getElementById("evaluation-detail")?.scrollIntoView({behavior:"smooth",block:"center"})}>评估详情</button><button onClick={() => document.getElementById("recommendation")?.scrollIntoView({behavior:"smooth",block:"center"})}>提示词建议</button><div className="tab-spacer"/><button className="compare-trigger" onClick={() => setShowCompare(!showCompare)}>{showCompare ? "关闭对比" : "对比初版"}</button></div>
            {showCompare && <div className="compare-box"><div><span>初始版本 · 第 1 轮</span><pre>{run.rounds[0].prompt}</pre><small>模型 {run.rounds[0].evaluation?.score ?? "—"} · Jev {run.rounds[0].jev_probability == null ? "—" : `${Math.round(run.rounds[0].jev_probability*100)}%`}</small><pre className="compare-output">{run.rounds[0].output || "未提供输出"}</pre></div><div><span>当前版本 · 第 {current.round_no} 轮</span><pre>{current.prompt}</pre><small>模型 {current.evaluation?.score ?? "—"} · Jev {jevPct == null ? "—" : `${jevPct}%`}</small><pre className="compare-output">{current.output || "未提供输出"}</pre></div></div>}
            <div className="output-card"><div className="content-section-heading"><div><span className="content-dot"/><h3>实际生成内容</h3><small>目标模型输出</small></div>{current.output && <button className="icon-text-button" onClick={() => copy(current.output || "")}><Icon><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></Icon>复制</button>}</div>{current.output ? <pre className="output-text">{current.output}</pre> : <div className="empty-output">{current.error?.message || "该轮没有生成输出。"}</div>}</div>
            <div id="evaluation-detail" className="evaluation-panel panel"><div className="content-section-heading"><div><span className="evaluation-icon">✓</span><h3>模型独立评估</h3><small>不使用 Jev 概率作为评分依据</small></div></div>{current.evaluation ? <div className="evaluation-columns"><div><h4>做得好的地方</h4>{current.evaluation.strengths.length ? <ul className="good-list">{current.evaluation.strengths.map((x,i)=><li key={i}>{x}</li>)}</ul> : <p className="muted-copy">无</p>}</div><div><h4>问题与遗漏</h4>{current.evaluation.issues.length || current.evaluation.missing_requirements.length ? <ul className="issue-list">{[...current.evaluation.issues,...current.evaluation.missing_requirements].map((x,i)=><li key={i}>{x}</li>)}</ul> : <p className="good-empty">未发现明确问题</p>}</div></div> : <div className="empty-output">{current.error?.message || "评估结果尚不可用。"}</div>}</div>
            <div id="recommendation" className="recommendation-panel"><div className="content-section-heading"><div><span className="recommendation-icon">↗</span><h3>优化建议</h3><small>由选择的目标模型生成</small></div>{current.next_prompt && <div className="recommendation-actions"><button className="icon-text-button" onClick={() => copy(current.next_prompt || "")}><Icon><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></Icon>复制</button><button className="outline-button adopt-button" onClick={() => adopt(current.next_prompt || "")}>采用此版本</button></div>}</div>{current.advice ? <><p className="advice-text">{current.advice}</p><div className="reason-box"><strong>修改原因</strong><span>{current.reason}</span></div><div className="prompt-version"><div className="version-heading"><div><span>推荐的新提示词</span><small>完整版本 · 第 {current.round_no+1} 版</small></div><button className="edit-prompt-button" onClick={() => {setDraftPrompt(current.next_prompt || "");setEditPrompt(true);}}>编辑后采用</button></div>{editPrompt ? <><textarea className="code-input recommended-input" value={draftPrompt} onChange={e => setDraftPrompt(e.target.value)}/><div className="edit-actions"><button className="text-button" onClick={() => setEditPrompt(false)}>取消</button><button className="primary-small" onClick={() => adopt(draftPrompt)}>采用编辑版本</button></div></> : <pre className="recommended-text">{current.next_prompt}</pre>}</div></> : current.status === "target_reached" ? <div className="good-empty">本轮已达到设定阈值，无需继续优化。</div> : <div className="empty-output">{current.error?.message || "本轮优化建议未生成。"}</div>}</div>
            <div className="disclaimer"><span>i</span><p>模型输出和评估都可能有误，结果仅供调试参考。Jev 概率针对本次输出，不代表提示词未来生成成功的真实统计概率。Token 用量以 API 返回值为准。</p></div>
          </div>}
        </div> : <div className="empty-state panel"><div className="empty-illustration"><div className="empty-lattice">{Array.from({length:9},(_,i)=><i key={i}/>)}</div></div><h3>准备好开始第一轮评测</h3><p>填写模型 Key 和 Jev Key 后运行。也可以先查看预置演示，了解评测结果和自动优化时间线。</p><button className="quiet-button" onClick={loadDemo}>体验演示数据</button></div>}
      </section>
      <footer className="footer"><span>PromptLattice <i>v1.0</i></span><span>本地优先 · 无账号 · 不保存 API Key</span><a href="https://github.com/jerrystark7/PromptLattice" target="_blank" rel="noreferrer">GitHub ↗</a></footer>
    </div>
  </main>;
}
