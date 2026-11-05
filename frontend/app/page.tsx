"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { adoptRoundPrompt, buildMessages, demoRounds, demoRun, hasValidNextPrompt, optimizationHistoryForRun, removeSecrets, restoreRoundConfig, supportedParameterPatch } from "@/lib/workflow.mjs";

const API = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
type Provider = "deepseek" | "qwen" | "moonshot" | "openai";
type PromptMode = "user" | "system";
type ThemePreference = "system" | "light" | "dark";
type RunStatus = "idle" | "running" | "stopping" | "completed" | "target_reached" | "max_rounds" | "stopped" | "repeated_prompt" | "failed" | "interrupted";
type Params = { temperature: number; top_p: number; top_k: number; max_tokens: number; thinking: string };
type Evaluation = { score: number; criterion_scores?: { requirements: number; factuality: number; format: number; style: number }; criterion_weights?: { requirements: number; factuality: number; format: number; style: number }; hard_constraint_failures?: string[]; strengths: string[]; issues: string[]; missing_requirements: string[] };
type Usage = { input_tokens?: number | null; output_tokens?: number | null; total_tokens?: number | null };
type Round = { round_no: number; created_at?: string; status: string; prompt: string; prompt_mode?: PromptMode; test_input: string; expected: string; reference: string; provider: string; region?: string; model: string; parameters?: Partial<Params>; parameter_support?: Record<string, Capability>; thinking_support?: { available: boolean; selected: string; choices: string[] }; output?: string; jev_probability?: number; jev_checks?: Record<string, number | null>; evaluation?: Evaluation; advice?: string; reason?: string; next_prompt?: string; parameter_advice?: string; parameter_reason?: string; next_parameters?: Partial<Params>; parameter_warnings?: string[]; optimization_history_rounds?: number; usage?: Record<string, Usage | Usage[] | null>; duration_ms?: number; stop_reason?: string; error?: { kind: string; message: string }; threshold?: number; max_rounds?: number; auto_optimize?: boolean };
type Run = { id: string; created_at?: string; status: string; mode?: string; initial_prompt: string; latest_prompt: string; stop_reason: string; rounds: Round[]; round_count?: number; config?: Record<string, any> };
type Capability = { supported: boolean; reason: string; api_parameter?: string; warning?: string; minimum?: number | null; maximum?: number | null };
type Capabilities = { parameters: Record<keyof Omit<Params, "thinking">, Capability>; warnings: string[]; thinking: { available: boolean; selected: string; choices: string[] } };
type ProviderCatalog = Record<string, Record<string, { label: string; base_url: string; models: string[] }>>;
type Draft = { prompt: string; promptMode: PromptMode; expected: string; testInput: string; reference: string; provider: Provider; region: string; model: string; params: Params; threshold: number; maxRounds: number; mode: "manual" | "auto"; run: Run | null; activeRound: number; status: RunStatus; stage: string; activeTab: ResultTab; runOrigin: "current" | "history" | "demo" };
type ResultTab = "output" | "evaluation" | "recommendation";

const providerNames: Record<Provider, string> = { deepseek: "DeepSeek", qwen: "通义千问 Qwen", moonshot: "Moonshot / Kimi", openai: "OpenAI GPT" };
const parameterLabels = { temperature: "Temperature", top_p: "Top-p", top_k: "Top-k", max_tokens: "最大输出 Token", thinking: "推理模式" } as const;
const jevCheckLabels: Record<string, string> = { requirements_met: "内容要求", format_and_length_met: "格式与篇幅", grounded_in_input: "事实有据", style_and_tone_met: "语气与风格" };
const evaluationCriterionLabels = { requirements: "要求符合", factuality: "事实有据", format: "格式与篇幅", style: "语气与清晰度" } as const;
const initialCatalog: ProviderCatalog = {
  deepseek: { default: { label: "默认入口", base_url: "https://api.deepseek.com", models: ["deepseek-flash", "deepseek-v4-pro"] } },
  qwen: {
    cn: { label: "中国内地", base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1", models: ["qwen3.8-max", "qwen3.8-flash", "qwen-plus", "qwen-max"] },
    international: { label: "国际站", base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", models: ["qwen3.8-max", "qwen3.8-flash", "qwen-plus", "qwen-max"] },
  },
  moonshot: {
    cn: { label: "中国内地", base_url: "https://api.moonshot.cn/v1", models: ["kimi-k3.1-pro", "kimi-k3", "kimi-k2.6", "kimi-k2.5"] },
    international: { label: "国际站", base_url: "https://api.moonshot.ai/v1", models: ["kimi-k3.1-pro", "kimi-k3", "kimi-k2.6", "kimi-k2.5"] },
  },
  openai: { default: { label: "默认入口", base_url: "https://api.openai.com/v1", models: ["gpt-4.1-mini", "gpt-4.1", "gpt-5.4"] } },
};
const defaults: Params = { temperature: 0.7, top_p: 1, top_k: 40, max_tokens: 2048, thinking: "default" };
const THEME_KEY = "promptlattice.theme";

function statusText(status: string) {
  const labels: Record<string, string> = {
    idle: "准备就绪", running: "运行中", stopping: "正在停止", completed: "已完成", target_reached: "已达到目标",
    max_rounds: "达到最大轮数", stopped: "已停止", repeated_prompt: "重复提示词，已停止",
    failed: "运行失败", interrupted: "运行中断",
  };
  return labels[status] || status;
}

function elapsed(round: Round) { return round.duration_ms == null ? "耗时未提供" : `${(round.duration_ms / 1000).toFixed(1)} 秒`; }

function tokenTotal(round: Round, only?: "model" | "jev") {
  const entries = Object.entries(round.usage || {})
    .filter(([key]) => only === "jev" ? key === "jev" : only === "model" ? key !== "jev" : true)
    .flatMap(([, value]) => Array.isArray(value) ? value : value ? [value] : []);
  let total = 0, count = 0;
  for (const usage of entries) {
    if (usage.total_tokens != null) { total += usage.total_tokens; count++; }
    else if (usage.input_tokens != null || usage.output_tokens != null) { total += (usage.input_tokens || 0) + (usage.output_tokens || 0); count++; }
  }
  return count ? `${total.toLocaleString()} tokens${count < entries.length ? " · 部分未提供" : ""}` : "未提供";
}

function roundDate(round: Round, isDemo: boolean) {
  if (isDemo) return "演示数据";
  if (!round.created_at) return "时间未提供";
  const date = new Date(round.created_at);
  return Number.isNaN(date.valueOf()) ? "时间未提供" : date.toLocaleString([], { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function Icon({ kind }: { kind: "logo" | "settings" | "play" | "stop" | "copy" | "arrow" | "spark" | "close" | "check" | "external" }) {
  const shapes: Record<string, React.ReactNode> = {
    logo: <><rect x="4.5" y="4.5" width="6.2" height="6.2" rx="1.5" fill="currentColor" stroke="none"/><rect x="13.3" y="4.5" width="6.2" height="6.2" rx="1.5" fill="currentColor" stroke="none"/><rect x="4.5" y="13.3" width="6.2" height="6.2" rx="1.5" fill="currentColor" stroke="none"/><rect x="13.3" y="13.3" width="6.2" height="6.2" rx="1.5" fill="currentColor" stroke="none"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1-1.8 1.8-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5v.2h-2.6v-.2a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1-1.8-1.8.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H6.5v-2.6h.2a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1 1.8-1.8.1.1a1.7 1.7 0 0 0 1.8.3 1.7 1.7 0 0 0 1-1.5v-.2h2.6v.2a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1 1.8 1.8-.1.1a1.7 1.7 0 0 0-.3 1.8 1.7 1.7 0 0 0 1.5 1h.2V14h-.2a1.7 1.7 0 0 0-1.5 1Z"/></>,
    play: <path d="m8 5 11 7-11 7z" fill="currentColor" stroke="none"/>,
    stop: <rect x="6" y="6" width="12" height="12" rx="1" fill="currentColor" stroke="none"/>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></>,
    arrow: <path d="m6 9 6 6 6-6"/>, spark: <><path d="m12 3 1.6 5.4L19 10l-5.4 1.6L12 17l-1.6-5.4L5 10l5.4-1.6L12 3Z"/><path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/></>, close: <path d="m6 6 12 12M18 6 6 18"/>, check: <path d="m5 12 4 4L19 6"/>, external: <><path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5"/></>,
  };
  return <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{shapes[kind]}</svg>;
}

function ThemeSettings() {
  const [open, setOpen] = useState(false);
  const [preference, setPreference] = useState<ThemePreference>(() => {
    if (typeof window === "undefined") return "system";
    const stored = window.localStorage.getItem(THEME_KEY);
    return stored === "light" || stored === "dark" || stored === "system" ? stored : "system";
  });
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = preference === "system" ? (media.matches ? "dark" : "light") : preference;
    };
    apply();
    media.addEventListener?.("change", apply);
    return () => media.removeEventListener?.("change", apply);
  }, [preference]);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: PointerEvent) => { if (!rootRef.current?.contains(event.target as Node)) setOpen(false); };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); triggerRef.current?.focus(); }
      if ((event.key === "ArrowDown" || event.key === "ArrowUp") && menuRef.current) {
        event.preventDefault();
        const items = [...menuRef.current.querySelectorAll<HTMLButtonElement>("button[role=menuitemradio]")];
        const index = items.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === "ArrowDown" ? (index + 1) % items.length : (index - 1 + items.length) % items.length;
        items[next]?.focus();
      }
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    menuRef.current?.querySelector<HTMLButtonElement>("[aria-checked=true]")?.focus();
    return () => { document.removeEventListener("pointerdown", onPointer); document.removeEventListener("keydown", onKey); };
  }, [open]);

  function selectTheme(value: ThemePreference) {
    setPreference(value);
    window.localStorage.setItem(THEME_KEY, value);
  }

  return <div className="settings-anchor" ref={rootRef}>
    {open && <div className="settings-menu" role="menu" aria-label="外观设置" ref={menuRef}>
      <div className="settings-menu-title">外观设置</div>
      {([ ["light", "浅色模式"], ["dark", "深色模式"], ["system", "跟随系统"] ] as [ThemePreference, string][]).map(([value, label]) => <button key={value} type="button" role="menuitemradio" aria-checked={preference === value} className={`theme-choice ${preference === value ? "selected" : ""}`} onClick={() => selectTheme(value)}>
        <span className={`theme-swatch ${value}`} aria-hidden="true"/><span>{label}</span>{preference === value && <span className="theme-check">✓</span>}
      </button>)}
      <div className="settings-menu-divider"/>
      <button type="button" role="menuitem" className="theme-choice close-menu" onClick={() => { setOpen(false); triggerRef.current?.focus(); }}>关闭菜单 <kbd>Esc</kbd></button>
    </div>}
    <button ref={triggerRef} type="button" className="settings-trigger" aria-haspopup="menu" aria-expanded={open} aria-label="打开 PromptLattice 设置" onClick={() => setOpen(value => !value)}>
      <Icon kind="settings"/><span>设置</span>
    </button>
  </div>;
}

export default function Home() {
  const [catalog, setCatalog] = useState<ProviderCatalog>(initialCatalog);
  const [prompt, setPrompt] = useState("你是一名专业的产品文案。介绍以下产品。"), [expected, setExpected] = useState("用专业但亲切的中文介绍产品，约 120 字；说明已知卖点，不夸大未提供的功能。"), [testInput, setTestInput] = useState("轻量通勤背包，18L，防泼水，适合日常通勤。"), [reference, setReference] = useState("");
  const [promptMode, setPromptMode] = useState<PromptMode>("user");
  const [provider, setProvider] = useState<Provider>("deepseek"), [region, setRegion] = useState("default"), [model, setModel] = useState("deepseek-flash");
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({}), [jevKey, setJevKey] = useState(""), [showKey, setShowKey] = useState(false), [showJevKey, setShowJevKey] = useState(false);
  const [params, setParams] = useState<Params>({ ...defaults, thinking: "enabled" }), [caps, setCaps] = useState<Capabilities | null>(null), [capsLoading, setCapsLoading] = useState(false);
  const [mode, setMode] = useState<"manual" | "auto">("manual"), [threshold, setThreshold] = useState(85), [maxRounds, setMaxRounds] = useState(15);
  const [status, setStatus] = useState<RunStatus>("idle"), [stage, setStage] = useState("准备就绪"), [error, setError] = useState(""), [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [run, setRun] = useState<Run | null>(null), [activeRound, setActiveRound] = useState(0), [history, setHistory] = useState<Run[]>([]), [runOrigin, setRunOrigin] = useState<Draft["runOrigin"]>("current");
  const [activeTab, setActiveTab] = useState<ResultTab>("output"), [showCompare, setShowCompare] = useState(false), [showModelConfig, setShowModelConfig] = useState(true), [showKeys, setShowKeys] = useState(true);
  const [testState, setTestState] = useState<{ status: "idle" | "testing" | "success" | "error"; message: string }>({ status: "idle", message: "" });
  const [notice, setNotice] = useState("");
  const [isDemo, setIsDemo] = useState(false), [demoConfirm, setDemoConfirm] = useState(false), [pendingDemo, setPendingDemo] = useState(false), [isStopping, setIsStopping] = useState(false);
  const [editingPrompt, setEditingPrompt] = useState(false), [editedPrompt, setEditedPrompt] = useState("");
  const [showFullConfig, setShowFullConfig] = useState(false);
  const [snapshot, setSnapshot] = useState<Draft | null>(null);
  const [exportMenu, setExportMenu] = useState(false);
  const abortRef = useRef<AbortController | null>(null), runIdRef = useRef(""), stopWhenStarted = useRef(false), runRef = useRef<Run | null>(null), snapshotRef = useRef<Draft | null>(null);

  const providerKeys = Object.keys(catalog) as Provider[];
  const currentRegions = catalog[provider] || {};
  const currentRegion = currentRegions[region] || Object.values(currentRegions)[0];
  const currentApiKey = apiKeys[`${provider}:${region}`] || "";
  const activeRoundData = run?.rounds[activeRound] || null;
  const previousRound = activeRound > 0 ? run?.rounds[activeRound - 1] || null : null;
  const busy = status === "running" || status === "stopping";
  const messages = useMemo(() => buildMessages(prompt, testInput, promptMode), [prompt, testInput, promptMode]);
  const roundCount = run?.rounds.length || 0;
  const isHistorical = runOrigin === "history";
  const jevPercent = activeRoundData?.jev_probability == null ? null : Math.round(activeRoundData.jev_probability * 100);
  const latestVisiblePrompt = activeRoundData?.next_prompt?.trim() && hasValidNextPrompt(activeRoundData) ? activeRoundData.next_prompt : activeRoundData?.prompt || prompt;

  function updateRun(next: Run | null | ((previous: Run | null) => Run | null)) {
    const value = typeof next === "function" ? next(runRef.current) : next;
    runRef.current = value;
    setRun(value);
  }

  function draftSnapshot(): Draft {
    return { prompt, promptMode, expected, testInput, reference, provider, region, model, params: { ...params }, threshold, maxRounds, mode, run: runRef.current, activeRound, status, stage, activeTab, runOrigin };
  }

  function showNotice(message: string) {
    setNotice(message);
    window.setTimeout(() => setNotice(""), 2200);
  }

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/api/providers`).then(response => response.ok ? response.json() : null).then(data => {
      if (!cancelled && data && typeof data === "object") setCatalog(data as ProviderCatalog);
    }).catch(() => undefined);
    fetch(`${API}/api/runs/reconcile`, { method: "POST" }).catch(() => undefined).finally(() => { if (!cancelled) refreshHistory(); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!history.some(item => item.status === "running" || item.status === "stopping")) return;
    const timer = window.setInterval(async () => {
      const items = await refreshHistory();
      const selected = runRef.current;
      if (runOrigin !== "history" || !selected) return;
      const latest = items.find(item => item.id === selected.id);
      if (!latest || latest.status === "running" || latest.status === "stopping") return;
      const response = await fetch(`${API}/api/runs/${selected.id}`).catch(() => null);
      if (!response?.ok) return;
      const detail = await response.json() as Run;
      updateRun(detail); setStatus(detail.status as RunStatus); setIsStopping(false); setStage(detail.stop_reason || statusText(detail.status));
      setActiveRound(Math.max(0, detail.rounds.length - 1));
    }, 2200);
    return () => window.clearInterval(timer);
    // refreshHistory and updateRun intentionally use the current selected run ref.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history, runOrigin]);

  useEffect(() => {
    const controller = new AbortController();
    setCaps(null); setCapsLoading(true);
    const query = new URLSearchParams({ provider, model, region, thinking: params.thinking });
    fetch(`${API}/api/capabilities?${query}`, { signal: controller.signal }).then(response => response.ok ? response.json() : null).then(data => {
      if (data) setCaps(data as Capabilities);
    }).catch(() => undefined).finally(() => { if (!controller.signal.aborted) setCapsLoading(false); });
    return () => controller.abort();
  }, [provider, model, region, params.thinking]);

  useEffect(() => {
    if (!caps) return;
    setParams(previous => {
      const next = { ...previous };
      for (const key of ["temperature", "top_p", "top_k", "max_tokens"] as const) {
        const capability = caps.parameters[key];
        if (!capability?.supported) continue;
        if (capability.minimum != null && next[key] < capability.minimum) next[key] = capability.minimum;
        if (capability.maximum != null && next[key] > capability.maximum) next[key] = capability.maximum;
      }
      return next;
    });
  }, [caps]);

  useEffect(() => {
    if (pendingDemo && !busy && !abortRef.current) {
      setPendingDemo(false);
      enterDemoNow();
    }
    // enterDemoNow intentionally captures the settled editor and result state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingDemo, busy]);

  useEffect(() => {
    if (showFullConfig && !activeRoundData) setShowFullConfig(false);
  }, [showFullConfig, activeRoundData]);

  const statusIcon = status === "target_reached" ? "check" : status === "running" ? "play" : status === "failed" || status === "interrupted" ? "close" : "spark";

  function setDraftFieldErrors(next: Record<string, string>) { setFieldErrors(next); }

  function validateDraft(): boolean {
    const next: Record<string, string> = {};
    if (!prompt.trim()) next.prompt = "请输入当前提示词。";
    if (!expected.trim()) next.expected = "请描述期望效果，作为评测依据。";
    if (!model.trim()) next.model = "请输入模型 ID。";
    if (promptMode === "system" && !testInput.trim()) next.testInput = "系统提示词模式需要填写测试输入。";
    if (!currentApiKey.trim()) next.apiKey = `请填写 ${providerNames[provider]} API Key。`;
    if (!jevKey.trim()) next.jevKey = "请填写 Jev API Key。";
    setDraftFieldErrors(next);
    return Object.keys(next).length === 0;
  }

  async function refreshHistory(): Promise<Run[]> {
    try {
      const response = await fetch(`${API}/api/runs`);
      if (!response.ok) return [];
      const items = await response.json() as Run[];
      setHistory(items);
      return items;
    } catch { /* Backend can be stopped while the UI remains open. */ }
    return [];
  }

  async function stopOnServer(runId: string) {
    if (!runId) { stopWhenStarted.current = true; return; }
    try { await fetch(`${API}/api/runs/${runId}/stop`, { method: "DELETE" }); } catch { /* The stream still checks cancellation at each stage. */ }
  }

  async function stopRun() {
    if (!busy || isStopping) return;
    setIsStopping(true);
    setStage("正在停止；当前 API 请求结束后不会启动后续步骤或轮次。");
    if (runIdRef.current) await stopOnServer(runIdRef.current);
    else stopWhenStarted.current = true;
  }

  async function startRun(event?: FormEvent, automatic = mode === "auto") {
    event?.preventDefault();
    if (busy || isDemo || pendingDemo || !validateDraft()) return;
    setError(""); setStage("正在连接服务"); setStatus("running"); setIsStopping(false); setActiveTab("output"); setFieldErrors({}); setTestState({ status: "idle", message: "" });
    setRunOrigin("current"); runIdRef.current = ""; stopWhenStarted.current = false;
    const controller = new AbortController(); abortRef.current = controller;
    const payload = {
      prompt, prompt_mode: promptMode, expected, test_input: testInput, reference,
      model: { provider, region, model, ...params },
      target_api_key: currentApiKey, jev_api_key: jevKey,
      auto_optimize: automatic, threshold, max_rounds: automatic ? maxRounds : 1,
      optimization_history: optimizationHistoryForRun(isDemo ? null : runRef.current, {
        provider, region, model, prompt_mode: promptMode, test_input: testInput, expected, reference,
      }),
    };
    let receivedFinished = false, runId = "";
    try {
      const response = await fetch(`${API}/api/runs/stream`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal });
      if (!response.ok || !response.body) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail?.message || detail?.detail || "无法启动评测，请检查后端服务和输入内容。");
      }
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let buffer = "";
      const processBlock = async (block: string) => {
        const eventName = block.match(/^event:\s*(.+)$/m)?.[1];
        const dataText = block.split("\n").filter(line => line.startsWith("data:")).map(line => line.slice(5).trim()).join("\n");
        if (!eventName || !dataText) return;
        const data = JSON.parse(dataText);
        if (eventName === "started") {
          runId = data.run_id; runIdRef.current = runId;
          updateRun({ id: runId, status: "running", mode: automatic ? "auto" : "manual", initial_prompt: prompt, latest_prompt: prompt, stop_reason: "", rounds: [], config: payload.model });
          setRunOrigin("current"); setActiveRound(0);
          if (stopWhenStarted.current) { stopWhenStarted.current = false; setIsStopping(true); await stopOnServer(runId); }
        } else if (eventName === "stage") {
          setStage(data.message || "正在运行");
        } else if (eventName === "round") {
          updateRun(previous => {
            if (!previous || previous.id !== runId) return previous;
            const rounds = [...previous.rounds.filter(item => item.round_no !== data.round.round_no), data.round].sort((a, b) => a.round_no - b.round_no);
            return { ...previous, rounds, latest_prompt: data.round.next_prompt || data.round.prompt };
          });
          setActiveRound(Math.max(0, data.round.round_no - 1));
          if (automatic && data.round.status === "completed" && data.round.next_prompt) {
            setPrompt(data.round.next_prompt);
            if (data.round.next_parameters) setParams(previous => ({ ...previous, ...data.round.next_parameters }));
          }
        } else if (eventName === "error") {
          setError(data.message || "运行失败");
        } else if (eventName === "finished") {
          receivedFinished = true;
          setStatus(data.status as RunStatus); setStage(data.stop_reason || statusText(data.status)); setIsStopping(false);
          updateRun(previous => previous ? { ...previous, status: data.status, stop_reason: data.stop_reason || "" } : previous);
        }
      };
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n"); buffer = blocks.pop() || "";
        for (const block of blocks) await processBlock(block);
      }
      if (buffer.trim()) await processBlock(buffer);
      if (runId) {
        await refreshHistory();
        const latest = await fetch(`${API}/api/runs/${runId}`).then(value => value.ok ? value.json() : null).catch(() => null) as Run | null;
        if (latest) { updateRun(latest); setActiveRound(Math.max(0, latest.rounds.length - 1)); }
      }
      if (!receivedFinished) {
        setStatus("interrupted"); setStage("运行连接中断，任务无法从页面恢复；已保留服务端已保存的轮次。");
        updateRun(previous => previous ? { ...previous, status: "interrupted", stop_reason: "运行连接中断，无法恢复。" } : previous);
        if (runId) await stopOnServer(runId);
      }
    } catch (exception) {
      setStatus((exception as Error).name === "AbortError" ? "interrupted" : "failed");
      const message = (exception as Error).name === "AbortError" ? "运行连接中断，未完成请求无法恢复。" : ((exception as Error).message || "请求失败");
      setError(message); setStage((exception as Error).name === "AbortError" ? "运行中断" : "运行失败");
      if (runId) {
        await refreshHistory();
        const latest = await fetch(`${API}/api/runs/${runId}`).then(value => value.ok ? value.json() : null).catch(() => null) as Run | null;
        if (latest) updateRun(latest);
      }
    } finally {
      abortRef.current = null; runIdRef.current = ""; setIsStopping(false);
      if (pendingDemo) { setPendingDemo(false); enterDemoNow(); }
    }
  }

  function selectProvider(nextProvider: Provider) {
    const nextCatalog = catalog[nextProvider] || {};
    const nextRegion = Object.keys(nextCatalog)[0] || "default";
    setProvider(nextProvider); setRegion(nextRegion); setModel(nextCatalog[nextRegion]?.models?.[0] || "");
    setTestState({ status: "idle", message: "" }); setError(""); setFieldErrors(prev => ({ ...prev, apiKey: "" }));
    setParams(prev => ({ ...prev, thinking: nextProvider === "deepseek" ? "enabled" : "default" }));
  }

  async function testConnection() {
    if (!currentApiKey.trim()) { setFieldErrors(prev => ({ ...prev, apiKey: `请填写 ${providerNames[provider]} API Key。` })); setTestState({ status: "error", message: "请先填写当前服务商与区域对应的 API Key。" }); return; }
    setTestState({ status: "testing", message: "正在发送一条短请求；可能产生少量 API 费用。" });
    try {
      const response = await fetch(`${API}/api/test-connection`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: { provider, region, model, ...params }, target_api_key: currentApiKey }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result?.detail?.message || "连接失败，请检查 Key、区域和模型 ID。");
      setTestState({ status: "success", message: `连接成功 · 回复：${result.reply || "已收到响应"} · ${result.usage?.total_tokens ?? "未提供"} tokens` });
    } catch (exception) {
      setTestState({ status: "error", message: (exception as Error).message || "连接失败，请检查网络与服务状态。" });
    }
  }

  function enterDemoNow() {
    if (isDemo) return;
    const saved = draftSnapshot(); snapshotRef.current = saved; setSnapshot(saved);
    setIsDemo(true); setError(""); setTestState({ status: "idle", message: "" }); setStage("演示数据 · 第 2 轮首次达到 85% 阈值");
    updateRun(demoRun as Run); setRunOrigin("demo"); setActiveRound(1); setStatus("target_reached"); setMode("auto"); setThreshold(85); setMaxRounds(15);
    setPrompt(demoRounds[1].prompt); setPromptMode("user"); setExpected(demoRounds[1].expected); setTestInput(demoRounds[1].test_input); setReference("");
    setProvider("qwen"); setRegion("cn"); setModel("qwen-plus"); setParams({ ...defaults, temperature: 0.7, top_p: 0.8, top_k: 20 });
    setActiveTab("output"); setShowCompare(false); setEditingPrompt(false); setFieldErrors({});
  }

  function requestDemo() {
    if (isDemo) return;
    if (busy) { setDemoConfirm(true); return; }
    enterDemoNow();
  }

  function exitDemo() {
    const saved = snapshotRef.current;
    if (saved) {
      setPrompt(saved.prompt); setPromptMode(saved.promptMode); setExpected(saved.expected); setTestInput(saved.testInput); setReference(saved.reference);
      setProvider(saved.provider); setRegion(saved.region); setModel(saved.model); setParams(saved.params); setThreshold(saved.threshold); setMaxRounds(saved.maxRounds); setMode(saved.mode);
      updateRun(saved.run); setActiveRound(saved.activeRound); setStatus(saved.status); setStage(saved.stage); setActiveTab(saved.activeTab); setRunOrigin(saved.runOrigin);
    } else {
      updateRun(null); setStatus("idle"); setStage("准备就绪");
    }
    snapshotRef.current = null; setSnapshot(null); setIsDemo(false); setDemoConfirm(false); setError("");
  }

  async function openHistory(runId: string) {
    try {
      const response = await fetch(`${API}/api/runs/${runId}`);
      if (!response.ok) return;
      const value = await response.json() as Run;
      if (value.status === "running") {
        const response = await fetch(`${API}/api/runs/${runId}/stop`, { method: "DELETE" }).catch(() => null);
        const result = await response?.json().catch(() => null);
        value.status = result?.stopping ? "stopping" : "interrupted";
        value.stop_reason = result?.stopping ? "正在等待当前供应商请求结束；页面无法恢复原 SSE 连接。" : "服务端任务已中断，已完成结果已保留。";
      }
      setIsDemo(false); snapshotRef.current = null; setSnapshot(null); updateRun(value); setRunOrigin("history");
      setActiveRound(Math.max(0, value.rounds.length - 1)); setStatus(value.status as RunStatus); setIsStopping(value.status === "stopping"); setStage(value.stop_reason || statusText(value.status)); setActiveTab("output"); setError("");
    } catch { setError("无法打开这条运行记录，请检查后端服务。"); }
  }

  function historicalRoundActionPrompt(value: string, round: Round) {
    const adopted = adoptRoundPrompt({ prompt, promptMode, expected, testInput, reference }, value, round, runOrigin);
    setPrompt(adopted.prompt); setPromptMode(adopted.promptMode);
    setEditingPrompt(false); setShowCompare(false); showNotice(isHistorical ? "已只替换提示词；其他测试配置保持不变。" : "已采用提示词，测试输入、模型与参数已保留。");
  }

  function canAdoptParameterSuggestion(round: Round) {
    return Boolean(round.next_parameters && caps
      && round.provider === provider
      && round.model === model
      && (round.region || "default") === (region || "default"));
  }

  function parameterSuggestionPatch(round: Round): Partial<Params> {
    if (!canAdoptParameterSuggestion(round) || !round.next_parameters || !caps) return {};
    return supportedParameterPatch(round.next_parameters, caps) as Partial<Params>;
  }

  function adoptParameterSuggestion(round: Round) {
    const patch = parameterSuggestionPatch(round);
    if (!Object.keys(patch).length) {
      showNotice("参数建议仅适用于本轮使用的模型；当前没有可直接采用的支持参数。");
      return;
    }
    setParams(previous => ({ ...previous, ...patch }));
    showNotice("已采用支持的参数建议；提示词和其他测试配置保持不变。");
  }

  function adoptPromptAndParameters(value: string, round: Round) {
    if (!value.trim()) return;
    const adopted = adoptRoundPrompt({ prompt, promptMode, expected, testInput, reference }, value, round, runOrigin);
    setPrompt(adopted.prompt); setPromptMode(adopted.promptMode); setEditingPrompt(false); setShowCompare(false);
    const patch = parameterSuggestionPatch(round);
    if (Object.keys(patch).length) {
      setParams(previous => ({ ...previous, ...patch }));
      showNotice("已采用提示词和支持的参数建议；其他测试配置保持不变。");
    } else {
      showNotice("已采用提示词；当前模型与本轮不同或能力未就绪，参数未自动更改。");
    }
  }

  async function copyText(value: string, label = "内容") {
    try { await navigator.clipboard.writeText(value); showNotice(`${label}已复制`); }
    catch { setError("无法访问剪贴板，请手动选择并复制内容。"); }
  }

  function useWholeRoundConfig(round: Round) {
    const restored = restoreRoundConfig(round);
    setPrompt(restored.prompt); setPromptMode(restored.promptMode); setTestInput(restored.testInput); setExpected(restored.expected); setReference(restored.reference);
    setProvider(restored.provider as Provider); setRegion(restored.region); setModel(restored.model); setParams(restored.params); setThreshold(restored.threshold); setMaxRounds(restored.maxRounds); setMode(restored.mode as "manual" | "auto");
    setEditingPrompt(false); setShowCompare(false); setFieldErrors({}); setTestState({ status: "idle", message: "" });
    showNotice("已恢复本轮测试配置；API Key 未保存，请确认当前服务凭据。");
  }

  function saveEditedPrompt(round: Round) {
    if (!editedPrompt.trim()) { setError("新提示词不能为空。"); return; }
    historicalRoundActionPrompt(editedPrompt, round);
    setError("");
  }

  function exportRun(format: "json" | "markdown") {
    if (!run || isDemo) return;
    const safeRun = removeSecrets(run) as Run;
    const content = format === "json" ? JSON.stringify(safeRun, null, 2) : [
      "# PromptLattice 运行记录", "", `- 状态：${statusText(run.status)}`, `- 模式：${run.mode === "auto" ? "自动优化" : "单轮评测"}`, `- 停止原因：${run.stop_reason || "—"}`, "",
      ...run.rounds.flatMap(round => [
        `## 第 ${round.round_no} 轮 · ${statusText(round.status)}`, "", `- 模型：${round.provider} / ${round.model}`, `- 提示词角色：${round.prompt_mode === "system" ? "System" : "User"}`, `- 本轮实际参数：${JSON.stringify(round.parameters || {})}`,
        `- Jev 达标判断概率：${round.jev_probability == null ? "未提供" : `${Math.round(round.jev_probability * 100)}%`}`, `- Jev 分项概率：${Object.entries(round.jev_checks || {}).map(([key, value]) => `${jevCheckLabels[key] || key} ${value == null ? "未提供" : `${Math.round(value * 100)}%`}`).join("；") || "未提供"}`, `- 模型符合度：${round.evaluation?.score ?? "未提供"}/100`,
        `- 耗时：${elapsed(round)}`, `- Token：${tokenTotal(round)}`, "", "### 本轮提示词", "```", round.prompt, "```", "", "### 测试输入", "```", round.test_input || "（未提供）", "```", "",
        "### 期望效果", "```", round.expected, "```", "", "### 实际输出", "```", round.output || "（未生成）", "```", "", "### 模型评估", JSON.stringify(round.evaluation || {}, null, 2), "",
        "### 修改建议", round.advice || "—", "", "### 修改原因", round.reason || "—", "", "### 参数建议", round.parameter_advice || "—", "", "### 参数理由", round.parameter_reason || "—", "", "### 参数能力与校验", JSON.stringify({ parameter_support: round.parameter_support || {}, thinking_support: round.thinking_support || {}, parameter_warnings: round.parameter_warnings || [], next_parameters: round.next_parameters || {} }, null, 2), "", "### 推荐的下一版提示词", "```", round.next_prompt || "（未生成或已停止）", "```", "",
      ]),
    ].join("\n");
    const blob = new Blob([content], { type: format === "json" ? "application/json" : "text/markdown" });
    const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = `promptlattice-${run.id}.${format === "json" ? "json" : "md"}`; link.click(); URL.revokeObjectURL(url);
    showNotice(`${format.toUpperCase()} 记录已导出`);
  }

  const visibleTabItems: [ResultTab, string][] = [["output", "生成结果"], ["evaluation", "评估详情"], ["recommendation", "优化建议"]];
  const currentProviderLabel = providerNames[provider];
  const activeRegionLabel = currentRegion?.label || region;

  return <main className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" aria-label="PromptLattice 首页"><span className="brand-mark"><Icon kind="logo"/></span><span className="brand-wordmark">Prompt<span className="brand-lattice">Lattice</span></span></a>
      <div className="top-context"><span className="context-dot"/>提示词调试工作台<span className="context-divider"/>本地会话</div>
      <div className="top-actions">
        {isDemo ? <button type="button" className="demo-exit-button" onClick={exitDemo}>退出演示并恢复草稿</button> : <button type="button" className="demo-button" onClick={requestDemo}>体验演示</button>}
        <span className="top-separator"/><span className="local-badge">本地</span>
      </div>
    </header>

    <div className="page-wrap" id="top">
      <section className="page-heading">
        <div><div className="eyebrow">本地运行 · 提示词工程</div><h1>提示词评测工作台</h1><p>使用所选模型生成内容，再由 Jev 与模型分别评估。</p></div>
        <div className="heading-state"><span className={`status-pill ${status}`}><i/>{statusText(status)}</span><span className="secure-note">Key 仅随请求发送 · 不写入记录</span></div>
      </section>

      {isDemo && <div className="demo-banner" role="status"><span className="demo-banner-label">演示数据</span><span>预置的两轮示例；第 2 轮两项评分首次达到 85%，不会调用模型或写入历史。</span><button type="button" onClick={exitDemo}>退出演示并恢复草稿</button></div>}
      {pendingDemo && <div className="pending-banner" role="status">已请求停止当前运行；当前 API 请求结束后会进入演示，已完成结果会保留。</div>}

      <form className="workbench" onSubmit={event => startRun(event, mode === "auto")}>
        <div className="editor-column">
          <section className="panel task-panel" aria-labelledby="task-title">
            <div className="panel-heading"><div><div className="step-label">测试场景</div><h2 id="task-title">提示词与评测依据</h2><p>期望效果只用于评估，不会混入目标模型的生成请求。</p></div><span className="required-legend"><b>*</b> 必填</span></div>
            <label className="field-label" htmlFor="prompt">当前提示词</label>
            <div className="segmented-control" role="group" aria-label="提示词使用方式">
              <button type="button" className={promptMode === "user" ? "selected" : ""} aria-pressed={promptMode === "user"} disabled={isDemo} onClick={() => setPromptMode("user")}>作为用户消息</button>
              <button type="button" className={promptMode === "system" ? "selected" : ""} aria-pressed={promptMode === "system"} disabled={isDemo} onClick={() => setPromptMode("system")}>作为系统提示词</button>
            </div>
            <textarea id="prompt" className="prompt-editor" value={prompt} onChange={event => setPrompt(event.target.value)} placeholder="写下要测试的提示词…" aria-invalid={!!fieldErrors.prompt} disabled={isDemo} spellCheck={false}/>
            {fieldErrors.prompt && <p className="field-error">{fieldErrors.prompt}</p>}
            <label className="field-label field-spaced" htmlFor="expected">期望效果 <span className="required-star">*</span></label>
            <textarea id="expected" className="text-editor expected-editor" value={expected} onChange={event => setExpected(event.target.value)} placeholder="描述语气、结构、长度和明确约束。" aria-invalid={!!fieldErrors.expected} disabled={isDemo}/>
            {fieldErrors.expected && <p className="field-error">{fieldErrors.expected}</p>}
            <div className="field-grid">
              <div><label className="field-label" htmlFor="test-input">测试输入 <span className="optional-label">可选</span></label><textarea id="test-input" className="text-editor compact-editor" value={testInput} onChange={event => setTestInput(event.target.value)} placeholder="用于模拟真实场景的输入…" aria-invalid={!!fieldErrors.testInput} disabled={isDemo}/>{fieldErrors.testInput && <p className="field-error">{fieldErrors.testInput}</p>}</div>
              <div><label className="field-label" htmlFor="reference">参考答案 <span className="optional-label">评测用</span></label><textarea id="reference" className="text-editor compact-editor" value={reference} onChange={event => setReference(event.target.value)} placeholder="可选；只发给 Jev 和模型评估，不发给生成模型。" disabled={isDemo}/></div>
            </div>
            {promptMode === "user" && testInput.trim() && <details className="message-preview" open>
              <summary><span className="preview-status-dot"/>实际发送给目标模型 <span className="role-chip">user</span><span className="preview-summary-hint">查看消息结构</span></summary>
              <pre>{JSON.stringify(messages, null, 2)}</pre>
            </details>}
            {promptMode === "system" && <div className={`info-callout ${fieldErrors.testInput ? "invalid-callout" : ""}`}><span>i</span><p>提示词以 <code>system</code> 角色发送，测试输入以 <code>user</code> 角色发送。此模式必须填写测试输入。</p></div>}
          </section>

          <section className="panel model-panel" aria-labelledby="model-title">
            <div className="panel-heading model-heading">
              <div><div className="step-label">模型与参数</div><h2 id="model-title">选择生成与评估模型</h2><p className="selected-model-line"><span className="selected-model-dot"/>{currentProviderLabel} · {model || "输入模型 ID"} · {activeRegionLabel}</p></div>
              <button type="button" className="subtle-button" aria-expanded={showModelConfig} onClick={() => setShowModelConfig(value => !value)}>{showModelConfig ? "收起设置" : "展开设置"}<Icon kind="arrow"/></button>
            </div>
            {showModelConfig && <>
              <div className="model-fields">
                <div><label className="field-label" htmlFor="provider">服务商</label><select id="provider" className="form-control" value={provider} onChange={event => selectProvider(event.target.value as Provider)} disabled={isDemo}>{providerKeys.map(key => <option key={key} value={key}>{providerNames[key] || key}</option>)}</select></div>
                {Object.keys(currentRegions).length > 1 && <div><label className="field-label" htmlFor="region">API 区域</label><select id="region" className="form-control" value={region} onChange={event => { setRegion(event.target.value); setTestState({ status: "idle", message: "" }); setError(""); }} disabled={isDemo}>{Object.entries(currentRegions).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}</select></div>}
                <div className="model-id-field"><label className="field-label" htmlFor="model">模型 ID</label><input id="model" className="form-control" list="model-suggestions" value={model} onChange={event => setModel(event.target.value)} placeholder="可填写服务商支持的模型 ID" aria-invalid={!!fieldErrors.model} disabled={isDemo}/><datalist id="model-suggestions">{(currentRegion?.models || []).map(name => <option key={name} value={name}/>)}</datalist>{fieldErrors.model && <p className="field-error">{fieldErrors.model}</p>}</div>
              </div>

              {caps?.thinking.available && <div className="thinking-setting"><div><strong>推理模式</strong><span>按当前模型的官方接口参数设置</span></div><select className="form-control compact-select" aria-label="推理模式" value={params.thinking} onChange={event => setParams(value => ({ ...value, thinking: event.target.value }))} disabled={isDemo}>
                {caps.thinking.choices.map(choice => <option key={choice} value={choice}>{choice === "enabled" ? "开启思考" : choice === "disabled" ? "关闭思考" : "使用服务默认"}</option>)}
              </select></div>}

              <div className="parameter-header"><div><strong>生成参数</strong><span>{capsLoading ? "正在读取能力配置…" : "不支持的参数会禁用且不会发送"}</span></div><span className="mono-caption">接口能力 · 模型行为可能不同</span></div>
              <div className="parameter-grid">
                {([ ["temperature", "Temperature", 0, 2, 0.1], ["top_p", "Top-p", 0.05, 1, 0.05], ["top_k", "Top-k", 1, 100, 1], ["max_tokens", "最大输出 Token", 1, 32768, 1] ] as [keyof Omit<Params, "thinking">, string, number, number, number][]).map(([key, label, min, max, step]) => {
                  const capability = caps?.parameters?.[key];
                  const isSupported = capability?.supported ?? key !== "top_k";
                  const actualMin = capability?.minimum ?? min;
                  const actualMax = capability?.maximum ?? max;
                  return <label className={`parameter-control ${isSupported ? "" : "disabled"}`} key={key}>
                    <span className="parameter-title">{label}{!isSupported && <em>不支持</em>}</span>
                    <div className="parameter-inputs"><input type="range" min={actualMin} max={actualMax} step={step} value={params[key]} disabled={!isSupported || isDemo} aria-label={`${label} 滑块`} onChange={event => setParams(value => ({ ...value, [key]: Number(event.target.value) }))}/><input className="parameter-number" type="number" min={actualMin} max={actualMax} step={step} value={params[key]} disabled={!isSupported || isDemo} aria-label={`${label} 数值`} onChange={event => setParams(value => ({ ...value, [key]: Number(event.target.value) }))}/></div>
                    {!isSupported ? <small>{capability?.reason || "接口能力未确认，平台不发送该参数。"}</small> : capability?.warning ? <small className="parameter-warning">{capability.warning}</small> : <small>{capability?.api_parameter ? `请求字段：${capability.api_parameter}` : "接口层支持；具体模型可能有限制。"}</small>}
                  </label>;
                })}
              </div>
              {!!caps?.warnings.length && <div className="model-warning"><span>!</span><p>{caps.warnings.join(" ")}</p></div>}

              <div className="key-section-heading"><div><strong>API Key</strong><span>只在本页面内存中保留，刷新后需重新填写</span></div><button type="button" className="subtle-button" aria-expanded={showKeys} onClick={() => setShowKeys(value => !value)}>{showKeys ? "收起" : "展开"}<Icon kind="arrow"/></button></div>
              {showKeys && (isDemo ? <div className="demo-readonly-note">演示不会发送 API 请求，也不会读取或覆盖你填写过的 Key。</div> : <div className="key-grid">
                <div><label className="field-label" htmlFor="model-key">{currentProviderLabel} · {activeRegionLabel} API Key <span className="required-star">*</span></label><div className="secret-control"><input id="model-key" className="form-control" type={showKey ? "text" : "password"} autoComplete="off" spellCheck={false} value={currentApiKey} onChange={event => { setApiKeys(value => ({ ...value, [`${provider}:${region}`]: event.target.value })); setFieldErrors(prev => ({ ...prev, apiKey: "" })); }} placeholder="粘贴当前服务商与区域对应的 Key" aria-invalid={!!fieldErrors.apiKey} disabled={isDemo}/><button type="button" onClick={() => setShowKey(value => !value)} aria-label={showKey ? "隐藏模型 API Key" : "显示模型 API Key"}>{showKey ? "隐藏" : "显示"}</button></div>{fieldErrors.apiKey && <p className="field-error">{fieldErrors.apiKey}</p>}</div>
                <div><label className="field-label" htmlFor="jev-key">Jev API Key <span className="required-star">*</span></label><div className="secret-control"><input id="jev-key" className="form-control" type={showJevKey ? "text" : "password"} autoComplete="off" spellCheck={false} value={jevKey} onChange={event => { setJevKey(event.target.value); setFieldErrors(prev => ({ ...prev, jevKey: "" })); }} placeholder="TypeSafe Jev API Key" aria-invalid={!!fieldErrors.jevKey}/><button type="button" onClick={() => setShowJevKey(value => !value)} aria-label={showJevKey ? "隐藏 Jev API Key" : "显示 Jev API Key"}>{showJevKey ? "隐藏" : "显示"}</button></div>{fieldErrors.jevKey && <p className="field-error">{fieldErrors.jevKey}</p>}</div>
                <div className="key-disclaimer"><span>⌑</span><p>Key 通过 POST 发送到本地后端，不进入 URL、浏览器持久存储、SQLite、日志或导出。提示词、测试输入、期望效果和生成结果会按流程发送给选中的模型服务商和 Jev。</p></div>
                <button type="button" className="secondary-button connection-button" disabled={busy || testState.status === "testing"} onClick={testConnection}>{testState.status === "testing" ? "正在测试连接…" : "测试模型连接"}<span>可能产生少量 API 费用</span></button>
                {testState.message && <p className={`connection-message ${testState.status}`} role="status">{testState.status === "success" && <Icon kind="check"/>}{testState.message}</p>}
              </div>)}
            </>}
          </section>

          <section className="panel run-settings-panel" aria-labelledby="run-settings-title">
            <div className="panel-heading"><div><div className="step-label">运行设置</div><h2 id="run-settings-title">单轮评测或自动优化</h2></div></div>
            <div className="mode-switch" role="group" aria-label="运行模式"><button type="button" className={mode === "manual" ? "selected" : ""} aria-pressed={mode === "manual"} disabled={isDemo || busy} onClick={() => setMode("manual")}>单轮评测</button><button type="button" className={mode === "auto" ? "selected" : ""} aria-pressed={mode === "auto"} disabled={isDemo || busy} onClick={() => setMode("auto")}>自动优化</button></div>
            {mode === "auto" && <div className="auto-options"><p>每轮按“生成 → Jev 判断 → 模型评分 → 未达标时改写”执行。达到双阈值或停止条件后结束。</p><div className="auto-fields"><label>目标阈值 <span><input type="number" min="0" max="100" value={threshold} disabled={isDemo || busy} onChange={event => setThreshold(Math.max(0, Math.min(100, Number(event.target.value))))}/> %</span></label><label>最大轮数 <span><input type="number" min="1" max="15" value={maxRounds} disabled={isDemo || busy} onChange={event => setMaxRounds(Math.max(1, Math.min(15, Number(event.target.value))))}/> 轮</span></label></div><div className="cost-note"><b>!</b><span>最大轮数包含原始提示词的第 1 轮。每轮会调用目标模型和 Jev，实际费用由对应服务商收取。</span></div></div>}
            <div className="request-summary"><div><span>本次模型</span><strong>{currentProviderLabel} · {model || "未选择"}</strong></div><div><span>提示词角色</span><strong>{promptMode === "system" ? "系统 + 用户" : "用户消息"}</strong></div><div><span>Jev 评测依据</span><strong>期望效果{reference.trim() ? " · 参考答案" : ""}</strong></div></div>
          </section>
        </div>

        <section className="results-column" aria-labelledby="results-title">
          <div className="panel result-panel">
            <div className="results-header"><div><div className="result-title-line"><span className="step-label">运行结果</span>{isDemo && <span className="demo-tag">演示数据</span>}{run && !isDemo && <span className="run-id">运行 {run.id.slice(0, 8).toUpperCase()}</span>}</div><h2 id="results-title">{activeRoundData ? `第 ${activeRoundData.round_no} 轮` : "结果与版本"}</h2><p>{run ? `${roundCount} 轮 · ${statusText(run.status)}${run.stop_reason ? ` · ${run.stop_reason}` : ""}` : "运行后在这里查看输出、评分和提示词版本。"}</p></div>
              {run && !isDemo && <div className="export-actions"><button type="button" className="text-action" aria-expanded={exportMenu} onClick={() => setExportMenu(value => !value)}>导出⌄</button>{exportMenu && <div className="export-popover"><button type="button" onClick={() => { exportRun("json"); setExportMenu(false); }}>导出 JSON</button><button type="button" onClick={() => { exportRun("markdown"); setExportMenu(false); }}>导出 Markdown</button></div>}</div>}
            </div>

            {activeRoundData ? <>
              <div className="result-meta"><div><span className="round-label">第 {String(activeRoundData.round_no).padStart(2, "0")} 轮</span><span>{providerNames[activeRoundData.provider as Provider] || activeRoundData.provider} · {activeRoundData.model}</span><span className="role-chip">{activeRoundData.prompt_mode === "system" ? "系统" : "用户"}</span></div><span>{elapsed(activeRoundData)} · 模型 {tokenTotal(activeRoundData, "model")} · Jev {tokenTotal(activeRoundData, "jev")}</span></div>
              <div className="score-grid">
                <div className="score-card"><div className="score-heading"><span>Jev 达标判断概率</span><small>本次输出 · NOUL</small></div><strong>{jevPercent == null ? "—" : <>{jevPercent}<small>%</small></>}</strong><div className="score-track"><i style={{ width: `${jevPercent ?? 0}%` }}/></div><p>Jev 对本次生成结果的判断，不是未来成功率。</p>{activeRoundData.jev_checks && Object.values(activeRoundData.jev_checks).some(value => value != null) && <details className="jev-checks"><summary><span>查看分项判断</span><small>独立判断，不合成总概率</small></summary><div className="jev-check-grid">{Object.entries(activeRoundData.jev_checks).map(([key, value]) => <div key={key}><span>{jevCheckLabels[key] || key}</span><strong>{value == null ? "未提供" : `${Math.round(value * 100)}%`}</strong></div>)}</div><p>这些是 Jev 对不同要求分别回答“是”的概率，不是自然语言解释。</p></details>}</div>
                <div className="score-card"><div className="score-heading"><span>模型符合度</span><small>{activeRoundData.evaluation ? "独立评分" : "评估未完成"}</small></div><strong>{activeRoundData.evaluation?.score ?? "—"}<small>{activeRoundData.evaluation ? "/100" : ""}</small></strong><div className="score-track"><i style={{ width: `${activeRoundData.evaluation?.score ?? 0}%` }}/></div><p>{previousRound?.evaluation && activeRoundData.evaluation ? `较上一轮 ${activeRoundData.evaluation.score - previousRound.evaluation.score > 0 ? "+" : ""}${activeRoundData.evaluation.score - previousRound.evaluation.score} 分` : "按明确维度计分；不使用 Jev 概率。"}</p></div>
              </div>

              {showCompare && <div className="comparison-panel"><div className="comparison-heading"><div><strong>版本对比</strong><span>初版与当前所选轮次</span></div><button type="button" className="text-action" onClick={() => setShowCompare(false)}>关闭对比</button></div><div className="comparison-grid">{[run!.rounds[0], activeRoundData].map((item, index) => <div className="comparison-item" key={`${item.round_no}-${index}`}><div className="comparison-label">{index === 0 ? "初始版本" : `第 ${item.round_no} 轮 · 本轮提示词`}</div><pre>{item.prompt}</pre><div className="comparison-scores">模型 {item.evaluation?.score ?? "—"} / Jev {item.jev_probability == null ? "—" : `${Math.round(item.jev_probability * 100)}%`}</div><details><summary>查看实际输出</summary><pre>{item.output || "尚无输出"}</pre></details></div>)}</div></div>}

              <div className="result-tabs" role="tablist" aria-label="结果详情">
                {visibleTabItems.map(([key, label]) => <button key={key} id={`tab-${key}`} type="button" role="tab" aria-selected={activeTab === key} aria-controls="tab-panel" tabIndex={activeTab === key ? 0 : -1} className={activeTab === key ? "active" : ""} onClick={() => setActiveTab(key)}>{label}</button>)}
                <button type="button" className={`compare-toggle ${showCompare ? "selected" : ""}`} aria-pressed={showCompare} onClick={() => setShowCompare(value => !value)}>{showCompare ? "关闭对比" : "对比初版"}</button>
              </div>
              <div id="tab-panel" role="tabpanel" aria-labelledby={`tab-${activeTab}`} className="tab-content">
                {activeTab === "output" && <section className="content-pane"><div className="content-pane-heading"><div><span className="pane-mark output-mark"/><div><h3>实际生成内容</h3><p>目标模型本轮返回</p></div></div>{activeRoundData.output && <button type="button" className="secondary-button compact-action" onClick={() => copyText(activeRoundData.output || "", "生成结果")}>复制</button>}</div>{activeRoundData.output ? <pre className="long-content">{activeRoundData.output}</pre> : <div className="pane-empty">{activeRoundData.error?.message || (activeRoundData.status === "running" ? "等待生成内容…" : "本轮没有生成输出。")}</div>}
                  <details className="prompt-sent-details"><summary>本轮实际发送的提示词结构</summary><div className="sent-messages-label">测试输入：{activeRoundData.test_input || "（未提供）"}</div><pre>{JSON.stringify(buildMessages(activeRoundData.prompt, activeRoundData.test_input, activeRoundData.prompt_mode || "user"), null, 2)}</pre><div className="sent-messages-note">期望效果和参考答案不会加入目标模型的生成请求。</div></details>
                </section>}
                {activeTab === "evaluation" && <section className="content-pane"><div className="content-pane-heading"><div><span className="pane-mark evaluation-mark">✓</span><div><h3>模型独立评估</h3><p>依据本轮提示词约束与期望效果；不使用 Jev 分数</p></div></div></div>{activeRoundData.evaluation ? <><div className="evaluation-blocks">{activeRoundData.evaluation.criterion_scores && <div className="criterion-score-panel"><h4>评分维度</h4><div className="criterion-score-grid">{(Object.keys(evaluationCriterionLabels) as (keyof typeof evaluationCriterionLabels)[]).map(key => <div key={key}><span>{evaluationCriterionLabels[key]}<small>{activeRoundData.evaluation?.criterion_weights?.[key] ?? 0}%</small></span><strong>{activeRoundData.evaluation?.criterion_scores?.[key] ?? "—"}</strong></div>)}</div><p>总分按维度权重计算；只要模型识别到硬性约束失败，总分最高为 59。</p></div>}{!!activeRoundData.evaluation.hard_constraint_failures?.length && <div className="hard-constraint-panel"><h4>未满足的硬性约束</h4><ul className="issue-list">{activeRoundData.evaluation.hard_constraint_failures.map((item, index) => <li key={index}>{item}</li>)}</ul></div>}<div><h4>做得好的地方</h4>{activeRoundData.evaluation.strengths.length ? <ul className="positive-list">{activeRoundData.evaluation.strengths.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="quiet-copy">未列出优点。</p>}</div><div><h4>问题</h4>{activeRoundData.evaluation.issues.length ? <ul className="issue-list">{activeRoundData.evaluation.issues.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="quiet-copy">未发现问题。</p>}</div><div className="missing-requirements"><h4>遗漏的明确要求</h4>{activeRoundData.evaluation.missing_requirements.length ? <ul className="issue-list">{activeRoundData.evaluation.missing_requirements.map((item, index) => <li key={index}>{item}</li>)}</ul> : <p className="positive-copy">未发现遗漏。</p>}</div></div></> : <div className="pane-empty">{activeRoundData.status === "running" ? "模型正在独立评估…" : activeRoundData.error?.message || "本轮评估结果未生成。"}</div>}</section>}
              {activeTab === "recommendation" && <section className="content-pane recommendation-content"><div className="content-pane-heading"><div><span className="pane-mark recommendation-mark"><Icon kind="spark"/></span><div><h3>提示词与参数建议</h3><p>目标模型生成建议；此前轮次只用于建议，不加入本轮生成{(activeRoundData.optimization_history_rounds || 0) > 0 ? ` · 已参考此前 ${activeRoundData.optimization_history_rounds} 轮` : ""}</p></div></div></div>
                  {hasValidNextPrompt(activeRoundData) ? <>
                    <div className="advice-block"><div><strong>修改意见</strong><p>{activeRoundData.advice || "模型未提供单独的修改意见。"}</p></div><div><strong>修改原因</strong><p>{activeRoundData.reason || "模型未提供修改原因。"}</p></div></div>
                    <div className="next-prompt-card"><div className="next-prompt-heading"><div><strong>推荐的下一版提示词</strong><span>完整版本 · 供下一轮或手动采用</span></div><div className="prompt-actions"><button type="button" className="text-action" onClick={() => copyText(activeRoundData.next_prompt || "", "新提示词")}>复制</button><button type="button" className="secondary-button compact-action" onClick={() => historicalRoundActionPrompt(activeRoundData.next_prompt || "", activeRoundData)}>只采用提示词</button></div></div>
                      {editingPrompt ? <><textarea className="prompt-edit-area" value={editedPrompt} onChange={event => setEditedPrompt(event.target.value)} aria-label="编辑推荐提示词"/><div className="edit-prompt-actions"><button type="button" className="text-action" onClick={() => setEditingPrompt(false)}>取消</button><button type="button" className="primary-small" onClick={() => saveEditedPrompt(activeRoundData)}>采用编辑后的提示词</button></div></> : <><pre className="next-prompt-text">{activeRoundData.next_prompt}</pre><div className="edit-prompt-footer"><button type="button" className="text-action" onClick={() => { setEditedPrompt(activeRoundData.next_prompt || ""); setEditingPrompt(true); }}>编辑后采用</button>{isHistorical && <span>只替换提示词，不改测试输入、角色、模型或参数。</span>}</div></>}
                    </div>
                    {activeRoundData.next_parameters && <div className="parameter-recommendation"><div className="parameter-recommendation-heading"><div><strong>下一轮参数建议</strong><span>按当前模型能力筛选；自动优化会直接使用这些参数</span></div><span className="parameter-model-tag">{providerNames[activeRoundData.provider as Provider] || activeRoundData.provider} · {activeRoundData.model}</span></div><p>{activeRoundData.parameter_advice || "模型未提供参数建议。"}</p><p className="parameter-reason">{activeRoundData.parameter_reason || "模型未提供参数调整理由。"}</p><div className="parameter-suggestion-grid">{(["temperature", "top_p", "top_k", "max_tokens", ...(activeRoundData.thinking_support?.available ? ["thinking"] : [])] as (keyof typeof parameterLabels)[]).map(key => { const supported = key === "thinking" ? activeRoundData.thinking_support?.available : activeRoundData.parameter_support?.[key]?.supported; const before = activeRoundData.parameters?.[key]; const after = activeRoundData.next_parameters?.[key] ?? before; const format = (value: string | number | undefined) => key === "thinking" ? value === "enabled" ? "开启" : value === "disabled" ? "关闭" : value === "default" ? "默认" : value ?? "—" : value ?? "—"; return <div key={key}><span>{parameterLabels[key]}</span><strong>{format(before)}<i>→</i>{format(after)}</strong><small>{supported === false ? "当前模式不支持 · 保持原值" : before === after ? "保持不变" : "建议调整"}</small></div>; })}</div>{!!activeRoundData.parameter_warnings?.length && <ul className="parameter-warning-list">{activeRoundData.parameter_warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}<div className="parameter-recommendation-actions"><button type="button" className="secondary-button compact-action" onClick={() => adoptParameterSuggestion(activeRoundData)} disabled={!canAdoptParameterSuggestion(activeRoundData)}>只采用参数</button><button type="button" className="primary-small" onClick={() => adoptPromptAndParameters(activeRoundData.next_prompt || "", activeRoundData)} disabled={!canAdoptParameterSuggestion(activeRoundData)}>{isHistorical ? "同时采用提示词与参数" : "采用提示词与参数"}</button>{!canAdoptParameterSuggestion(activeRoundData) && <small>只能直接采用同一服务商、区域与模型的参数建议；其他情况可只采用提示词。</small>}</div></div>}
                    {isHistorical && <p className="restore-hint">需要重现实验场景？使用轮次详情中的“恢复整套测试配置”；API Key 不会从历史中恢复。</p>}
                  </> : <div className="terminal-result"><div className={`terminal-mark ${activeRoundData.status}`}><Icon kind={activeRoundData.status === "target_reached" ? "check" : activeRoundData.status === "failed" ? "close" : "spark"}/></div><div><strong>{activeRoundData.status === "target_reached" ? "本轮已达到目标" : activeRoundData.status === "max_rounds" ? "已达到最大运行轮数" : activeRoundData.status === "repeated_prompt" ? "生成了重复提示词，自动优化已停止" : activeRoundData.status === "stopped" ? "运行已停止" : activeRoundData.status === "interrupted" ? "运行中断，未完成请求无法恢复" : activeRoundData.status === "failed" ? "本轮运行失败" : activeRoundData.status === "running" ? "等待模型生成建议" : "本轮没有可用的新提示词"}</strong><p>{activeRoundData.stop_reason || activeRoundData.error?.message || "本轮没有生成有效的下一版提示词。"}</p></div><div className="current-prompt-card"><div><strong>当前可用提示词</strong><span>本轮实际使用</span></div><pre>{activeRoundData.prompt}</pre><div><button type="button" className="text-action" onClick={() => copyText(activeRoundData.prompt, "当前提示词")}>复制当前提示词</button><button type="button" className="secondary-button compact-action" onClick={() => historicalRoundActionPrompt(activeRoundData.prompt, activeRoundData)}>{isHistorical ? "只采用提示词" : "采用当前版本"}</button></div></div></div>}
                </section>}
              </div>

              <details className="round-config-details" open={showFullConfig} onToggle={event => setShowFullConfig((event.currentTarget as HTMLDetailsElement).open)}>
                <summary><span>第 {activeRoundData.round_no} 轮详情</span><span>本轮提示词、测试场景、调用参数与用量</span></summary>
                <div className="round-config-grid"><div><h4>本轮提示词</h4><pre>{activeRoundData.prompt}</pre></div><div><h4>测试输入</h4><pre>{activeRoundData.test_input || "（未提供）"}</pre></div><div><h4>期望效果</h4><pre>{activeRoundData.expected}</pre></div><div><h4>参考答案</h4><pre>{activeRoundData.reference || "（未提供）"}</pre></div><div><h4>参数</h4><pre>{JSON.stringify(activeRoundData.parameters || {}, null, 2)}</pre></div></div>
                <div className="round-config-actions"><button type="button" className="text-action" onClick={() => copyText(activeRoundData.prompt, "本轮提示词")}>复制本轮提示词</button>{!isDemo && <button type="button" className="secondary-button compact-action" onClick={() => useWholeRoundConfig(activeRoundData)}>恢复整套测试配置</button>}</div>
              </details>
              <p className="result-disclaimer">模型输出与评估都可能有误，仅作调试参考。Jev 数值表示其对本次输出是否达标的判断，不是提示词未来多次生成的真实成功概率。Token 用量以 API 返回为准。</p>
            </> : <div className="result-empty"><h3>等待一次真实运行</h3><p>生成内容、两项评估和提示词版本会显示在这里。也可以先用演示数据体验完整流程。</p><button type="button" className="secondary-button" onClick={requestDemo}>体验演示运行</button></div>}
          </div>

          {run?.rounds.length ? <section className="panel timeline-panel" aria-label="运行版本时间线"><div className="timeline-header"><div><h3>迭代时间线</h3><p>{isDemo ? "演示数据 · 不写入真实历史" : "点击任意轮次查看该轮输出与评分"}</p></div><span>{run.rounds.length} 轮</span></div><div className="timeline-list">{run.rounds.map((round, index) => <button type="button" key={`${run.id}-${round.round_no}`} className={`timeline-item ${index === activeRound ? "selected" : ""}`} aria-current={index === activeRound ? "step" : undefined} onClick={() => { setActiveRound(index); if (isDemo) { setPrompt(round.prompt); setPromptMode(round.prompt_mode || "user"); } setActiveTab("output"); setShowCompare(false); setEditingPrompt(false); }}><span className={`timeline-index ${round.status}`}>{round.status === "target_reached" ? "✓" : round.round_no}</span><span className="timeline-main"><strong>第 {round.round_no} 轮 <i>{statusText(round.status)}</i></strong><small>{roundDate(round, isDemo)} · {round.model} · {elapsed(round)}</small></span><span className="timeline-score">{round.evaluation?.score ?? "—"}</span></button>)}</div></section> : null}

          <section className="panel history-panel"><div className="timeline-header"><div><h3>真实运行历史</h3><p>来自本地 SQLite；演示轮次不会出现在这里。</p></div><span>{history.length}</span></div>{history.length ? <div className="saved-runs-list">{history.map(item => <button type="button" key={item.id} className={`saved-run ${run?.id === item.id && !isDemo ? "selected" : ""}`} onClick={() => openHistory(item.id)}><span className={`saved-run-status ${item.status}`}/><span><strong>{item.initial_prompt?.slice(0, 50) || "提示词评测"}</strong><small>{item.created_at ? new Date(item.created_at).toLocaleString() : "时间未提供"} · {item.round_count ?? item.rounds?.length ?? 0} 轮 · {statusText(item.status)}</small></span><b>›</b></button>)}</div> : <div className="history-empty">完成真实评测后，会保存在本地 SQLite。演示数据不会写入这里。</div>}</section>
        </section>

        <div className="action-dock">
          <div className="dock-state"><span className={`status-indicator ${status}`}><i/></span><div><strong>{isDemo ? "演示模式 · 未调用 API" : isStopping ? "正在停止" : statusText(status)}</strong><small>{isDemo ? "退出演示会恢复进入前的草稿" : stage}</small></div></div>
          <div className="dock-buttons">{busy ? <button type="button" className="stop-button" onClick={stopRun} disabled={isStopping}><Icon kind="stop"/>{isStopping ? "正在停止…" : "停止"}</button> : <button type="submit" className="primary-button" disabled={isDemo || pendingDemo || testState.status === "testing"}><Icon kind={mode === "auto" ? "spark" : "play"}/>{mode === "auto" ? "开始自动优化" : "运行评测"}</button>}</div>
        </div>
      </form>

      {error && <div className="global-error" role="alert"><span>!</span><div><strong>需要处理的问题</strong><p>{error}</p></div><button type="button" aria-label="关闭错误提示" onClick={() => setError("")}><Icon kind="close"/></button></div>}
    </div>

    <ThemeSettings/>
    {notice && <div className="toast-notice" role="status">{notice}</div>}
    {demoConfirm && <div className="dialog-backdrop" role="presentation"><section className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="demo-confirm-title"><div className="dialog-kicker">运行中的任务</div><h2 id="demo-confirm-title">先停止当前评测再进入演示？</h2><p>会向本地后端请求停止。正在进行的供应商请求可能要等响应后才结束；已经完成的轮次会保留在运行历史中。</p><div className="dialog-actions"><button type="button" className="text-action" onClick={() => setDemoConfirm(false)}>继续当前任务</button><button type="button" className="primary-button" onClick={() => { setDemoConfirm(false); setPendingDemo(true); void stopRun(); }}>请求停止并进入演示</button></div></section></div>}
    <footer className="page-footer"><span>PromptLattice <i>v2</i></span><span>本地优先 · API Key 仅在内存中使用</span><a href="https://github.com/jerrystark7/PromptLattice" target="_blank" rel="noreferrer">GitHub <Icon kind="external"/></a></footer>
  </main>;
}
