import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import storage
from .engine import TIMEOUT, TaskError, chat, evaluate, generation_messages, improve, jev_assess
from .providers import PROVIDERS, capabilities, provider_for

STOP_FLAGS: dict[str, asyncio.Event] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.initialize()
    yield


app = FastAPI(title="PromptLattice API", version="1.0.0", lifespan=lifespan)
cors_origins = [origin.strip() for origin in os.getenv("PROMPTLATTICE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type"])


class ModelSettings(BaseModel):
    provider: str
    model: str = Field(min_length=1, max_length=160)
    region: str | None = None
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=1, ge=0, le=1)
    top_k: int | None = Field(default=40, ge=1, le=1000)
    max_tokens: int | None = Field(default=2048, ge=1, le=100000)
    thinking: str = "default"


class HistoryParameters(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    thinking: str | None = None


class HistoryEvaluation(BaseModel):
    score: int | None = Field(default=None, ge=0, le=100)
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    hard_constraint_failures: list[str] = Field(default_factory=list, max_length=8)
    strengths: list[str] = Field(default_factory=list, max_length=8)
    issues: list[str] = Field(default_factory=list, max_length=8)
    missing_requirements: list[str] = Field(default_factory=list, max_length=8)


class OptimizationHistoryRound(BaseModel):
    source_run_id: str = Field(default="", max_length=80)
    round_no: int = Field(ge=1, le=100)
    status: str = Field(default="completed", max_length=40)
    prompt: str = Field(default="", max_length=1800)
    prompt_mode: str = "user"
    test_input: str = Field(default="", max_length=1200)
    expected: str = Field(default="", max_length=1200)
    reference: str = Field(default="", max_length=600)
    parameters: HistoryParameters = Field(default_factory=HistoryParameters)
    output: str = Field(default="", max_length=1400)
    jev_probability: float | None = Field(default=None, ge=0, le=1)
    jev_checks: dict[str, float | None] = Field(default_factory=dict)
    evaluation: HistoryEvaluation = Field(default_factory=HistoryEvaluation)
    advice: str = Field(default="", max_length=500)
    reason: str = Field(default="", max_length=500)
    next_prompt: str = Field(default="", max_length=1800)
    parameter_advice: str = Field(default="", max_length=500)
    parameter_reason: str = Field(default="", max_length=500)
    next_parameters: HistoryParameters = Field(default_factory=HistoryParameters)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=30000)
    prompt_mode: str = "user"
    expected: str = Field(min_length=1, max_length=10000)
    test_input: str = Field(default="", max_length=20000)
    reference: str = Field(default="", max_length=20000)
    model: ModelSettings
    target_api_key: str = Field(min_length=1)
    jev_api_key: str = Field(min_length=1)
    auto_optimize: bool = False
    threshold: int = Field(default=85, ge=0, le=100)
    max_rounds: int = Field(default=15, ge=1, le=15)
    optimization_history: list[OptimizationHistoryRound] = Field(default_factory=list, max_length=15)

    def clean_config(self) -> dict[str, Any]:
        return {"provider": self.model.provider, "model": self.model.model, "region": self.model.region, "params": {k: getattr(self.model, k) for k in ("temperature", "top_p", "top_k", "max_tokens", "thinking")}}

    def target_config(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        config = self.clean_config()
        if params is not None:
            config["params"] = dict(params)
        return {**config, "target_api_key": self.target_api_key}


class ConnectionRequest(BaseModel):
    model: ModelSettings
    target_api_key: str = Field(min_length=1)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/providers")
async def provider_list() -> dict[str, Any]:
    return {key: {region: {"label": value.label, "base_url": value.base_url, "models": list(value.models)} for region, value in regions.items()} for key, regions in PROVIDERS.items()}


@app.get("/api/capabilities")
async def get_capabilities(provider: str, model: str, region: str | None = None, thinking: str = "default") -> dict[str, Any]:
    try:
        return capabilities(provider, model, region, thinking)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/test-connection")
async def test_connection(request: ConnectionRequest) -> dict[str, Any]:
    try:
        provider_for(request.model.provider, request.model.region)
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # DeepSeek defaults to thinking mode; disable it for this short
            # connectivity probe so a tiny token budget cannot be consumed by
            # reasoning without producing a user-facing answer.
            probe_params = {"temperature": 0, "max_tokens": 64}
            if request.model.provider == "deepseek":
                probe_params["thinking"] = "disabled"
            reply, usage = await chat(client, request.model.provider, request.model.model, request.model.region, request.target_api_key, [{"role": "user", "content": "Reply with the single word: connected"}], probe_params)
        return {"ok": True, "reply": reply[:100], "usage": usage}
    except ValueError as exc:
        raise HTTPException(400, detail={"kind": "invalid_provider", "message": str(exc)}) from exc
    except TaskError as exc:
        raise HTTPException(400, detail={"kind": exc.kind, "message": str(exc)}) from exc


@app.post("/api/runs/reconcile")
async def reconcile_runs() -> dict[str, Any]:
    """Stop in-process runs that a refreshed page cannot reattach to."""
    running = [item for item in storage.list_runs() if item.get("status") == "running"]
    run_ids = []
    for item in running:
        run_id = item["id"]
        flag = STOP_FLAGS.get(run_id)
        if flag:
            flag.set()
            storage.request_stop(run_id, "正在停止：页面刷新后无法恢复 SSE；当前供应商请求结束后会收敛状态。")
        else:
            storage.interrupt_run(run_id, "页面刷新后无法恢复原运行；服务端任务已中断，已完成结果已保留。")
        run_ids.append(run_id)
    return {"reconciled": run_ids}


@app.get("/api/runs")
async def run_list() -> list[dict[str, Any]]:
    return storage.list_runs()


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str) -> dict[str, Any]:
    result = storage.get_run(run_id)
    if not result:
        raise HTTPException(404, "运行记录不存在")
    return result


@app.delete("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, bool]:
    flag = STOP_FLAGS.get(run_id)
    if flag:
        flag.set()
        return {"stopping": True}
    return {"stopping": False}


def sse(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _history_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) <= limit:
        return text
    half = max(1, (limit - 28) // 2)
    return f"{text[:half]}\n…历史文本中间已省略…\n{text[-half:]}"


def _history_snapshot(detail: dict[str, Any], source_run_id: str) -> dict[str, Any]:
    evaluation = detail.get("evaluation") or {}
    return {
        "source_run_id": source_run_id,
        "round_no": detail.get("round_no", 1),
        "status": detail.get("status", "completed"),
        "prompt": _history_text(detail.get("prompt"), 1800),
        "prompt_mode": detail.get("prompt_mode", "user"),
        "test_input": _history_text(detail.get("test_input"), 1200),
        "expected": _history_text(detail.get("expected"), 1200),
        "reference": _history_text(detail.get("reference"), 600),
        "parameters": {key: (detail.get("parameters") or {}).get(key) for key in ("temperature", "top_p", "top_k", "max_tokens", "thinking")},
        "output": _history_text(detail.get("output"), 1400),
        "jev_probability": detail.get("jev_probability"),
        "jev_checks": {key: (detail.get("jev_checks") or {}).get(key) for key in ("requirements_met", "format_and_length_met", "grounded_in_input", "style_and_tone_met")},
        "evaluation": {
            "score": evaluation.get("score"),
            "criterion_scores": evaluation.get("criterion_scores", {}),
            "hard_constraint_failures": [_history_text(item, 240) for item in evaluation.get("hard_constraint_failures", [])[:8]],
            "strengths": [_history_text(item, 240) for item in evaluation.get("strengths", [])[:8]],
            "issues": [_history_text(item, 240) for item in evaluation.get("issues", [])[:8]],
            "missing_requirements": [_history_text(item, 240) for item in evaluation.get("missing_requirements", [])[:8]],
        },
        "advice": _history_text(detail.get("advice"), 500),
        "reason": _history_text(detail.get("reason"), 500),
        "next_prompt": _history_text(detail.get("next_prompt"), 1800),
        "parameter_advice": _history_text(detail.get("parameter_advice"), 500),
        "parameter_reason": _history_text(detail.get("parameter_reason"), 500),
        "next_parameters": {key: (detail.get("next_parameters") or {}).get(key) for key in ("temperature", "top_p", "top_k", "max_tokens", "thinking")},
    }


@app.post("/api/runs/stream")
async def stream_run(request: RunRequest) -> StreamingResponse:
    if request.prompt_mode not in ("user", "system"):
        raise HTTPException(422, "提示词使用方式无效")
    if request.prompt_mode == "system" and not request.test_input.strip():
        raise HTTPException(422, "系统提示词模式需要填写测试输入")
    try:
        provider_for(request.model.provider, request.model.region)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    run_id = str(uuid.uuid4())
    stop_flag = asyncio.Event()
    STOP_FLAGS[run_id] = stop_flag
    is_auto = request.auto_optimize
    stored_config = {
        **request.clean_config(),
        "prompt_mode": request.prompt_mode,
        "test_input": request.test_input,
        "expected": request.expected,
        "reference": request.reference,
        "threshold": request.threshold,
        "max_rounds": request.max_rounds if is_auto else 1,
        "auto_optimize": is_auto,
        "optimization_history": [item.model_dump(exclude_none=True) for item in request.optimization_history][-15:],
    }
    storage.create_run(run_id, "auto" if is_auto else "manual", request.prompt, stored_config)

    async def generate():
        current_prompt = request.prompt
        current_params = request.clean_config()["params"]
        optimization_history = [item.model_dump(exclude_none=True) for item in request.optimization_history][-15:]
        seen = {current_prompt.strip()}
        detail: dict[str, Any] | None = None
        try:
            yield sse("started", {"run_id": run_id, "mode": "auto" if is_auto else "manual", "max_rounds": request.max_rounds})
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                for round_no in range(1, request.max_rounds + 1):
                    if stop_flag.is_set():
                        storage.finish_run(run_id, "stopped", "用户手动停止")
                        yield sse("finished", {"run_id": run_id, "status": "stopped", "stop_reason": "用户手动停止"})
                        return
                    started = time.monotonic()
                    round_capabilities = capabilities(request.model.provider, request.model.model, request.model.region, current_params.get("thinking", "default"))
                    detail = {
                        "round_no": round_no, "status": "running", "prompt": current_prompt,
                        "prompt_mode": request.prompt_mode,
                        "test_input": request.test_input, "expected": request.expected, "reference": request.reference,
                        "provider": request.model.provider, "model": request.model.model, "region": request.model.region,
                        "parameters": dict(current_params),
                        "parameter_support": round_capabilities["parameters"],
                        "thinking_support": round_capabilities["thinking"],
                        "threshold": request.threshold, "max_rounds": request.max_rounds if is_auto else 1,
                        "auto_optimize": is_auto,
                        "optimization_history_rounds": len(optimization_history),
                        "usage": {"generation": None, "jev": None, "evaluation": None, "optimization": None},
                    }
                    detail["duration_ms"] = round((time.monotonic() - started) * 1000)
                    storage.save_round(run_id, round_no, detail)
                    try:
                        yield sse("stage", {"run_id": run_id, "round_no": round_no, "stage": "generation", "message": f"第 {round_no} 轮：正在生成内容"})
                        output, detail["usage"]["generation"] = await chat(client, request.model.provider, request.model.model, request.model.region, request.target_api_key, generation_messages(current_prompt, request.test_input, request.prompt_mode), current_params)
                        detail["output"] = output
                        detail["duration_ms"] = round((time.monotonic() - started) * 1000)
                        storage.save_round(run_id, round_no, detail)
                        yield sse("round", {"run_id": run_id, "round": detail})
                        if stop_flag.is_set():
                            detail.update({"status": "stopped", "stop_reason": "用户手动停止（当前请求已完成）"})
                            detail["duration_ms"] = round((time.monotonic() - started) * 1000)
                            storage.save_round(run_id, round_no, detail)
                            storage.finish_run(run_id, "stopped", detail["stop_reason"])
                            yield sse("round", {"run_id": run_id, "round": detail})
                            yield sse("finished", {"run_id": run_id, "status": "stopped", "stop_reason": detail["stop_reason"]})
                            return

                        yield sse("stage", {"run_id": run_id, "round_no": round_no, "stage": "jev", "message": "Jev 正在评估本次输出"})
                        jev_probability, jev_checks, detail["usage"]["jev"] = await jev_assess(client, request.jev_api_key, request.prompt, current_prompt, request.prompt_mode, request.test_input, request.expected, request.reference, output)
                        detail["jev_probability"] = jev_probability
                        detail["jev_checks"] = jev_checks
                        detail["duration_ms"] = round((time.monotonic() - started) * 1000)
                        storage.save_round(run_id, round_no, detail)
                        yield sse("round", {"run_id": run_id, "round": detail})
                        if stop_flag.is_set():
                            detail.update({"status": "stopped", "stop_reason": "用户手动停止（本次 Jev 请求完成后停止）"})
                        else:
                            yield sse("stage", {"run_id": run_id, "round_no": round_no, "stage": "evaluation", "message": "目标模型正在独立评分"})
                            model_result, detail["usage"]["evaluation"] = await evaluate(client, request.target_config(current_params), current_prompt, request.prompt_mode, request.test_input, request.expected, request.reference, output)
                            detail["evaluation"] = model_result
                            detail["duration_ms"] = round((time.monotonic() - started) * 1000)
                            storage.save_round(run_id, round_no, detail)
                            yield sse("round", {"run_id": run_id, "round": detail})
                            if stop_flag.is_set():
                                detail.update({"status": "stopped", "stop_reason": "用户手动停止（当前评分请求完成后停止）"})

                        if detail["status"] != "stopped":
                            reached = jev_probability * 100 >= request.threshold and model_result["score"] >= request.threshold
                            if is_auto and reached:
                                detail.update({"status": "target_reached", "stop_reason": "Jev 与模型评分均达到目标阈值"})
                            elif is_auto and round_no >= request.max_rounds:
                                detail.update({"status": "max_rounds", "stop_reason": "达到最大运行轮数"})
                            elif stop_flag.is_set():
                                detail.update({"status": "stopped", "stop_reason": "用户手动停止"})
                            else:
                                yield sse("stage", {"run_id": run_id, "round_no": round_no, "stage": "optimization", "message": "目标模型正在生成优化建议"})
                                recommendation, detail["usage"]["optimization"] = await improve(
                                    client, request.target_config(current_params), current_prompt, request.prompt_mode,
                                    request.test_input, request.expected, request.reference, output, jev_probability,
                                    model_result, round_capabilities, list(optimization_history),
                                )
                                detail.update({
                                    "advice": recommendation["advice"],
                                    "reason": recommendation["reason"],
                                    "next_prompt": recommendation["new_prompt"],
                                    "parameter_advice": recommendation["parameter_advice"],
                                    "parameter_reason": recommendation["parameter_reason"],
                                    "next_parameters": recommendation["next_parameters"],
                                    "parameter_warnings": recommendation["parameter_warnings"],
                                })
                                if stop_flag.is_set():
                                    detail.update({"status": "stopped", "stop_reason": "用户手动停止（当前改写请求完成后未启动下一轮）"})
                                elif is_auto and recommendation["new_prompt"].strip() in seen:
                                    detail.update({"status": "repeated_prompt", "stop_reason": "模型生成了已出现过的提示词，自动优化已停止"})
                                else:
                                    detail["status"] = "completed"

                        detail["duration_ms"] = round((time.monotonic() - started) * 1000)
                        storage.save_round(run_id, round_no, detail)
                        yield sse("round", {"run_id": run_id, "round": detail})
                        if detail["status"] in ("target_reached", "max_rounds", "stopped", "repeated_prompt"):
                            storage.finish_run(run_id, detail["status"], detail.get("stop_reason", ""))
                            yield sse("finished", {"run_id": run_id, "status": detail["status"], "stop_reason": detail.get("stop_reason", "")})
                            return
                        if is_auto:
                            optimization_history.append(_history_snapshot(detail, run_id))
                            optimization_history = optimization_history[-15:]
                            current_prompt = detail["next_prompt"]
                            current_params = dict(detail["next_parameters"])
                            seen.add(current_prompt.strip())
                    except TaskError as exc:
                        detail.update({"status": "failed", "error": {"kind": exc.kind, "message": str(exc)}, "duration_ms": round((time.monotonic() - started) * 1000)})
                        storage.save_round(run_id, round_no, detail)
                        yield sse("round", {"run_id": run_id, "round": detail})
                        storage.finish_run(run_id, "failed", str(exc))
                        yield sse("error", {"run_id": run_id, "kind": exc.kind, "message": str(exc)})
                        yield sse("finished", {"run_id": run_id, "status": "failed", "stop_reason": str(exc)})
                        return
                    except Exception as exc:
                        message = f"本轮处理失败：{exc.__class__.__name__}。已保存此前完成的部分结果。"
                        detail.update({"status": "failed", "error": {"kind": "internal_error", "message": message}, "duration_ms": round((time.monotonic() - started) * 1000)})
                        storage.save_round(run_id, round_no, detail)
                        yield sse("round", {"run_id": run_id, "round": detail})
                        storage.finish_run(run_id, "failed", message)
                        yield sse("error", {"run_id": run_id, "kind": "internal_error", "message": message})
                        yield sse("finished", {"run_id": run_id, "status": "failed", "stop_reason": message})
                        return
            storage.finish_run(run_id, "completed", "完成")
            yield sse("finished", {"run_id": run_id, "status": "completed", "stop_reason": "完成"})
        except asyncio.CancelledError:
            stop_flag.set()
            if detail is not None:
                detail.update({"status": "interrupted", "stop_reason": "客户端连接中断；当前任务无法恢复。"})
                detail["duration_ms"] = round((time.monotonic() - started) * 1000) if "started" in locals() else None
                storage.save_round(run_id, detail["round_no"], detail)
            storage.finish_run(run_id, "interrupted", "客户端连接中断；当前任务无法恢复。")
            raise
        except Exception as exc:
            storage.finish_run(run_id, "failed", "服务端处理失败")
            yield sse("error", {"run_id": run_id, "kind": "server_error", "message": f"服务端处理失败：{exc.__class__.__name__}"})
            yield sse("finished", {"run_id": run_id, "status": "failed", "stop_reason": "服务端处理失败"})
        finally:
            STOP_FLAGS.pop(run_id, None)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
