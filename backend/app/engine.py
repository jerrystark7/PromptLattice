import json
import os
import re
from typing import Any

import httpx

from .providers import build_chat_payload

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = os.getenv("PROMPTLATTICE_JEV_MODEL", "jev-latest")
TIMEOUT = httpx.Timeout(90, connect=15)


class TaskError(Exception):
    def __init__(self, message: str, kind: str = "request_error"):
        super().__init__(message)
        self.kind = kind


def provider_error(status: int, body: str, service: str = "目标模型") -> TaskError:
    try:
        obj = json.loads(body)
        message = str((obj.get("error") or {}).get("message") or obj.get("message") or body)
    except Exception:
        message = body[:400]
    low = message.lower()
    if status in (401, 403):
        return TaskError(f"{service} API Key 无效或没有访问权限，请检查 Key 与账号权限。", "invalid_key")
    if status == 402:
        return TaskError(f"{service} 账户余额不足或需要开通计费，请检查服务商控制台。", "insufficient_balance")
    if status == 429:
        return TaskError("请求限流或额度不足。请稍后重试并检查服务商余额。", "rate_limit")
    if status == 404 or any(s in low for s in ("model not found", "model unavailable", "unknown model")):
        return TaskError("模型 ID 不存在或当前账号无法使用，请检查名称和服务区域。", "model_unavailable")
    if status == 400 and any(s in low for s in ("parameter", "unsupported", "top_k", "max_tokens", "temperature")):
        return TaskError(f"模型拒绝了请求参数：{message[:240]}", "unsupported_parameter")
    if status >= 500:
        return TaskError("模型服务暂时不可用，请稍后重试。", "provider_error")
    return TaskError(f"API 请求失败（HTTP {status}）：{message[:240]}", "request_error")


def usage_record(raw: dict[str, Any]) -> dict[str, int | None]:
    return {"input_tokens": raw.get("prompt_tokens", raw.get("input_tokens")), "output_tokens": raw.get("completion_tokens", raw.get("output_tokens")), "total_tokens": raw.get("total_tokens")}


async def chat(client: httpx.AsyncClient, provider: str, model: str, region: str | None, key: str, messages: list[dict[str, str]], params: dict[str, Any]) -> tuple[str, dict[str, int | None]]:
    url, payload = build_chat_payload(provider, model, region, messages, params)
    try:
        response = await client.post(url, json=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    except httpx.TimeoutException as exc:
        raise TaskError("模型请求超时，请重试或降低输出长度。", "timeout") from exc
    except httpx.RequestError as exc:
        raise TaskError(f"无法连接模型服务：{exc.__class__.__name__}", "connection_error") from exc
    if response.is_error:
        raise provider_error(response.status_code, response.text)
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise TaskError("模型返回格式不符合 Chat Completions 规范。", "invalid_response") from exc
    return content, usage_record(data.get("usage") or {})


def generation_messages(prompt: str, test_input: str, prompt_mode: str) -> list[dict[str, str]]:
    if prompt_mode == "system":
        return [{"role": "system", "content": prompt}, {"role": "user", "content": test_input}]
    user_message = prompt if not test_input.strip() else f"{prompt}\n\n---\n测试输入：\n{test_input}"
    return [{"role": "user", "content": user_message}]


async def jev_assess(client: httpx.AsyncClient, api_key: str, original_prompt: str, prompt: str, prompt_mode: str, test_input: str, expected: str, reference: str, output: str) -> tuple[float, dict[str, int | None]]:
    state = {"original_prompt": original_prompt, "current_prompt": prompt, "prompt_mode": prompt_mode, "test_input": test_input or "（未提供）", "expected_effect": expected, "reference_answer": reference or "（未提供）", "actual_generation": output}
    question_name = "is_satisfactory"
    body = {"model": JEV_MODEL, "state": json.dumps(state, ensure_ascii=False), "questions": {question_name: {"type": "noul", "instructions": "Does actual_generation satisfy expected_effect and all of its explicit constraints? Use reference_answer as evaluation evidence only when supplied. Judge this output only, not future performance. Answer yes only if material requirements are met."}}}
    try:
        response = await client.post(JEV_URL, json=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    except httpx.TimeoutException as exc:
        raise TaskError("Jev 请求超时。", "jev_timeout") from exc
    except httpx.RequestError as exc:
        raise TaskError(f"无法连接 Jev 服务：{exc.__class__.__name__}", "jev_connection_error") from exc
    if response.is_error:
        raise provider_error(response.status_code, response.text, "Jev")
    data = response.json()
    answer = (data.get("answers") or {}).get(question_name) or {}
    probability = answer.get("noul") if isinstance(answer, dict) else answer
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= float(probability) <= 1:
        raise TaskError("Jev 响应缺少有效的 answers.<name>.noul 字段，本轮不生成虚假分数。", "jev_invalid_response")
    raw_usage = data.get("usage") or {}
    return float(probability), {"input_tokens": raw_usage.get("input_tokens"), "output_tokens": raw_usage.get("output_tokens"), "total_tokens": raw_usage.get("total_tokens")}


def parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("没有 JSON 对象")
    result = json.loads(cleaned[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("输出不是对象")
    return result


def model_config(config: dict[str, Any]) -> tuple[str, str, str | None, str, dict[str, Any]]:
    return config["provider"], config["model"], config.get("region"), config["target_api_key"], config["params"]


async def structured_call(client: httpx.AsyncClient, config: dict[str, Any], messages: list[dict[str, str]], required: dict[str, type]) -> tuple[dict[str, Any], list[dict[str, int | None]]]:
    provider, model, region, key, params = model_config(config)
    answer, usage = await chat(client, provider, model, region, key, messages, params)
    usages = [usage]
    for attempt in range(2):
        try:
            result = parse_json(answer)
            for field, value_type in required.items():
                value = result.get(field)
                valid = isinstance(value, value_type)
                if value_type is int and isinstance(value, (float, int)) and not isinstance(value, bool):
                    valid = True
                if field not in result or not valid:
                    raise ValueError(f"字段 {field} 类型无效")
            return result, usages
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt:
                break
            retry_messages = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": f"上次 JSON 无法解析或字段错误（{exc}）。请修复，并只返回满足原要求的 JSON 对象。"}]
            answer, usage = await chat(client, provider, model, region, key, retry_messages, params)
            usages.append(usage)
    raise TaskError("模型连续两次未返回有效 JSON，请确认模型支持按指令输出结构化内容。", "invalid_json")


async def evaluate(client: httpx.AsyncClient, config: dict[str, Any], prompt: str, test_input: str, expected: str, reference: str, output: str) -> tuple[dict[str, Any], list[dict[str, int | None]]]:
    content = json.dumps({"expected_effect": expected, "reference_answer": reference or None, "test_input": test_input or None, "actual_output": output}, ensure_ascii=False)
    result, usage = await structured_call(client, config, [
        {"role": "system", "content": "Independently assess this output. The reference answer is evaluation evidence only. Do not infer facts. Do not use any external score. Return JSON only: score integer 0-100, strengths array of strings, issues array of strings, missing_requirements array of strings."},
        {"role": "user", "content": content},
    ], {"score": int, "strengths": list, "issues": list, "missing_requirements": list})
    result["score"] = max(0, min(100, round(result["score"])))
    for field in ("strengths", "issues", "missing_requirements"):
        result[field] = [str(x) for x in result[field]]
    return result, usage


async def improve(client: httpx.AsyncClient, config: dict[str, Any], prompt: str, test_input: str, expected: str, reference: str, output: str, probability: float, evaluation: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, int | None]]]:
    content = {"current_prompt": prompt, "test_input": test_input or None, "expected_effect": expected, "reference_answer_for_evaluation_only": reference or None, "actual_output": output, "jev_probability_for_this_output": probability, "model_evaluation": evaluation}
    result, usage = await structured_call(client, config, [
        {"role": "system", "content": "Improve the prompt, preserving intent and constraints. Do not copy the reference answer into the new prompt or overfit one sample. Do not invent facts. Return JSON only with advice (string), reason (string), and new_prompt (string). new_prompt must be a complete prompt, not a diff."},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
    ], {"advice": str, "reason": str, "new_prompt": str})
    if not result["new_prompt"].strip():
        raise TaskError("模型返回的新提示词为空。", "invalid_prompt")
    return {k: result[k].strip() for k in ("advice", "reason", "new_prompt")}, usage
