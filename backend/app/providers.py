from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    label: str
    base_url: str
    models: tuple[str, ...]


PROVIDERS: dict[str, dict[str, ProviderConfig]] = {
    "deepseek": {
        "default": ProviderConfig("DeepSeek", "https://api.deepseek.com", ("deepseek-flash", "deepseek-v4-pro")),
    },
    "qwen": {
        "cn": ProviderConfig("通义千问 · 中国内地", "https://dashscope.aliyuncs.com/compatible-mode/v1", ("qwen3.8-max", "qwen3.8-flash", "qwen-plus", "qwen-max")),
        "international": ProviderConfig("通义千问 · 国际站", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", ("qwen3.8-max", "qwen3.8-flash", "qwen-plus", "qwen-max")),
    },
    "moonshot": {
        "cn": ProviderConfig("Moonshot · 中国内地", "https://api.moonshot.cn/v1", ("kimi-k3.1-pro", "kimi-k3", "kimi-k2.6", "kimi-k2.5")),
        "international": ProviderConfig("Moonshot · 国际站", "https://api.moonshot.ai/v1", ("kimi-k3.1-pro", "kimi-k3", "kimi-k2.6", "kimi-k2.5")),
    },
    "openai": {
        "default": ProviderConfig("OpenAI", "https://api.openai.com/v1", ("gpt-4.1-mini", "gpt-4.1", "gpt-5.4")),
    },
}


def provider_for(provider: str, region: str | None) -> ProviderConfig:
    regions = PROVIDERS.get(provider)
    if not regions:
        raise ValueError("不支持的模型服务商")
    region_key = region or ("default" if "default" in regions else "cn")
    if region_key not in regions:
        raise ValueError("所选服务区域无效")
    return regions[region_key]


def _capability(supported: bool, reason: str = "", api_parameter: str = "", warning: str = "", minimum: float | int | None = None, maximum: float | int | None = None) -> dict[str, Any]:
    return {"supported": supported, "reason": reason, "api_parameter": api_parameter, "warning": warning, "minimum": minimum, "maximum": maximum}


_QWEN_TOP_K_MODELS = {"qwen3.8-max", "qwen3.8-flash", "qwen-plus", "qwen-max", "qwen-turbo"}


def _is_qwen_model(model: str) -> bool:
    # Top-k is a DashScope extension with model-level support. Custom IDs are
    # deliberately excluded until the user selects a documented model ID.
    return model.lower() in _QWEN_TOP_K_MODELS


def _qwen_mixed_thinking(model: str) -> bool:
    value = model.lower()
    return value.startswith(("qwen3.8", "qwen3.7", "qwen3.6", "qwen3.5", "qwen3-", "qwen3_", "qwen3-vl", "qwq-"))


def _openai_reasoning_model(model: str) -> bool:
    value = model.lower()
    return value.startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))


def capabilities(provider: str, model: str, region: str | None, thinking: str = "default") -> dict[str, Any]:
    config = provider_for(provider, region)
    normalized = model.lower()
    warning = ""
    thinking_control: dict[str, Any] = {"available": False, "selected": "default", "choices": []}
    max_parameter = "max_tokens"
    params: dict[str, dict[str, Any]] = {
        "temperature": _capability(True, api_parameter="temperature"),
        "top_p": _capability(True, api_parameter="top_p"),
        "top_k": _capability(False, "此接口或模型未确认支持 Top-k；平台不会发送。"),
        "max_tokens": _capability(True, api_parameter="max_tokens"),
    }

    if provider == "deepseek":
        if normalized in {"deepseek-flash", "deepseek-v4-pro"}:
            thinking_control = {"available": True, "selected": thinking if thinking in ("enabled", "disabled") else "enabled", "choices": ["enabled", "disabled"]}
            mode = thinking_control["selected"]
            if mode == "enabled":
                params["temperature"] = _capability(False, "DeepSeek 思考模式不使用 Temperature；平台不会发送。")
                if normalized == "deepseek-flash":
                    params["top_p"] = _capability(True, api_parameter="top_p", warning="deepseek-flash 思考模式的有效范围为 0.95–1.0。", minimum=0.95, maximum=1)
                else:
                    params["top_p"] = _capability(False, "deepseek-v4-pro 思考模式且未启用工具调用时会忽略 Top-p；平台不会发送。")
            elif normalized == "deepseek-flash":
                params["top_p"] = _capability(False, "deepseek-flash 非思考模式固定使用 Top-p 1.0；平台不会发送。")
        else:
            warning = "该 DeepSeek 模型 ID 不在当前官方模型示例中；通用参数可能因模型模式而被忽略或拒绝。"
    elif provider == "qwen":
        if _is_qwen_model(normalized):
            params["top_k"] = _capability(True, api_parameter="top_k", warning="Top-k 是 DashScope 扩展参数，不属于标准 OpenAI 参数；具体支持范围取决于模型 ID。")
        else:
            warning = "自定义模型 ID 未匹配到已核对的 Qwen 模型族，Top-k 不会发送。"
        if _qwen_mixed_thinking(normalized):
            thinking_control = {"available": True, "selected": thinking if thinking in ("enabled", "disabled", "default") else "default", "choices": ["default", "enabled", "disabled"]}
            if thinking_control["selected"] == "enabled":
                warning = "Qwen 思考模式下部分模型会限制 Temperature 或将 Top-p 调整至至少 0.95；请以所选具体模型文档为准。"
                if normalized.startswith(("qwen3.8", "qwen3.7", "qwen3.6", "qwen3.5", "qwen3-")):
                    params["top_p"]["minimum"] = 0.95
        elif not _is_qwen_model(normalized):
            warning = "自定义模型 ID 的思考模式与采样参数行为未知；只发送 OpenAI 兼容接口基础参数。"
    elif provider == "moonshot":
        # The Moonshot Chat API documents OpenAI-compatible sampling controls,
        # but does not document top_k. Its current API marks max_tokens deprecated.
        max_parameter = "max_completion_tokens"
        params["max_tokens"] = _capability(True, api_parameter=max_parameter)
        if normalized.startswith(("kimi-k3", "kimi/k3")):
            warning = "Kimi K3 系列固定启用思考模式；Temperature / Top-p 的具体效果随模型版本而异。"
        elif not normalized.startswith(("kimi-", "kimi/")):
            warning = "自定义模型 ID 的参数行为未知；Top-k 不会发送，最大输出长度使用 max_completion_tokens。"
    elif provider == "openai":
        max_parameter = "max_completion_tokens"
        params["max_tokens"] = _capability(True, api_parameter=max_parameter)
        if _openai_reasoning_model(normalized):
            params["temperature"] = _capability(False, "所选 OpenAI 推理模型在当前未配置 reasoning_effort=none；平台不会发送 Temperature。")
            params["top_p"] = _capability(False, "所选 OpenAI 推理模型在当前未配置 reasoning_effort=none；平台不会发送 Top-p。")
            warning = "推理模型使用 max_completion_tokens；Temperature 与 Top-p 已禁用，避免参数不兼容。"
        elif not normalized.startswith(("gpt-4", "chatgpt-")):
            warning = "自定义模型 ID 的参数行为未知；平台按 Chat Completions 接口使用 max_completion_tokens。"

    params["max_tokens"]["reason"] = f"最大输出长度通过 {max_parameter} 发送。"
    warnings = [warning] if warning else []
    warnings.extend(item["warning"] for item in params.values() if item.get("warning"))
    return {
        "provider": provider,
        "model": model,
        "thinking": thinking_control,
        "parameters": params,
        "warnings": list(dict.fromkeys(warnings)),
    }


def build_chat_payload(provider: str, model: str, region: str | None, messages: list[dict[str, str]], values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = provider_for(provider, region)
    settings = capabilities(provider, model, region, str(values.get("thinking") or "default"))
    supported = settings["parameters"]
    payload: dict[str, Any] = {"model": model, "messages": messages}
    for name in ("temperature", "top_p", "top_k", "max_tokens"):
        value = values.get(name)
        if value is not None and supported[name]["supported"]:
            minimum, maximum = supported[name].get("minimum"), supported[name].get("maximum")
            if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
                bounds = f"{minimum if minimum is not None else '无下限'}–{maximum if maximum is not None else '无上限'}"
                raise ValueError(f"{name}={value} 超出 {model} 当前模式支持范围（{bounds}）")
            api_name = supported[name]["api_parameter"] or name
            payload[api_name] = value

    mode = values.get("thinking")
    if provider == "deepseek" and mode in ("enabled", "disabled") and settings["thinking"]["available"]:
        # Raw HTTP requests use the same top-level OpenAI-compatible field the
        # official DeepSeek SDK exposes through extra_body.
        payload["thinking"] = {"type": mode}
    elif provider == "qwen" and mode in ("enabled", "disabled") and settings["thinking"]["available"]:
        payload["enable_thinking"] = mode == "enabled"

    return f"{config.base_url}/chat/completions", payload
