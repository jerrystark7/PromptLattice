from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderConfig:
    label: str
    base_url: str
    models: tuple[str, ...]
    top_k: bool = False


PROVIDERS: dict[str, dict[str, ProviderConfig]] = {
    "deepseek": {"default": ProviderConfig("DeepSeek", "https://api.deepseek.com", ("deepseek-flash", "deepseek-v4-pro"))},
    "qwen": {
        "cn": ProviderConfig("通义千问 · 中国内地", "https://dashscope.aliyuncs.com/compatible-mode/v1", ("qwen-plus", "qwen-max", "qwen-turbo", "qwen3.5-plus"), True),
        "international": ProviderConfig("通义千问 · 国际站", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1", ("qwen-plus", "qwen-max", "qwen-turbo"), True),
    },
    "moonshot": {
        "cn": ProviderConfig("Moonshot · 中国内地", "https://api.moonshot.cn/v1", ("kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k")),
        "international": ProviderConfig("Moonshot · 国际站", "https://api.moonshot.ai/v1", ("kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k")),
    },
    "openai": {"default": ProviderConfig("OpenAI", "https://api.openai.com/v1", ("gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"))},
}


def provider_for(provider: str, region: str | None) -> ProviderConfig:
    regions = PROVIDERS.get(provider)
    if not regions:
        raise ValueError("不支持的模型服务商")
    region_key = region or ("default" if "default" in regions else "cn")
    if region_key not in regions:
        raise ValueError("所选服务区域无效")
    return regions[region_key]


def capabilities(provider: str, model: str, region: str | None) -> dict[str, Any]:
    config = provider_for(provider, region)
    unsupported_topk = "该服务商接口未说明支持 Top-k；平台不会发送此参数。"
    params = {
        "temperature": {"supported": True, "reason": ""},
        "top_p": {"supported": True, "reason": ""},
        "top_k": {"supported": config.top_k, "reason": "" if config.top_k else unsupported_topk},
        "max_tokens": {"supported": True, "reason": "通过 Chat Completions 的 max_tokens 发送。"},
    }
    warnings: list[str] = []
    normalized = model.lower()
    if provider == "deepseek" and ("reasoner" in normalized or "thinking" in normalized):
        for name in ("temperature", "top_p"):
            params[name] = {"supported": False, "reason": "当前推理模式不接受此采样参数，平台不会发送。"}
    if provider == "qwen" and ("qwen3" in normalized or "qwen3.5" in normalized):
        warnings.append("部分 Qwen 推理模式可能忽略采样参数，具体行为取决于所选模型。")
    if provider == "openai" and normalized.startswith(("o1", "o3", "o4")):
        for name in ("temperature", "top_p"):
            params[name] = {"supported": False, "reason": "此 OpenAI 推理模型不支持该采样参数。"}
        warnings.append("部分 OpenAI 推理模型使用 max_completion_tokens 语义；若接口拒绝 max_tokens，请更换模型或调整服务端配置。")
    return {"provider": provider, "model": model, "parameters": params, "warnings": warnings}


def build_chat_payload(provider: str, model: str, region: str | None, messages: list[dict[str, str]], values: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = provider_for(provider, region)
    supported = capabilities(provider, model, region)["parameters"]
    payload: dict[str, Any] = {"model": model, "messages": messages}
    for name in ("temperature", "top_p", "top_k", "max_tokens"):
        if values.get(name) is not None and supported[name]["supported"]:
            payload[name] = values[name]
    return f"{config.base_url}/chat/completions", payload
