# PromptLattice

Prompt Eval tool

PromptLattice 是一个本地优先的提示词评测与迭代工作台。使用真实模型生成内容，由 TypeSafe Jev 和所选模型分别评估，再生成可直接采用的新提示词。自动模式通过 SSE 显示进度并将完成轮次保存到 SQLite。

## 环境要求

- Python 3.10 或更高版本
- Node.js 20.9 或更高版本、npm
- DeepSeek、Qwen、Moonshot 或 OpenAI 的 API Key，以及 TypeSafe Jev API Key，才能运行真实评测

## 本地启动

在项目根目录打开两个终端。

终端一：

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

终端二：

```bash
cd frontend
npm install
npm run dev
```

打开 <http://localhost:3000>。服务端 API 文档位于 <http://localhost:8000/docs>。运行历史默认写入 `backend/data/promptlattice.db`，该目录已加入 `.gitignore`。可用环境变量 `PROMPTLATTICE_DB` 指定数据库路径、`PROMPTLATTICE_JEV_MODEL` 设置 Jev 模型（默认 `jev-latest`）、`PROMPTLATTICE_CORS_ORIGINS` 设置逗号分隔的前端来源白名单。前端默认连接 `http://localhost:8000`，也可设置 `NEXT_PUBLIC_API_BASE_URL` 指定后端地址。

## 支持的模型

| 服务商 | 区域 | 默认模型示例 | API 入口 | 参数说明 |
| --- | --- | --- | --- | --- |
| DeepSeek | 默认 | `deepseek-flash`、`deepseek-v4-pro` | `api.deepseek.com` | Temperature、Top-p、最大输出 Token；不发送 Top-k |
| 通义千问 Qwen | 中国内地 / 国际站 | `qwen-plus`、`qwen-max`、`qwen-turbo` | DashScope 兼容接口 | 支持 Temperature、Top-p、Top-k、最大输出 Token；推理模型可能忽略采样参数 |
| Moonshot / Kimi | 中国内地 / 国际站 | `kimi-k2.5`、`moonshot-v1-8k`、`moonshot-v1-32k` | `.cn` / `.ai` 区域入口 | Temperature、Top-p、最大输出 Token；不发送 Top-k |
| OpenAI GPT | 默认 | `gpt-4.1-mini`、`gpt-4.1`、`gpt-4o-mini` | `api.openai.com` | Temperature、Top-p、最大输出 Token；不发送 Top-k |

模型选项只是便于上手的示例。服务商列表可能调整，模型 ID 输入框允许手动填写；能否使用最终取决于服务商账号、地区和模型权限。参数能力在后端集中管理；不支持的参数会禁用且不发送。服务商可能忽略特定模型的参数，界面会对已知推理模式显示提示。连接测试会单独为 DeepSeek 关闭思考模式并留出足够的短回复 Token；正常评测遵从模型默认模式。若模型只返回推理内容而没有最终回答，服务端会提示增加最大输出 Token，不会把推理文本当作生成结果。

## API Key 与数据处理

- 模型 API Key 和 Jev Key 只在页面内存中暂存，并经 HTTPS 部署时的 POST 请求发往自己的后端；不会放入 URL、SQLite、服务端应用日志或导出文件。服务关闭或页面刷新后需重新输入。
- 服务端评测期间将提示词、测试输入、期望效果、参考答案和生成内容发送给用户所选的模型服务商或 Jev，以完成对应步骤。
- **生产环境必须使用 HTTPS**，并限制后端可访问来源及网络出口。第一版面向本地单用户部署，不包含账号、云同步或多租户隔离。
- 测试连接会向所选模型发送一条短请求，可能产生少量 API 费用。自动优化会重复调用目标模型和 Jev；费用由各服务商按其价格收取，本项目不估算费用。

## Jev 评估含义

每轮使用 TypeSafe 官方 `POST https://api.typesafe.ai/v1/systemone`，通过 Noul 问题判断本轮输出是否满足期望效果与明确约束。读取 TypeSafe 响应的 `answers.<question-name>.noul` 概率，界面将其显示为 **Jev 达标判断概率**。这代表 Jev 对本次生成内容的判断，不是提示词未来生成成功的真实统计概率。模型输出和评估都可能有误，只适合作为调试参考。Jev 不生成自然语言修改建议；独立评分和提示词优化由目标模型完成。

## 自动优化和记录

- 第 1 轮使用当前提示词；最大轮数包含第 1 轮，范围 1–15。
- 每轮生成后调用 Jev 和目标模型独立评分。两项均达到阈值时停止；达到最大轮数、手动停止、重复提示词或 API 失败时也会停止并记录原因。
- 每轮结束即写入 SQLite，保存提示词、输入、期望、输出、评估、建议、用量和耗时。API 未返回 Token 用量时会显示“未提供”。API Key 从记录结构中过滤，不会进入数据库或 JSON / Markdown 导出。
- 停止请求会在正在进行的 API 调用返回后生效，不再启动后续步骤或轮次；网络请求本身不保证能被供应商侧撤销。

## 演示模式

点击“体验演示”可加载预置的三轮结果、评分和建议，无需 API Key 或网络模型调用。页面持续标注“演示数据”；演示结果不写入真实运行记录，也不经过真实评测代码。

## 官方接口文档

- [TypeSafe System One API / Noul 响应格式](https://api.typesafe.ai/redoc) · [OpenAPI Swagger](https://api.typesafe.ai/docs)
- [DeepSeek API 文档](https://api-docs.deepseek.com/)
- [DeepSeek Chat Completions 接口与 thinking 参数](https://api-docs.deepseek.com/api/create-chat-completion) · [DeepSeek Responses 接口](https://api-docs.deepseek.com/api/create-response)
- [阿里云百炼 Model Studio 兼容模式](https://help.aliyun.com/zh/model-studio/compatible-mode)
- [Moonshot / Kimi API 快速开始](https://platform.moonshot.ai/docs/guide/start-using-kimi-api) · [中国区文档](https://platform.moonshot.cn/docs/intro)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)

## 已知限制

- 请求通过各服务商的 OpenAI 兼容 Chat Completions 接口发送，并将 Chat Completions 与常见 Responses API 文本响应统一解析。若某个模型或网关只开放 Responses API、或使用专有请求参数，当前版本仍需该服务商提供 Chat Completions 兼容入口。
- 自动优化使用单条测试输入迭代；分数可能受采样随机性和评审模型判断影响，不能代替代表性测试集或人工复核。
- 首版不提供多用户鉴权、API Key 加密持久化、云端数据库、费用预估或后台任务队列。若重启后端，正在进行的 SSE 任务无法恢复；已完成轮次仍保留。
- TypeSafe Key 需在 TypeSafe 开发者控制台申请；本地未提供 Key 时不能完成真实模型或 Jev 联调。
