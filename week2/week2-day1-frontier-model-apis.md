# Week 2 — Frontier Model APIs

> 课程：Ed Donner - AI Engineer Core Track
> 日期：2026-04-07

## 核心论点

### 1. OpenAI Chat Completions API 的核心参数

`openai.chat.completions.create()` 是 OpenAI 的基础调用接口。

**必需参数：**
- **model** — 模型名称，如 `"gpt-4o"`, `"gpt-5-mini"`
- **messages** — 对话消息列表，每条包含 `role`（`system`/`user`/`assistant`）和 `content`

**常用可选参数：**
- **temperature**（0-2，默认1）— 控制随机性。0=确定性输出，越高越随机
- **max_tokens** — 生成的最大 token 数
- **stream** — 是否流式返回（见论点 2）
- **top_p** — nucleus sampling，与 temperature 二选一调
- **stop** — 遇到指定字符串时停止生成
- **presence_penalty / frequency_penalty** — 控制重复度
- **response_format** — 指定输出格式，如 `{"type": "json_object"}`
- **seed** — 固定随机种子，使输出更可复现
- **tools / tool_choice** — function calling 相关
- **reasoning_effort** — 仅 o 系列 / GPT-5 系列推理模型可用（见论点 3）

---

### 2. Stream（流式传输）

**`stream=False`（默认）：** 等模型生成完所有内容后一次性返回。

**`stream=True`：** 模型一边生成一边返回，token 逐个往外吐，像 ChatGPT 网页版那样文字逐渐出现。

流式的好处：
- **首个 token 延迟（Time to First Token, TTFT）** 大幅降低
- 用户体验好，不用盯着空白页等
- 长回复时用户可以边看边决定要不要中断

流式返回的每个 chunk 里是 `delta`（增量）而不是完整的 `message`，需要自己拼接。面向用户的应用（聊天界面、Gradio UI）基本都用 stream；批量处理、自动化脚本不需要。

---

### 3. reasoning_effort 与推理模型

**reasoning_effort** 是专门给推理模型用的参数，控制模型"思考多深"。

- OpenAI o 系列 / GPT-5 系列支持：`"low"` / `"medium"` / `"high"`（GPT-5 完整版还有 `"none"`, `"minimal"`, `"xhigh"`）
- 对 gpt-4o 等非推理模型传此参数会报错或被忽略
- 核心权衡：effort 越低 → 速度越快、token 消耗越少、成本越低，但推理质量可能下降

**Claude 的推理模式** 通过 `thinking` 参数控制：

| 写法 | 行为 |
|------|------|
| 不传 `thinking` | 普通模式，直接回答 |
| `thinking={"type": "adaptive"}` | 模型自己决定要不要思考（新版推荐） |
| `thinking={"type": "enabled", "budget_tokens": N}` | 强制开启思考（旧版） |

`adaptive` 是开关（授权模型"你可以思考"），`output_config.effort` 是倾向性引导（"这个任务多重要"）。即使 `effort: "max"`，模型遇到简单问题还是可能不思考，因为 adaptive 让模型保留判断权。

Claude 的优势：思考过程**透明可见**，返回结果里能直接看到推理链。

---

### 4. LangChain：LLM 应用开发框架

**LangChain** 是一个 Python 框架，核心目的是让你更容易地用 LLM 构建应用。它把不同模型的支持拆成独立的包：

| 包名 | 对接 |
|------|------|
| `langchain_openai` | OpenAI (GPT 系列) |
| `langchain_anthropic` | Anthropic (Claude 系列) |
| `langchain_google_genai` | Google (Gemini 系列) |
| `langchain_core` | 核心抽象（所有包共享的基础接口） |

**`ChatOpenAI`** 是 `langchain_openai` 包里的类，封装了 OpenAI 的 Chat Completions API。所有 LangChain 模型类都继承自 `BaseChatModel`，提供统一接口。

统一接口的威力：换模型只需改一行（`ChatOpenAI` → `ChatAnthropic`），下游代码不用动。

**LangChain 本身免费开源**，靠商业产品赚钱：LangSmith（监控调试平台）、LangGraph Cloud（部署托管）。经典的开源 + 商业化模式。

类比：LangChain 之于 LLM API = React 之于 DOM API。

---

### 5. LangChain 的四种调用方式

LangChain 的模型参数（temperature、max_tokens 等）在**创建实例时**指定，`invoke()` 只接收输入内容。这样设计是因为在 chain/agent 里同一个 llm 对象会被反复调用，模型配置是固定的，每次变的只是输入。

`invoke()` 的输入很灵活：字符串、Message 对象列表、原生 OpenAI 格式的 dict 列表都可以，LangChain 自动转换。

| 方法 | 场景 | 返回 |
|------|------|------|
| `.invoke(input)` | 最常用，同步调用 | 一个 `AIMessage` |
| `.stream(input)` | 聊天 UI，实时显示 | 迭代器，逐个 `AIMessageChunk` |
| `.batch([inputs])` | 批量并行处理 | `AIMessage` 列表 |
| `.ainvoke(input)` | 异步应用（FastAPI 等） | `AIMessage`（需 await） |

`batch()` 内部是并行的，3 个请求同时发，总耗时 ≈ 单个请求时间。还有对应的异步版 `astream()`、`abatch()`。

---

### 6. LiteLLM：轻量级 API 统一层

**LiteLLM** 和 LangChain 解决类似问题（统一不同 LLM 的接口），但思路完全不同：

- **LangChain** 建一套自己的抽象体系，你要学它的写法
- **LiteLLM** 直接模仿 OpenAI 的 API 格式，用同一套 OpenAI 写法调所有模型

```python
from litellm import completion
completion(model="openai/gpt-4.1", messages=[...])       # OpenAI
completion(model="anthropic/claude-...", messages=[...])  # Claude
completion(model="gemini/gemini-...", messages=[...])     # Gemini
```

返回值也是 OpenAI 格式：`response.choices[0].message.content`。

model 字符串用 `provider/model_name` 格式区分调用哪家。LiteLLM 本身不收费，费用由对应的模型提供商收取。

| | LangChain | LiteLLM |
|---|---|---|
| 定位 | 全功能框架 | 只做 API 统一 |
| 学习成本 | 高 | 会 OpenAI API 就会用 |
| 适合场景 | 构建复杂 LLM 应用 | 快速对比不同模型 |

类比：LangChain = 瑞士军刀，LiteLLM = 万能转接头。

---

### 7. Prompt Caching（提示缓存）

**Prompt Caching** 是模型提供商（Anthropic、OpenAI、Google）的服务器端功能，不是客户端库的功能。用什么库（原生 SDK、LangChain、LiteLLM）都能触发。

**解决的问题：** 反复调用 API 时 prompt 前缀大量重复（如长 system prompt），每次都重新处理浪费算力和钱。

**工作原理：**
1. tokenization（文本 → token 序列）— 每次都做，成本极低
2. 对 token 序列算 hash（指纹）— 极快
3. 查缓存表：命中 → 跳过 GPU 计算，用缓存的中间结果（KV Cache）；没命中 → 正常计算并存入缓存

真正贵的是 Transformer 的 GPU 计算，缓存跳过的就是这步。

**各家策略不同：**
- **OpenAI** — 自动缓存，缓存命中的 token 打 5 折。要求精确前缀匹配
- **Anthropic** — 需要显式指定缓存内容，"预热"缓存多付 25%，之后复用打 1 折
- **Google (Gemini)** — 支持隐式和显式两种方式

受益最大的场景：长 system prompt、多轮对话、RAG 应用、批量处理。

各家返回的 usage 里都会显示 cached tokens 数量（如 `response.usage.prompt_tokens_details.cached_tokens`）。

---

### 8. 聊天机器人对话：角色互换与 messages 构造

课程用 GPT 和 Claude 做了一个对抗对话实验：GPT 被设为爱抬杠的喷子，Claude 被设为和事佬。

**核心设计 — 角色互换：** 同一句话在两边的 role 是反过来的。

| | 在 call_gpt 里 | 在 call_claude 里 |
|---|---|---|
| GPT 说的话 | `assistant`（我说的） | `user`（对方说的） |
| Claude 说的话 | `user`（对方说的） | `assistant`（我说的） |

因为 API 的规则是：`assistant` = 我自己，`user` = 对方。

**`zip` 配对技巧：** `gpt_messages` 和 `claude_messages` 长度可能差 1（GPT 刚说完新的一条，Claude 还没回）。`zip` 以短的为准，多出来的丢掉。然后用 `messages.append({"role": "user", "content": gpt_messages[-1]})` 单独补上最新那条没被配对的消息。两者配合刚好完整。

这个 messages 列表结构是构建**对话式 AI 助手**的基础 — 通过维护完整的对话历史，让模型能保持上下文。

## 关键术语

| 术语 | 解释 |
|------|------|
| **Chat Completions API** | OpenAI 的聊天补全接口，输入 messages 列表，输出模型回复 |
| **stream** | 流式传输，模型一边生成一边返回，而非等全部生成完 |
| **TTFT (Time to First Token)** | 首个 token 延迟，流式传输可大幅降低 |
| **reasoning_effort** | 控制推理模型思考深度的参数 |
| **adaptive thinking** | Claude 的自适应思考模式，模型自行判断是否需要深度推理 |
| **LangChain** | LLM 应用开发框架，提供统一接口和丰富的组件生态 |
| **ChatOpenAI** | LangChain 对 OpenAI Chat API 的封装类 |
| **BaseChatModel** | LangChain 核心抽象类，所有模型类的父类 |
| **invoke()** | LangChain 的统一同步调用方法 |
| **AIMessage** | LangChain 统一的模型回复对象 |
| **LiteLLM** | 轻量级 API 统一层，用 OpenAI 格式调所有模型 |
| **Prompt Caching** | 模型提供商的服务器端缓存功能，避免重复计算相同的 prompt 前缀 |
| **KV Cache** | Transformer 计算的中间结果缓存，Prompt Caching 跳过的就是这部分 |
| **LangSmith** | LangChain 的商业产品，用于监控、调试、评估 LLM 应用 |
| **zip** | Python 内置函数，将多个列表按位配对，以最短的为准 |
