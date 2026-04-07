# Week 1 — Transformers, Emergent Intelligence 与 AI 演进

> 课程：Ed Donner - AI Engineer Core Track
> 日期：2026-04-06

## 核心论点

### 1. Transformer 赢了 LSTM，不是因为更聪明，而是因为能并行

Transformer 之前，最主流的序列模型是 **LSTM**（Long Short-Term Memory，长短期记忆网络），一种 **RNN**（Recurrent Neural Network，循环神经网络）。LSTM 通过 **cell state**（细胞状态）机制解决了普通 RNN 的 **梯度消失**（gradient vanishing）问题，让重要信息可以跨越很长的序列传递。

但 LSTM 有致命问题：必须顺序处理，前一步输出喂给下一步，**无法并行计算**，训练极慢，规模很难做大。

Transformer 是**完全不同的架构设计**，核心机制是 **Self-Attention**（自注意力）：对序列中的每个词，直接计算它和所有其他词的关联程度，不需要逐步传递。

|            | LSTM                             | Transformer              |
| ---------- | -------------------------------- | ------------------------ |
| 处理方式   | 顺序处理（一个词接一个词）       | 并行处理（所有词同时看） |
| 核心机制   | 门控（forget/input/output gate） | Self-Attention           |
| 理解上下文 | 信息沿时间步逐步传递             | 任意两个词直接建立关联   |

Transformer 的优势是两层的：

- **工程上**：能并行，训练快，能做大
- **架构上**：任意距离的词可以直接"对话"，不受距离限制

论文标题 "**Attention Is All You Need**" 的含义：不需要 LSTM 那些复杂的递推机制，只用更简单的 attention 就够了，因为它能 scale。

类比：LSTM 像手艺精湛但只能一个人干活的工匠，Transformer 像技术稍简单但能同时雇一万人干活的工厂。工厂赢了。

> 补充：Ed 在课上主要强调了并行化优势，但 self-attention 在建模长距离依赖上的能力也是 Transformer 成功的重要原因，不仅仅是"简化"。

### 2. 世界的反应：从震惊到质疑

2023 年 ChatGPT 爆发后，公众经历两个阶段：

**第一阶段：震惊。** Ed 说他做 AI 是本职工作，之前从没有朋友问他"什么是 transformer"，突然所有人都在问。

**第二阶段：反弹。** 代表性事件是论文 "**On the Dangers of Stochastic Parrots**"（随机鹦鹉的危险）。批评者的核心论点：LLM 只是统计模型，只是在预测"最可能的下一个词"，本质上是"加强版输入法联想"（predictive text on steroids）。人们把统计输出当真理，很危险。

这个批评的逻辑在**小模型**上完全成立——小模型确实只能生成"看起来通顺但经常胡说八道"的文本。但论文没有预见到**规模本身会带来质的飞跃**。

### 3. Emergent Intelligence：规模带来的质变

LLM 本质就是给定输入序列，预测最可能的下一个 **token**（词片段）。能预测出"看起来像样的文本"不奇怪。

**真正奇怪的是**：预测出来的内容经常**恰好是正确的**。给它数学题，预期它生成"看起来像数学答案的文字"，但它给出的是真正的正确答案。

这就是 **emergent intelligence**（涌现智能）——当神经网络规模大到一定程度，不仅生成"合理的"token，还生成"智能的"token。具体表现：

- 逻辑推理
- 代码生成
- 多语言翻译（即使某些语言训练数据很少）
- **Few-shot learning**（少样本学习，只给几个例子就能学会新任务）

关键点：我们理解它 **how** it works（统计原理都清楚），但不完全理解 **why** it works so well。连 OpenAI 等前沿实验室的人也对此困惑。

类比：蚂蚁单个很简单，但几百万只组成的蚁群能建造复杂巢穴、组织精密分工——个体的简单规则在足够大的规模下，涌现出"智能"的集体行为。

### 4. 从 Prompt Engineering 到 Context Engineering

AI 应用的发展阶段：

**Prompt Engineering**（提示工程）：曾经是高薪职位，现在人人都会，不再是独立职业。

**Copilots**（副驾驶）：人和 LLM 协作完成工作（GitHub Copilot、Microsoft Copilot）。关键突破是人和 AI 可以**协同工作**。

**Context Engineering**（上下文工程）：prompt engineering 的进化版。不只是写好 prompt，而是**全面构建 LLM 看到的整个输入**：

```
┌─────────────────────────────────┐
│  System Prompt（系统指令）         │
│  用户历史对话                      │
│  检索到的文档片段（RAG）            │
│  工具调用的结果                     │
│  用户当前的问题                     │
└─────────────────────────────────┘
          ↓
      LLM 预测输出
```

核心原则：**LLM 只能基于它看到的输入来预测输出**。想要正确的机票价格，价格就必须在输入里。想要回答公司问题，公司文档就必须在输入里。

**RAG**（Retrieval-Augmented Generation，检索增强生成）和 **tools**（工具调用）本质上都是 context engineering 的手段——往 LLM 输入里塞更好的信息。

### 5. Agentic AI：让 LLM 自己决定下一步做什么

**Agentic AI**（智能体 AI）的两个常见定义：

1. **LLM 控制工作流**：LLM 负责决定接下来做什么，包括调用其他 LLM、使用工具
2. **LLM 在循环中运行 + 有工具**：LLM 被反复调用，每次可以使用工具执行操作

关键词是 **autonomy**（自主性）。当 LLM 输出包含"我接下来要做 X"的指令时，它在某种意义上自主选择了行为。

底层机制不变：输入序列进去，输出序列出来。只不过输出包含了"下一步行动的指令"，系统执行后把结果再喂回去，形成循环。

Claude Code 就是典型的 agentic AI——列出 to-do list，逐步规划执行，本质是一系列 LLM 调用在循环中完成任务。

## 关键术语

| 术语                                 | 解释                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| LSTM (Long Short-Term Memory)        | Transformer 之前最主流的序列模型，能处理长距离依赖但无法并行 |
| RNN (Recurrent Neural Network)       | 循环神经网络，LSTM 的上位概念                                |
| Transformer                          | 基于 self-attention 的并行架构，当前 LLM 的基础              |
| Self-Attention                       | 序列中每个词直接计算与所有其他词的关联，不受距离限制         |
| Token                                | 词片段，LLM 处理文本的基本单位                               |
| Emergent Intelligence                | 涌现智能，模型规模足够大后自动出现的"智能"能力               |
| Stochastic Parrots                   | "随机鹦鹉"，批评 LLM 只是统计预测而非真正理解                |
| Few-shot Learning                    | 少样本学习，只给几个例子就能完成新任务                       |
| Prompt Engineering                   | 研究如何写好 prompt 让 LLM 给出更好回答                      |
| Context Engineering                  | 全面设计 LLM 输入的内容和结构，prompt engineering 的进化     |
| RAG (Retrieval-Augmented Generation) | 检索增强生成，从外部数据源检索信息注入 LLM 输入              |
| Agentic AI                           | 智能体 AI，LLM 在循环中运行并自主决定下一步行动              |
| Autonomy                             | 自主性，LLM 选择自己下一步行为的能力                         |
| Copilot                              | 副驾驶模式，人与 AI 协作完成工作                             |
