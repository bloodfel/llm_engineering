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

#### 1.1 LSTM 的能力与局限

LSTM 在 Transformer 出现前统治 NLP 近十年（2014-2017）。它通过门控机制（forget gate, input gate, output gate）控制信息的保留和遗忘，像一条传送带让重要信息跨越长序列传递。能力确实强，但顺序处理的本质限制了它的规模上限。

#### 1.2 Transformer 的架构创新

Transformer 不是 LSTM 的改进版，而是完全不同的思路。Self-Attention 让序列中任意两个词直接建立关联——比如 "The cat sat on the mat because **it** was tired" 中，self-attention 可以直接让 "it" 和 "cat" 关联，不管隔多远。LSTM 的信息必须经过中间所有时间步传递，距离越远越容易丢失。

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

### 6. Parameters：模型的"知识容量"

**Parameter**（参数）就是神经网络里的一个个数字，每个数字代表模型从训练数据中学到的一小片"知识"。模型做预测时，就是把输入数据经过这些参数的层层计算，最终得出输出。

参数量的演进（对数级增长）：

| 模型               | 参数量                 |
| ------------------ | ---------------------- |
| 传统机器学习       | 20-200 个              |
| GPT-1              | 1.17 亿                |
| GPT-2              | 15 亿                  |
| GPT-3              | 1750 亿                |
| GPT-4              | 1.76 万亿              |
| 最新 frontier 模型 | 未公开（可能数十万亿） |

常用缩写：**M** = Million（百万），**B** = Billion（十亿），**T** = Trillion（万亿）。如 "Llama 3.2 3B" = 30 亿参数。

同一家公司的模型有不同档位（如 Haiku < Sonnet < Opus），档位 = 参数量大小 = 计算成本 = 价格。Frontier 模型（Claude、GPT、Gemini）均不公开具体参数量。

#### 6.1 为什么参数少的模型可以更强？

Gemma 270M 比 GPT-2 1.5B 强，原因有三层：

- **训练数据**：现在的小模型用的数据量远超当年，且经过精心筛选清洗。Llama 3.2 3B 用了 15 万亿 token 训练，GPT-2 只用了几百亿
- **训练方法**：**RLHF**（Reinforcement Learning from Human Feedback，人类反馈强化学习）、**Knowledge Distillation**（知识蒸馏，用大模型当老师训练小模型）等技术让学习效率大幅提升
- **架构微调**：**GQA**（Grouped Query Attention）、**RoPE**（Rotary Position Embedding）等改进让同等参数下效果更好

类比：同样 2.0L 排量的发动机，2024 年造的比 2005 年造的马力大得多——不是排量变了，是材料、工艺、设计全面进步了。

### 7. Training Time Scaling vs Inference Time Scaling：两条正交的提升路径

提升模型表现有两条**独立的**路径：

**Training Time Scaling**（训练时扩展）：更大的模型 + 更多训练数据 + 更多算力。**Chinchilla Scaling Laws**（龙猫缩放定律）指出参数量和能有效吸收的训练数据量大致成正比。过去几年的主旋律。

**Inference Time Scaling**（推理时扩展）：模型大小不变，在使用时用技巧让它表现更好。两个典型手段：

1. **Reasoning**（推理/思考）：让模型先生成思考过程再回答。本质是给模型一个"外部草稿纸"——生成的中间步骤变成后续 token 可参考的输入
2. **往输入里塞更多信息**：如 RAG，让模型有更多可参考的数据

主要 reasoning 技术：

| 技术                        | 做法                         | 代表                                                |
| --------------------------- | ---------------------------- | --------------------------------------------------- |
| **CoT**（Chain of Thought） | prompt 里加"请一步步思考"    | 2022 年 Google 提出                                 |
| **内置 reasoning**          | 模型训练时就学会先思考再回答 | OpenAI o1/o3、Claude extended thinking、DeepSeek-R1 |
| **Self-Consistency**        | CoT 生成多个答案，投票取共识 | 多次采样                                            |
| **Tree of Thoughts**        | 分支探索多条思路             | 更复杂的推理任务                                    |

这两条路径是**正交的**，可以同时做。过去全靠训练时扩展（堆大模型），最近一两年 inference time scaling 爆发——先是 RAG，再是各种 reasoning 技术。

### 8. 开源模型参数量地图：建立选型直觉

| 模型      | 参数量     | 备注                         |
| --------- | ---------- | ---------------------------- |
| Llama 3.2 | 3B         | 和 GPT-2 差不多大，但强得多  |
| Llama 3.1 | 8B         | 编号更小但模型更大           |
| Llama 3.3 | 3.3B       |                              |
| Qwen      | 20B / 120B | 开源，两个尺寸               |
| DeepSeek  | 671B       | MoE 架构，不是每次用全部参数 |

> 补充：Ed 提到 Llama 4 "2.45 billion"，实际 Llama 4 Scout 是 17B 激活 / 109B 总参数（MoE），Maverick 是 17B 激活 / 400B 总参数。具体数字以各模型官方公布为准。

实际选型参考（开源模型本地部署）：

| 参数量 | 典型场景                   | 硬件要求                         |
| ------ | -------------------------- | -------------------------------- |
| 1-3B   | 手机端、边缘设备、简单任务 | 普通手机/笔记本                  |
| 7-8B   | 大多数日常任务的甜蜜点     | 一张消费级 GPU 或 Mac 24GB+      |
| 13-34B | 复杂推理、代码生成         | 多张 GPU 或 Mac 48GB+            |
| 70B+   | 接近 frontier 水平         | 专业 GPU 集群                    |
| 671B   | 研究/商用级                | 大量算力，MoE 让实际推理成本可控 |

**MoE**（Mixture of Experts，混合专家模型）：模型内部包含多个"小专家"，根据问题类型只激活其中一部分。DeepSeek 671B 总参数虽多，但单次推理不用全部计算。

## 关键术语

| 术语                                 | 解释                                                           |
| ------------------------------------ | -------------------------------------------------------------- |
| LSTM (Long Short-Term Memory)        | Transformer 之前最主流的序列模型，能处理长距离依赖但无法并行   |
| RNN (Recurrent Neural Network)       | 循环神经网络，LSTM 的上位概念                                  |
| Transformer                          | 基于 self-attention 的并行架构，当前 LLM 的基础                |
| Self-Attention                       | 序列中每个词直接计算与所有其他词的关联，不受距离限制           |
| Token                                | 词片段，LLM 处理文本的基本单位                                 |
| Parameter                            | 参数，神经网络中的数字，代表学到的一小片"知识"                 |
| B / M / T                            | Billion（十亿）/ Million（百万）/ Trillion（万亿），参数量单位 |
| Emergent Intelligence                | 涌现智能，模型规模足够大后自动出现的"智能"能力                 |
| Stochastic Parrots                   | "随机鹦鹉"，批评 LLM 只是统计预测而非真正理解                  |
| Few-shot Learning                    | 少样本学习，只给几个例子就能完成新任务                         |
| Prompt Engineering                   | 研究如何写好 prompt 让 LLM 给出更好回答                        |
| Context Engineering                  | 全面设计 LLM 输入的内容和结构，prompt engineering 的进化       |
| RAG (Retrieval-Augmented Generation) | 检索增强生成，从外部数据源检索信息注入 LLM 输入                |
| Agentic AI                           | 智能体 AI，LLM 在循环中运行并自主决定下一步行动                |
| Autonomy                             | 自主性，LLM 选择自己下一步行为的能力                           |
| Copilot                              | 副驾驶模式，人与 AI 协作完成工作                               |
| RLHF                                 | 人类反馈强化学习，用人类偏好数据训练模型                       |
| Knowledge Distillation               | 知识蒸馏，用大模型当老师训练小模型                             |
| Quantization                         | 量化，压缩模型参数精度（如 16-bit → 4-bit）以减小体积          |
| Training Time Scaling                | 训练时扩展，通过更大模型/更多数据提升性能                      |
| Inference Time Scaling               | 推理时扩展，通过 reasoning、RAG 等技巧在使用时提升性能         |
| Chinchilla Scaling Laws              | 龙猫缩放定律，参数量与有效训练数据量大致成正比                 |
| CoT (Chain of Thought)               | 思维链，让模型一步步思考再回答                                 |
| MoE (Mixture of Experts)             | 混合专家模型，根据问题只激活部分子模型                         |

### 9. Token 的实际运作细节

#### 9.1 常见词 vs 罕见词的拆分

常见词通常一个词对应一个 token：

```
"an important sentence for my class of AI engineers"
→ 每个词各占 1 个 token，共 9 个 token（50 个字符）
```

罕见词会被拆成多个 token：

```
"exquisitely"  → "ex" + "quis" + "ite" + "ly"（4 个 token）
"handcrafted"  → "hand" + "crafted"（2 个 token）
"witchcraft"   → "witch" + "craft"（2 个 token）
"LLM"          → "LL" + "M"（2 个 token，tokenizer 建造时 LLM 还不是热词）
```

#### 9.2 "词首 token" vs "词中 token"

Token 会区分**是不是一个词的开头**。词首 token 包含前面的空格：

- `" important"` → 一个 token（带空格，表示新词开头）
- `"important"` → 另一个 token（不带空格，表示某个词的中间部分）

所以 "unimportant" 拆成 `"un"` + `"important"`（不带空格版），和独立出现的 `" important"`（带空格版）是**不同的 token**。

#### 9.3 数字的 token 化

数字按三位一组拆分：

```
3.141592653589793 → "3" + "." + "141" + "592" + "653" + "589" + "793"
```

GPT 词表里为每个三位数（000-999）都分配了一个 token。早期 GPT 算三位数加法很准、四位数就不行——因为三位数是一个 token，可以直接记住 "token A + token B = token C"；四位数跨多个 token，需要真正"理解"进位。

#### 9.4 Token 方案为什么比单词方案好

单词方案：每个词形独占一个词表位置（play, playing, played, player... 各占一个），词表随词形数量线性增长。

Token 方案：词根和词缀分开存（play, ing, ed, er...），排列组合覆盖大量词形。像乐高积木——几百种基础块拼出无数造型。

#### 9.5 实用换算

| 换算关系      | 数值                     |
| ------------- | ------------------------ |
| 1 token ≈     | 4 个字符                 |
| 1000 tokens ≈ | 750 个英文单词           |
| 莎士比亚全集  | ≈ 90 万词 ≈ 120 万 token |

注意：上面适用于普通英文。**代码、数学、科学术语**消耗更多 token（接近 1 token/字符），因为变量名、符号会被拆得更碎。

### 10. Context Window：模型能"看"多少内容的上限

**Context window** 就是模型一次能处理的最大 token 数量。输入 + 输出的所有 token 加起来不能超过这个数。

需要装进 context window 的不只是最新消息，而是**整个对话历史 + 所有生成的输出**：

```
┌─ Context Window ──────────────────────────┐
│  System prompt                             │
│  用户第一条消息                              │
│  模型第一次回复                              │
│  用户第二条消息                              │
│  模型第二次回复                              │
│  ...（所有历史）                             │
│  用户最新的消息                              │
│  模型正在生成的回复（逐 token 增长）          │
└───────────────────────────────────────────┘
          全部加起来 ≤ context window
```

聊得越久，历史越长，留给新回复的空间越小。超过上限会报错。

各模型 context window：

| 模型             | Context Window                             |
| ---------------- | ------------------------------------------ |
| GPT-5            | 400K token                                 |
| Claude           | 200K token                                 |
| GPT-OS（开源）   | \~130K token                               |
| Gemini 2.5 Flash | **1M token**（能装下几乎整本莎士比亚全集） |

Context window 大小直接影响 inference time scaling 的能力——RAG 塞文档、**multi-shot prompting**（多示例提示，在输入中给多个示例问答供模型参考）塞大量示例，都需要消耗 context window 空间。

### 11. API 计费的完整逻辑

#### 11.1 基本规则

按 input token 和 output token **分别计费**，单位是"每百万 token"：

| 模型       | Input 价格/百万 token | Output 价格/百万 token |
| ---------- | --------------------- | ---------------------- |
| GPT-5      | \$1.25                | \$10                   |
| GPT-5 Nano | \$0.05                | \$0.40                 |

日常单次对话（几十到几百 token）成本极低，不到一分钱。这门课建议准备 \$5 API 预算，够整个课程用。

#### 11.2 两个容易踩的坑

**坑 1：Input token 包含完整对话历史。** 聊了 20 轮，第 21 轮发消息时前 20 轮全部算 input token，全部要付钱。这不是"不公平"——你需要模型看到完整历史才能给出好回答，计算量确实在那里。

**坑 2：Output token 包含你看不到的 reasoning。** 像 OpenAI o1/o3 这种 reasoning 模型，会先生成一大段"思考过程"再给最终答案。这些思考 token **你看不到，但要付钱**。会导致成本有不可预测性。

#### 11.3 缓存（Caching）

短时间内发送相同输入，部分 input token 不需要重新计算，可以省钱。GPT 自动缓存，Claude 的缓存机制稍复杂，后续课程会讲。

### 12. LLM 的"记忆"是假的

每次 API 调用都是**完全无状态的**（stateless）。模型不会记住上一次对话。

"记忆"的实现方式：把之前的对话历史全部塞进 messages 列表里重新发送。ChatGPT 网页版做的事情和你手动构造 messages 列表完全一样——每次发消息，前端自动把整个聊天记录打包发给 API。

```Python
messages = [
    {"role": "user", "content": "我叫 Ed"},
    {"role": "assistant", "content": "你好 Ed！"},  # 把上轮回复带上
    {"role": "user", "content": "我叫什么？"},       # 模型能基于历史回答
]
```

三种 role：**system**（系统指令）、**user**（用户消息）、**assistant**（模型回复）。你甚至可以伪造对话历史，模型照样当真。

### 13. 从 API 调用到 Transformer 的完整链路

```
你的代码发送 messages
       ↓
   API 服务器收到
       ↓
   Tokenizer 把文字拆成 token ID 序列
       ↓
   Token ID 序列喂进 Transformer
       ↓
   Transformer 通过 Self-Attention（多层叠加）处理：
     - 浅层：识别语法、词性
     - 中层：理解语义关系
     - 深层：综合推理
       ↓
   输出概率分布，选出下一个 token
       ↓
   把新 token 拼回序列，再跑一遍 Transformer
       ↓
   重复，直到输出 EOS（结束标记）或达到 max_tokens
       ↓
   把所有生成的 token 解码回文字，返回给你的代码
```

所有主流 LLM（GPT、Claude、Gemini）底层都是 Transformer 架构，都是这个流程。区别在于训练数据、参数量、架构细节优化和后训练策略（RLHF 等）。

### 14. Self-Attention 的内部机制：Q、K、V

#### 14.1 Token 怎么变成向量？

每个 token 就是一个编号（如 "猫" = 5678）。模型里有一张 **Embedding Table**（嵌入表），每个编号对应一行数字（比如 768 个数字），叫做这个 token 的**向量**（vector）。训练开始时这些数字是随机的，训练过程中逐步调整。

#### 14.2 Q、K、V 是什么？

每个 token 的向量通过三个参数矩阵变换出三个新向量：

| 向量 | 全称 | 作用 | 类比 |
|------|------|------|------|
| **Q** | Query | 我在**找**什么信息 | 你拿着一个问题去找答案 |
| **K** | Key | 我能**被找到**的标签 | 每本书封面的关键词 |
| **V** | Value | 我实际**包含**的内容 | 书里面的正文 |

完整流程：

```
Token 向量 × Wq → Q（我在找什么）
Token 向量 × Wk → K（我的标签）
Token 向量 × Wv → V（我的内容）
      ↓
Q · K → 点积 → 关联度分数
      ↓
Softmax → 注意力权重（0-1，总和为1）
      ↓
权重 × V → 加权求和 → 融合了相关 token 信息的新向量
```

例子：处理 "The cat sat because it was tired" 中的 "it" 时，"it" 的 Q 和所有 token 的 K 做点积，发现和 "cat" 的 K 关联度最高（0.82），所以 "it" 的输出中 82% 的信息来自 "cat" 的 V。

#### 14.3 为什么点积能衡量相关性？

点积公式：A · B = |A| × |B| × cos(θ)。两个向量方向越相似（θ 越小），cos(θ) 越大，点积越大。训练过程调整 Wq 和 Wk，让应该相关的 token 对的 Q 和 K 方向相似。

#### 14.4 为什么要分 Q、K、V 三个？

因为"怎么找"和"找到后拿什么"是两件事。Q 和 K 负责决定**看谁**（"它"找指代对象），V 负责决定**拿什么信息**（"猫"的语义特征：是动物、会累等）。分开让模型更灵活。

#### 14.5 Multi-Head Attention

实际不是只做一次 Q-K-V，而是同时做**多次**（多头注意力）。每个"头"用不同的参数矩阵，关注不同类型的关系（语法、位置、语义等），最后拼起来。

### 15. 模型是怎么训练的？

#### 15.1 训练数据和"正确答案"

训练数据是互联网上爬来的海量文本（几万亿 token）。正确答案不需要人标注——文本里每个位置的下一个词天然就是答案：

```
输入："The cat sat on the"  →  正确答案："mat"（原文里本来就有）
```

#### 15.2 训练循环

所有参数（embedding table、Wq、Wk、Wv 等）从**随机数**开始，然后重复几十亿次：

1. 喂一句话，模型预测下一个 token
2. 和正确答案比较，用 **cross-entropy loss**（交叉熵损失）算误差——正确答案被分配的概率越低，误差越大
3. 用 **backpropagation**（反向传播）把误差往回传，沿途每个参数微调一点点
4. 换下一批文本，重复

没有人告诉模型"它和猫相关"。是预测任务本身逼模型学会了什么该和什么关联——如果 "it" 不注意 "cat"，就预测不出 "tired"，误差就大，参数就会被调整到让 "it" 更关注 "cat"。

#### 15.3 训练完成后

所有参数固定下来，推理时一个都不动。这就是为什么 inference time scaling 不是"调参数"——参数已经锁死了。

### 16. Attention 的两种类型

| 类型 | 谁看谁 | 目的 | 用在哪 |
|------|--------|------|--------|
| **Self-Attention** | 一个序列**自己内部**互相看 | 理解序列本身（如"它"指代谁） | LLM 预测下一个 token |
| **Cross-Attention** | 一个序列去看**另一个序列** | 从外部序列获取信息 | 翻译时生成英文去看中文原文 |

Self-attention 是 Transformer 论文（2017）的关键创新。之前 attention 只用在两个序列之间（cross-attention），没人让一个序列自己和自己算。论文证明了：只用 self-attention 就能理解序列，不需要 LSTM。

LLM（ChatGPT、Claude）只做"预测下一个 token"，只有一个序列，所以只需要 self-attention，不需要 cross-attention。

### 17. "Attention Is All You Need" 的历史背景

Transformer 不是凭空发明的，而是把已有组件重新组合：

| 组件 | 之前就有了 |
|------|-----------|
| Attention 机制 | 2014 年 Bahdanau 提出 |
| Q、K、V 框架 | 信息检索领域的老概念 |
| 反向传播 | 几十年历史 |
| Embedding（词向量） | Word2Vec（2013） |

论文 8 位作者（Google Brain / Google Research）的关键创新：
1. 把 attention 从配角变成**唯一机制**——去掉 RNN，只用 attention
2. 发明 **self-attention**——让一个序列自己和自己算
3. **Multi-head attention**——多组 Q-K-V 同时捕捉不同类型的关系
4. 证明了简化后的架构能 scale——并行化带来的规模优势远超预期

## 关键术语

| 术语                                 | 解释                                                           |
| ------------------------------------ | -------------------------------------------------------------- |
| LSTM (Long Short-Term Memory)        | Transformer 之前最主流的序列模型，能处理长距离依赖但无法并行   |
| RNN (Recurrent Neural Network)       | 循环神经网络，LSTM 的上位概念                                  |
| Transformer                          | 基于 self-attention 的并行架构，当前 LLM 的基础                |
| Self-Attention                       | 序列中每个词直接计算与所有其他词的关联，不受距离限制           |
| Token                                | 词片段，LLM 处理文本的基本单位                                 |
| Tokenizer                            | 将文字拆分为 token 的工具，每个模型有自己的固定词表            |
| EOS (End of Sequence)                | 序列结束标记，模型生成此 token 时停止输出                      |
| Parameter                            | 参数，神经网络中的数字，代表学到的一小片"知识"                 |
| B / M / T                            | Billion（十亿）/ Million（百万）/ Trillion（万亿），参数量单位 |
| Emergent Intelligence                | 涌现智能，模型规模足够大后自动出现的"智能"能力                 |
| Stochastic Parrots                   | "随机鹦鹉"，批评 LLM 只是统计预测而非真正理解                  |
| Few-shot Learning                    | 少样本学习，只给几个例子就能完成新任务                         |
| Multi-shot Prompting                 | 在输入中提供多个示例问答，引导模型生成更好的回答               |
| Prompt Engineering                   | 研究如何写好 prompt 让 LLM 给出更好回答                        |
| Context Engineering                  | 全面设计 LLM 输入的内容和结构，prompt engineering 的进化       |
| Context Window                       | 模型一次能处理的最大 token 数量（输入+输出总和）               |
| RAG (Retrieval-Augmented Generation) | 检索增强生成，从外部数据源检索信息注入 LLM 输入                |
| Agentic AI                           | 智能体 AI，LLM 在循环中运行并自主决定下一步行动                |
| Autonomy                             | 自主性，LLM 选择自己下一步行为的能力                           |
| Copilot                              | 副驾驶模式，人与 AI 协作完成工作                               |
| RLHF                                 | 人类反馈强化学习，用人类偏好数据训练模型                       |
| Knowledge Distillation               | 知识蒸馏，用大模型当老师训练小模型                             |
| Quantization                         | 量化，压缩模型参数精度（如 16-bit → 4-bit）以减小体积          |
| Training Time Scaling                | 训练时扩展，通过更大模型/更多数据提升性能                      |
| Inference Time Scaling               | 推理时扩展，通过 reasoning、RAG 等技巧在使用时提升性能         |
| Chinchilla Scaling Laws              | 龙猫缩放定律，参数量与有效训练数据量大致成正比                 |
| CoT (Chain of Thought)               | 思维链，让模型一步步思考再回答                                 |
| MoE (Mixture of Experts)             | 混合专家模型，根据问题只激活部分子模型                         |
| Stateless                            | 无状态，每次 API 调用都是独立的，模型不记住之前的对话          |
| Caching                              | 缓存，重复输入时复用之前的计算结果以降低成本                   |
| API                                  | 程序调用模型的接口，按 token 使用量计费                        |
| Embedding Table                      | 嵌入表，将 token 编号映射为向量的查找表                        |
| Vector (向量)                        | 一组数字，表示 token 在高维空间中的位置和含义                  |
| Q / K / V                            | Query / Key / Value，self-attention 中每个 token 的三种角色    |
| Dot Product (点积)                   | 两个向量对应位置相乘再求和，衡量方向相似度                     |
| Attention Weights                    | 注意力权重，softmax 后的 0-1 分数，表示各 token 的重要程度     |
| Multi-Head Attention                 | 多头注意力，同时用多组 Q-K-V 捕捉不同类型的关系               |
| Cross-Attention                      | 跨序列注意力，一个序列去另一个序列里找信息（如翻译）           |
| Cross-Entropy Loss                   | 交叉熵损失，衡量预测概率分布和正确答案之间的差距               |
| Backpropagation                      | 反向传播，将误差从输出往回传递并微调每个参数                   |

## 我的疑问 / 待深入

- Frontier 模型（Claude、GPT、Gemini）具体参数量均未公开，业界普遍认为在数千亿到万亿级别
- Claude 的 caching 机制具体怎么运作？（后续课程会讲）
