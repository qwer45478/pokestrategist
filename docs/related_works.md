# Related Works

这份文档记录当前与 pokestrategist 最相关的 6 个开源项目，供后续做路线比较、工程借鉴和实验取舍时参考。

这里的“相关”不是按 GitHub star 数量排序，而是按下列标准综合判断：

- 是否真的在做宝可梦对战决策，而不只是提供模拟器、环境或聊天 bot。
- 是否包含可学习的决策模型，而不只是硬编码规则。
- 是否具备比较完整的训练、推理或在线对战闭环。
- 是否对 pokestrategist 当前主线有实质性对照价值。

当前 pokestrategist 的主线参考点见 [research_analysis.md](research_analysis.md)。简化地说，pokestrategist 的核心目标不是“从文本里直接生成动作”，也不是“把对战压成一个低维 MDP 做粗粒度 RL”，而是基于 replay 构造结构化监督，围绕 honest legal-action candidate set 做精确动作排序，并显式建模 hidden posterior、semantic line、response、typed consequence、switch subgame 和 frontier。


## 1. 总览排序

按与 pokestrategist 的整体相似度，从高到低大致可以排成：

1. AlphaMon
2. PokeLLMon
3. poke_RL
4. alphaPoke
5. reinforcement-learning-pokemon-bot
6. showdownbot / Percymon

这个排序并不表示性能高低，而是表示它们对 pokestrategist 当前问题定义的贴近程度。


## 2. 总览对比表

| 项目 | GitHub | 任务形态 | 主方法 | 与 pokestrategist 最像的地方 | 与 pokestrategist 的关键差异 |
| --- | --- | --- | --- | --- | --- |
| AlphaMon | [davidemodolo/alphamon](https://github.com/davidemodolo/alphamon) | 从 replay 训练可对战模型，并接入 Showdown 推理与 bot | 结构化 token + decoder transformer + value head + Monte Carlo search | 有完整 replay 管线、训练管线、推理栈和在线 bot | 主线是自回归 token 预测，不是显式 legal-action ranking |
| PokeLLMon | [git-disl/PokeLLMon](https://github.com/git-disl/PokeLLMon) | LLM 直接读状态文本并做在线对战 | GPT + ICRL + 知识增强 + self-consistency | 目标都是做真正能打的 Showdown battle agent | 核心是 prompt 驱动文本代理，不是结构化监督模型 |
| poke_RL | [leolellisr/poke_RL](https://github.com/leolellisr/poke_RL) | 在 Showdown / poke-env 中训练 RL 对战 agent | Tabular RL + DQN/Double-DQN/PPO/REINFORCE | 明确把宝可梦对战当成学习型决策问题 | 状态和动作抽象很粗，且大量依赖手工 MDP 设计 |
| alphaPoke | [MatteoH2O1999/alphaPoke](https://github.com/MatteoH2O1999/alphaPoke) | 可部署、可挑战的 RL battle bot | RL bot + GUI 挑战界面 | 真正打通了模型到对战执行的用户链路 | README 公开的研究细节较少，更像课程项目产物 |
| reinforcement-learning-pokemon-bot | [hsahovic/reinforcement-learning-pokemon-bot](https://github.com/hsahovic/reinforcement-learning-pokemon-bot) | 早期 RL battle bot | Deep policy RL in pruned/full environment | 早期端到端对战学习代理，对后续生态有历史意义 | 已弃用，建模和工程都明显早期化 |
| showdownbot / Percymon | [rameshvarun/showdownbot](https://github.com/rameshvarun/showdownbot) | 早期 Showdown AI / bot | minimax / greedy / random / 可选 net | 是一个真正的 battle AI 壳，而不是纯聊天 bot | 版本陈旧，对现代 Showdown 和现代 ML 路线参考有限 |


## 3. 详细记录

## 3.1 AlphaMon

仓库： [davidemodolo/alphamon](https://github.com/davidemodolo/alphamon)

### 项目定位

AlphaMon 是一个面向 Pokemon Showdown doubles 格式的 transformer-based battle policy model。它不是简单的 bot 框架，而是完整覆盖了 replay harvesting、数据清洗、tokenization、训练、推理服务、搜索和自动对战客户端的一条龙系统。

公开 README 显示，它当前主线仍在活跃维护，数据管线、训练管线和推理栈都处于可用状态，扩散分支则仅作为实验基线保留。

### 核心方法

- 从 Showdown API 抓取 replay，并做去重、格式过滤、质量检查。
- 将 battle log 转成结构化 battle tokens，而不是直接喂原始自由文本。
- 使用 decoder-only transformer 作为主干。
- 同时训练 policy head 和 value head。
- 预训练和微调分阶段进行，微调时对动作 token 做更高权重。
- 推理阶段在模型之上再包一层 Monte Carlo search 做动作排序。
- 支持 Flask inference server 和自动 Showdown bot。

### 为什么与 pokestrategist 相近

它是这 6 个项目里最接近 pokestrategist 当前工程形态的一个，原因主要有三点：

- 它不是只做 environment，也不是只做一个规则 bot，而是有完整 replay 到训练再到对战推理的闭环。
- 它重视数据工程，明确做 replay harvesting、质量清洗、缓存和 checkpoint 管理。
- 它把搜索层放在模型上方，而不是假设模型一次前向就足够做最终决策。

### 与 pokestrategist 的关键差异

尽管相近，AlphaMon 与 pokestrategist 的建模哲学差异也很大：

- AlphaMon 主线是自回归 token 预测，动作本质上还是序列生成问题。
- pokestrategist 主线是显式 legal-action candidate ranking，评分发生在 honest candidate set 内。
- AlphaMon 的中间结构主要是 token 序列与搜索；pokestrategist 则显式建模 hidden posterior、semantic line、response、typed consequence、switch subgame 和 frontier。
- AlphaMon 更像“生成式 policy + 搜索”，pokestrategist 更像“结构化状态 + 候选动作打分 + 受约束排序”。

### 对 pokestrategist 最有借鉴价值的部分

- replay harvesting、去重、缓存和大规模数据集构建流程。
- 把训练、推理服务、自动对战 bot 做成一套连贯工程，而不是分散的脚本。
- 在推理阶段加入上层搜索，而不是只依赖单次前向分数。
- 明确地区分预训练、微调和在线推理配置。

### 主要限制

- token 自回归路线对宝可梦这种高规则性、强合法性约束任务不一定是最自然的主线。
- doubles 与 pokestrategist 当前聚焦的 singles 主线并不一致。
- 它更依赖序列建模与搜索组合，而不是显式可解释的中间决策结构。

### 综合评价

如果后续只选一个外部项目做深入代码对照，AlphaMon 是优先级最高的对象。它最适合帮助 pokestrategist 对照“完整训练/推理/部署栈应该长什么样”，但不一定适合直接照搬其主干模型形式。


## 3.2 PokeLLMon

仓库： [git-disl/PokeLLMon](https://github.com/git-disl/PokeLLMon)

### 项目定位

PokeLLMon 是一个基于 GPT 的宝可梦对战 agent。它的重点不是从 replay 离线训练结构化策略网络，而是把当前 battle state、历史日志和外部知识转成文本输入，让 LLM 直接给出动作，并通过在线提示增强去减少错误。

公开论文和仓库描述中，它的三个核心策略是：

- In-Context Reinforcement Learning (ICRL)
- Knowledge-Augmented Generation (KAG)
- Consistent Action Generation / self-consistency

### 核心方法

- 把当前对战状态和历史 turn log 翻译成文本。
- 调用 GPT 生成动作。
- 把上几回合反馈追加回上下文，以文本反馈的形式在线修正决策。
- 检索 type 优劣、move/ability effect 等外部知识，减少 hallucination。
- 独立生成多个动作并投票，以降低高压局面下的 panic switching。

### 为什么与 pokestrategist 相近

- 它和 pokestrategist 一样，目标都不是做一个离线学术玩具，而是真正打通 Showdown 在线对战代理。
- 它同样强调宝可梦规则知识不能缺位，且战斗决策要结合历史上下文。
- 它的失败分析也与 pokestrategist 有交集，例如错误换人、短视和对对手下一步响应建模不足。

### 与 pokestrategist 的关键差异

- PokeLLMon 的核心决策单元是 LLM 文本生成；pokestrategist 的核心决策单元是结构化 legal-action 排序。
- PokeLLMon 依赖在线提示工程和外部 API；pokestrategist 依赖可训练的本地模型和结构化监督。
- PokeLLMon 的解释性主要来自语言解释；pokestrategist 的解释性主要来自显式中间变量和各个头部输出。
- PokeLLMon 更像“prompted agent”；pokestrategist 更像“specialized decision model”。

### 对 pokestrategist 最有借鉴价值的部分

- 在线 battle agent 的执行链路与错误分析框架。
- 如何把规则知识和即时反馈有效注入决策，而不是只靠静态模型参数。
- 如何识别并压制 panic switching 这类推理期不稳定现象。

### 主要限制

- 高度依赖外部大模型接口，复现实验和成本控制都较弱。
- 主要优势来自提示增强与知识注入，不等于形成了一个稳定、可完全本地复现实验的决策学习架构。
- 对结构化候选动作排序、hidden posterior 和可校准概率输出支持较弱。

### 综合评价

PokeLLMon 最适合作为“在线代理范式”的参考，而不是 pokestrategist 当前主线模型的直接模板。它提醒我们哪些问题会在真实对战环境中暴露，但不提供一条可以直接替代当前结构化建模路线的工程方案。


## 3.3 poke_RL

仓库： [leolellisr/poke_RL](https://github.com/leolellisr/poke_RL)

### 项目定位

poke_RL 是一个较完整的宝可梦对战 RL 基准项目，目标是用 classical RL 与 deep RL 方法训练 Pokémon battle agents，并基于 poke-env 和 Showdown 做训练与验证。

它的价值主要不在于当代最强结果，而在于把“宝可梦对战作为 RL 问题”这件事做得相对系统。

### 核心方法

- 使用 poke-env 作为与 Showdown 通信的 Python 环境。
- 实现了 Monte Carlo Control、Q-Learning、SARSA(lambda)、DQN、Double-DQN、PPO、REINFORCE 等多条基线。
- 手工定义 MDP 状态和奖励。
- 在 stochastic 环境和经过改造的 deterministic 环境中都进行实验。

### 为什么与 pokestrategist 相近

- 它同样把宝可梦对战当成学习型决策任务，而不是纯规则程序。
- 它同样重视“move vs switch”这类基础动作决策问题。
- 它也认真讨论了环境随机性、状态表示和训练稳定性。

### 与 pokestrategist 的关键差异

- poke_RL 的状态是手工压缩后的 MDP 特征，动作空间也被强行离散成固定 9 动作。
- 它更接近传统 RL benchmark，而不是 replay-supervised exact legal-action ranking。
- 它没有显式 hidden posterior、semantic line、response 或 frontier 这些结构。
- 为了构造 deterministic 环境，它还修改了 Showdown 并移除了不少真实随机机制，这和 pokestrategist 的现实对战约束并不一致。

### 对 pokestrategist 最有借鉴价值的部分

- RL 基线和 reward design 的组织方式。
- deterministic vs stochastic 环境对训练稳定性的影响分析。
- 作为对照实验时，可以帮助判断结构化 imitation/ranking 路线相对传统 RL 的价值。

### 主要限制

- 大量任务定义依赖手工压缩状态和修改后的环境。
- 动作空间和状态表达过于粗粒度，难以承载现代 singles 中大量高阶战术细节。
- 与当前 exact legal-action 主线相比，表达能力和可解释粒度都较弱。

### 综合评价

poke_RL 是很好的历史基线与对照系，不是最像 pokestrategist 的最终形态，但非常适合作为“传统 RL 路线能走到哪里”的参照物。


## 3.4 alphaPoke

仓库： [MatteoH2O1999/alphaPoke](https://github.com/MatteoH2O1999/alphaPoke)

### 项目定位

alphaPoke 是一个基于 reinforcement learning 的 Pokémon Showdown battle bot 项目，并额外提供了 GUI 界面，方便人类账户直接挑战 bot。README 也明确说明这是一个课程项目，并在 release 中附带 scientific report。

### 核心方法

- 使用 RL 技术训练对战 bot。
- 直接面向 Showdown 账号和挑战流程做封装。
- 提供 GUI，而不是仅靠命令行脚本。

### 为什么与 pokestrategist 相近

- 它和 pokestrategist 一样，目标都是“能实际接入 Showdown 并对战”的 agent，而不是只在离线 notebook 里做分类。
- 它在工程上重视部署体验和可挑战性，而不是只做训练代码。

### 与 pokestrategist 的关键差异

- 公开 README 中关于具体模型、状态表示、动作建模和训练目标的细节很少。
- 它整体更像一个以部署和课程交付为中心的 RL bot 项目，而不是当前这种强调结构化中间变量与精确动作绑定的研究型工程。
- pokestrategist 当前在数据构建、监督信号和中间结构上都明显更深入。

### 对 pokestrategist 最有借鉴价值的部分

- bot 交互流程与可用性包装。
- 如何把训练结果更方便地暴露给人工测试者。
- release 和 report 打包思路。

### 主要限制

- 研究细节公开不足。
- README 自己就说明是 uni course project，很多内部文件接口并不承诺稳定。
- 更适合参考部署与演示层，而不是主建模层。

### 综合评价

alphaPoke 是一个“可操作 battle bot 工程”的参考对象，而不是 pokestrategist 当前主线建模的最优对照对象。


## 3.5 reinforcement-learning-pokemon-bot

仓库： [hsahovic/reinforcement-learning-pokemon-bot](https://github.com/hsahovic/reinforcement-learning-pokemon-bot)

### 项目定位

这是一个较早期的 RL Pokémon battle bot 项目。仓库 README 直接说明它已经弃用，并指出后来发展出了更成熟的 [poke-env](https://github.com/hsahovic/poke-env)。

因此，它今天的主要价值不在结果本身，而在历史和架构演化路径。

### 核心方法

- 自建环境层与 player/network/model manager 分层。
- 在 pruned environment 中使用 deep policy RL。
- 提供了训练脚本和预训练 agent 运行方式。
- 文档中提到 pruned environment 下的 policy network 能较明显超过 random player，而 full environment 下模型不收敛。

### 为什么与 pokestrategist 相近

- 它是一个真正的学习型对战代理，而不是单纯的规则 bot。
- 它尝试把 battle state、玩家交互和模型管理拆成较清晰的工程边界。

### 与 pokestrategist 的关键差异

- 它非常早期，环境和模型都还停留在“先做一个能跑通的 RL battle bot”阶段。
- 它没有 replay-supervised 数据管线，也没有当前 pokestrategist 这种对 hidden info、candidate set 和 structured heads 的系统化建模。
- 它自己也承认 full environment 下模型不收敛。

### 对 pokestrategist 最有借鉴价值的部分

- 早期 end-to-end battle bot 的工程拆分方式。
- 为什么很多早期 RL 项目会先退到 pruned environment。
- 从早期 battle bot 工程演化出通用环境库的路径。

### 主要限制

- 已弃用。
- 面向旧版 Showdown 环境。
- 建模能力和工程成熟度都与当前主线差距较大。

### 综合评价

这是一个值得保留的历史参考，但不适合作为当前主线设计的直接对标对象。


## 3.6 showdownbot / Percymon

仓库： [rameshvarun/showdownbot](https://github.com/rameshvarun/showdownbot)

### 项目定位

Percymon 是一个更早期的 Pokémon Showdown AI。它基于 Node.js，可以直接连到 Showdown 对战，支持 minimax、greedy、random 等决策算法，也允许使用或更新网络配置。

### 核心方法

- 提供 bot 连接和账号管理。
- 支持若干决策算法切换。
- 附带 web console 与在线控制能力。

### 为什么与 pokestrategist 相近

- 它至少是真正的 battle AI，而不是聊天机器人或单纯的模拟器工具。
- 它讨论的是“如何在 Showdown 里做动作决策”，而不是泛泛的宝可梦数据处理。

### 与 pokestrategist 的关键差异

- README 明确说明它是为旧版 Pokémon Showdown 构建的，已不能直接用于新版本。
- 建模方式整体偏早期 heuristic/search 风格，与当前结构化学习路线距离很大。
- 它对现代数据管线、结构化监督和概率建模帮助有限。

### 对 pokestrategist 最有借鉴价值的部分

- battle bot 的最小可运行外壳。
- 决策算法切换与在线控制界面。
- 早期 Showdown AI 项目的系统组织方式。

### 主要限制

- 版本明显过时。
- 研究价值更多在历史参照，而不是可直接复用的现代实现。
- 对当前 singles 精确动作排序主线帮助有限。

### 综合评价

Percymon 可以保留为早期 battle AI 样本，但不应作为后续主要借鉴对象。


## 4. 邻近但不列入这 6 个最相关项目的基础设施

下面几个仓库很重要，但它们不属于“与 pokestrategist 最相似的学习型项目”，所以不放进上面的 6 个主条目里：

### 4.1 poke-env

仓库： [hsahovic/poke-env](https://github.com/hsahovic/poke-env)

它是 Python 侧最重要的 Showdown bot / RL 环境库之一。大量 RL battle bot 都基于它构建。对 pokestrategist 的价值主要在于在线对战环境、player abstraction 和脚手架，而不是决策建模本身。

### 4.2 Pokemon Showdown

仓库： [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown)

这是事实标准模拟器和服务器，是几乎所有项目的底层依赖。它决定了 battle protocol、choice request、规则实现和 replay 来源，但它本身不是 AI 项目。

### 4.3 pkmn/engine

仓库： [pkmn/engine](https://github.com/pkmn/engine)

这是一个高性能 battle simulation engine，面向 tooling、embedded systems 和 AI use cases。它的主要价值在于更快、更可控的模拟内核。如果未来 pokestrategist 需要更强的 rollout/search 或大规模离线模拟，它可能是很值得评估的底层替代件。


## 5. 对 pokestrategist 的实际启发

把这 6 个项目放在一起看，可以得出几个比较清晰的判断：

1. 真正同时覆盖“数据、模型、推理、在线对战”的开源项目并不多，AlphaMon 和 PokeLLMon 是最强的两个直接参照。
2. 传统 RL 路线在宝可梦对战上长期存在状态压缩粗、环境简化强、泛化有限的问题，这也是 pokestrategist 选择结构化监督与 exact legal-action ranking 的重要合理性。
3. LLM agent 路线证明了在线 battle agent 的可行性，但它的优势更多在外部知识注入和推理期稳定化，而不是替代结构化决策模型。
4. 当前开源生态里，像 pokestrategist 这样强调 honest candidate set、hidden posterior、semantic line、typed consequence 和 switch subgame 的项目并不常见，这意味着该路线具有一定原创空间。


## 6. 后续可跟踪方向

如果后续要继续对这些 related works 做更细的长期跟踪，建议重点追下面三条线：

1. AlphaMon 是否在 doubles 之外扩展到更强的 structured search 或 explicit candidate ranking。
2. PokeLLMon 及其后继工作是否把 prompt agent 进一步结构化，尤其是是否显式建模对手响应和长期计划。
3. 新出现的 replay-driven Pokemon decision projects 是否开始从“序列生成”转向“显式候选动作排序”。
