这份文档会把相关论文分成两条线：

1. 世界模型与潜在动力学线：World Models, PlaNet, Dreamer, DreamerV3, Sparse Imagination, Latent Diffusion Planning
2. 时间三联决策线：DRQN, Decision Transformer / Trajectory Transformer / PDT, Predictron, I2A, MuZero, EfficientZero, Q-Transformer, Bayes Adaptive MCTS

这两条线和 pokestrategist 的关系并不一样：

- 世界模型线更像是在回答“如何把历史与当前观测压成可更新的内部状态，并预测未来转移”。
- 时间三联决策线更像是在回答“如何让历史信息、当前状态和未来后果共同决定此刻动作”。

## 2. 为什么这些论文和 pokestrategist 相关

你当前的研究已经逐步收敛到下面这个判断：

- 最终目标不是“预测对手下一手标签”，而是“对我方候选动作做排序”。
- 排序不能只靠一个静态 policy head，而要拆成默认稳健解、对手信念更新、战略门控和有限幅度重排。
- 因为 Pokemon 对战有隐藏信息、部分可观测性、长时依赖和策略性误导，所以系统不能只做瞬时分类，而需要内部状态、对手信念和反事实推演能力。

这正是部分可观测决策、世界模型、序列建模和有限规划这些文献最相关的地方。

如果把 pokestrategist 的研究目标翻译成现代 model-based RL / planning 的语言，可以写成：

$$
\text{从部分可观测历史 } (o_{\le t}, a_{<t}) \text{ 中学习潜在状态 } z_t,
$$

$$
\text{再用 } z_t \text{ 产生默认动作分布、对手响应分布、战略门控和价值估计。}
$$

进一步，理想系统还应当具备：

- history analysis：把过往 turn、reveals、换人节奏和行为证据压成可持续记忆
- belief update：对手隐藏配置的后验更新
- latent dynamics：给定我方动作和对手响应后的潜在状态转移
- prediction heads：默认策略、对手策略、价值、风险、终止概率
- imagination / planning：在潜在空间里对候选动作做有限步前瞻

这正是下面几类论文各自擅长的部分。

## 3. 核心框架：历史 + 当下 + 未来 -> 当前动作



### 3.1 核心结构

把你的意思展开，可以写成四步：

1. 历史分析 history analysis：从过去回合、reveals、换人、出招模式里提炼长期证据
2. 当下状态 current state：把当前公共局面和隐藏信息后验压成可决策状态
3. 未来预测 future prediction：估计不同候选动作带来的短中期后果
4. 瞬时决策 instant decision：用前面三者共同决定这一手动作排序

形式上更接近下面这个抽象式子：

$$
m_t = \mathrm{Enc}_\theta(o_{\le t}, a_{<t}, r_{<t})
$$

$$
z_t = \mathrm{State}_\theta(m_t, o_t)
$$

$$
\hat \tau_{t:t+H} = \mathrm{Predict}_\theta(z_t)
$$

$$
a_t = \mathrm{Decide}_\theta(m_t, z_t, \hat \tau_{t:t+H})
$$

这里最关键的不是“三个模块刚好并排摆着”，而是：

- 历史不能丢，因为这是部分可观测博弈
- 当前状态不能只看显式盘面，而要包含 belief state
- 未来预测不能只是辅助标签，而要真正进入当前动作决策

### 3.2 文献中的对应框架

在现有文献中，这个框架通常落在下面几类方法的交叉处：

1. **POMDP / belief-state control**
   历史观测不是噪声，而是用来更新当前 belief state 的依据。

2. **Recurrent / sequence decision models**
   用 RNN 或 Transformer 把历史轨迹编码进当前决策上下文。

3. **World model / latent dynamics**
   用内部状态和转移模型去预测未来若干步后果。

4. **Imagination / planning / receding-horizon decision**
   用未来预测来反过来决定“这一手现在该怎么下”。

从控制视角看，这和 receding-horizon control / model predictive control 的思想是相通的；从深度学习文献看，则更常以 belief-state + world model + planning 或 sequence modeling 的形式出现。


## 4. 世界模型线的关键论文

### 4.1 World Models

- 论文：David Ha, Jürgen Schmidhuber, World Models, 2018
- 链接：https://arxiv.org/abs/1803.10122

#### 提出了什么思想

这篇论文的核心思想是：

- 不要让控制器直接从原始高维观测学决策。
- 先学一个内部世界模型，把“眼前看到了什么”和“接下来可能发生什么”压成低维潜在变量。
- 再让一个很小的 controller 基于这个世界模型做动作。

作者把系统拆成了三个部分：

- V: Vision，用 VAE 压缩当前观测
- M: Memory，用 MDN-RNN 预测潜在状态的未来演化
- C: Controller，用很小的策略模块根据潜在状态和记忆做动作

这篇论文的经典观点是：

> 策略模块可以很小，复杂性应尽量放在世界模型里。

#### 使用了什么算法

- 表征学习：VAE
- 时序预测：MDN-RNN，也就是带混合高斯输出的 RNN
- 控制器优化：CMA-ES，而不是标准 policy gradient
- 训练方式：先学世界模型，再学控制器；甚至可以在“梦境”里训练策略再迁移回真实环境

#### 对 pokestrategist 的启发

这篇论文最直接的启发不是“用 VAE”，而是下面三点：

1. **把复杂性从决策头移到内部状态模型**
   pokestrategist 现在已经发现，单纯预测对手标签并不能解决最终动作排序问题。World Models 告诉我们，更合理的方向是先把“对战世界”压成一个内部可推演状态，再从这个状态出发做动作排序。

2. **默认策略可以很小，但世界状态模型必须强**
   你当前的 self-prior + strategic residual 其实就有这个味道：默认头不需要包办一切，关键是 belief state 和未来推演要足够好。

3. **模型偏差会被策略利用**
   这篇论文明确讨论了“agent 会 exploit 自己的 world model”。这和 Pokemon 场景高度相关：如果你的对手模型、未来转移模型或价值模型有系统偏差，策略很快就会学会钻漏洞。因此你现在坚持 bounded residual、风险惩罚和门控，是对的。

#### 直接可迁移的部分

- 把当前 belief-state encoder 视为 V+M 的原型
- 在潜在空间里建模未来 turn 的公共状态变化和隐变量更新
- 给战略模块看的不只是当前状态，还包括 latent memory

#### 不应直接照搬的部分

- VAE + MDN-RNN 是 2018 年方案，现在更推荐 RSSM 一类随机-确定性混合状态模型
- CMA-ES 不适合你当前这个监督与结构化标签较多的设置
- 直接在“梦境对战环境”里训练完整策略，对 Pokemon 这种隐藏信息博弈来说风险太大，容易学到 exploit model bias 的坏策略

### 4.2 PlaNet

- 论文：Danijar Hafner et al., Learning Latent Dynamics for Planning from Pixels, 2019
- 链接：https://arxiv.org/abs/1811.04551

#### 提出了什么思想

PlaNet 的核心思想是：

- 不在像素空间做规划，而是在潜在空间做规划
- 这个潜在状态必须同时具备短期可预测性和长期可规划性

它比 World Models 更进一步的地方是：

- 世界模型不是只为压缩和生成而学，而是明确为 planning 服务
- 状态模型同时包含 deterministic 和 stochastic 成分，用来更好地处理部分可观测性和不确定性

#### 使用了什么算法

- latent dynamics model：随机 + 确定性混合状态空间模型
- planning：在潜在空间里做 online planning
- 训练目标：多步变分学习目标 latent overshooting

#### 对 pokestrategist 的启发

PlaNet 对你尤其重要，因为 Pokemon 对战本质上就是部分可观测问题。

最有用的启发有三点：

1. **隐藏信息不要硬塞成确定状态，而要显式建成 stochastic latent state**
   这和你文档里的 $b_t(h_t \mid x_t, E_t)$ 非常一致。对手配置、隐藏后排、未 reveal 招式、Tera 倾向，都应该进入随机潜变量，而不是被当前观测直接确定。

2. **规划应在潜在空间进行，而不是在手工标签空间里拼接规则**
   你最终想要的是 candidate-action scorer。如果未来能稳定枚举候选动作，那么 PlaNet 式 latent rollout 会比单纯 family 分类更自然。

3. **部分可观测性下，记忆不是可选项，而是状态定义的一部分**
   对战历史、reveals、换人偏好、对手 Tera 时机，都是 latent belief 的一部分。

#### 对 pokestrategist 的具体映射

可以把未来的 pokestrategist latent state 写成：

$$
z_t = (z_t^{public}, z_t^{belief}, z_t^{tempo})
$$

其中：

- $z_t^{public}$ 表示公共场地与显式队伍状态
- $z_t^{belief}$ 表示对手隐藏配置后验
- $z_t^{tempo}$ 表示当前节奏、战略压力、可利用窗口等高阶摘要

### 4.3 Dreamer

- 论文：Danijar Hafner et al., Dream to Control: Learning Behaviors by Latent Imagination, 2020
- 链接：https://arxiv.org/abs/1912.01603

#### 提出了什么思想

Dreamer 的关键思想是：

- 世界模型不是只用于 planning，也可以直接用于想象未来轨迹
- 策略和值函数可以在潜在空间 imagined trajectories 上训练
- 这样既有 model-based 的样本效率，又避免每步都做昂贵的 online search

#### 使用了什么算法

- RSSM 类 latent world model
- imagination rollout：在 latent state 里展开未来轨迹
- actor-critic：在 imagined trajectories 上训练行为策略和值函数
- 关键点：通过 latent imagination 传播梯度，而不是只做 sampling-based planning

#### 对 pokestrategist 的启发

Dreamer 和你现在的 default + strategic residual 结构特别接近，因为它强调的不是“替代掉策略”，而是“给策略一个可想象未来的内部环境”。

你可以直接把它映射成：

- self-prior：默认策略头
- strategic residual：相当于一个局部 imagination-based correction
- value head：对未来局面做短视野或中短视野评估

也就是说，Dreamer 的思想最适合支持你现在的这条线：

> 默认先验先给出稳健候选，再由 imagined latent futures 去做 bounded reranking。

#### 适合你的地方

- 不需要一开始就上 MCTS
- 可以先在 top-k 候选上做有限步 imagination
- 很适合部分可观测 + 长时依赖问题

#### 不足

- Dreamer 更擅长单智能体连续控制
- Pokemon 是双边博弈，而且是 simultaneous action + hidden information，必须把对手响应分布一起纳入 imagination

### 4.4 DreamerV3

- 论文：Danijar Hafner et al., Mastering Diverse Domains through World Models, 2023/2024
- 链接：https://arxiv.org/abs/2301.04104

#### 提出了什么思想

DreamerV3 的重要意义在于：

- 世界模型不是只能在单一 benchmark 上调很多超参后才有效
- 通过 normalization、balancing、transformation 等稳健训练技巧，世界模型可以在大量任务上用一套配置工作

#### 使用了什么算法

- 仍然是 Dreamer 系列的 latent world model + imagination 学习路线
- 重点不在全新结构，而在让训练更稳、更泛化、更少依赖手调

#### 对 pokestrategist 的启发

它最重要的启发不是“再换一个世界模型”，而是：

1. **真正难的往往不是提出一个 world model，而是让它在多种局面分布上稳定训练。**
2. **训练稳定性技术和损失平衡，本身就是架构的一部分。**

这和你当前实验非常一致。你已经看到：

- opponent auxiliary 会带来 trade-off
- strategic gate 只有在 default-family anchor 足够强时才有效
- default / gate / residual 的权重配比会显著改变系统行为

DreamerV3 告诉我们，这不是“调参细节”，而是这个方向成败的核心。

### 4.5 Sparse Imagination for Efficient Visual World Model Planning

- 论文： Junha Chun, Youngjoon Jeong, Taesup Kim, Sparse Imagination for Efficient Visual World Model Planning, 2025/2026
- 链接：https://arxiv.org/abs/2506.01392

#### 提出了什么思想

这篇论文的重点不是再发明一种全新的 world model，而是指出：

- 一旦开始在 learned model 上做多步 imagination，真正的瓶颈往往会从“模型会不会想”变成“模型想一次太贵”
- rollout 时并不是每个 token、每个局面因子都必须被完整处理
- 可以用稀疏 imagination 在控制计算量的同时保留足够的规划质量

#### 使用了什么算法

- transformer-based visual world model
- randomized grouped attention
- 在 latent rollout 期间按预算做稀疏 token 处理
- 以更低的推理成本维持 planning 效果

#### 对 pokestrategist 的启发

这篇论文对你最重要的意义不是“视觉 world model”，而是：

1. **如果未来真的做 TopK 候选上的多步 rollout，计算预算会很快成为主约束。**
2. **候选前瞻未必需要每一步都展开完整状态。** 可以优先保留对当前决策最关键的摘要，例如：
   - 我方与对方 active 的局面摘要
   - hazards / weather / terrain 等公共盘面
   - Tera 可用性
   - 对手隐藏配置 belief summary
3. **稀疏前瞻应被看成 planning 后端的效率层，而不是表示学习主线。**

#### 不宜过度解读的地方

- 它主要解决的是 rollout 的计算效率，不是隐藏信息建模本身
- 在 pokestrategist 还没有稳定 latent transition branch 之前，不应把它放在 PlaNet / Dreamer 前面

### 4.6 Latent Diffusion Planning for Imitation Learning

- 论文： Amber Xie, Oleh Rybkin, Dorsa Sadigh, Chelsea Finn, Latent Diffusion Planning for Imitation Learning, 2025
- 链接：https://arxiv.org/abs/2504.16925

#### 提出了什么思想

LDP 的核心思想是：

- 不要把“规划未来”和“把未来翻译成动作”硬绑在一个头里
- 先在 learned latent space 里规划未来状态
- 再用 inverse dynamics 把 latent plan 反解成动作

它特别强调两类在 imitation / offline 场景里常被浪费的数据：

- action-free demonstrations
- suboptimal data

#### 使用了什么算法

- 先用 VAE 学压缩 latent space
- 再用 diffusion planner 生成未来 latent state 轨迹
- 再用 diffusion inverse dynamics 把 latent futures 映射回动作

#### 对 pokestrategist 的启发

这篇论文和你很相关，因为它提供了一个与 MuZero / sequence planner 都不同的模板：

1. **先规划想达到的未来局面，再反解当前动作。**
2. **低质量 replay 和没有明确奖励的 replay 也有价值。** 它们不一定适合直接 imitation，但很适合用来学“未来局面长什么样”。
3. **它天然支持“planner”和“action decoder”分离。** 这和你当前想保留 default anchor、再做 bounded reranking 的偏好是一致的。

#### 局限

- 原论文主要在视觉 imitation learning 场景，不是隐藏信息双边博弈
- diffusion planner 的训练与推理成本都更高，短期内不如 bounded reranker 实用

## 5. 时间三联决策线的关键论文

### 5.1 DRQN

- 论文： Matthew Hausknecht, Peter Stone, Deep Recurrent Q-Learning for Partially Observable MDPs, 2015/2017
- 链接：https://arxiv.org/abs/1507.06527

#### 提出了什么思想

DRQN 的核心思想很直接：

- 在部分可观测环境里，单帧或单时刻输入不足以支持可靠决策
- 决策模型必须带记忆，才能把历史证据积累起来

它通过在 DQN 中引入 LSTM，把“过去看到过什么”显式纳入当前动作价值估计。

#### 使用了什么算法

- 用 recurrent layer 替换 DQN 的一部分全连接层
- 把观测序列编码成隐藏状态
- 用隐藏状态预测当前动作价值

#### 对 pokestrategist 的启发

这篇论文不是在教你怎么做世界模型，而是在提醒你一件更基础的事：

> “历史分析”本身就是模型结构的一部分，而不是额外可选特征。

对 Pokemon 来说，这一点尤其明显，因为：

- 对手隐藏后排和未 reveal 招式，只能靠历史证据缩小后验
- 同一个当前盘面，在不同历史上下文下意义完全不同
- Tera 时机、换人倾向、诱导模式都要从历史里抽出来

如果按这个核心框架来分，DRQN 最对应的是“历史分析”这一块。

### 5.2 Decision Transformer / Trajectory Transformer

- 论文： Lili Chen et al., Decision Transformer: Reinforcement Learning via Sequence Modeling, 2021
- 链接：https://arxiv.org/abs/2106.01345
- 论文： Michael Janner et al., Offline Reinforcement Learning as One Big Sequence Modeling Problem, 2021
- 链接：https://arxiv.org/abs/2106.02039

#### 提出了什么思想

这条线的关键思想是：

- 可以把决策问题直接看成序列建模问题
- 当前动作不是孤立预测出来的，而是从“过去轨迹 + 目标未来收益”中自回归地产生

Decision Transformer 更强调：

- 给定 past states、past actions 和 desired return，直接输出当前动作

Trajectory Transformer 更强调：

- 学整个轨迹分布，再用 beam search 在预测轨迹里选更好的行动序列

#### 使用了什么算法

- 因果 Transformer
- 轨迹 token 化建模
- 自回归预测动作或未来轨迹
- Trajectory Transformer 还把 beam search 当作规划器使用

#### 对 pokestrategist 的启发

这条线的重要性在于，它提供了一个不完全依赖显式 dynamics head 的替代思路：

- 如果你手头主要是 replay 数据，那么“把对战看成序列”本身就很自然
- 你可以让模型从 battle history 直接生成当前动作分布或候选排序
- 也可以把短未来轨迹当作 token 序列，做 trajectory-level reranking

但它对 pokestrategist 也有明显局限：

- 它对隐藏信息的不确定性表达通常不如 belief-state world model 显式
- 它更擅长离线轨迹建模，不一定天然适合双边 simultaneous-move 博弈

所以对你来说，这条线更适合作为：

- 一个可比较的基线方向
- 一个 history encoder 的实现参考
- 一个未来可以和 belief-state world model 融合的候选模块

#### 补充：Future-conditioned Unsupervised Pretraining for Decision Transformer

- 论文： Zhihui Xie et al., Future-conditioned Unsupervised Pretraining for Decision Transformer, ICML 2023
- 链接：https://arxiv.org/abs/2305.16683

#### 提出了什么思想

这篇论文提出 Pretrained Decision Transformer, PDT。它的关键思想是：

- 即使没有 reward label，也可以先做 decision-model pretraining
- 训练时把未来轨迹信息当作 privileged context，用来预测当前动作
- 等到需要 return-conditioned 决策时，再把 return 映射到可能的 future embeddings 上做微调

#### 使用了什么算法

- Decision Transformer 风格的因果 Transformer
- future-conditioned action prediction
- reward-free / suboptimal offline data 上的预训练
- 与 return-conditioned finetuning 的衔接

#### 对 pokestrategist 的启发

PDT 对你尤其有价值，因为 Pokemon replay 往往满足下面两个条件：

1. **数据很多，但可靠 reward 很少。**
2. **不少 replay 是 suboptimal 的，但历史与未来结构本身仍然有学习价值。**

这意味着你完全可以先做：

- history encoder 的 future-conditioned 预训练
- belief-state 主干的 reward-free 轨迹建模预训练
- 再接 self-prior、gate、value、reranker 这些有监督头做 finetune

#### 局限

- 它仍然偏 sequence model，不会自动给你显式 belief uncertainty
- 它更适合做预训练或强基线，而不是直接替代 belief-state world model

#### 补充：Q-Transformer

- 论文： Yevgen Chebotar et al., Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions, 2023
- 链接：https://arxiv.org/abs/2309.10150

#### 提出了什么思想

Q-Transformer 的核心思想是：

- 不只用 Transformer 做 imitation / sequence modeling，也可以直接做 Q-learning
- 把动作拆成自回归 token 序列，让 Q-value 也变成可序列建模的对象
- 用 offline temporal-difference backup 去学高容量的 autoregressive Q-function

#### 使用了什么算法

- Transformer 作为 autoregressive Q-function
- 动作维度离散化并 token 化
- 离线 TD backup
- 大规模 offline data 上的多任务训练

#### 对 pokestrategist 的启发

这篇论文对你最直接的启发是：

1. **候选动作重排不一定只能是 imitation-style residual，也可以变成 value-guided autoregressive scoring。**
2. **Pokemon 动作天然可以拆成结构化 token。** 例如：
   - action head
   - move family
   - tera flag
   - switch target
3. **如果未来想从“默认候选排序”走向“动作条件价值打分”，Q-Transformer 是比 plain policy head 更贴切的参考。**

#### 局限

- 原论文主要针对机器人连续控制离散化后的动作序列
- pokestrategist 还要额外处理 hidden information 与 simultaneous move，这部分不被 Q-Transformer 直接解决

### 5.3 Predictron

- 论文： David Silver et al., The Predictron: End-To-End Learning and Planning, 2017
- 链接：https://arxiv.org/abs/1612.08810

#### 提出了什么思想

Predictron 的核心是：

- 内部模型不一定要忠实模拟真实世界的所有细节
- 只要它能更准确地预测 value，它就是有用的 planning model

也就是说，模型不是为了还原世界而存在，而是为了 **提高决策所需量的预测质量**。

#### 使用了什么算法

- 用一个 fully abstract Markov reward process 在内部展开 imagined planning steps
- 每一步积累 internal rewards 和 internal values
- 端到端训练，使内部 rollout 的汇总值逼近真实 value

#### 对 pokestrategist 的启发

这篇论文特别适合修正一个常见误区：

> pokestrategist 不一定需要先学一个“完整真实对战模拟器”，才能开始利用 world model 思想。

更现实的路径是：

- 先学一个只对动作排序有用的抽象模型
- 这个模型不必还原所有细节，只需要在 action ranking 所需的变量上足够准确

这和你当前的结构高度匹配。你当前的研究已经接受了：

- 先不追求完整 candidate-level simulator
- 先追求 self-prior + strategic correction + value-guided reranking

这其实就是 Predictron 式思路：

- abstract model first
- decision usefulness first
- full realism later

### 5.4 I2A

- 论文： Théophane Weber et al., Imagination-Augmented Agents for Deep Reinforcement Learning, 2017
- 链接：https://arxiv.org/abs/1707.06203

#### 提出了什么思想

I2A 的关键思想是：

- 不规定“模型应该如何被使用”
- 让 agent 自己去学习如何解释 imagined trajectories，并把这些 imagined predictions 当作 policy 的额外上下文

#### 使用了什么算法

- 先学 environment model
- 用 model rollout 生成 imagined trajectories
- 再用一个 rollout encoder / policy network 去解释这些 imagined 结果

#### 对 pokestrategist 的启发

I2A 和你现在的思路非常像，因为你并不想让未来的世界模型直接接管决策，而是想让它：

- 提供额外上下文
- 在默认候选集附近做修正
- 由主策略决定如何使用这些修正信号

换句话说，I2A 更像：

> “默认策略 + 想象结果作为补充证据”

这和你现在的 strategic residual 很接近。

如果用 pokestrategist 的语言重写 I2A 思路，就是：

- 默认 self-prior 先给出 TopK 候选
- 对每个候选，在 latent world model 中 rollout 若干步
- rollout encoder 把这些 imagined futures 编成向量
- residual head 学习如何使用这些 imagined futures 去重排候选

### 5.5 MuZero

- 论文： Julian Schrittwieser et al., Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model, 2019/2020
- 链接：https://arxiv.org/abs/1911.08265

#### 提出了什么思想

MuZero 的核心思想是：

- 不需要一个显式、可解释、完整还原环境的模型
- 只学习 planning 最需要的量：reward、policy、value
- 再把它接进树搜索

这是“世界模型”和“规划”真正深度结合的经典代表。

#### 使用了什么算法

- representation function：历史观测 $\to$ latent state
- dynamics function：latent state + action $\to$ next latent state + reward
- prediction function：latent state $\to$ policy + value
- Monte Carlo Tree Search：在 learned latent dynamics 上做搜索

#### 为什么它是“未来预测驱动瞬时决策”的强实现

MuZero 不是你这个概念的唯一对应，但它确实是最完整的现代实现之一，因为它把：

1. 当前潜在状态
2. 未来动作后果展开
3. 当前动作选择

放进了同一个 planning loop 里。对你的定义来说，MuZero 的价值在于：它清楚展示了“未来预测不是旁路分析，而是可以直接进入此刻动作选择”。

#### 对 pokestrategist 的启发

MuZero 对你最重要的启发不是“去复刻 Atari MCTS”，而是下面这个结构模板：

$$
z_t = h_\theta(x_t, E_t)
$$

$$
z_{t+1}, \hat r_t = g_\theta(z_t, a_t^{self}, a_t^{opp})
$$

$$
(\hat \pi_t^{self}, \hat \pi_t^{opp}, \hat v_t, \hat g_t) = f_\theta(z_t)
$$

这里和标准 MuZero 不同的地方是：

- Pokemon 是 simultaneous move，对动力学来说最好输入 joint action 或 opponent response sample
- Pokemon 是部分可观测，对 latent state 应该是 belief state 而不是 fully observed state
- 你还需要战略门控 $\hat g_t$，因为并不希望每一步都大幅特化

所以更准确地说，pokestrategist 若未来走到这一类方案，需要的是：

> **belief-state MuZero**，而不是 vanilla MuZero。

### 5.6 EfficientZero

- 论文： Weirui Ye et al., Mastering Atari Games with Limited Data, 2021
- 链接：https://arxiv.org/abs/2111.00210

#### 提出了什么思想

EfficientZero 是 MuZero 路线里很重要的一篇，因为它强调：

- learned planning model 不只是能工作
- 它还可以在少量数据下更高效地工作

#### 使用了什么算法

- 以 MuZero 为基础
- 加强 value / consistency / data-efficiency 设计
- 目标是在小数据 regime 下也能稳定学出有效规划

#### 对 pokestrategist 的启发

这篇论文对你最大的意义是现实层面的：

- 你的高质量 Pokemon replay 数据不可能像 Atari 环境那样无限自博弈生成
- 所以 sample efficiency 和监督信号设计会非常关键

换句话说，pokestrategist 即使未来真的走向 MuZero-style 规划，也更应该关注 EfficientZero 这类“少数据可学”的变体，而不是只看 MuZero 原始设定。

### 5.7 Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning

- 论文： Jiayu Chen, Le Xu, Wentse Chen, Jeff Schneider, Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning, ICLR 2026
- 链接：https://arxiv.org/abs/2410.11234

#### 提出了什么思想

这篇论文的重要性在于，它明确把 offline model-based RL 里的模型不确定性提到台前：

- 离线数据并不能唯一确定真实环境
- 因此 learned world model 不应被当成确定真值
- 更合理的做法是把问题视为 BAMDP，再把 search 建在“对环境不确定”的前提上

#### 使用了什么算法

- Bayes Adaptive MDP formulation
- learned stochastic world model
- Bayes Adaptive MCTS
- 把 search 当作 policy improvement operator 接入 offline MBRL

#### 对 pokestrategist 的启发

这篇论文对 pokestrategist 很关键，因为你的问题同时具备：

- 离线 replay 数据
- 模型不确定性
- 部分可观测性
- 未来想做有限搜索或有限 rollout

因此，如果你未来真的进入 search 路线，一个很重要的判断是：

> **应当优先做 uncertainty-aware search，而不是把 learned belief dynamics 当成确定对战模拟器。**

这意味着未来的 belief-state MuZero / finite search，更应该吸收 Bayes Adaptive MCTS 的思想，而不是只照搬 deterministic MCTS。

## 6. 如果把这些论文映射回 pokestrategist，最合理的结构是什么

结合上面两条研究线，一个最适合 pokestrategist 的长期结构可以写成：

### 6.1 模块分解

1. **History analysis + belief-state representation**
   输入：当前公共状态、己方完整信息、历史行为证据
   输出：潜在状态 $z_t$

2. **Opponent belief / latent dynamics**
   输入：$z_t$、我方动作、对手响应样本或响应分布摘要
   输出：$z_{t+1}$、局面变化、代理 reward/value targets

3. **Prediction heads**
   从 $z_t$ 或 imagined $z_{t+k}$ 预测：
   - 我方默认策略 $\pi_{self}^0$
   - 对手默认/个体化响应 $\pi_{opp}$
   - value / risk / terminal proxy
   - strategic gate $g_t$

4. **Instantaneous decision / bounded reranker**
   不直接搜索全部动作树，而是在默认 TopK 候选上做有限步 imagination、风险评估和重排，最终输出“这一手”的即时动作排序。

在更重的实现里，这一层的后端不一定只有一种形式：

- 可以是 bounded reranker
- 可以是 Bayes-adaptive finite search
- 也可以是 LDP 风格的“先规划未来 latent state，再反解当前动作”

### 6.2 与你当前架构的对应关系

你当前的架构已经有了这个长期结构的前两层雏形：

- 当前 evidence aggregation 与 belief transformer 共同构成 history analysis + representation 原型
- 当前 opponent belief / evidence aggregation 是 belief update 原型
- 当前 self-prior 是默认策略头原型
- 当前 strategic gate + residual scaffold 是 bounded reranker 原型

所以从学术定位上说，你现在不是在“拍脑袋造一个新结构”，而是在往一条很明确的 model-based decision architecture 靠近，只是还没把 latent dynamics 和 planning 完整接上。

## 7. 这些论文具体能怎样改造你当前研究

### 7.1 最近期可做：Dreamer / I2A / Sparse Imagination 风格的有限 imagination reranking

这是最适合你当前代码阶段的方向。

做法是：

1. 保留当前 self-prior 作为默认 TopK 候选生成器
2. 学一个 latent transition model，预测若干步后的公共状态摘要与价值 proxy
3. 对每个候选动作 rollout 若干条 imagined opponent responses
4. 把 imagined futures 编码成 residual features
5. 用 residual head 做 bounded reranking

这条路线最接近 Dreamer + I2A，而不是完整 MuZero。

如果未来 rollout 成本开始成为瓶颈，那么下一步不是立刻放弃 imagination，而是可以借鉴 Sparse Imagination：

- 只保留最关键的局面 token 或 belief summary 做前瞻
- 对不同候选分配不同预算
- 先把有限步 imagination 做到可负担，再讨论更深搜索

#### 7.1.1 信息组的结构边界不应按原始字段切，而应按高水平对战中的决策对象切

如果参考 Smogon 的 OU 资源页、对战入门攻略、SV OU Role Compendium 和 SV OU Speed Tiers，可以看到高水平单打玩家反复使用的是同一组分析框架：

- Team Preview 先识别双方 archetype、主 wincon 和必须保住的 check
- 每回合围绕 risk vs. reward、short-term vs. long-term 做取舍
- 用 wallbreaker / setup sweeper / wall / pivot / hazard control / speed control 这些 role 去理解局面，而不是把所有种族值、招式、道具一视同仁地平铺
- 用 checks / counters / offensive synergy / defensive synergy 去判断“这个动作后下一拍会发生什么”

对 pokestrategist 来说，这意味着近期主线不应是“把 12 个精灵 token 的所有字段尽量压成一个全局向量”，而应是：

> 保留完整 belief state，但在决策层按高手共识的语义信息组做条件读取。

更具体地，当前最合理的信息组边界不是按“属性 / 技能 / 道具”三类原始字段切，而是按下面七类决策对象切：

1. **即时进攻与推进组**
   - 我方当前 active 的攻击侧能力、速度线、招式覆盖、强化状态、Tera 攻击窗口
   - 对方当前 active 的承伤面、免疫 / 抗性、是否会被 2HKO / OHKO
   - 对方后排里能稳定吃下当前进攻的典型 switch-in
   - 这一组对应高手常说的 wallbreaking、making progress、forcing progress

2. **防守安全与反制组**
   - 对方当前 active 的即时击杀威胁、先制与强化威胁、状态威胁
   - 我方当前 active 和可换入后排的承伤、安全落点、恢复能力、Unaware / Regenerator / Intimidate 一类的 defensive utility
   - 这一组对应 checks、counters、defensive backbone、safe pivot

3. **速度控制与节奏组**
   - 双方当前速度线、Boost 后速度、Choice Scarf、Booster Energy、天气 / 戏法空间、priority
   - 哪一侧在这个回合及下一回合更可能先手、复仇、清场或被迫换人
   - 这一组必须单列，因为 OU 社区长期把 speed tiers 单独维护成资源，说明速度不是普通数值，而是独立的决策轴

4. **场面控制与功能招式组**
   - Stealth Rock / Spikes / Toxic Spikes、Defog / Rapid Spin / Court Change / Magic Bounce
   - Screens、Encore、Taunt、Knock Off、Trick / Switcheroo、phazing、status spread
   - 这一组对应 hazard game、utility、tempo denial，不应被混进普通攻击特征里

5. **资源与残局路径组**
   - 双方剩余血量、已倒下成员、Tera 是否可用、一次性资源是否已消耗、残局清场条件是否成熟
   - 类似 Supreme Overlord、Multiscale、Disguise、Focus Sash 这类与残局价值强相关的资源状态
   - 这一组对应短期与长期之间的桥梁，决定“这回合吃亏是否换来更大的终局收益”

6. **隐藏信息与对手范围组**
   - 未揭示招式、未确认道具、未确认特性、未确认 Tera Type
   - 基于 OU role compendium、sample teams、viability trends 得到的对手角色先验和常见响应范围
   - 这一组在 Pokemon 尤其关键，因为高手的很多判断并不是在已知真值上做，而是在角色范围上做

7. **全局对局计划组**
   - Team Preview 阶段形成的双方 archetype、我方主 wincon、对方必须拆掉的 key stop、我方必须保留的 check
   - 这不是普通 per-turn feature，而是一个跨回合更新的 game-plan memory
   - 它决定模型是否把当前动作看成“立刻赚血量”还是“为下一只 cleaner 清路”

上面这些边界并不是拍脑袋分组，而是把高手长期稳定使用的分析对象直接翻译成模型接口。这样做的目的不是替代学习，而是把学习限制在更合理的统计空间里。

#### 7.1.2 更适合 pokestrategist 的近期结构：typed summaries + 软路由 + shared fallback

基于上面的信息组，当前最合适的模型优化方法不是全局降维，也不是硬规则关闭某些神经元，而是：

1. 用共享 belief-state backbone 保留完整 token 级状态
2. 从完整状态里抽出按信息组组织的 typed summaries
3. 对每个候选动作单独做软路由，决定它该主要读哪些信息组
4. 保留一个共享 fallback 通道，避免路由错一点就让系统失明

可以把它写成：

$$
H_t = \operatorname{BeliefEncoder}(x_t, h_{<t})
$$

$$
u_k = \operatorname{GroupPool}_k(H_t), \quad k \in \{\text{offense, defense, speed, utility, resource, hidden, plan}\}
$$

$$
g(a, x_t) = \operatorname{entmax}\left(W_g [u_{global}; e(a); z_{plan}]\right)
$$

$$
s(a \mid x_t) = s_{prior}(a \mid x_t) + s_{shared}(a, H_t) + \sum_k g_k(a, x_t)\,\Delta_k(a, u_k)
$$

这里最关键的点有三个：

- `s_prior` 继续保留当前 self-prior，负责给出稳健默认候选
- `g_k(a, x_t)` 是软门控，不是 0/1 硬规则；它可以接近稀疏，但不应完全硬切
- `s_shared` 是共享兜底通道，用来处理路由不确定或多组都相关的局面

这会把当前模型从“所有决策都读一个共享大头”改成“先提候选，再按候选动作类型去读不同证据”。

更具体地，近期最值得落地的结构不是 neuron-level masking，而是下面这套更稳定、也更可解释的部件：

1. **Typed group tokens**
   - 从完整 belief tokens 中用 cross-attention 或 slot pooling 产出 7 个 group tokens
   - 不删除原始信息，只是把读法显式化

2. **Candidate intent router**
   - 对每个 TopK 候选动作输出一组 group 权重
   - 建议用 sparse softmax / entmax / top-2 gating 这类稀疏软路由，而不是硬开关

3. **Group experts**
   - 每个信息组只负责自己最擅长的标量或小向量，例如 progress、safety、tempo、resource gain、hidden-info risk
   - 不再要求一个共享 MLP 从 `cls_state` 里同时恢复所有逻辑

4. **Shared fallback expert**
   - 用于处理多组交错、路由不确定或当前分组没覆盖好的局面
   - 这是防止“路由准确率不够高时整个决策崩掉”的关键保险丝

5. **Action-conditioned future branch**
   - 未来头不再只预测“这个状态总体未来如何”
   - 而要预测“如果选这个候选动作，短期推进 / 生存 / 速度 / 资源会怎样变化”

这也解释了为什么单纯的 state-only future bundle 很容易出现一种现象：辅助 future 标签能学到，但主决策指标不升反降。它预测了未来，却没有把未来和候选动作、速度轴、check/counter 结构绑定起来。

#### 7.1.3 训练时不要让人类替模型决定答案，而是让人类决定结构与初始化

在这个设计里，人类规则和机器学习最合理的分工是：

- **人类决定结构边界**：哪些信息组应该存在，哪些专家负责什么问题
- **模型学习激活强度**：当前这个候选动作要读哪几组，各组占多大权重
- **共享通道兜底**：避免路由错误造成灾难性失真

因此，近期训练不该是“我们手写如果进攻就只看这几项”，而应是：

1. **用弱规则做 router 初始化，而不是做最终规则**
   - 例如 attack 候选在初始化时偏向 offense / defense / speed
   - switch 候选在初始化时偏向 defense / resource / hidden
   - setup 候选在初始化时偏向 offense / speed / resource / hidden

2. **给 router 加轻量弱监督**
   - 现有 strategic intent multi-head 可以作为第一版软标签
   - action head、switch target、future value、hazard 状态、速度是否足以 revenge kill 都可以提供 group-level 弱监督

3. **加约束而不是追求完全自由学习**
   - 保底共享通道最小权重
   - 路由熵正则，避免永远平均分配或永远单一路由
   - expert load balancing，避免所有样本都压到一个专家
   - group dropout，避免某组成为投机性捷径

4. **把路由本身当成评估对象**
   - attack 候选是否更常激活 offense / speed
   - switch 候选是否更常激活 defense / resource
   - 未揭示信息更多时，hidden-info 组是否上升
   - revenge-kill 与 clean 场景里，speed 组是否明显抬升

如果这些诊断都不成立，就说明问题不是“模型还不够大”，而是分组边界、弱监督或候选定义还不对。

### 7.2 中期可做：PlaNet / Dreamer / LDP 风格的 belief-state world model

这里的关键不是“学下一帧”，而是学：

- 下一回合公共可观测状态
- 对手隐藏配置后验更新
- 节奏和战略窗口变化
- 短期局面价值变化

训练目标可以包括：

- next public state reconstruction
- opponent action family prediction
- reward proxy / material delta / alive-only value
- terminal / KO / forced-switch proxy

如果后续发现：

- 低质量 replay 很多
- 某些数据更适合学未来局面，而不是直接学动作

那么 LDP 提供了一个中期分支：

- 先在 latent space 里学“未来想去哪里”
- 再用 inverse model 把目标未来局面反解成 move / switch / tera

### 7.3 长期可做：MuZero / Bayes Adaptive MCTS 风格的 belief-state planning

如果未来数据层能稳定枚举完整合法动作，那么最自然的升级方向就是：

- representation：当前 belief state
- dynamics：joint-action latent transition
- prediction：policy/value/risk/gate
- search：在我方 TopK 与对手 TopJ 响应上做有限深度搜索

注意这里不应直接照搬 vanilla MuZero，而应当改成：

- belief-state MuZero
- simultaneous-action MuZero
- bounded-exploitation MuZero

如果这一步主要依赖离线 replay 而不是大规模自博弈，那么比 vanilla MCTS 更贴切的参考其实是 Bayes Adaptive MCTS：

- 搜索节点里应显式反映模型不确定性
- 对 hidden-info belief 的分歧不应只在搜索外部用 penalty 处理
- search 应被视为 uncertainty-aware policy improvement，而不是“把不确定模型当真环境来搜”

## 8. 哪几篇最值得你优先读

如果只按“对 pokestrategist 的启发价值”排序，我建议这样读：

1. **Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL**
   因为它最直接对应你当前的约束：离线 replay、模型不确定性、有限搜索。

2. **MuZero**
   因为它给出了“表征 + 动力学 + 预测 + 搜索”的完整结构模板。

3. **PlaNet**
   因为它最适合你理解“部分可观测 + latent belief dynamics”应该怎样建模。

4. **Dreamer**
   因为它最像你当前的默认策略 + 有限 imagination 修正路线。

5. **DRQN**
   因为它最直接强调“历史分析”不是可选附加项，而是部分可观测决策的基础结构。

6. **I2A**
   因为它最像你当前 strategic residual / candidate reranker 的概念结构。

7. **Future-conditioned Unsupervised Pretraining for Decision Transformer**
   因为它告诉你 reward-free replay 和 suboptimal data 也可以先拿来做 future-conditioned 预训练。

8. **Q-Transformer**
   因为它提供了“离线数据 + 自回归 Q + 结构化动作分解”的候选动作打分思路。

9. **Latent Diffusion Planning for Imitation Learning**
   因为它展示了如何把“先规划未来状态，再反解动作”做成模块化 latent planner。

10. **Decision Transformer / Trajectory Transformer**
   因为它们提供了把 replay 直接当时序决策序列建模的替代路线。

11. **EfficientZero**
   因为它提醒你：少数据、有限 replay、训练稳定性，才是从论文到实用系统的关键问题。

12. **Sparse Imagination for Efficient Visual World Model Planning**
   因为它提醒你一旦引入多步 imagination，计算预算与稀疏前瞻会立刻成为工程瓶颈。

13. **World Models**
   因为它提供了最直观的思想原型和风险提醒，尤其是 model exploitation。

## 9. 对你当前研究最重要的最终判断

如果把这些论文都压缩成一句对 pokestrategist 最有用的话，那就是：

> 你的方向并不是继续做“更复杂的动作分类器”，而是逐步把 pokestrategist 变成一个带有历史分析、belief state、未来后果估计和 bounded reranking 的即时决策系统。

更具体地说：

- DRQN / POMDP 视角告诉你：历史分析不是附加信息，而是当前状态定义的一部分。
- World Models / PlaNet / Dreamer 告诉你：应该学习一个可压缩、可更新、可 imagination 的内部对战世界。
- Predictron / I2A / MuZero 告诉你：未来预测必须真正反馈到当前动作排序，而不是停留在辅助标签层。
- Decision Transformer / Trajectory Transformer 告诉你：如果主要依赖 replay 数据，序列建模本身也可以成为决策器或规划器的一部分。
- PDT / Q-Transformer 告诉你：离线 replay 不只是做 imitation，也可以做 future-conditioned 预训练和 autoregressive Q 打分。
- Sparse Imagination / Bayes Adaptive MCTS / LDP 告诉你：一旦开始在 learned model 上做前瞻，计算预算、模型不确定性和对 action-free / suboptimal 数据的利用方式都会变成主设计变量。

所以，从学术对应关系上看，你当前最合理的研究定位是：

> **从监督式 self-prior 起步，逐步过渡到 history-aware belief-state model + bounded imagination reranking / value-guided candidate scoring，再根据数据规模、动作枚举能力和不确定性建模能力决定是否演化到 Bayes-adaptive search、latent diffusion planner 或 sequence-model planner。**

这条路线和你现在已经做出来的 default policy、strategic gate、residual reranker，并不冲突，反而是自然延伸。

## 10. 一个直接可执行的研究路线图

### 阶段 A：先把“历史分析”做成一等模块

- 强化当前 evidence aggregation，让模型显式编码 turn history
- 把 reveals、换人链、Tera 试探、残局节奏变成结构化历史 token 或记忆状态
- 先验证“更强历史编码”是否能稳定提升当前 self-prior 排序

### 阶段 B：把当前系统提升为最小世界模型版本

- 保留 self-prior 默认头
- 先把 `offense / defense / speed / utility / resource / hidden / plan` 七类信息组做成 typed summaries
- 先把 candidate scorer 升级为带 shared fallback 的 soft-routing reranker
- 增加 latent belief transition model
- 预测 next public state / opponent response / short-horizon value
- 在 TopK 候选上做 imagination reranking

如果手头 reward-free replay 或低质量 replay 很多，还可以在这一阶段前插入一个 PDT 风格的 future-conditioned 预训练步骤，先把 history encoder 和共享主干预热起来。

这一步最接近 PlaNet + Dreamer + I2A。

### 阶段 C：把 strategic gate 从启发式监督提升为 learned strategic intent

- 从 critical-turn heuristic 升级到显式战略意图桶
- 让 gate 不只判断“是否复杂”，还判断“为什么复杂”
- 让 gate 和 router 共享一部分 intent / risk / hidden-info 弱监督
- 让 residual 与战略意图变量 $z_t$ 绑定

### 阶段 D：为“未来预测 -> 当前动作”选择实现后端

- 方案 1：MuZero-style 有限深度搜索
- 方案 2：Trajectory Transformer 风格的 sequence planner
- 方案 3：继续保留 bounded reranker，只扩大 imagination 深度和候选集
- 方案 4：Latent Diffusion Planning 风格的 planner + inverse dynamics
- 方案 5：Bayes Adaptive MCTS 风格的 uncertainty-aware offline search

### 阶段 E：再决定是否需要 full simulator

这一步不是前提，而是后续可能的增强项。Predictron 和 MuZero 都告诉我们：

- 决策有用的抽象模型，往往比高保真但难训练的完整模拟器更重要。

## 11. 针对现有系统的具体修改建议

下面这些建议不是“从头重写一个新系统”，而是基于现有代码结构，按论文里最成熟、最可迁移的部分做增量改造。

### 11.1 不要推倒重来，先保留四个已经验证有效的核心资产

结合当前实验结果，下面四样东西不应被推翻：

1. Belief-State Transformer 主干
2. self-prior 默认结构化分布
3. strategic gate + bounded residual 的隔离结构
4. 独立的短视野 state-value 训练入口

原因很简单：

- DRQN / POMDP 线说明历史与 belief state 本来就该是主干的一部分
- I2A / Dreamer 线说明“默认策略 + 有限修正”本身就是合理范式
- 你自己的实验也已经表明，default anchor 是必要条件，不能直接让“战略分支”接管主决策

因此，最优策略不是重建全新模型，而是把现有系统逐步升级成更强的 history-aware belief-state planner。

### 11.2 第一优先级：把历史分析从简单累计更新升级为显式 temporal encoder

当前代码里，`src/pokestrategist/models/belief_state_transformer.py` 的 `_belief_token()` 主要做的是：

- 先从 global features 初始化一个 opponent belief prior
- 再按历史窗口顺序，用 `opponent_belief_update()` 对历史 token 逐步更新

这个结构是对的，但还比较“轻”。按照 DRQN、Decision Transformer 和 POMDP 这条线，第一优先级应当是：

1. 扩充 history token 的语义，而不只放动作类型 / move / switch slot
2. 给历史分支单独一个 temporal encoder，而不是只做逐步状态更新
3. 把历史编码结果显式并入 gate、value 和 residual 分支，而不是只产出一个 belief token

建议的具体改法：

- 给历史事件增加更多结构化标记：revealed move 数、最近是否 Tera、换人链类型、残局阶段、危险场地变化
- 在 belief update 前面加一个小型 GRU 或 causal Transformer，先把历史压成 `history_summary`
- 让 `history_summary` 同时送入 opponent belief prior、strategic gate 和后续 value head

这一步借鉴的主要是 DRQN 和序列建模路线。目标不是让模型“记得更多”，而是让历史成为显式的一等输入，而不是隐含在少量 hand-crafted 更新里。

### 11.3 第二优先级：把 strategic gate 从 critical-turn 启发式升级为 learned intent / risk gate

当前 `src/pokestrategist/training/belief_state_dataset.py` 里的 `_strategic_gate_target()` 本质上还是一个 heuristic target：

- low HP
- boost pressure
- trick room / board pressure
- hidden info 与早期 turn 组合条件

这个启发式很适合做第一阶段 scaffold，但不适合当最终门控定义。按照 I2A、Dreamer 和 EfficientZero 的思路，下一步应该把 gate 改成“学习到的偏离必要性”而不是“手工定义的 critical turn”。

建议的具体改法：

1. 先把单一 gate 改成多头 gate：risk、tempo、hidden-info、endgame 四类 soft targets
2. 让总 gate 由这些子头聚合，而不是只用一个 BCE
3. 用 value disagreement 或 residual gain 构造一部分自蒸馏目标

也就是说，可以让 gate 学习：

$$
g_t \approx \Pr(\text{默认策略不足} \mid x_t, h_t)
$$

而不只是学习“当前是否像一个 critical turn”。

这一改动应保留你现有的 residual clip 和 default anchor，不要因为 gate 更智能了就取消边界约束。

### 11.4 第三优先级：增加 world-model auxiliary branch，而不是直接上完整搜索

按照 PlaNet 和 Dreamer 的路线，你当前最值得补的一块不是 MCTS，而是一个小型 latent transition branch。

当前系统已经有：

- belief-state encoder
- action-family 主任务
- 独立的 state-value baseline

最自然的升级方式是把这些拼起来，增加一个共享主干下的 world-model 辅助头，预测：

1. next public state summary
2. opponent response family
3. short-horizon value
4. terminal / forced-switch / KO proxy

建议的具体改法：

- 不要先新开一个庞大的 planner 分支
- 先在 `src/pokestrategist/cli/train_action_family.py` 里加可选 auxiliary loss
- 把 `src/pokestrategist/cli/train_state_value.py` 里的 value 监督逻辑逐步并入共享 encoder 训练

这一阶段最像 PlaNet / Dreamer，而不是 MuZero。重点是学“下一步会怎样”，不是学“整棵树怎么搜”。

如果你想在这个阶段更充分利用 reward-free replay 或低质量 replay，那么 PDT 也是一个很合适的补充：

- 先用 future-conditioned 预训练共享 encoder
- 再接 action-family / value / gate 辅助头做监督微调

### 11.5 第四优先级：把 residual 从 state-only logit 修正升级为 action-conditional reranker

当前 `src/pokestrategist/models/belief_state_transformer.py` 里的 strategic residual 是：

- 直接从 `cls_state` 产出 head / family / tera residual logits
- 再用 gate 缩放并 clip

这一步已经验证有价值，但它仍然是“状态条件修正”，还不是论文里更强的 I2A / Predictron 式“候选动作条件修正”。

下一阶段建议这样改：

1. 先用当前 self-prior 选出 TopK family 候选
2. 为每个候选构造 candidate-specific feature
3. 用 latent transition head rollout 1 到 2 步 imagined futures
4. 用一个小 reranker 对 TopK 候选单独打分

这里最关键的变化是：

- residual 不再只看当前状态
- residual 要看“如果我选这个候选，会发生什么”

更具体地，这里的 `candidate-specific feature` 不应继续是“把整个 `cls_state` 和候选 embedding 拼起来丢给一个共享 MLP”，而应升级成：

1. 从共享主干里抽出 `offense / defense / speed / utility / resource / hidden / plan` 七个 group token
2. 对每个候选动作预测一组稀疏 soft routing 权重
3. 让各组 expert 分别输出 `progress`、`safety`、`tempo`、`resource_gain`、`hidden_risk` 一类的可解释子分数
4. 再把这些子分数和共享 fallback 分数合成为最终 reranker score

这比“全局降维后统一打分”更符合高手实际对局的思考顺序：

- 先问这个候选的主要目标是什么
- 再问这类目标最依赖哪些证据
- 最后在局面允许的风险范围内给出排序

如果未来分支也继续保留，那么它最适合接到这些 group expert 后面，而不是继续做 state-only future head。也就是说，模型应学的是：

- 这个进攻候选能否制造 progress
- 这个换人候选是否真的安全
- 这个强化候选是否会被更快的 revenge killer 立刻掐死
- 这个 support 候选是否能为既定 wincon 清路

而不是只学“当前状态总体上看未来值是高还是低”。

这一步最直接对应 I2A 和 Predictron，也是你当前系统从“更强分类器”走向“真正即时决策器”的分水岭。

如果后续继续把 reranker 做深，有两条新增文献非常值得参考：

- **Q-Transformer**：把 move family / tera / switch target 拆成动作 token，用 autoregressive Q 做候选打分，而不只是 imitation-style residual
- **Sparse Imagination**：当 TopK rollout 变贵时，只对最关键的状态摘要做稀疏前瞻，降低 reranking 成本

### 11.6 第五优先级：把 state-value 分支从独立基线升级为共享 backbone 的校准器

你现在已经有 `src/pokestrategist/cli/train_state_value.py` 这条独立价值训练链路，这是很有用的，不应该废弃。但从论文角度看，它更适合作为共享主干上的辅助头，而不是长期独立训练的平行模型。

建议的具体改法：

1. 先保留独立 value baseline，继续作为 sanity check
2. 再增加一个 shared-encoder value head，和 action-family 一起多任务训练
3. 用 value uncertainty 或 value disagreement 反哺 gate / reranker

这一步主要借鉴 Dreamer 和 EfficientZero：

- value 不是最后才接上的评估器
- value 应该是当前动作修正的重要证据源

### 11.7 第六优先级：把 PDT / Decision Transformer 当作对照实验，而不是主线替代方案

PDT / Decision Transformer / Trajectory Transformer 对你有参考价值，但我不建议现在把它们当主线替代掉 belief-state world model。更合理的用法是：

1. 做一个 history-to-action 的 sequence baseline
2. 再做一个 PDT 风格的 future-conditioned 预训练版本
3. 比较它和当前 self-prior 在离线 replay 上的排序表现
4. 检查它是否在某些局面族上更擅长吸收长历史证据

如果它在特定子问题上更强，可以把它的 history encoder 部分吸收到现有主干，而不是整个改成 sequence-only policy。

### 11.8 当前阶段不建议直接做的事

基于现有论文和你当前系统状态，下面几件事暂时不建议优先做：

1. 直接复刻完整 MuZero + naive MCTS
2. 先造一个高保真完整对战模拟器再开始训练
3. 去掉 default anchor，让战略分支直接主导输出
4. 用更复杂的 heuristic 继续堆 gate target，而不改变 gate 的训练定义

原因不是这些方向永远不对，而是它们目前都跳过了你最缺的中间层：

- 可训练的历史编码
- 可监督的 latent transition
- action-conditional reranking
- shared value calibration

即使未来真的走向搜索，也更应优先参考 Bayes Adaptive MCTS 这类 uncertainty-aware offline search，而不是把 learned model 当成确定真环境直接硬搜。

### 11.9 一个最小可行改造顺序

如果要用最小风险方式改现有系统，我建议按下面顺序推进：

1. 先升级历史编码，再跑 self-prior 对比
2. 再把当前状态按 `offense / defense / speed / utility / resource / hidden / plan` 七类信息组显式做成 typed summaries
3. 然后把 candidate scorer 改成带 shared fallback 的 soft-routing reranker，而不是继续扩大全共享 MLP
4. 再把 heuristic gate 改成 multi-head learned gate，并让 gate / router 共享一部分弱监督
5. 最后补 1-step latent transition + action-conditioned future auxiliary，把 imagined future 接到 candidate-level scorer，而不是继续做 state-only future head

只有这五步都跑通后，才值得讨论是否进入 MuZero-style 搜索或更重的 sequence planner。

这条顺序的核心思想是：

> 先借鉴现有论文中最成熟、最稳的模块化思想，把你当前系统升级成一个带 history、belief、prediction 和 reranking 的强基线；只有当这个强基线已经清楚暴露瓶颈时，才去做更重的规划创新。
