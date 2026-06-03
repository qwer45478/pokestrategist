# pokestrategist 研究过程记录

本文承接 [research_analysis.md](research_analysis.md) 中拆出的过程性内容，只保留三类信息：

- 每个阶段试图解决什么问题
- 当前代码和训练主线发生了什么变化
- 为什么某条路线继续推进，或为什么在这里停止

本文不再保留以下细节：

- 逐 run 路径与临时产物引用
- 命令行与 smoke 执行流水
- 大段逐指标罗列
- 已经失效的临时结果目录引用

## 1. 全局研究原则

整个项目到目前为止一直围绕四条原则推进：

1. 目标不是做一个静态分类器，而是逼近高手在 belief state 条件下的动作决策。
2. 关键算法最终都要落到具体动作，但 exact 化必须按模块分阶段推进。
3. 每个子块只做一次实现和最多一两轮局部调优；一旦结论明确，就转向下一个相邻问题。
4. 主线优先保留稳定骨架，再让更细粒度的 exact 信号逐步替换 coarse 模块。

## 2. 2026-04-19 到 2026-04-20：从粗二分类过渡到结构化动作主线

### 2.1 稳态化训练与评估基础设施

最早期的工作先没有改任务定义，而是补训练稳定性与评估诊断：

- 引入样本级加权采样，用于把训练重心拉向关键回合和长尾局面。
- 引入 validation temperature scaling，先修概率校准问题。
- 引入关键回合切片诊断，区分 routine turns 与真正影响决策的 turns。

这一阶段的收敛结论是：

- 温度校准是最稳定、最确定的收益项。
- 第一版 critical-turn 启发式过宽，需要收紧定义后才有分析价值。
- `critical_longtail` 在当时的特征空间里扰动过强，不适合作为主线默认策略。

因此，训练主线先收敛到“稳健默认训练 + 校准 + 分桶诊断”，而不是继续围绕加权策略做细抠。

### 2.2 监督目标从 `move vs switch` 推进到 structured action family

在确认二分类目标过粗之后，主线开始推进到结构化动作监督：

- 数据侧补齐了 `action_head`、`action_family`、`move_family`、`switch_target` 等标签。
- 训练与评估链路切换到 `action_family` 任务。
- replay-level grouped split、多分类 calibration、top-k 和按桶诊断被完整保留。

- 当前工程链路可以在不重写数据和评估协议的前提下，继续向更接近真实动作空间的方向推进。
- `action_family` 成为从粗任务过渡到候选动作打分模型的桥梁层。

### 2.3 数据审计与观测边界

在同一阶段也明确了当前数据边界：

- 已有 replay 数据足够做原型验证，但还不是完整的高分段全量对战库。
- 早期数据在 item、ability、field state 等关键信息上不完整。
- 因此，模型瓶颈不能简单归因为架构本身，观测边界本身也是限制因素。

这个判断影响了后续方法论：

- 先做能在现有观测边界内验证收益的中间模块。
- 先证明 belief-state 表达和结构化动作建模真的有增益，再扩大系统边界。

## 3. 2026-04-20：Belief-State Transformer 与 hierarchical 主线确立

### 3.1 Backbone 从 MLP 迁移到实体中心编码

- Transformer 方向本身没有被证伪。
- 但第一版 MVP 没有直接打赢当时的 MLP 基线。
- 它更像是“结构表示尚未调顺”，而不是“MLP 已经足够”。

因此，当时没有立刻往上叠更高耦合的 value / planning 头，而是先转去修监督目标本身。

### 3.2 hierarchical 目标接通后，真正的收益来自目标与选模修正

随后主线从 flat family softmax 转到 hierarchical objective：

- 共享同一个 belief-state encoder。
- 拆成 `head`、`move_family`、`tera` 等多头，再组合回 family 评估口径。
- 训练目标从单头 softmax 变成更贴近动作结构的联合监督。

这条线的实际演化不是“换完 hierarchical 就赢”，而是经历了三步收敛：

2. 真正有效的修正来自多任务 loss 重平衡，而不是更激进的 decoder 改写。
3. 进一步诊断后发现，原来的 leaf-level `macro_recall` 已经与真正想优化的行为边界脱节，因此选模指标改为更合理的 `base_family_macro_recall`。

到这一轮结束时，主线结论已经稳定为：

- 默认 hierarchical 行为应对齐到基础 family 的行为质量，而不是被稀有 tera leaf 牵着走。

这也是后来 `Belief-State Transformer + hierarchical objective + base-family aligned selection` 成为主线骨架的原因。
在动作主线初步稳定后，value 模块没有直接并回主干，而是先做独立验证：

- 单独训练 belief-state state-value 模型。
- 比较 `final_win` 与短视野 `horizon_material`。
- 再比较不同 horizon 和不同 material heuristic。
- 短视野目标明显比最终胜负更适合作为第一版 value 信号。
- 当前 backbone 最擅长的是更局部、噪声更低的价值代理。
- 两回合尺度比更长 horizon 更稳。
- 在当前短视野任务里，真正有信息量的局部资源信号更像“人数差”，而不是细碎 HP 波动。

因此，value 主线当时收敛到：

- 先做独立的短视野 state value。
- 默认 heuristic 收敛到 `turn2 + alive-only` 一类更稳定的局部目标。
- 暂时不把高耦合动作条件 Q 头直接塞回动作主任务。

## 5. 2026-05-05：exact readiness 审计后，主线明确保持模糊骨架

在补完 per-sample evidence chain 之后，项目对 exact-action readiness 做了一次系统审计。结果很明确：

- exact label 覆盖已经大幅提升。
1. 当前 formal 主线继续停留在 family-level 的结构化动作排序。
2. exact 化不直接接管主任务，而是按模块逐项替换。
3. 替换顺序应优先从已经接近 exact 的子块开始，再逐步靠近完整动作排序。

这一步非常关键，因为后面所有 exactification 工作，都是在“保留 fuzzy mainline，不做一次性全切 exact”的前提下展开的。

## 6. 2026-05-06：expert-line 与 hybrid 尝试的收敛判断
- 但 pure expert reranker 不能粗暴替代 grouped local router。
- 第一版“grouped residual + expert residual 直接相加”的 hybrid 也没有形成主任务突破。

因此这一块最后收敛到的不是“放弃 expert”，而是：

- expert-style planning 适合作为偏置、门控或先验。
- 它不适合作为与 grouped scorer 同强度并列的第二套主评分器。

这也解释了后面为什么研究重心重新转回“在 grouped mainline 周围逐块 exact 化”，而不是继续深挖第一版 hybrid 结构。


这一步的目标是：在不改主任务、不改 proposal entry 的前提下，把 exact switch-target 分布变成 grouped reranker 可消费的局部信号。

结论：

- 接口、payload 和 checkpoint 链路都已经打通。
- 这证明 exact switch 可以作为一个独立 branch 接入当前主线。
随后又把 recoverable exact-candidate support 接成 grouped reranker 的另一个 late branch，并补了 confidence / lift 两类局部调优。

结论：
因此这块按既定节奏停止，不继续细调。

### 7.3 proposal-stage exact-candidate

在 late exact-candidate 没有形成净收益之后，研究重点转向 proposal stage：让 exact-candidate 不再晚接入 reranker，而是先影响谁能进入 TopK 候选集。
- 方向是对的。
- exact-candidate 更适合作为 candidate-entry bias，而不是 late reranker feature。
- proposal-only 的简单形式已经比混合 late-branch 的版本更合理。
- 在这一块内部继续做 budget / lift 微调没有带来新的决定性收益。

因此，这条线保留下来的不是某个复杂变体，而是“exact-candidate 先参与候选集入场”这个方法判断。

### 7.4 proposal-stage exact-move

随后转向 exact-move proposal。出发点很明确：exact-move 更可能适合做 candidate-set entry bias，而不是 late compatibility hint。

结论：

- 这是当时最有价值的新 exact 化子块之一。
- 它第一次把很强的覆盖能力和很强的 top1 指向性同时放进一条 proposal 路线。
- 在单支 exact proposal 里，move 方向提供了最强的“具体动作入场偏置”信号。
- 这一块只做了一轮最贴近的局部调优后就已经足够，不值得继续深挖微调。

因此，exact-move proposal 被保留下来，成为后续组合 proposal 的核心信号之一。

### 7.5 proposal-stage exact-switch

再往后转到 proposal-stage exact-switch，让最强的具体 switch target 更早进入候选集筛选。

结论：

- 这也是一个明确有效的新子块。
- switch 信号不适合裸 best-prob 直接上推；用 confidence 形态处理后更稳定。
- 它代表的是“高覆盖 / 高 switch-quality”的候选集路线，而不是更尖锐的 top1 线。

因此，exact-switch proposal 的 confidence 版本保留下来，成为后续 proposal 组合的另一条核心信号。

### 7.6 二元 proposal 组合的收敛判断

在 move、candidate、switch 三条单支 proposal 信号都摸清之后，项目对几种二元组合做了快速试探。

最终收敛判断如下：

- `move + switch`：是更均衡的折中，但没有形成新主线突破；adaptive weighting 也没有带来净收益。
- `candidate + switch`：是一个有价值的折中点，说明 candidate precision 和 switch coverage 可以有限兼容；但 adaptive 版同样没有再前进。
- `move + candidate`：这是这一轮 proposal-stage exactification 组合里的最强局部最优；固定版已经很强，adaptive 只带来很小但可接受的改进。

这一步之后，proposal 组合块的主线判断已经很清楚：

- 两个信号的高质量组合是有意义的。
- 但“固定权重相加”或“最自然的 adaptive mixing”都不是通用解。
- 当前真正值得保留的是 `move + candidate` 这条组合，而不是把所有 proposal bias 一起堆上去。

### 7.7 三信号 proposal 探针

在 `move + candidate` 已经收敛之后，又严格受限地做了一次 `move + candidate + switch` fixed 探针。

结论：

- 第三个 switch 信号确实被成功接入。
- 但它没有超过当前最优的 `move + candidate` 组合。
- 这说明 proposal 组合块的问题已经不是“再多加一个已知有效信号”就会自然更强，而是不同信号之间开始出现明显干扰。

因此，三信号块在 fixed 探针后立即停止，不继续扩成 adaptive 三信号。

### 7.8 reranker-stage 的 late move + candidate

在 proposal 组合块收束后，研究又回到 reranker-stage exactification：把 late exact-move 与 late exact-candidate 一起接入 grouped reranker。

结论：

- 这是第一个在不改 proposal entry 的前提下，把 late-branch reranker 推回到较强覆盖区间的新块。

## 8. 2026-05-07：v1 基线稳定后，recoverable exact candidate 的第二轮思路调整

在最近 10,000 局 metamon 数据上，当前 clean v1 决策模型第一次获得了足够稳定的数据规模和可重复训练结果。这使得后续 exactification 路线可以直接在当前 v1 主线里做局部实验，而不需要再借助旧 action-family 系统。

### 8.1 先建立正式 v1 基线，再判断 recoverable 强化是否值得继续推进

在 lazy-loading 训练路径稳定后，v1 基线在 10,000 局近期高分 replay 上完成了第一次正式 CUDA 训练。结论是：

- 数据规模本身已经足够支撑当前 v1 路线做正式对比。
- 单纯扩大样本量以后，coarse 决策主链已经明显比小数据基线更强。
- 因此，下一步瓶颈更像是模型设计，而不是继续单纯堆更多 replay 或更多 epoch。

### 8.2 第一轮 recoverable 强化为什么停止

第一轮 recoverable 实验采用了一个相对直接的补强方式：

- 让 recoverable entry bias 更早进入主评分链；
- 对 recoverable 子集加入统一的 local ranking loss；
- 再加统一的 hard-negative pairwise loss。

这条线的正式实验结论是否定的：

- recoverable move-like 子集的局部命中有改善；
- 但 non-recoverable 样本被过度拉向 recoverable 候选；
- 尤其 switch 行为受损明显，`switch_target_accuracy` 出现实质回退。

因此，这条线的问题不是“recoverable exact candidate 没价值”，而是“把 recoverable 作为全局统一强化项”太粗暴。

### 8.3 第二轮假设：条件启用的简单 exactification 比统一加力更合理

下一轮的工作假设收敛为：

- 主干仍由 coarse scorer 负责，不轻易让 exact 分支主导全部样本；
- recoverable exact candidate 只在高置信、局部合适的样本上进入主评分链；
- move 和 switch 不再共享完全相同的 recoverable 强化逻辑。

更具体地说，第二轮不再把 recoverable 当成“全局都该更强”的统一项，而是改成一种更克制的、条件启用的专家式补强：

1. exact entry bias 不再注入所有 recoverable candidate latent，而是只作为更简单的 gated score 项出现；
2. recoverable local ranking 先只作用在 move-like recoverable 子集，不再统一约束 switch 子集；
3. switch 仍暂时交给当前较稳的 coarse + switch-target 路线，避免 exactification 过早伤到现有优势。

这条假设在写入 analysis 之前，必须先完成一轮正式代码实验并验证：

- 是否能保住 switch 相关指标；
- 是否能保住 non-recoverable 样本的总体 top1；
- 是否还能留下 recoverable move 子集上的局部收益。

在这些结果没有明确之前，这一轮 gated move-only exactification 只保留在 worklog，不写入 analysis 作为当前默认设计。
- candidate 方向改成 residual lift 之后，明显优于 raw support。
- 但即使如此，它仍然没有逼近当前 proposal-stage 最优组合的整体质量。

因此，这一块也按“实现 + 一轮最贴近调优”收束。后续如果再回到 reranker-stage exactification，应从 lift 版继续，而不是重新回到 raw 版。

## 8. 当前收敛状态

截至目前，工作流已经收敛到下面这条主线：

1. 状态骨架仍然是 `Belief-State Transformer`。
2. 主任务仍然是 family-level 的结构化动作排序，而不是直接切到 full exact action。
3. grouped reranker 仍然是当前稳定的局部决策核心。
4. exact 化采用模块替换：谁先能稳定提供增益，谁先进入 proposal 或 reranker 的局部部件。

就当前证据看：

- proposal-stage exact 化普遍比 late-stage exact feature 更有前景。
- `move + candidate` 是目前最强的 proposal 组合块。
- late reranker 方向可以继续保留，但更适合作为次级研究线，而不是当前主线替代方案。

## 9. 下一步约束

后续继续推进时，应遵守已经验证过的节奏约束：

1. 不要再回到已经明确收束的微调块上做第三轮、第四轮微调。
2. 新 exact 子块优先从“能影响 candidate-entry 或局部排序边界”的位置切入，而不是先做更深但更弱的末端修饰。
3. 在 legal action enumeration 和 recoverable move candidate 仍不完整之前，不把主任务整体切到 exact action prediction。
4. 任何新块都应继续沿用“先实现、再做最多一两轮局部调优、随后尽快转移”的策略。

这份 worklog 之后只作为阶段决策日志维护，不再回填逐次实验流水。更细的过程细节若确有保留价值，应先整理成按主题归纳的短结论，再决定是否写回本文。

## 10. 2026-05-08：从 state-level future auxiliary 转向动作条件局部未来重构

### 10.1 为什么当前 v1 还不是真正的“局部未来驱动决策”

在 clean v1 路线稳定并完成 honest-v4 评估协议修复之后，主线性能瓶颈已经从“候选集里找不到正确动作”逐步转向“正确动作已经进入候选集，但排序仍然不够好”。

这一轮重新审视后，确认了当前 future 机制的结构性限制：

- 当前 `future_target` 是按 replay 实际轨迹构造的下一决策摘要；
- 当前模型只用 `state_context -> state_future_mean/state_future_tail` 预测这个摘要；
- 当前动作总分并不真正消费每个候选动作各自的未来摘要。

因此，现有 future 头只是一个 state-level auxiliary，帮助共享状态编码器知道“这类局面未来大概会怎么演化”，但它回答不了真正关键的问题：

> 如果当前我选动作 `a_1` 而不是 `a_2`，未来局部影响会有什么差异？

这意味着模型仍然主要在做 `state + candidate -> score` 的静态排序，而不是 `state + candidate -> local consequence -> score` 的因果式局部决策。

### 10.2 本轮设计目标

在不引入完整 simulator rollout、也不要求精确多回合对手分支树监督的前提下，本轮重构采用一个可落地的第一版近似：

> 动作条件局部未来 + 计划条件效用混合。

具体地，把每个合法候选动作都映射到一个局部未来摘要：

$$
\hat y_t(a) = f_\theta(\xi_t, a)
$$

其中 $\hat y_t(a)$ 仍然保留当前工程里已经稳定存在的八维摘要接口：

- `delta_plan`
- `delta_belief`
- `delta_resource`
- `delta_phase`
- `delta_unlock`
- `safe_score`
- `tempo_score`
- `convert_score`

同时再预测一个 tail-risk 版本：

$$
\hat y_t^{tail}(a) = g_\theta(\xi_t, a)
$$

它不再代表完整世界模型的多步分布，而是代表“若把未来压成少数局部风险摘要维度时，这个动作潜在的负面尾部影响有多大”。

### 10.3 计划条件效用混合

为了避免“同一个局部未来摘要在不同阶段/计划下被同样解释”，本轮不直接把 `candidate_future_mean` 线性塞回总分，而是引入 plan-conditioned utility mixture：

$$
S_t^{local}(a)
=
\langle \hat y_t(a), w_t^{plan} \rangle
-
\langle |\hat y_t^{tail}(a)|, r_t^{plan} \rangle
$$

其中：

- $w_t^{plan}$ 来自当前 `plan_posterior` 对计划效用向量的加权混合；
- $r_t^{plan}$ 来自当前 `plan_posterior` 对风险惩罚向量的加权混合。

这样做的意图是：

- 在 `preserve` / `stabilize` 更强时，模型应更看重风险下界与资源保留；
- 在 `pressure` / `convert` 更强时，模型应更看重 unlock、tempo 与 conversion 进展；
- 计划状态不再只作为独立辅助头存在，而开始真正参与候选动作排序。

### 10.4 第一版可监督近似

离线 replay 数据没有给出“每个未选动作的真实未来”，因此这轮实现不做伪精确的全候选监督，而采用一个明确但克制的近似：

- 只对 gold action 对应的候选行施加局部未来监督；
- 其余候选动作仅通过共享参数泛化获得局部未来预测；
- 原有 state-level future head 保留，用于维持更稳定的全局局面演化表征。

## 11. 2026-05-08：把 Showdown/Smogon 静态资料真正接入 v1 主线

### 11.1 这轮修改解决的不是“知识缺失”，而是“知识没有进入候选打分主链”

在补齐 `data/static/gen9` 全量固定规则事实和 `data/static/gen9ou` usage/moveset 统计之后，代码层面最初只完成了两件事：

- 固定规则事实进入了 `static_rules.py`，因此 species / move / matchup 的规则特征已经可用；
- 外部 usage 统计被下载并整理成独立 JSON，但还没有真正进入当前 v1 决策主链。

这意味着当时模型虽然“拥有这些资料”，但它们并没有直接改变 hidden candidate 的生成，也没有直接改变 legal candidate 的排序输入。因此，本轮工作不是再增加新的资料文件，而是把已有资料放进最有决策价值的位置。

### 11.2 本轮主线接入位置

这一轮选择了两个最直接影响 exact legal-action 排序的位置：

1. hidden candidate proposal；
2. candidate scoring prior block。

具体实现是：

- 新增 `usage_priors.py`，把 `data/static/gen9ou/usage.json` 解析成 species-level meta prior；
- 对每个 species 提供 `usage_percent`、top moves、items、abilities、tera types 和 move-family 聚合 usage；
- `DecisionTensorDataset` 在 hidden move proposal 时，不再只依赖 train-only hidden prior，而是把 train prior 与 Smogon top moves / top families 合并排序；
- legal candidate 现在额外携带一组外部 prior 特征：species usage、move usage、family usage、以及 item/ability/tera 的 set-context score；
- `PokeStrategistDecisionModel` 的 candidate prior block 不再只有“训练 prior + support”两维，而是改成“训练先验 + 外部 meta prior”的联合输入。

这个设计保持了原有 exact legal-action 主线不变：

- fixed rule facts 继续通过静态规则特征进入模型；
- external usage 只作为 candidate prior / proposal bias，不替代 legal action 枚举；
- replay 中真实观测到的动作、revealed action 和 legal switch 仍然不受 usage 截断控制。

### 11.3 正式实验结论

在 `pokestrategist_v1_metamon_gen9ou_1550_recent10000_honestv4.jsonl` 上完成正式 3-epoch CUDA 训练后，这条 usage-aware 主线相对于前一条 `honestv4_vnext_gatedfuture_v1` 主线给出了明确的正向结果：

- `covered_top1` 从约 `0.3371` 提升到 `0.3735`；
- `covered_top3` 从约 `0.6902` 提升到 `0.7138`；
- `gold_top1` 从约 `0.3091` 提升到 `0.3327`；
- `switch_target_species_accuracy` 从约 `0.3069` 提升到 `0.3509`；
- `ranking_score` 从约 `0.5139` 提升到 `0.5331`；
- `selection_score` 从约 `0.4884` 提升到 `0.5140`。

这说明把外部 usage/moveset 资料接到 hidden candidate proposal 和 candidate prior block 是对的，收益不是只体现在某个单独辅助头，而是已经落到当前主评估口径上。

### 11.4 新出现的权衡

这条线也暴露了新的结构性权衡：

- `gold_candidate_coverage` 从约 `0.9169` 下降到 `0.8907`；
- `false_switch_rate_on_move` 从约 `0.2599` 上升到 `0.2890`；
- `ECE` 从约 `0.0171` 上升到 `0.0245`，虽然仍然处在较低水平。

这表明外部 meta prior 现在更像是在做“更尖锐的候选排序”，而不是单纯扩大候选覆盖。也就是说，这轮改动在当前形态下主要提升了命中后的排序质量和 switch 决策质量，但仍然可能把一部分 hidden move 覆盖挤掉。

因此，当前收敛判断不是“回退 usage prior”，而是：

- usage-aware proposal / scoring 已经证明应该保留在主线；
- 下一步若继续优化，应优先修 `candidate coverage` 与 `move-side false switch` 的新副作用；
- 最自然的后续方向，是把 train prior 与 external usage prior 从“线性混合”推进到更条件化的 gated mixture，而不是重新把外部资料移出主线。

这条近似的含义非常明确：

- 本轮不是完整反事实世界模型；
- 但它已经把最关键的缺失部件补上了，即 `candidate-conditioned local consequence`；
- 如果这一步有效，下一轮再继续往 `candidate-conditioned opponent response` 扩展才有意义。

### 10.5 本轮代码化范围

本轮正式实现只包含四个结构变化：

1. 保留原有 `state_future_*` 头作为全局 auxiliary。
2. 新增 `candidate_future_mean` / `candidate_future_tail` 头，对每个合法候选动作输出局部未来摘要。
3. 新增 `plan_value_embedding` / `plan_tail_embedding`，把 `plan_posterior` 混合成当前局面的效用权重与风险权重。
4. 把 `local_score_weight` / `local_tail_weight` 真正接入 `legal_action_scores`，使局部未来首次进入候选排序主链。

这意味着 clean v1 从这一轮开始，不再只是“有 future 辅助监督的动作排序器”，而是进入第一版“动作条件局部未来决策器”。

### 10.6 本轮没有实现的部分

为了保持一次迭代只解决一个主问题，本轮明确不做以下扩张：

- 不引入完整对手响应树；
- 不引入显式 `P(o \mid x, a)` 的候选条件响应分布；
- 不引入多步 rollout 蒙特卡洛；
- 不引入 full exact simulator 近似；
- 不引入额外的自然语言 explanation head。

这些部分仍属于同一大方向，但必须建立在“候选动作自己的局部未来已经能稳定进入排序主链”之后。

### 10.7 第一轮代码实验结论

本轮已经把 `candidate-conditioned local future` 正式接进了代码：

- 为每个合法候选动作新增 `candidate_future_mean` / `candidate_future_tail` 头；
- 用 gold action 对应候选行做局部未来监督；
- 把计划后验混合成未来收益权重与风险权重；
- 尝试把局部未来分数直接接入 `legal_action_scores` 主链。

在真实 `honest-v4` 数据上的第一轮 smoke 给出了一个非常明确的判断：

1. 方向本身没有被证伪，说明 `candidate-conditioned future` 不是空信号。
2. 但“直接进入主分数”的第一版实现干扰了现有 coarse scorer，尤其明显伤到 switch 路径。
3. 即使把局部未来限制为只作用于 move-like 候选，第一版参数化仍未形成净收益。
4. 即使把局部未来退回成 auxiliary-only 监督，这一版实现也还会明显扰动主任务，因此暂时不能把 `candidate future loss` 设成默认训练项。

因此，这一轮的工程决策不应理解为回退，而应理解为第一版融合方式过粗。下一步的正确处理不是把该模块降为长期实验开关，而是把拖累它的旧耦合方式一起改掉：

- `candidate-conditioned future` 继续保留为 vNext 主线部件；
- future 分支不再直接把梯度回灌到旧 `action_encoder`，避免辅助 future loss 扭曲旧候选表示；
- local future 进入主分数前必须先做候选集内标准化和有界化，避免未校准 raw future score 压倒 imitation / branch scorer；
- local future 通过 learned gate 作为 residual 进入主链，而不是无门控全量相加；
- `local_score_weight`、`local_tail_weight`、`local_future_weight` 恢复为非零默认值，使 vNext 不再回退到旧 scorer。

换句话说，本轮之后的判断是：旧方法瓶颈已经明确，动作条件局部未来必须继续作为主线推进；需要修改的是粗暴融合方式和旧 scorer 的梯度耦合，而不是放弃这个方向。

### 10.8 vNext 门控残差版本的正式结果

根据上述判断，随后实现了 vNext 版门控残差融合：

- future 分支输入使用 detached action-state pair，避免 `candidate_future_loss` 直接扭曲旧 action encoder；
- local future score 在候选集内做标准化，再经 `tanh` 有界化；
- local future 通过 learned gate 作为 residual 进入主链；
- `local_score_weight`、`local_tail_weight`、`local_future_weight` 恢复为非零默认值。

这一版在完整 3 epoch honest-v4 正式训练中得到的结论是：

- `covered_top3` 从旧 honest-v4 的约 0.6850 提升到约 0.6902；
- `covered_top1` 从约 0.3363 提升到约 0.3371；
- `ranking_score` 从约 0.5128 提升到约 0.5139；
- `switch_target_species_accuracy` 从约 0.3240 回落到约 0.3069；
- `false_switch_rate_on_move` 基本持平，约 0.260。

这不是突破性大幅提升，但它改变了上一轮的结论：candidate-conditioned future 在经过隔离、标准化和门控后，已经可以作为主链残差带来小幅 ranking 收益，而不是只能作为不可用实验开关存在。

因此，当前 vNext 判断更新为：

- 动作条件局部未来应继续保留为默认主线部件；
- 下一步不是回退，而是解决它和 switch 专家之间的协同问题；
- 未来模块的正确方向应从“单一 residual”继续升级为 `move future scorer` 与 `switch future/resource scorer` 分治，或者引入 opponent-response-conditioned future 来减少 switch 退化。

## 12. 2026-05-09：candidate-side 之后继续完成 state-side 全面解耦

### 12.1 为什么这一步必须继续往 state side 走

在 candidate-side 已经拆成 `structure / identity / prior / meta / matchup` 五个子塔之后，主链里仍然保留着一个明显的耦合瓶颈：

- `self / opp / history / numeric` 仍然先被压进一个共享 `history_state`；
- `plan / belief / resource / phase / unlock` 仍然从同一个共享状态直接出头；
- `route / response / future / branch / support / rule` 仍然主要复用同一个 `state_context`。

这会导致一个结构问题：即使 candidate 侧已经解耦，不同语义的 state 证据仍然在同一个上下文里相互污染，尤其是 switch 路由、对手响应、局部未来和规则价值这些本来就应分治的子任务。

因此这一轮的目标不是再加新头，而是把当前 v1 主链里剩余最大的共享 state bottleneck 一次拆开。

### 12.2 本轮代码改动范围

这轮实现做了三件核心事：

1. 把 observation state 拆成 `self / opp / history / numeric` 四个输入子塔，再融合成 `observation_state`。
2. 在 `observation_state` 之上，为 `plan / belief / resource / phase / unlock` 分别建立独立 reason-state 子塔，而不是共用一个 `history_state` 线性出头。
3. 再往上把候选评分链真正拆成独立 state contexts：

- `decision_context` 供 imitation 主打分使用；
- `route_context` 只供 `head_router` 使用；
- `response_context` 只供 state response 与 candidate response 使用；
- `future_context` 只供 state future 与 candidate future 使用；
- `branch_context` 只供 revealed/hidden/abstract/switch experts 使用；
- `support_context` 只供 recoverable support 使用；
- `rule_context` 只供 move/switch rule scorer 使用。

同时，candidate response logits 不再复用 rule pair，而是改成独立的 `response_pair`。

### 12.3 回归验证

这轮改动先后通过了三层验证：

- 聚焦前向测试：`tests/test_decision_model_v1.py`；
- train/eval CLI 聚焦测试：`tests/test_train_v1_cli.py`；
- 全量 pytest 回归：10 个测试全部通过。

此外，`tests/test_decision_model_v1.py` 已新增 state-side 子塔和各类 state context 的显式 shape 断言，因此后续若再把这些子空间重新耦合起来，测试会直接暴露接口回退。

### 12.4 pilot 结果

在 `4096` 样本、`1 epoch` 的 CUDA pilot 上，这条 full-decouple 线路相对前一条 candidate-only decoupled pilot 给出明确的 move-side ranking 改善：

- `gold_top1` 约 `0.2083 -> 0.2418`；
- `covered_top1` 约 `0.2332 -> 0.2654`；
- `covered_top3` 约 `0.5565 -> 0.5794`；
- `gold_candidate_coverage` 约 `0.8935 -> 0.9113`；
- `top1_family_accuracy` 约 `0.4338 -> 0.4995`；
- `false_switch_rate_on_move` 约 `0.6241 -> 0.3798`。

但 pilot 也同时暴露出 switch 侧代价：

- `switch_target_accuracy` 约 `0.3214 -> 0.2413`；
- `head_routing_accuracy` 约 `0.4255 -> 0.3633`。

因此 pilot 阶段的判断是：state-side 全面解耦确实抓到了旧共享 state 的噪声问题，但当前分治方式对 switch routing 的保护还不够。

### 12.5 正式 3 epoch 结果与当前结论

在完整 `honest-v4 + usage prior` 正式训练上，`runs/pokestrategist_v1_metamon_recent10000_honestv4_usageprior_fulldecouple_v1` 与当前正式主线 `usageprior_v1` 的对照结果如下：

- `covered_top1`: `0.3735 -> 0.3624`
- `covered_top3`: `0.7138 -> 0.7042`
- `gold_top1`: `0.3327 -> 0.3227`
- `switch_target_species_accuracy`: `0.3509 -> 0.3155`
- `ranking_score`: `0.5331 -> 0.5262`
- `selection_score`: `0.5140 -> 0.5019`

但这轮也有两个明确正向信号：

- `false_switch_rate_on_move`: `0.2890 -> 0.2832`
- `ECE`: `0.0245 -> 0.0106`

因此这一轮的结论不是“全面解耦方向错误”，而是：

1. 把 state bottleneck 拆开是合理的，至少它明显改善了校准和 move-side 假 switch 偏置。
2. 但当前 route / switch / response / future 的上下文切分还不够好，特别是 switch-target 和 covered ranking 一起回落，说明分治边界还没有对齐真实监督。
3. 当前 full-decouple 版本还不能直接替换 `usageprior_v1` 成为新的正式主线。

下一步最自然的修正方向已经明确为：

- 不回退 candidate/state 解耦本身；
- 优先修 `route_context` 与 `branch_context` 的 switch 协同，而不是重新把所有 state 合回一个共享向量；
- 在 trainer/loss 侧引入更显式的 decoupled context 对齐监督，否则最终 imitation 排序仍会把这些独立子空间重新拉回混合用途。

### 12.6 Full-Closure 正式首轮结果

在把 analysis 中剩余结构环节一次性接进代码之后，先用旧版 `data/processed/pokestrategist_v1_metamon_gen9ou_1550_recent10000_honestv4.jsonl` 跑了一轮 CUDA 训练。流程虽然成功，但后续检查发现那份旧 JSONL 早于最新 schema：新增的 `particle_posterior`、`reveal_likelihood` 和 `typed_consequence` 目标都会在读取时退化成默认零值，所以那一轮结果不能当作新结构的正式验证。

随后用最新 `dataset_builder` 从 `data/raw/metamon_gen9ou_1550_recent10000.jsonl` 重建了 `data/processed/pokestrategist_v1_metamon_gen9ou_1550_recent10000_honestv4_fullclosure_20260510.jsonl`，样本数保持不变：`10000` replays、`542613` samples。对重建后数据做抽样检查可以确认新增目标已经恢复为非平凡分布：`particle` 目标均值标准差约 `0.1228`，`reveal` 各轴均值约 `[0.3934, 0.0268, 0.0054, 0.0358, 0.6346]`，typed consequence 各轴中心 bin 占比也不再退化为全 `1.0`。

在这份 rebuilt full-closure 数据上完成的正式 3 epoch CUDA 训练产物位于 `runs/pokestrategist_v1_metamon_recent10000_honestv4_fullclosure_20260510_rebuilt_v1`，best epoch 为 `3`，主要指标如下：

- `gold_top1 = 0.3144`
- `gold_in_top3 = 0.6154`
- `covered_top1 = 0.3549`
- `covered_top3 = 0.6947`
- `switch_target_accuracy = 0.2897`
- `line_accuracy = 0.6214`
- `particle_mae = 0.0711`
- `reveal_bce = 0.2882`
- `typed_consequence_axis_accuracy = 0.7525`
- `ECE = 0.0423`

相对上一条正式 `lineconsequence_v1` 主线，这轮 rebuilt full-closure 训练给出两个明确事实。

第一，新增结构已经被真正训练到了：`particle_mae`、`reveal_bce` 和 `typed_consequence_axis_accuracy` 都进入了非平凡区间，说明先前那种“全部接线但标签默认零”的假阳性已经被排除。

第二，当前默认损失权重下，full-closure 版整体还不是主线升级：`covered_top1` 从 `0.3723` 降到 `0.3549`，`covered_top3` 从 `0.7130` 降到 `0.6947`，`switch_target_accuracy` 从 `0.3371` 降到 `0.2897`，`line_accuracy` 从 `0.6358` 降到 `0.6214`，`ECE` 也从 `0.0200` 变差到 `0.0423`。换句话说，现在的问题已经不是“这些结构没有被训练”，而是“这些结构在当前 objective mixing 下干扰了主 ranking / switch 目标”。

因此这一轮的工程结论很清楚：

1. full-closure 架构已经完成第一次真实 formal run，不再需要把它视为“只完成实现、未训练验证”的分支；
2. 当前 full-closure 默认配方还不能替代 `lineconsequence_v1` 成为新的正式主线；
3. 下一步不应回退这些结构，而应优先做 loss-weight、selective coupling 与 auxiliary-to-ranking transfer 的修正，特别是减少 particle / reveal / consequence / frontier 对 exact ranking 与 switch target 的负迁移。