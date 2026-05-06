# LoRA 方向文献矩阵

> 研究主题：基于智能优化的自适应低秩大模型微调方法研究  
> 筛选来源：以 [Yuheng2000/Awesome-LoRA](https://github.com/Yuheng2000/Awesome-LoRA) 和 [ZJU-LLMs/Awesome-LoRAs](https://github.com/ZJU-LLMs/Awesome-LoRAs) 为入口，只保留与 LoRA、动态/自适应 rank、智能优化、量化、初始化相关的文献。  
> 核实原则：优先使用论文摘要、README、ACL/PMLR/HF paper page、官方代码仓库。未读到或未确认的信息标记为“待核实”。

## 筛选结论

本论文方向最相关的主线是：

1. **基础 baseline**：LoRA、QLoRA。
2. **自适应 rank / 动态 rank**：AdaLoRA、DyLoRA、DoRA、AutoLoRA、Bayesian-LoRA。
3. **智能优化或无梯度优化**：Derivative-Free Optimization for LoRA。
4. **优化过程改进**：LoRA+。
5. **初始化与 rank 重分配**：LoRA-GA、PiSSA、EVA。

## 文献矩阵

| 类别 | 论文名称 | 年份 | 核心问题 | 方法核心 | 是否有代码 | 是否适合作为 baseline | 和“智能优化 + 自适应 rank LoRA”的关系 | 我可以借鉴什么 | 不适合直接做什么 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 原始 LoRA | [LoRA: Low-Rank Adaptation of Large Language Models](https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/) | 2022 | 全参数微调成本高，多个任务保存完整模型不现实。 | 冻结预训练权重，在 Transformer 权重旁注入低秩矩阵，仅训练低秩增量。 | 有，[microsoft/LoRA](https://github.com/microsoft/LoRA)；HF PEFT 已支持。 | **必须 baseline**。统一 rank LoRA 是所有改进方法的参照。 | 你的 PSO rank_pattern 搜索应以 LoRA 的固定 rank 配置为对照。 | LoRA 模块位置、rank/alpha/dropout、训练参数量统计、无额外推理延迟的表述方式。 | 不应只复现 LoRA 后声称“自适应”；LoRA 本身不解决 layer-wise rank 分配。 |
| Dynamic Rank Allocation | [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://www.microsoft.com/en-us/research/publication/adaptive-budget-allocation-for-parameter-efficient-fine-tuning/) | 2023 | LoRA 将参数预算均匀分配到所有矩阵，忽略不同层/矩阵重要性。 | 用 SVD 形式参数化增量，根据重要性评分和预算调度剪掉不重要奇异值，实现自适应预算分配。 | 有，[QingruZhang/AdaLoRA](https://github.com/QingruZhang/AdaLoRA)；HF PEFT 有 AdaLoraConfig。 | **强 baseline**，尤其适合作为“自适应 rank”对比。 | 与你的方向高度重合：都是 layer-wise / matrix-wise rank 预算分配。区别是 AdaLoRA 基于训练中重要性评分，你计划用 PSO 搜索。 | 重要性评分、预算调度、target_rank/init_rank、低预算实验设计。 | 不适合直接照搬为“智能优化”；它不是群智能/黑盒优化框架。 |
| Dynamic Rank Allocation | [DyLoRA: Parameter-Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://aclanthology.org/2023.eacl-main.239/) | 2023 | 固定 rank LoRA 训练后不能灵活切换 rank，搜索 rank 需要重复训练。 | 一次训练支持一系列 rank，通过排序/组织 adapter 表示，让不同 rank 子模型可用。 | Awesome 仓库标注有代码；官方代码链接需进一步核实。 | **可作为补充 baseline**，尤其用于“无需搜索”的动态 rank 对比。 | 你做 PSO 搜索时，可把 DyLoRA 作为“训练一次覆盖多 rank”的反方向基线。 | rank 范围训练、不同 rank 共享权重、避免穷举训练的实验动机。 | 不适合直接作为 layer-wise rank_pattern 搜索方法；它更关注单个 adapter 在多个 rank 下可用。 |
| Dynamic Rank Allocation | [DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution](https://huggingface.co/papers/2405.17357) | 2024 | LoRA 旁路结构忽略不同权重矩阵对参数预算的差异化需求。 | 将高 rank LoRA 层拆成结构化单 rank 组件，训练中按重要性动态剪枝/分配预算。 | 论文页标注代码可用；具体仓库和可复现性待核实。 | **可作为动态 rank baseline**，但要注意名称容易与 Weight-Decomposed LoRA 的 DoRA 混淆。 | 与 PSO rank_pattern 搜索高度相关，都是在固定预算下选择哪些 rank 组件保留。 | 单 rank 组件拆分、重要性剪枝、相同存储预算下对比。 | 不适合在论文中不加说明地简称 DoRA，需明确是 Dynamic Rank Distribution，不是另一篇 Weight-Decomposed LoRA。 |
| QLoRA | [QLoRA: Efficient Finetuning of Quantized LLMs](https://huggingface.co/papers/2305.14314) | 2023 | 大模型 LoRA 微调仍受显存限制，65B 级模型难以在单卡训练。 | 冻结 4-bit 量化基础模型，通过 LoRA 反传；引入 NF4、double quantization、paged optimizers。 | 有，[artidoro/qlora](https://github.com/artidoro/qlora)。 | **必须 baseline**，尤其是本项目支持 QLoRA 时。 | 你的自适应 rank 搜索需要报告在 LoRA 和 QLoRA 下是否稳定；QLoRA 还能作为显存效率 baseline。 | 4-bit 配置、NF4、paged optimizer、显存/性能权衡的实验记录方式。 | 不适合把 QLoRA 当作 rank 自适应方法；它主要解决量化与显存。 |
| LoRA+ | [LoRA+: Efficient Low Rank Adaptation of Large Models](https://proceedings.mlr.press/v235/hayou24a.html) | 2024 | LoRA 中 A/B 两个 adapter 矩阵使用相同学习率可能导致大宽度模型特征学习不足。 | 对 LoRA A/B 设置不同学习率比例，在不增加参数量的情况下改进优化过程。 | 有，[nikhil-ghosh-berkeley/loraplus](https://github.com/nikhil-ghosh-berkeley/loraplus)。 | **适合作优化 baseline**，与 rank 搜索正交。 | PSO 找 rank，LoRA+ 调学习率；可组合成“rank_pattern + LoRA+ optimizer”消融。 | A/B 分组参数、学习率比例 `loraplus_lr_ratio`、同计算成本对比。 | 不适合替代自适应 rank；它不决定每层 rank。 |
| AutoLoRA | [AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning](https://aclanthology.org/2024.naacl-long.282/) | 2024 | LoRA 统一 rank 和穷举 rank 搜索成本高，且可能导致次优性能。 | 将每个 rank-1 分量关联选择变量，用训练/验证双层或元学习过程学习选择变量，再阈值化确定 rank。 | 论文页未读到明确官方代码；Awesome-LoRA 有代码链接但需核实。 | **理论上适合作强 baseline**；若代码不可用，可作为方法论对比。 | 与你的方向最直接：都是自动确定每个 LoRA 层 rank。AutoLoRA 用元学习，你计划用 PSO。 | 选择变量建模、验证集驱动 rank 选择、rank-1 分量粒度的搜索空间。 | 不适合直接照抄元学习过程；你的创新点应强调 PSO/群智能搜索、成本约束和可复现实验。 |
| Derivative-Free Optimization | [Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models](https://ar5iv.labs.arxiv.org/html/2403.01754) | 2024 | LoRA 虽减少参数，但仍需梯度反传；黑盒/少样本场景中希望避免梯度计算。 | 在 self-attention 层加入低秩模块，用 CMA-ES、Fireworks Algorithm 等无梯度优化方法交替优化低秩模块。 | 论文 HTML 脚注给出代码：[stan-anony/derivative_free_lora_rank](https://github.com/stan-anony/derivative_free_lora_rank)；仓库可用性待核实。 | **适合作“智能优化 baseline/启发”**，但不是主流 SFT baseline。 | 与你的“智能优化”高度相关，PSO 同属黑盒/群智能优化范式，可借鉴其把层级低秩优化拆成子问题。 | DFO 问题建模、API/forward-only 成本、少样本设置、分层交替优化。 | 不适合直接照搬到 Qwen SFT；该方法更偏少样本分类/黑盒优化，生成式指令微调需重新验证。 |
| Bayesian-LoRA | [Bayesian-LoRA: LoRA based Parameter Efficient Fine-Tuning using Optimal Quantization levels and Rank Values trough Differentiable Bayesian Gates](https://huggingface.co/papers/2406.13046) | 2024 | 统一 rank 和缺少量化联合优化会影响效率；希望同时选择 rank 与 quantization level。 | 对 rank 值和量化级别引入先验，通过可微 Bayesian gates 学习每个低秩矩阵的 rank 与量化配置。 | 未在主要来源读到官方代码，标记为**待核实**。 | **不建议作为首批可复现 baseline**；适合写相关工作和方法启发。 | 与你的方向非常近：rank 搜索 + 资源约束 + 量化，但它是可微贝叶斯门控，你是 PSO。 | 把 rank 和 bit 操作/量化成本联合进目标函数；用先验或惩罚项表达资源偏好。 | 不适合在无代码和未复现实验前作为核心对比结论；也不宜把 DeBERTa/GLUE 结果直接迁移到 Qwen/GSM8K。 |
| Bayesian LoRA / 校准 | [Bayesian Low-rank Adaptation for Large Language Models](https://huggingface.co/papers/2308.13111) | 2023 | 小数据微调后的 LLM 容易过度自信，需要不确定性估计和校准。 | Laplace-LoRA 对 LoRA 参数做 Laplace approximation，改进微调后模型校准。 | 有，[adamxyang/laplace-lora](https://github.com/adamxyang/laplace-lora)。 | **不适合作 rank baseline**；适合作“不确定性/贝叶斯 LoRA”相关工作。 | 与自适应 rank 不是同一问题，但可启发你把搜索不确定性或验证集置信度纳入目标函数。 | Bayesian posterior、校准指标、少样本不确定性分析。 | 不适合直接拿来解决 layer-wise rank_pattern 搜索。 |
| LoRA-GA | [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://huggingface.co/papers/2407.05000) | 2024 | LoRA 随机/零初始化导致早期收敛慢，可能拉高总训练成本。 | 在小批训练样本上估计梯度，对梯度矩阵做分解，用梯度方向初始化 LoRA adapter。 | 有，[Outsider565/LoRA-GA](https://github.com/Outsider565/LoRA-GA)。 | **适合作初始化 baseline**，尤其可与固定 rank、PSO rank 对比。 | PSO 搜 rank 时，初始化方式会影响每个候选 rank 的评估公平性；LoRA-GA 可作为强初始化对照。 | 用少量样本估计梯度、初始化后再标准 LoRA 训练、初始化成本单独记录。 | 不适合直接宣称解决 rank 分配；它主要优化初始化和收敛。 |
| PiSSA 初始化 | [PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://huggingface.co/papers/2404.02948) | 2024 | LoRA 的 Gaussian/zero 初始化可能收敛慢；量化场景中初始化误差也重要。 | 对预训练权重做 SVD，用主奇异值/奇异向量初始化可训练部分，剩余部分冻结；兼容量化。 | 有，[GraphPKU/PiSSA](https://github.com/GraphPKU/PiSSA)。 | **强初始化 baseline**，尤其适合 GSM8K/Qwen 类实验。 | 与你的方法可组合：PSO 搜 rank_pattern，PiSSA 决定每个 rank 下如何初始化。 | SVD 初始化、QPiSSA、初始化耗时、同 rank 下 LoRA vs PiSSA 对比。 | 不适合把 PiSSA 当作智能优化；它不搜索 rank_pattern。 |
| EVA 初始化与 rank 重分配 | [One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation](https://huggingface.co/papers/2410.07170) | 2024 | LoRA 通常随机初始化且各层 rank 均匀，导致收敛和参数效率可能不理想。 | 用下游数据 activation 做 SVD，按 explained variance 初始化 LoRA，并在层间重分配 rank。 | 有，[ml-jku/EVA](https://github.com/ml-jku/EVA)；HF PEFT 文档也提供 EVA 配置与初始化接口。 | **非常适合作强 baseline**，因为同时涉及初始化和 rank redistribution。 | 与你的论文方向高度重合：都是数据/任务相关的 layer-wise rank 分配。EVA 用激活方差，你用 PSO 目标函数搜索。 | activation SVD、variance-based rank redistribution、初始化前向采样、rank 预算约束。 | 不适合忽略其数据驱动初始化成本；与 PSO 比较时应统一总预算和额外前向/搜索开销。 |

## 与本论文实验设计的对应关系

| 实验问题 | 推荐对比方法 | 目的 |
| --- | --- | --- |
| 固定 rank LoRA 是否足够？ | LoRA r=4/8/16 | 建立基础性能、参数量、显存基线。 |
| 量化是否影响 rank 搜索？ | QLoRA r=4/8/16 | 区分 rank_pattern 收益和 4-bit 量化收益。 |
| 训练中自适应 rank 是否强于 PSO 搜索？ | AdaLoRA、AutoLoRA、DoRA | 对比“可微/重要性/元学习”与“群智能黑盒搜索”。 |
| 搜索-free 动态 rank 是否更划算？ | DyLoRA | 对比一次训练覆盖多个 rank 与多候选搜索。 |
| 优化器差异是否影响结论？ | LoRA+ | 排除学习率分组带来的收益混淆。 |
| 初始化是否影响 rank_pattern 评估？ | LoRA-GA、PiSSA、EVA | 判断 PSO 搜到的 rank 是否依赖初始化。 |
| 智能优化是否合理？ | Derivative-Free LoRA | 为 PSO、CMA-ES、FWA 等黑盒优化提供相关工作依据。 |
| 是否能联合 rank 与压缩成本？ | Bayesian-LoRA、QLoRA | 启发目标函数加入 bit operations、显存和训练时间惩罚项。 |

## 8 篇必须精读论文清单

1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)  
   原始 baseline，必须掌握 LoRA 参数化、rank-deficiency 动机和实验设置。

2. [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)  
   与“自适应 rank/预算分配”最核心相关，是你的方法必须比较或讨论的对象。

3. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)  
   项目支持 QLoRA，必须理解 NF4、double quantization、paged optimizer 和 LoRA 组合方式。

4. [AutoLoRA: Automatically Tuning Matrix Ranks in Low-Rank Adaptation Based on Meta Learning](https://aclanthology.org/2024.naacl-long.282/)  
   直接处理自动 rank tuning，可作为 PSO rank 搜索的近邻方法。

5. [Derivative-Free Optimization for Low-Rank Adaptation in Large Language Models](https://arxiv.org/abs/2403.01754)  
   与“智能优化/无梯度优化 + LoRA”最直接相关，可支撑 PSO 方法动机。

6. [LoRA+: Efficient Low Rank Adaptation of Large Models](https://proceedings.mlr.press/v235/hayou24a.html)  
   重要优化 baseline，能帮助区分 rank 搜索收益和优化器/学习率收益。

7. [PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models](https://arxiv.org/abs/2404.02948)  
   强初始化 baseline，尤其适合与 QLoRA/GSM8K 实验结合。

8. [One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation](https://arxiv.org/abs/2410.07170)  
   同时涉及数据驱动初始化和 rank 重分配，是最接近“自适应 rank LoRA”的强相关工作之一。

## 待核实事项

- AutoLoRA 的官方可复现实验代码：Awesome-LoRA 中有代码入口，但 ACL/HF paper page 未直接确认官方仓库。
- DoRA 动态 rank 分配代码仓库的稳定性与实现细节：需避免与 Weight-Decomposed LoRA 的 DoRA 混淆。
- Bayesian-LoRA 是否有公开官方代码，以及其 GLUE/DeBERTa 设置能否迁移到 Qwen/GSM8K。
- Derivative-Free LoRA 代码仓库是否仍可访问、是否能在当前 Transformers/PEFT 版本下运行。
- EVA 的会议版本与最终题名/年份：当前按 arXiv/HF paper page 记为 2024，正式出版信息需写论文时再核实。

