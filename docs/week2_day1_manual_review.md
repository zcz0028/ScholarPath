# ScholarPath Week 2 Day 1 人工复核

## 1. 复核目标

本次复核以 `outputs/week2_day1_diagnostics` 中的离线诊断结果为依据，目标是定位 B4 Top100 中 zero-recall queries 的具体失败阶段，并为第二周后续召回增强确定优先级。复核过程只读取第一周已有输出和 gold 标准答案，不调用 OpenAlex API，也不改变正式评测口径。

本次复核需要回答三个问题：

1. 正确论文是否曾进入 B0、B1.2、B2 或 B4 融合候选池；
2. zero-recall 是由候选召回失败还是语义重排失败造成；
3. 对于真实召回失败查询，后续应采用哪一类定向救援策略。

## 2. 自动诊断结果

- 查询总数：50
- 目标阶段：`b4_top100`
- B4 Top100 总 TP：61
- B4 Top100 有命中的查询：24
- B4 Top100 zero-recall queries：26
- `retrieval_miss`：24
- `ranking_loss_top100`：2
- `possible_matching_issue`：0
- API 调用数：0
- 评测模式：`strict`

总体上，26 条 zero-recall queries 中有 24 条属于候选召回失败，只有 2 条属于候选已进入 B4 Pool、但在语义重排后落到 Top100 之外。因此，第二周的主要瓶颈位于候选召回阶段，而不是 Selector 或 Constraint Guard。

## 3. P0：排序失败查询复核

### 3.1 RealScholarQuery_25

- 原始查询：`Video aesthetics score, using multimodal large models.`
- Gold 数量：1
- 命中的 Gold 论文：`Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels`
- 自动识别锚点：`Video`、`multimodal`
- B0 Top100：命中 1 篇，最佳排名 45
- B1.2 Top100：命中 1 篇，最佳排名 43
- B2 Top100：命中 1 篇，最佳排名 44
- B4 Pool：命中 1 篇，融合池顺序排名 42
- B4 Top100：命中 0 篇
- 失败类型：`ranking_loss_top100`
- 优先级：P0

初步判断：该正确论文在 B0、B1.2、B2 三条召回链路中均已出现，并成功进入 B4 融合候选池，因此不存在候选召回不足。论文在融合池中的顺序排名为 42，但经过 B4 轻量语义重排后未进入 Top100，说明其语义重排得分被明显压低。原查询非常短，且“video aesthetics score”与论文标题中的“visual scoring”存在表达差异，现有重排特征可能没有充分识别 `video aesthetics assessment`、`visual quality assessment`、`visual scoring` 和 `LMM-based scoring` 之间的同义关系。

后续处理：

- 审计该论文的 fusion score、semantic score 和最终排序分数；
- 对比重排前排名 42 与重排后实际排名；
- 增加 `video aesthetics assessment`、`visual scoring`、`video quality assessment` 等任务别名；
- 提高任务短语和模型实体的精确锚点权重；
- 不增加新的 OpenAlex API 调用。

建议标签：`reranking_score_audit + alias_expansion`

### 3.2 RealScholarQuery_30

- 原始查询：`I would like to find some research papers about test time training topic, in LLM research area.`
- Gold 数量：6
- 命中的 Gold 论文：`Self-Refine: Iterative Refinement with Self-Feedback`
- 自动识别锚点：`LLM`
- B0 Top100：命中 1 篇，最佳排名 43
- B1.2 Top100：命中 1 篇，最佳排名 42
- B2 Top100：命中 0 篇
- B4 Pool：命中 1 篇，融合池顺序排名 45
- B4 Top100：命中 0 篇
- 失败类型：`ranking_loss_top100`
- 优先级：P0

初步判断：该查询的正确论文已经由 B0 和 B1.2 召回，并进入 B4 融合候选池，因此主要问题同样位于重排阶段。当前锚点提取器只识别出 `LLM`，没有将查询中的 `test time training` 稳定识别为方法锚点。与此同时，Gold 论文标题使用的是 `Self-Refine` 和 `Iterative Refinement with Self-Feedback`，与用户查询中的 `test time training` 并非字面同义，导致轻量语义重排可能低估其相关性。

后续处理：

- 审计该论文在 B4 中的 fusion score、semantic score 和最终排名；
- 将 `test time training`、`test-time adaptation`、`self-refinement`、`iterative refinement`、`self-feedback` 建立为受控方法别名；
- 检查锚点提取器为何未识别 `test time training`；
- 对方法别名命中增加重排加分，但不直接放宽 strict 匹配；
- 不增加新的 OpenAlex API 调用。

建议标签：`reranking_score_audit + method_alias_expansion`

## 4. Retrieval-miss 查询人工分类

以下 24 条查询在 B0、B1.2、B2 和 B4 Pool 中均未命中任何 Gold 论文，属于真实候选召回失败。表中的“初步失败原因”和“救援检索式”是根据原始查询与自动锚点做出的离线分析，后续仍需通过消融实验验证。

| QID | Gold 数量 | 主要学术锚点 | 初步失败原因 | 推荐救援策略 | 建议救援检索式 |
|---|---:|---|---|---|---|
| `RealScholarQuery_1` | 29 | `in-context learning` | 查询改写保留了宽泛表达，但没有围绕“预训练过程中如何形成 ICL 能力”构造机制型检索式，也未覆盖 induction heads、implicit learning 等常见表述。 | `method_model_retrieval + alias_expansion` | `in-context learning emergence during pretraining`；`induction heads mechanism of in-context learning` |
| `RealScholarQuery_3` | 42 | `multimodal`、`visual`、`audio`、`audio-visual`、`exclude survey` | 查询同时包含视觉、音频和视听预训练三种模态，现有检索可能被宽泛的 multimodal 结果淹没；排除 survey 不应在召回阶段过早过滤。 | `modality_composition_retrieval + alias_expansion` | `audio visual multimodal foundation model pretraining`；`vision audio audio-visual large-scale pre-trained model` |
| `RealScholarQuery_7` | 35 | `video`；隐含任务为长视频描述 | `long video description` 不是稳定术语，相关工作常使用 long-form video captioning、dense video captioning、movie description 等名称。 | `task_method_retrieval + alias_expansion` | `long-form video captioning minutes-long video`；`long video description dense video captioning` |
| `RealScholarQuery_12` | 17 | `transformer`、`3D`、`video generation` | 任务、模型和模态虽然明确，但现有子查询没有形成高约束组合，容易返回普通 2D 视频生成或 3D 场景理解论文。 | `method_model_retrieval + task_method_retrieval` | `transformer 3D video generation`；`transformer-based 4D dynamic scene generation` |
| `RealScholarQuery_16` | 8 | `document-level event extraction`、`trigger-free`、`without human-annotated triggers` | 关键限定“无触发词/无人工触发标注”未被提取为方法锚点，检索退化为普通文档级事件抽取。 | `task_method_retrieval + alias_expansion` | `trigger-free document-level event extraction`；`document event extraction without trigger annotations` |
| `RealScholarQuery_17` | 9 | `in-context learning`、`LLMs`、`NER`、`RE`、`EE`、与 SFT 小模型比较 | 查询是带结论倾向的比较型问题，现有检索可能只搜索 ICL 或信息抽取，未构造 ICL 与 supervised fine-tuning 的对比检索。 | `method_model_retrieval + comparison_aware_retrieval` | `in-context learning versus supervised fine-tuning information extraction`；`LLM ICL NER RE EE small model comparison` |
| `RealScholarQuery_20` | 4 | `LLMs`；隐含任务为自动文献综述/多论文关系分析 | 查询使用“支持某个主张”的长描述，没有落到 automated literature review、survey generation、scientific synthesis 等稳定任务术语。 | `task_method_retrieval + alias_expansion + controlled_citation_expansion` | `LLM automated literature review survey generation`；`large language model multi-paper synthesis scientific survey writing` |
| `RealScholarQuery_21` | 2 | `SFT`；隐含方法为同一 prompt 的多响应训练 | “same prompt with different responses”没有被识别为 response diversity、multiple responses、preference/instruction data augmentation 等方法概念。 | `method_model_retrieval + alias_expansion` | `multiple responses per prompt supervised fine-tuning`；`response diversity improves instruction tuning` |
| `RealScholarQuery_22` | 3 | `machine translation`、`commonsense` | 查询极短，普通关键词组合容易返回常识推理或通用机器翻译论文，没有覆盖 commonsense-aware translation、contextual knowledge 等别名。 | `task_method_retrieval + alias_expansion` | `commonsense reasoning for machine translation`；`commonsense-aware neural machine translation` |
| `RealScholarQuery_23` | 2 | `reinforcement learning`、`diffusion models`、`video generation` | 现有检索可能分别召回强化学习、扩散模型或视频生成论文，但没有稳定构造三者的交叉检索。 | `method_model_retrieval + task_method_retrieval` | `reinforcement learning optimize diffusion video generation`；`reward fine-tuning diffusion model for text-to-video generation` |
| `RealScholarQuery_28` | 4 | `HumanEval`、`MBPP`、`code_contests`、难度区间比较 | 数据集名已出现，但比较关系没有被结构化利用；当前系统没有围绕“难于 A/B、易于 C”的难度区间构造 benchmark 查询。 | `benchmark_aware_retrieval + comparison_aware_retrieval` | `code generation benchmark harder than HumanEval MBPP`；`program synthesis benchmark between HumanEval and CodeContests difficulty` |
| `RealScholarQuery_32` | 16 | `neural network based quantum Monte Carlo` | 自动锚点为空，说明领域术语没有进入词表；宽泛改写可能将 quantum Monte Carlo 与 neural network 分离。 | `exact_phrase_retrieval + domain_alias_expansion + controlled_citation_expansion` | `neural network quantum Monte Carlo`；`neural quantum states variational Monte Carlo` |
| `RealScholarQuery_34` | 7 | `3D scene understanding`、`3D AIGC`、`foundation models` | `AIGC` 被识别为命名实体，但没有展开为 3D generative foundation model、3D world model 等常见术语。 | `method_model_retrieval + alias_expansion` | `3D scene understanding generative foundation model`；`3D AIGC foundation model scene understanding` |
| `RealScholarQuery_35` | 7 | `LLM`、`quantized pretraining` | “quantized pretraining”是关键组合，但当前只识别出 LLM，未覆盖 low-bit pretraining、quantization-aware pretraining 等同义方法。 | `method_model_retrieval + exact_phrase_retrieval + alias_expansion` | `quantized pretraining large language model`；`low-bit quantization-aware LLM pretraining` |
| `RealScholarQuery_36` | 33 | `identity preservation`、`video generation` | 相关论文常使用 identity-consistent、subject-driven、personalized video generation 等术语，单一字面检索覆盖不足。 | `task_method_retrieval + alias_expansion` | `identity-preserving video generation`；`subject-driven identity-consistent text-to-video generation` |
| `RealScholarQuery_38` | 17 | `image`；隐含概念为编码表示/分布 | `image encoding distributions` 语义模糊，可能指 latent distribution、visual token distribution、representation distribution 或 image codec，现有系统缺少歧义拆分。 | `alias_expansion + controlled_citation_expansion` | `image latent representation distribution encoding`；`visual tokenization encoding distribution` |
| `RealScholarQuery_39` | 2 | `synthetic data`、`LLM`、`long thought data`、高质量/多样/困难推理数据 | 关键短语 `long thought data` 不是标准术语，应映射到 long chain-of-thought、reasoning trace、synthetic reasoning data 等表达。 | `task_method_retrieval + alias_expansion` | `synthetic long chain-of-thought data generation for LLM`；`automatic high-quality reasoning trace data synthesis` |
| `RealScholarQuery_42` | 10 | `video understanding`、`select frames` | 相关工作常使用 keyframe selection、frame sampling、adaptive frame selection、temporal token pruning 等术语，当前未建立别名。 | `task_method_retrieval + alias_expansion` | `adaptive frame selection for video understanding`；`keyframe sampling temporal token selection video understanding` |
| `RealScholarQuery_44` | 25 | `Crypto-based Private Learning`、`privacy-preserving machine learning` | 查询中的领域概念未被自动锚点识别；相关工作通常按 cryptographic ML、secure computation、homomorphic encryption、MPC 分类。 | `exact_phrase_retrieval + method_alias_expansion` | `cryptographic privacy-preserving machine learning`；`secure multiparty computation homomorphic encryption private machine learning` |
| `RealScholarQuery_45` | 58 | `controllability`、`video generation` | Gold 范围较大，而“可控性”包含运动、相机、布局、姿态、轨迹、身份等多个子方向；单一宽泛查询难以覆盖。 | `task_decomposition_retrieval + alias_expansion + controlled_citation_expansion` | `controllable video generation survey benchmark methods`；`motion camera trajectory layout controlled text-to-video generation` |
| `RealScholarQuery_46` | 65 | `robot decision making`、`task planning`、`datasets`、`benchmarks` | 任务词被识别，但 dataset/benchmark 意图没有作为高优先级召回通道；机器人规划还涉及 embodied AI、long-horizon planning 等别名。 | `benchmark_aware_retrieval + dataset_anchor_retrieval + alias_expansion` | `robot task planning decision making datasets benchmarks`；`embodied AI long-horizon planning benchmark` |
| `RealScholarQuery_47` | 4 | `LLM agents`、`financial tasks`、`evaluation`、`benchmark` | 当前只识别出 LLM，没有保持“agent 而非普通 LLM”的硬限定，也没有形成金融任务 benchmark 检索式。 | `benchmark_aware_retrieval + method_model_retrieval` | `LLM agent benchmark financial tasks`；`evaluation benchmark autonomous language agents for finance` |
| `RealScholarQuery_48` | 8 | `large language models`、`stock exchange analysis`、`factor mining` | “mining factors”在量化金融中通常称为 alpha factor discovery、factor generation、quantitative signal mining，现有查询缺少领域别名。 | `task_method_retrieval + domain_alias_expansion` | `large language model alpha factor mining stock market`；`LLM quantitative factor discovery financial signals` |
| `RealScholarQuery_49` | 8 | `vision-language models`、`agents`、`PC games`、`automatically play` | 当前锚点只识别 vision-language 与 PC，没有稳定保留 agent、game playing、computer control 等任务组合，也未覆盖 VLM agent、vision-language-action 等别名。 | `method_model_retrieval + task_method_retrieval + alias_expansion` | `vision-language agent PC game playing`；`multimodal VLM agent computer game control` |

## 5. 救援策略汇总

根据上述人工分类，24 条真实召回失败查询可归纳为以下方向。由于一条查询可能同时采用两种策略，因此各类数量允许重叠。

- 明确方法、模型或学术实体驱动的召回：约 12 条；
- 任务与方法组合检索：约 15 条；
- 数据集或 Benchmark 感知召回：3 条；
- 比较关系结构化检索：2 条；
- 领域或任务别名扩展：约 18 条；
- 精确短语/标题式锚点检索：4 条；
- 适合受控引文扩展的宽泛或歧义查询：5 条；
- 当前需要人工匹配审计的查询：0 条。

从分布上看，最普遍的问题不是查询中完全没有有效信息，而是系统没有将自然语言中的任务别名、领域术语和比较关系转化为高价值检索式。第二周的召回增强应优先实现“学术锚点提取 + 别名扩展 + 至多两条定向救援查询”，而不是继续无差别增加子查询数量。

## 6. Day 1 结论

在 B4 Top100 结果中，共有 26 条 zero-recall queries。其中，24 条属于真实候选召回失败，即正确论文没有进入 B4 融合候选池；2 条属于排序失败，即正确论文已经进入 B4 候选池，但经过轻量语义重排后落到 Top100 之外。当前疑似 DOI、URL、arXiv 或标题版本匹配问题数量为 0，整个诊断过程没有调用外部 API。

两条排序失败查询 `RealScholarQuery_25` 和 `RealScholarQuery_30` 应单独进行 B4 重排分数审计，不需要新增召回调用。其余 24 条查询应作为第二周召回增强的主要对象，优先围绕数据集名、方法名、任务名、模型名、比较关系和领域别名生成定向检索式。

因此，后续工作顺序确定为：

1. Day 2 完善 DOI、URL、OpenAlex ID、arXiv ID 及预印本/正式版本的统一解析与去重逻辑；
2. Day 3 基于本复核表实现 Academic Anchor Extractor 和 Anchor-aware Query Planner；
3. Day 4 只对低覆盖查询触发至多两条 Rescue Retrieval，并记录新增 API 调用与成本；
4. Day 5 对 anchor retrieval、citation expansion 和 reranking repair 进行独立消融，验证 zero-recall、TP、FP、F1 与成本变化。

## 7. Day 1 验收记录

- [x] `query_stage_diagnostics.jsonl` 包含 50 条记录；
- [x] `zero_recall_taxonomy.jsonl` 包含 26 条记录；
- [x] B4 Top100 TP 复现为 61；
- [x] `retrieval_miss` 复现为 24；
- [x] `ranking_loss_top100` 复现为 2；
- [x] `possible_matching_issue` 为 0；
- [x] API 调用数为 0；
- [x] 2 条排序失败查询已完成单独复核；
- [x] 24 条召回失败查询已完成初步分类；
- [x] 每条召回失败查询均给出至少一种建议救援策略；
- [x] Day 1 代码与文档已提交 Git。
