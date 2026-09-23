---
name: user-voice-taxonomy
description: 从 CSV、Excel、Parquet、JSONL 或飞书表格中的大量用户原声，自动完成数据清洗、AI 特征抽取、中文语义向量聚类、三级标签体系生成与 MECE 治理、覆盖率迭代、Laya 分层批量打标、低置信拒识、人工复核、金标评估/微调/温度校准，并最终交付固定 6 张子表的飞书电子表格。用户提到用户原声/VOC/反馈/投诉/评论/工单的聚类、标签发现、标签体系、自动打标、问题分布、TOP 问题或 Laya 时使用。
---

# 用户原声智能标签体系与自动打标

## 核心原则

- 让当前执行 Skill 的 AI 直接承担开放式语义任务：特征提取、簇理解、标签命名、定义边界与 MECE 评审；不要再调用外部大模型 API。
- 让脚本承担确定性任务：读取、清洗、去重、向量、聚类、阈值诊断、标签校验、Laya 推理、拒识、统计和评估。
- 标签发现与生产打标必须分离。候选标签未经治理和版本冻结，不得直接用于正式全量打标。
- 最终产物必须是一张飞书电子表格，固定包含 6 张子表。执行 AI 必须先加载 `sheet` Skill，再直接调用它提供的飞书表格 MCP/CLI 创建、填写、校验和交付；分析脚本不负责上传或管理飞书表格。
- 原文、证据、record_id、模型版本、标签版本和阈值全程可追溯。禁止把低置信结果当成确定事实。

详细 Schema、阈值口径和子表定义见 [references/contracts.md](references/contracts.md)。需要准备金标、训练或升级模型时，先读 [references/training-plan.md](references/training-plan.md)。

## 环境与默认路径

- 脚本：`scripts/user_voice_pipeline.py`
- Laya 训练：`scripts/laya_train.py`
- 默认 Laya multilingual：优先使用项目中的 `laya_models/laya-multilingual`；不存在时询问本地路径或下载。
- 默认中文向量模型：优先使用项目中的 `embedding_models/bge-small-zh-v1.5`；不存在时下载 `BAAI/bge-small-zh-v1.5` 到单独目录。
- 依赖：`pandas openpyxl pyarrow sentence-transformers scikit-learn laya>=0.3.7 torch safetensors`。首次运行先检查 PyPI 最新稳定版与本地版本；本地版本落后时升级并完成兼容性冒烟测试，禁止默认安装源里缓存的旧版。
- 每次运行使用独立输出目录，不覆盖用户源文件。

## 运行模式

根据用户目标选择：

1. `discover`：首次分析或标签需要重建。执行数据准备 → AI 特征抽取 → BGE 聚类 → AI 标签树生成 → 标签校验与覆盖率迭代。
2. `label`：已有冻结标签树。执行数据准备/特征校验 → Laya 分层打标 → 拒识 → 结果与分布。
3. `review`：已有历史结果。复查覆盖率、混淆、未知样本、漂移和版本升级建议。
4. `full`：默认，依次完成 discover + label + review，并交付飞书表格。

## 完整工作流

### 1. 检查输入与范围

- 用户提供文件或表格时，先完整读取结构，明确原声列、ID 列、上下文字段和数据量。
- 若没有用户数据但要求验证 Skill，使用 `assets/sample_user_voice.csv`，并明确这是内置演示数据，不把结果当业务结论。
- 表格任务必须同时遵循系统的 `sheet` Skill；读取本地 Excel 时执行其预检，读取飞书表格时使用对应 lark-cli 流程。

### 2. 数据准备

运行：

```bash
python3 scripts/user_voice_pipeline.py prepare \
  --input INPUT --text-col 原声 --id-col record_id \
  --context-cols 渠道 时间 --output-dir RUN_DIR --batch-size 50
```

检查 `data_profile.json`、`rejected_records.jsonl` 和 `ai_feature_batches.jsonl`。

### 3. 由执行 AI 生成语义特征

逐批读取 `ai_feature_batches.jsonl`，按 `references/contracts.md` 生成 `semantic_features.jsonl`。必须逐条覆盖，不得跳行；证据片段必须在同行原文连续命中。完成后运行 `cluster`，脚本会再次全量校验证据和记录覆盖。

### 4. 中文向量聚类与阈值选择

```bash
python3 scripts/user_voice_pipeline.py cluster \
  --work-dir RUN_DIR --features RUN_DIR/semantic_features.jsonl \
  --embedding-model BGE_PATH --min-cluster-size 2
```

- 不传 `--similarity-threshold` 时自动扫描 0.62—0.80，并综合轮廓系数、噪声率和最大簇占比选择。
- 只有业务明确给出阈值时才手动覆盖。
- 读取 `threshold_diagnostics.json`、`cluster_manifest.json`、`cluster_summaries.json` 和代表原声；若最大簇过大、噪声过高或结构不稳定，先调整阈值/最小簇规模再继续。

### 5. 由执行 AI 生成并治理标签树

基于聚类摘要、中心样本、边界样本、噪声样本和业务上下文生成 `taxonomy.json`。执行 AI 必须：

- 构造一级→二级→三级“属于”关系；情绪、严重度、诉求等正交维度不混入主问题树。
- 合并同义、拆分混杂、补齐缺口；每个标签写完整定义卡。
- 对高覆盖标签检查是否存在稳定子结构；对低覆盖标签先判断风险和业务价值，不机械删除。
- 保留未知/拒识机制，不为追求覆盖率强行落标。

校验：

```bash
python3 scripts/user_voice_pipeline.py validate-taxonomy \
  --taxonomy RUN_DIR/taxonomy.json --embedding-model BGE_PATH
```

结构错误必须修复；同级相似度告警必须由执行 AI 读取对应标签定义和样本后决定合并、重写边界或保留理由。

### 6. 金标评估、按需微调与校准

**微调不是每次任务的必经步骤。** 日常批次必须直接复用当前已发布的 Laya checkpoint 和阈值；禁止因为来了新数据就重新训练。

只有同时满足以下条件，才进入微调：

1. 用户明确要求训练/升级模型，或监控发现现有模型未达到约定的 Macro-F1、召回率、ECE、未知率/复核率门槛；
2. 标签体系已经冻结为新版本，且变化足以影响分类边界；
3. 已获得足量、人工确认并完成训练/验证/独立测试隔离的金标数据。

不满足时：

- 没有人工金标：使用当前已发布 checkpoint 做候选分类，保留拒识，并在运行说明中写明“未使用本批数据训练模型”。
- 只有大模型伪标签：可以用于候选样本扩充，但不能独立触发正式微调，也不能冒充最终测试集。
- 新数据只是常规增量批次：仅推理、统计和漂移监控，不训练。

需要微调时，按 `references/contracts.md` 生成 gold JSONL，运行：

```bash
python3 scripts/laya_train.py \
  --gold GOLD.jsonl --taxonomy RUN_DIR/taxonomy.json \
  --base-model LAYA_ROOT --subfolder multilingual \
  --output-dir RUN_DIR/laya_finetuned --epochs 2 --freeze-encoder
```

脚本按 record_id 隔离训练/验证，报告 Accuracy、Macro-F1、ECE、NLL，并拟合 choice 温度后写出可被 `laya.load()` 读取的新 checkpoint。正式发布前仍需保留独立测试集；不能用大模型伪标签冒充最终测试集。

### 7. Laya 分层打标、拒识与结果生成

```bash
python3 scripts/user_voice_pipeline.py label-report \
  --work-dir RUN_DIR --features RUN_DIR/semantic_features.jsonl \
  --taxonomy RUN_DIR/taxonomy.json --classifier laya \
  --laya-model LAYA_ROOT --laya-subfolder multilingual \
  --embedding-model BGE_PATH --confidence-threshold 0.50 \
  --margin-threshold 0.10
```

- 每个父节点仅比较其直接子标签；禁止一次平铺超过 20 个候选。
- 多意图原声按 issue span 分拆打标。
- Top1 概率低于阈值或 Top1-Top2 差值过小，写入人工复核队列。
- 若 Laya 未安装/权重不可用，可用 `--classifier embedding` 做明确标注的降级冒烟测试；不得把降级结果宣称为 Laya 结果。

### 8. 覆盖率迭代

读取 `label_distribution.json`、`coverage_suggestions.json`、`review_queue.jsonl`：

- 过高标签：检查聚类是否存在稳定、可行动的子结构后再拆分。
- 过低标签：检查是否高风险或新兴问题；否则合并或改为观察项。
- 未知率超过 10%：对未知样本重新聚类并生成新标签候选。
- 发现标签树变化时升级 taxonomy_version，保留旧标签映射和变更原因，再重跑全量打标与回归集。

### 9. 通过 `sheet` Skill 直接交付飞书表格

这是强制步骤，但不要在本 Skill 内重写飞书表格 API 或上传脚本。

1. 先发现并完整读取当前环境的 `sheet/SKILL.md`，严格遵循其创建、写入、样式、回读、自检和交付协议。
2. 将以下本地结构化结果整理为 6 个 DataFrame/二维表：
   - `run_manifest.json` + `data_profile.json` + `threshold_diagnostics.json` → `运行说明`
   - `tagged_data.jsonl` → `打标明细`
   - `taxonomy.json` → `标签体系`
   - `label_distribution.json` + `coverage_suggestions.json` → `标签分布`
   - `top_problems.json` → `TOP问题`
   - `review_queue.jsonl` → `人工复核队列`
3. 使用 `sheet` Skill 提供的飞书表格 MCP/CLI，直接新建一张飞书工作簿并批量写入这 6 张子表。优先采用 typed 多子表写入；数值与百分比保留真实类型，record_id 等标识符写文本。
4. 按 `sheet` Skill 规则设置最小充分样式：表头、列宽、换行、长明细冻结首行；汇总占比使用飞书公式或由明细复算后写入，并在运行说明中写清口径。
5. 回读工作簿结构、每张子表的首/中/末关键区域和公式结果；检查标签分布数量合计等于打标问题点数，复核队列数等于状态为 review 的明细数。
6. 运行 `sheet` Skill 要求的在线自检，取得可交付信号后，用 `present_files` 只交付飞书表格 URL。

只有飞书表格 MCP/Skill 暂时不可用时，才让 `label-report --output-xlsx` 生成本地 xlsx 作为故障兜底；恢复后仍须导入飞书，不能把本地文件当标准最终产物。

## 效果与质量门

- 数据：有效记录、去重、失败记录都有计数与明细。
- 语义：每个输入 record_id 有且只有一条特征结果；证据必须连续命中原文。
- 聚类：输出全部候选阈值诊断和最终选择；不隐瞒噪声簇。
- 标签：层级合法、ID 唯一、每个叶子标签有定义/边界/正例/反例/动作；同级重叠有检查。
- 打标：逐层概率、综合置信度、版本齐全；低置信和近似并列可拒识。
- 评估：有金标才报告 Macro-F1/ECE；无金标不得编造准确率。
- 报告：所有汇总由明细现算，代表原声可回溯，最终固定 6 张飞书子表。

## 遇到依赖或阻塞

只有以下情况才询问用户：无法识别真实原声列；需要正式微调但没有人工金标；飞书导入权限失败；用户要求的现有标签或业务口径未提供且不同选择会改变标签树。其余低影响参数使用上述默认值，并把实际取值写入”运行说明”。
