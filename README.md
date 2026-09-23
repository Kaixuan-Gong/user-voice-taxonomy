# User Voice Taxonomy Skill

用户原声智能标签体系与自动打标 Skill。从大量用户反馈中自动完成数据清洗、语义特征抽取、向量聚类、三级标签体系生成与治理、Laya 分层打标，最终交付飞书电子表格。

## 功能

- **数据准备**：读取 CSV/Excel/Parquet/JSONL/飞书表格，清洗去重
- **AI 特征抽取**：自动提取关键问题点、问题对象、用户诉求、情绪、严重度
- **中文语义聚类**：BGE 向量 + 层次聚类，自动选择最佳相似度阈值
- **三级标签体系**：AI 生成一级→二级→三级标签树，MECE 治理
- **Laya 分层打标**：基于 Laya 模型逐层分类，低置信自动拒识进人工复核
- **覆盖率迭代**：过高标签拆分、过低标签合并、未知样本重新聚类
- **飞书表格交付**：固定 6 张子表（运行说明、打标明细、标签体系、标签分布、TOP问题、人工复核队列）

## 安装

在豆包办公环境中执行：

```bash
npm_config_registry="https://bnpm.byted.org" npx -y agentbuddy@latest skill add <your-github-repo-url> -g -y
```

或者直接 clone 后放到 `.user_skills/` 目录。

## 模型依赖

首次运行前需要下载两个模型：

| 模型 | 大小 | 来源 |
|---|---|---|
| BGE-small-zh-v1.5（向量模型） | ~100MB | HuggingFace: `BAAI/bge-small-zh-v1.5` |
| Laya multilingual（分类模型） | ~650MB | HuggingFace: `convaiinnovations/laya` |

首次运行时会自动检测并下载，也可以手动放到以下目录：

```
embedding_models/bge-small-zh-v1.5/
laya_models/laya-multilingual/
```

## 快速开始

```bash
# 1. 数据准备
python3 scripts/user_voice_pipeline.py prepare \
  --input your_data.csv --text-col 原声 --id-col record_id \
  --context-cols 渠道 时间 --output-dir run_001

# 2. AI 读取 ai_feature_batches.jsonl，写 semantic_features.jsonl

# 3. 聚类
python3 scripts/user_voice_pipeline.py cluster \
  --work-dir run_001 --features run_001/semantic_features.jsonl \
  --embedding-model BAAI/bge-small-zh-v1.5 --min-cluster-size 2

# 4. AI 读取聚类结果，写 taxonomy.json

# 5. 校验标签体系
python3 scripts/user_voice_pipeline.py validate-taxonomy \
  --taxonomy run_001/taxonomy.json --embedding-model BAAI/bge-small-zh-v1.5

# 6. Laya 打标
python3 scripts/user_voice_pipeline.py label-report \
  --work-dir run_001 --features run_001/semantic_features.jsonl \
  --taxonomy run_001/taxonomy.json --classifier laya \
  --laya-model convaiinnovations --laya-subfolder multilingual \
  --embedding-model BAAI/bge-small-zh-v1.5
```

## 目录结构

```
user-voice-taxonomy-skill/
├── SKILL.md                    # Skill 主文档（触发与工作流）
├── scripts/
│   ├── user_voice_pipeline.py  # 主流水线脚本
│   └── laya_train.py           # Laya 微调脚本
├── references/
│   ├── contracts.md            # 数据与语义契约
│   ├── training-plan.md        # 模型训练方案
│   └── training-readiness.md   # 训练就绪检查
├── assets/
│   ├── sample_user_voice.csv  # 演示数据
│   ├── sample_taxonomy.json   # 示例标签体系
│   └── sample_semantic_features.jsonl  # 示例特征
├── embedding_models/           # BGE 向量模型（不入库）
└── laya_models/                # Laya 分类模型（不入库）
```

## 技术栈

- Python 3.10+
- pandas, scikit-learn, sentence-transformers
- laya (分类模型推理)
- PyTorch (MPS/CPU)
