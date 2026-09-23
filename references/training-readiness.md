# 当前训练能力与正式训练前补充项

## 已实现并验证

- Laya 分层 Choice 金标展开。
- 按 `record_id` 隔离训练集与验证集。
- 冻结 encoder、只训练决策头。
- Laya 全参数训练开关。
- Cross Entropy、梯度裁剪、Accuracy、Macro-F1、ECE、NLL。
- 验证集温度拟合。
- 新 checkpoint 保存与 `laya.load()` 重载。
- BGE-small-zh-v1.5 推理、向量聚类和阈值敏感性选择。
- Laya 与 BGE 在 Apple M5 MPS 上的完整优化器 step。

## 正式训练前必须补充

当前 `scripts/laya_train.py` 是可运行的训练基线，不应未经增强直接承担正式发布。拿到业务金标后需要补：

1. 四分数据集：train / validation / calibration / test；当前脚本只有 train / validation。
2. `group_id` 分组与时间外切分，防止同会话、同订单、近重复文本泄漏。
3. 每轮保存最佳 checkpoint、early stopping、训练中断恢复。
4. 类别不均衡处理和每标签指标、混淆矩阵。
5. 节点级而非仅全局的温度与拒识阈值选择。
6. 独立测试集只在最终验收时运行一次。
7. BGE 对比学习训练器、难负样本挖掘和 Recall@K/近邻纯度评估。
8. 训练配置、随机种子、依赖版本、数据哈希与模型哈希记录。

这些补充项依赖真实标签规模、分布和 `group_id/event_time` 字段，拿到数据审计结果后再定具体参数。
