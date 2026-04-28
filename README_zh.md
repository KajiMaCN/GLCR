# GLCR：面向三部图生物医学关联预测的全局-局部协同推理

[English README](README.md)

本仓库提供论文 **GLCR** 的代码实现以及处理后的基准数据输入：

**Let Bridges Speak: Global-Local Collaborative Reasoning for Tripartite Biomedical Association Prediction**

## 概述

三部图生物医学关联预测的目标，是借助中介层实体来推断两个端点实体之间的目标关联。常见情形包括 ncRNA-drug 关联预测，以及在疾病侧中介证据支持下的 drug-target 关联预测。

GLCR 将这一问题建模为 **全局-局部协同推理**。模型首先从完整三部图中为候选端点对建立全局结构锚点，再将中介证据组织为与当前端点对相关的 bridge tokens，最后通过显式通信、潜在补偿、可靠性校准和单一自适应效用门控，对全局与局部证据进行联合建模。

当前开源版本提供了论文四个主 benchmark 的主线复现代码与处理后的输入数据。

## 收录数据集

本仓库包含四个三部图生物医学 benchmark。对于每个数据集，模型预测 `primary_edge` 对应的端点关系，另外两类关系作为中介侧支持边。

| 数据集键名 | 预测的端点关系 | 中介侧支持关系 |
| --- | --- | --- |
| `ncRNADrug` | ncRNA-drug | ncRNA-phenotype, drug-phenotype |
| `openTargets` | drug-target | drug-disease, target-disease |
| `CTD` | chemical-gene | chemical-disease, gene-disease |
| `PrimeKG` | drug-target | drug-disease, target-disease |

每个数据集目录包含：

- `all_node_features.csv`：统一节点特征表，包含全局节点索引和数值特征
- 1 个主任务端点关系文件
- 2 个中介侧支持关系文件

## 方法简介

GLCR 的核心思想是：三部图关联预测不能只依赖全局拓扑，也不能只依赖局部中介证据，而应让两者显式协同。

- **全局结构建模**：学习图感知节点表示，并为每个候选端点对构造全局结构锚点。
- **局部中介证据建模**：将共享中介候选编码为 pair-conditioned bridge tokens，并聚合为显式局部推理状态。
- **潜在残差补偿**：补偿观测到的中介候选中未完全覆盖的 bridge 模式。
- **可靠性校准**：判断当前端点对的局部证据是否可信。
- **双 token 微子图通信**：在显式全局-局部交互前，仅保留显式桥接 token 与通信前局部 seed token 作为局部摘要。
- **效用感知自适应融合**：用单一共享门控统一调节粗桥接项和通信后的局部残差项。
- **训练目标**：与论文保持一致，并固定为三部分，也就是最终分类项、桥接校准正则项和桥接扰动项，不再暴露旧版本里的额外辅助损失。

## 仓库结构

```text
GLCR_open_source/
├── train_glcr.py
├── paper_configs.py
├── requirements.txt
├── model/
│   ├── GLCR.py
│   └── __init__.py
├── layer/
│   ├── GCN.py
│   └── __init__.py
├── tools/
│   ├── Datasets.py
│   ├── subgraph.py
│   ├── utils.py
│   └── __init__.py
└── datasets/
    ├── ncRNADrug/
    ├── openTargets/
    ├── CTD/
    └── PrimeKG/
```

## 环境安装

安装依赖：

```bash
pip install -r requirements.txt
```

当前版本整理与验证时使用的核心依赖版本为：

- `torch 2.6.0`
- `torch-geometric 2.7.0`
- `numpy 1.26.4`
- `pandas 3.0.1`
- `scikit-learn 1.8.0`

## 主实验复现

当前发布脚本对应本仓库中的主 benchmark 复现协议：

- `random` 划分
- `1:1` 正负样本平衡采样
- `5` 折交叉验证
- 在每个训练 fold 内再划分验证集
- 基于验证集按 `MCC` 自动选择测试阈值
- 四个数据集共享同一套 release 配置

完整运行示例：

```bash
python train_glcr.py \
  --dataset openTargets \
  --result_dir results/openTargets_main
```

可用数据集键名：

- `ncRNADrug`
- `openTargets`
- `CTD`
- `PrimeKG`

如果只想运行部分 fold，可以使用：

```bash
python train_glcr.py \
  --dataset CTD \
  --fold_indices 1,2 \
  --result_dir results/CTD_partial
```

## 训练与评估协议

对于每个 fold，支持图由以下两部分构成：

- 主任务端点关系中的训练折正样本
- 全部观测到的中介侧支持边

在图编码和 pair-context 提取之前，验证集和测试集中的正样本会从支持图中移除。模型在平衡后的端点对上训练，负样本来自未观测候选对。评估指标包括：

- `AUC`
- `AUPR`
- `F1`
- `MCC`
- `Accuracy`
- `Precision`
- `Recall`

## 输出结果

如果指定了 `--result_dir`，脚本会输出：

- `fold_metrics.csv`：每个 fold 的指标，以及均值和标准差
- `summary.json`：结构化汇总结果与运行元数据

训练日志默认写入 `logs/`，缓存默认写入 `cache/subgraph_features/`。

## 说明

- 本仓库对应论文中的 GLCR 模型实现。
- `datasets/` 中提供的是当前发布训练代码直接读取的处理后输入。
- 当前发布版本聚焦于本仓库提供的主 benchmark 主线复现流程。

## 引用

如果本仓库对你的研究有帮助，请引用：

```text
Let Bridges Speak: Global-Local Collaborative Reasoning for Tripartite Biomedical Association Prediction
```
