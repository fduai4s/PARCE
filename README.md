# PARCE 材料结构预测系统

PARCE 是一个面向材料结构类型预测的机器学习流程。项目以材料的声子频率或拉曼活性频率为输入，结合结构参数聚类与 RCNet 深度学习模型，预测材料所属的结构类型 ID（簇 ID），从而获取材料结构信息。

## 目录

- [项目概述](#项目概述)
- [运行环境](#运行环境)
- [数据集](#数据集)
- [目录结构](#目录结构)
- [使用流程](#使用流程)
- [主要参数](#主要参数)
- [输入输出](#输入输出)
- [示例](#示例)
- [注意事项](#注意事项)

## 项目概述

项目主要包含 5 个阶段：

1. **结构参数聚类**
   - 使用空间群、Pearson 符号、Wyckoff 序列、`c/a` 比率、`β` 角等结构参数进行 K-means 聚类。
   - 通过轮廓系数（Silhouette Score）和 SSE 拐点分析选择较优聚类方案。

2. **频率数据处理**
   - 对同一结构簇内的声子频率或拉曼频率进行标准化。
   - 对频率序列进行截断或补齐。
   - 计算簇内平均频率，得到簇代表性频率。

3. **亲和传播聚类**
   - 基于簇代表性频率进行 Affinity Propagation 聚类。
   - 构建频率特征与结构类型之间的训练数据。

4. **RCNet 模型训练**
   - 使用 RCNet 神经网络学习频率特征到结构类型 ID 的映射。
   - 支持批量训练、GPU 自动检测和并行训练。

5. **PARCE 集成预测**
   - 使用训练好的模型对新材料频率数据进行预测。
   - 输出预测结构类型 ID。

## 运行环境

建议使用 Conda 环境运行。

| 项目 | 推荐配置 |
| --- | --- |
| Conda 环境名 | `PARCE` |
| Python 版本 | `3.8.20` |
| 主要运行方式 | Jupyter Notebook + Python 脚本 |

### 创建并激活环境

```bash
conda create -n PARCE python=3.8.20
conda activate PARCE
```

### 安装依赖

```bash
pip install pandas numpy matplotlib
pip install scikit-learn umap-learn kneed
pip install tqdm joblib jupyter tensorboard
pip install torch torchvision torchaudio
```

> 如果需要使用 CUDA 版 PyTorch，请根据服务器 CUDA 版本安装对应的 PyTorch 包。

## 数据集

当前仓库包含两个示例数据集：

| 文件 | 说明 |
| --- | --- |
| `dataset/phono_freq_dataset.csv` | 声子频率数据集 |
| `dataset/raman_dataset.csv` | 拉曼活性频率数据集 |

常用字段如下：

| 字段 | 说明 |
| --- | --- |
| `id` / `material_id` | 材料 ID |
| `frequency` | 频率序列 |
| `space_group_number` | 空间群编号 |
| `pearson_symbol` | Pearson 符号 |
| `wyckoff_sequence` | Wyckoff 序列 |
| `c_a_ratio` | `c/a` 轴比 |
| `beta_angle` | `β` 角 |
| `cif` | CIF 结构信息 |

## 目录结构

```text
.
├── README.md
├── cluster.ipynb              # 结构参数聚类与频率聚类主流程
├── cluster_umap_1.py          # 聚类脚本
├── parce.ipynb                # PARCE 训练数据处理与 RCNet 文件夹生成
├── parce_batch.py             # RCNet 批量训练脚本
├── predict.ipynb              # 预测流程
├── dataset/
│   ├── phono_freq_dataset.csv # 声子频率示例数据
│   └── raman_dataset.csv      # 拉曼频率示例数据
├── examples/
│   ├── phonon_freq/           # 声子频率预测示例
│   └── raman/                 # 拉曼频率预测示例
└── template/
    ├── getdata.py             # RCNet 数据读取模块
    ├── network.py             # RCNet 网络结构
    ├── train.py               # RCNet 训练脚本模板
    └── utils.py               # RCNet 工具函数
```

运行过程中可能生成以下目录：

| 目录 | 说明 |
| --- | --- |
| `all_cluster_results*/` | 不同特征组合和簇数的 K-means 聚类结果 |
| `Kmeans_model/` | 保存的 K-means 模型 |
| `clustering_results/` | 选定聚类方案的结果 |
| `standardized_frequencies/` | 标准化后的频率数据 |
| `frequency_cuts/` | 截断后的频率数据 |
| `avg/` | 簇内平均频率结果 |
| `training_data/` | RCNet 训练数据 |
| `data_for_conversion/` | 数据转换临时目录 |
| `rcnet_training_umap/` | 批量生成的 RCNet 训练目录 |
| `model/` / `model_results/` | 训练得到的模型文件和结果 |
| `batch_logs/` | 批量训练日志 |
| `test/` | 预测测试数据与结果 |
| `id/` | ID 映射文件 |

## 使用流程

### 1. 激活环境

```bash
conda activate PARCE
```

### 2. 准备数据

将数据放入 `dataset/` 目录，并确认至少包含以下字段：

```text
frequency
space_group_number
pearson_symbol
wyckoff_sequence
c_a_ratio
beta_angle
```

如果使用 `cluster_umap_1.py`以在后台进行结构聚类，请先检查并修改脚本中的路径配置：

```python
DATASET_PATH = "dataset/your_dataset.csv"
BASE_OUTPUT_DIR = "all_cluster_results"
```

Notebook 中也可能包含数据路径配置，运行前请同步检查。

### 3. 结构参数聚类与频率聚类

打开并按顺序执行：

```bash
jupyter notebook cluster.ipynb
```

该步骤将完成：

- 结构参数特征编码；
- K-means 聚类；
- 聚类效果评估；（可运行cluster_umap_1.py代替）
- 频率标准化与截断；
- 亲和传播聚类；
- 训练数据初步生成。

### 4. 生成 RCNet 训练目录

打开并按顺序执行：

```bash
jupyter notebook parce.ipynb
```

该步骤将基于 `template/` 目录批量生成 RCNet 训练文件夹。

### 5. 批量训练 RCNet 模型

先检查将要运行的训练任务：

```bash
python parce_batch.py --dry-run
```

确认无误后开始训练：

```bash
python parce_batch.py --yes
```

常用参数示例：

```bash
# 指定训练目录
python parce_batch.py --rcnet-training-dir ./rcnet_training_umap --yes

# 指定特征组合和频率长度
python parce_batch.py --main-folders 124 125 --num-folders 6 12 18 --yes

# 强制并行训练
python parce_batch.py --parallel --max-workers 4 --yes

# 仅使用 CPU
python parce_batch.py --cpu-only --yes
```

### 6. 模型预测

打开并按顺序执行：

```bash
jupyter notebook predict.ipynb
```

预测结果通常输出到 `test/` 或 Notebook 中配置的结果目录。

## 主要参数

### 结构特征编码

| 编码 | 特征 |
| --- | --- |
| `1` | `space_group_number` |
| `2` | `pearson_symbol` |
| `3` | `wyckoff_sequence` |
| `4` | `c_a_ratio` |
| `5` | `beta_angle` |

示例：

- `124` 表示使用空间群、Pearson 符号和 `c/a` 比率；
- `125` 表示使用空间群、Pearson 符号和 `β` 角；
- `12345` 表示使用全部结构特征。

### 聚类参数

| 参数 | 默认/示例 |
| --- | --- |
| K-means 簇数范围 | `30-90`，步长 `5` |
| 评估指标 | Silhouette Score、SSE 拐点 |
| 频率截断长度 | 以 Notebook 或脚本配置为准 |
| 批量训练目录 | `./rcnet_training_umap` |

## 输入输出

### 输入

- 材料频率序列：声子频率或拉曼活性频率；
- 材料结构参数：空间群、Pearson 符号、Wyckoff 序列、`c/a` 比率、`β` 角。

### 输出

- 结构类型 ID（簇 ID）；
- K-means 聚类模型；
- RCNet 训练数据；
- RCNet 模型文件；
- 预测结果 CSV。

## 示例

| 示例目录 | 说明 |
| --- | --- |
| `examples/phonon_freq/` | 根据声子频率进行结构分类 |
| `examples/raman/` | 根据拉曼活性频率进行结构分类 |

## 注意事项

1. 建议保留标准文件名 `README.md`，不要使用 `READ.ME`。
2. 运行 Notebook 前，请先确认数据路径和输出路径是否为当前项目路径。
3. 部分脚本或 Notebook 可能包含历史绝对路径，迁移项目后需要手动修改。
4. 批量训练会生成大量模型、日志和中间数据，运行前请确认磁盘空间充足。
5. 如果服务器没有 GPU，`parce_batch.py` 会自动退回 CPU 模式，也可以使用 `--cpu-only` 强制使用 CPU。
