材料PARCE预测系统

运行环境

本项目需要在以下环境中运行：

环境名称: PARCE
Python版本: 3.8.20
主要依赖包:
  pandas
  numpy
  matplotlib
  scikit-learn
  umap-learn
  kneed
  tqdm
  joblib
  torch (PyTorch) - 用于RCNet神经网络训练
  re (正则表达式，Python内置)
  os (操作系统接口，Python内置)
  warnings (警告控制，Python内置)
  itertools (迭代工具，Python内置)
  argparse (命令行参数解析，Python内置)
  shutil (文件操作，Python内置)
  ast (抽象语法树，Python内置)

环境激活
conda activate PARCE

依赖包安装
如果环境中缺少某些包，可以使用以下命令安装，例如：
基础科学计算包
pip install pandas numpy matplotlib

机器学习包
pip install scikit-learn umap-learn kneed

工具包
pip install tqdm joblib

深度学习包（用于RCNet训练）
pip install torch torchvision torchaudio

项目概述

PARCE是一个基于材料声子频率预测材料结构性质的机器学习系统。该系统通过多层聚类和深度学习技术，实现从材料声子频率到结构性质的预测。

核心工作流程：

1. 结构参数聚类阶段
通过材料的结构参数（空间群、Pearson符号、Wyckoff序列、c/a比率、β角）进行K-means聚类
将具有相似结构性质的材料聚类到同一簇中，每个簇代表一种结构类型
优化特征组合和聚类簇数，找到最佳的结构分类方案

2. 声子频率处理阶段
对每个结构簇内的材料声子频率进行标准化处理
对声子频率进行截断
计算每个簇内材料的平均声子频率，得到簇代表性频率

3. 亲和传播聚类阶段
基于每个簇的代表性声子频率进行亲和传播聚类
生成训练数据集，建立声子频率与结构类型的对应关系

4. RCNet深度学习训练
使用RCNet神经网络架构训练预测模型
学习声子频率特征与结构类型ID之间的映射关系

5. PARCE集成预测
将训练好的模型集成到PARCE系统中
输入新材料的声子频率，输出预测的结构类型ID（簇ID）

聚类参数说明
特征编码: 1=space_group, 2=pearson_symbol, 3=wyckoff_sequence, 4=c_a_ratio, 5=beta_angle
聚类簇数范围: 30-90（步长为5），可随时修改
评估指标: 轮廓系数(Silhouette Score)和SSE拐点分析

目录结构

├── cluster.ipynb             结构参数与声子频率聚类主程序
├── parce.ipynb               PARCE系统训练数据处理
├── predict.ipynb             PARCE系统预测程序
├── test                      测试数据目录
│   └── test.csv              测试数据
├── dataset/                  输入数据目录
│   └── dataset.csv          材料数据集（包含结构特征和频率信息）
├── all_cluster_results/      所有聚类结果输出目录(下面是例子)
│   ├── 123/                 特征组合123的聚类结果
│   └── 124/                 特征组合124的聚类结果
├── Kmeans_model/            训练好的K-means模型保存目录
├── clustering_results/       选定的聚类结果数据目录（例子选择了124、125、12345）
├── frequency_cuts/          频率截断数据目录
├── standardized_frequencies/ 标准化频率数据目录
├── data_for_conversion/     数据转换临时目录
├── training_data/           训练数据目录
├── your_data/              用户自定义数据目录
├── template/               RCNet文件夹对应模板文件目录
├── model/                  已训练好的达标模型文件目录
├── rcnet_training/         RCNet训练相关目录，其中包含训练文件夹
├── avg/                    簇内平均声子频率结果目录(124组合，30截断长度)
├── batch_logs/             批处理日志目录
└── id/                     ID映射文件目录

使用说明

1. 数据准备

确保输入数据集 dataset/dataset.csv 包含以下必要字段：
material_id: 材料ID
frequency: 材料的声子频率
space_group_number: 空间群编号
pearson_symbol: Pearson符号
wyckoff_sequence: Wyckoff序列
c_a_ratio: c/a轴比
beta_angle: β角

2. 运行PARCE系统

步骤一：结构参数聚类分析
激活环境
conda activate PARCE

启动聚类分析
打开cluster.ipynb

按顺序执行所有单元格，系统将：
基于结构参数进行K-means聚类
自动处理所有3个及以上的特征组合
对每个组合进行30-90簇(可修改)的K-means聚类
生成轮廓系数和SSE分析图表
保存最优聚类结果

处理声子频率数据（标准化、截断、簇内平均）
进行亲和传播聚类生成训练集

步骤二：PARCE模型训练数据集生成
启动PARCE数据处理程序
 parce.ipynb

按顺序执行所有单元格，系统将：
对数据集进行预处理
根据模板文件夹，批量生成RCNet训练文件夹

3. 模型训练
启动PARCE训练程序：
python parce_batch.py

3. 模型预测
启动PARCE预测程序：
python predict.ipynb



输入输出与结果示例说明

输入数据
位置: dataset/dataset.csv
格式: CSV文件，包含材料的结构特征和声子频率信息
必要字段: material_id, frequency, space_group_number, pearson_symbol, wyckoff_sequence, c_a_ratio, beta_angle

PARCE系统输出
输入: 材料声子频率
输出: 预测的结构类型ID（簇ID）
应用: 通过声子频率预测材料的结构类型,获取材料结构信息

中间结果输出
聚类结果: all_cluster_results/ - 包含不同特征组合的K-means聚类分析结果
模型文件: Kmeans_model/ - 保存的K-means模型，用于结构聚类
训练数据: training_data/ - RCNet训练用的数据集
预测结果: test/test_results.csv - 预测结果输出

结果示例文件
examples/phonon_freq - 根据声子频率做结构分类
examples/raman - 根据拉曼活性频率做结构分类