![Alt text -w135](./LightCTR_LOGO.png)
## LightCTR Overview
LightCTR is a light-weight framework that combines mainstream algorithms of Click-Through-Rate Based Machine Learning and Deep Learning. The library is suitable for sparse data and designed for Stand-alone Multi-threaded Model Training.

Meanwhile, LightCTR is also an open source project that oriented to code readers. The clear execute logic will be of significance to leaners on Machine-Learning-related field.

#### 用于群体发现
群体发现作为点击率预估中用户画像的重要指标之一，LightCTR提供了相关的算法支持。可将离散id特征与连续特征组合输入`LightCTR::GBM`预先找群体，训练得到的每个叶子节点代表一个用户群，再使用`LightCTR::LR`或`LightCTR::MLP`对GBDT建立的低维0/1群体特征做高层分类。
当id特征过多使输入成为高维稀疏数据，便不再适合树模型，可通过`LightCTR::FM`将每个单个群体映射在低维空间中，向量内积即表示特征交叉的权重。FM相比LR引入一定的非线性，降低了数据稀疏下的过拟合的风险，或更进一步采用`LightCTR::NFM`结合DNN进行高维特征组合，达到更好的效果。

#### 用于行为序列
用户点击内容序列等用户的行为序列可通过`LightCTR::Embedding`得到行为的低维隐向量表示；隐向量数据经过平滑处理后，按时序输入循环神经网络`LightCTR::LSTM`，从用户行为序列中训练得到基础特征的表示；再基于变分自编码器`LightCTR::VAE`实现特征组合衍生，增强特征的表达能力；最后将特征表达输入`LightCTR::Softmax`分类器，利用样本标签训练生成用户评估，或导入`LightCTR::GMM`进行无监督聚类，作为用户画像的人群特征。

#### 用于内容分析
LightCTR可通过分析用户评论、兴趣得到推荐信息，作为点击率预估的参考。可使用`LightCTR::Embedding`预先训练含语意特征的词向量表，将切分后的词向量文本按时序输入`LightCTR::Attention_LSTM`训练得到句子的情感倾向，并在Attention层获得每个词对分类结果的重要性。  另一种方案是将由文本词向量构成的矩阵，输入卷积神经网络`LightCTR::CNN`用于提取句子局部相关性特征作为文本分类依据。当文本缺乏分类标签时，可使用`LightCTR::PLSA`无监督的获取文章主题分布，作为是否向用户推荐此文章的依据。

#### 分层模型融合
为了兼顾点击率预估的性能与效果，可使用不同模型逐层预测，如第一层采用在线学习、并引入稀疏性解的简单模型`LightCTR::FTRL_LR`，第二层采用`LightCTR::MLP`、`LightCTR::NFM`等复杂模型进行精细预测。

## List of Implemented Algorithms

* Factorization Machine
* Wide&Deep Neural Factorization Machine
* Gradient Boosting Tree Model
* Gaussian Mixture Clustering Model
* Topic Model PLSA
* Embedding Model
* Multi-Layer Perception
* Convolution Neural Network
* Attention-based Recurrent Neural Network
* Variational AutoEncoder

## Features
* Optimizer implemented by Mini-Batch GD, Negative sampling, Adagrad, FTRL, RMSprop, etc
* Regularization: L1, L2, Dropout, Partially connected
* Template-customized Activation and Loss function
* Evaluate methods including F1, AUC
* Multi-threaded training and Vectorized compiling

## Quick Start
* LightCTR depends on C++11 only
* Change configuration (e.g. Learning Rate) in `main.cpp`
* Build by Make and Debug by Xcode

## Welcome to Contribute
* Welcome everyone interested in machine learning or distributed system to contribute code, create issues or pull requests of new features.
* LightCTR is released under the Apache License, Version 2.0.

## Community
* Contact: kuangshi@kuangshi.info
