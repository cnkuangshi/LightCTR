![Alt text -w135](./LightCTR_LOGO.png)
## LightCTR Overview
LightCTR is a lightweight and scalable framework that combines mainstream algorithms of Click-Through-Rate prediction Based Machine Learning, Deep Learning and Philosophy of Parameter Server. The library is suitable for sparse data and designed for Large-scale Distributed Model Training.

Meanwhile, LightCTR is also an open source project that oriented to code readers. The clear execute logic will be of significance to leaners on Machine-Learning-related field.

#### 用于群体发现
点击率预估即是给合适的用户群体投放合适的内容，以达成促进广告收益或交易转化率的目的。具体操作来说，将收集到的用户点击与行为数据，用离散值与连续值结构化描述特征、归一化与De-bias等处理后，就要选择合适的模型，来对用户是否会对某一内容感兴趣并带来商业转化的概率进行评估；通常可将所有特征组合输入集成树模型`LightCTR::GBM`预先找群体，训练得到的每个叶子节点代表一个用户群，再使用`LightCTR::LR`或`LightCTR::MLP`对树模型建立的低维0/1群体特征做进一步分类。
当标识类别的离散特征过多使得输入变得高维稀疏，可能达到树模型与神经网络的处理瓶颈；一般可使用`LightCTR::FM`或`LightCTR::FFM`将离散特征做特征交叉训练，提升了特征利用率并降低了数据稀疏下的过拟合的风险，每维特征映射在低维空间中，也方便作为连续特征输入其他模型。
相比使用FM预训练特征低维映射后、再输入DNN中的两阶段训练，通过DNN在输入层按Field内部局部连接，在输入层直接端到端训练特征低维Embedding，可以更好的保证模型时效性；或采用将DNN嵌入FM模型中的`LightCTR::NFM`及其他相关变种，进行特征非线性高维组合，提升模型的表征能力，拥有更好的AUC表现。

#### 用于行为序列
用户点击内容序列往往蕴含内容间的局部相关性信息，如前后点击同一类商品或查看同一类网页，这些行为序列的局部关系可被`LightCTR::Embedding`建模捕获，得到点击内容或行为的低维隐向量表示；或基于变分自编码器`LightCTR::VAE`实现特征组合衍生，增强低维特征的表达能力；低维隐向量可被用来判断相关性或直接作为其他模型输入。进一步，序列数据经过平滑处理后，将训练好的行为隐向量按时序输入循环神经网络`LightCTR::LSTMUnit`，最后将RNN输出的特征表达输入`LightCTR::Softmax`分类器，利用预设的监督标签，训练对用户的评估模型或判别模型；
当预设标签覆盖率不足时，可将高维特征表达输入`LightCTR::GMM`进行无监督聚类，聚类簇概括为意图簇，作为用户意图匹配的依据，补充到用户画像的人群类别中。

#### 用于内容分析
用户评论、搜索广告页面上下文等文本也蕴含很多用户的兴趣信息可被挖掘，如搜索广告场景下用户关键词需要匹配语义相关度最高的拍卖词进而投放高转化率的广告，因此分析文本信息是点击率预估的重要依据。在提取整段文本语义信息方面，首先可使用`LightCTR::Embedding`预先训练词向量表，对文本中出现的词按词频做词向量加权并移除主成分；或参考Skip-thought方法结合负例采样，将文本中每个词向量按时序输入`LightCTR::LSTMUnit`，训练得到文本的语义特征表达，向量内积对应文本相关度。此外，参考DSSM将由文本词向量构成的矩阵，输入空洞卷积神经网络`LightCTR::CNN`用于提取句子局部语义相关性特征，训练对预设标签的判别模型。
当文本缺乏分类标签时，可使用`LightCTR::PLSA`无监督的获取文章主题分布，应用于按主题分布区分不同内容类别、计算长语料与短查询间的语义相似度，也可通过后验计算得到上下文中各词汇的重要程度，作为长文本关键词摘要。

#### 分层模型融合
更复杂的模型带来更好的表征能力，但同时也加大了计算时间消耗，响应时间与点击率呈强负相关，为了兼顾线上点击率预估的性能与效果，可使用不同模型逐层预测，如第一层采用在线学习、并引入稀疏解的简单模型`LightCTR::FTRL_LR`，第二层采用上文提到的输入层局部连接的`LightCTR::MLP`、或`LightCTR::NFM`等复杂模型进行精细预测。在系统层面，抽取并缓存DNN模型中最后一组全连接层权值或输出，作为用户或商品的固定表达，使用`LightCTR::ANN`近邻向量检索的TopN结果作为推荐召回，在最大化CTR/ROI的同时，降低线上推理的平均响应时间。此外，LightCTR在探索通过模型参数分位点压缩、二值网络等方法，在不损失预测精度前提下大幅提升计算效率。

#### 多机多线程并行计算
当模型参数量超过单机内存容量、或单机训练效率达不到时效性要求时，LightCTR基于参数服务器与异步梯度下降理论，支持可扩展性的模型参数集群训练。集群分为Master, ParamServer与Worker三种角色；一个集群有一个Master负责集群启动与运行状态的维护，大规模模型参数以DHT散布在多个ParamServer上，与多个负责模型数据并行梯度运算的Worker协同，每轮训练都先从ParamServer拉取(Pull)一个样本Batch的参数，运算得到的参数梯度推送(Push)到ParamServer进行梯度汇总，ParamServer通过梯度截断、延迟梯度补偿、动量修正等手段，异步无锁的迭代更新参数。参数在ParamServer上紧凑存储，通过特征命中率与权值大小进行特征优选与淘汰，提升集群内通信效率。LightCTR分布式集群采取心跳监控、消息重传等容错方式。

## List of Implemented Algorithms

* Factorization Machine
* Field-aware Factorization Machine
* Wide&Deep Neural Factorization Machine
* Gradient Boosting Tree Model
* Gaussian Mixture Clustering Model
* Topic Model PLSA
* Embedding Model
* Multi-Layer Perception
* Dilated Convolution Neural Network
* Self-Attention Recurrent Neural Network
* Variational AutoEncoder
* Approximate Nearest Neighbors Retrieval

## Features
* Optimizer implemented by Mini-Batch GD, Negative sampling, Adagrad, FTRL, RMSprop, Adam, etc
* Regularization: L1, L2, Dropout, Partially connected
* Template-customized Activation and Loss function
* Evaluate methods including F1, AUC
* Compressing Neural Network with Pruning and Quantization
* Support distributed model training based on Parameter Server
* Gradient clipping, momentum correction and Asynchronous SGD with Delay compensation
* Shared parameters Key-Value pairs store in physical nodes by DHT
* Lock-free Multi-threaded training and Vectorized compiling(AVX)

## Quick Start
* LightCTR depends on C++11 and ZeroMQ only
* Change configuration (e.g. Learning Rate, Data source) in `main.cpp`
* run `./build.sh` to start training task
* Current CI Status: [![Build Status](https://travis-ci.org/cnkuangshi/LightCTR.svg?branch=master)](https://travis-ci.org/cnkuangshi/LightCTR)

## Welcome to Contribute
* Welcome everyone interested in machine learning or distributed system to contribute code, create issues or pull requests of new features.
* LightCTR is released under the Apache License, Version 2.0.

## Community
* Contact: kuangshi@kuangshi.info
