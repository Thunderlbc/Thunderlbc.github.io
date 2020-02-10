---
layout: article
title: "Paper —— 1 —— A Deep Learning based Comment Sequence Labeling System for Answer Selection Challenge X. Zhou’15 SemEval 2015"
description: "论文整理"
category: [Paper,CQA,CNN,QA,SL]
comments: true
mathjax: true
tags: CNN,CQA,Sequence Labeling
aside:
  toc: true
---
> 文章主要是SemEval 2015测评中,答案选择挑战问题中的参赛论文. 基于14年Hu et.al.在NIPS上的一篇文章, 提出了在CNN提取问题q与评论c的交互信息的基础上, 使用RNN来对不同评论之间的的语义关联进行建模, 以及标出Good类的评论, 取得了不错的效果.


<!--more-->

##一. Intro

此次阅读的文章,所涉及到的挑战赛有如下内容:

* Subtask A: Given a question (short title + extended description), and several community answers, classify each of the answers as
definitely relevant (good),
potentially useful (potential), or
bad or irrelevant (bad, dialog, non-English, other)

* Subtask B: Given a YES/NO question (short title + extended description), and a list of community answers, decide whether the global answer to the question should be yes, no or unsure, based on the individual good answers. This subtask is only available for English.

对于这种Community Question Answering(CQA)平台下产生的数据来说, 如果可以有一种方法将众多的答案中好的答案挑出, 坏的答案过滤, 那么对于接下来可能的工作,如建立知识库, 信息检索等都有很大帮助.

一般来说, 大多数的解决方法, 都是通过衡量问题以及每一个回答之间的语义相关度来进行区分.

在之前传统的方法中, 有些是通过特征工程(feature-engineering)构造有效特征, 结合经典模型,如LR或者SVM等, 来度量语义相似度.  在此之上, 有些工作不仅仅只是使用了文本上的一些信息, 同时也结合了结构上的信息, 综合得到一个关于<问题-答案>对的表示.

与此同时, 也有一些方法是通过建立User-Profile的方式, 挖掘用户的行为习惯以及信息, 来对之前的方法进行增益, 取得了不错的效果.

近几年来, 基于神经网络的方法, 在NLP领域取得了不少的成果. 2013年Hu 等人挖掘文本以及非文本信息,作为一个Multi-DBN模型的输入,学得对于每一个<问题, 答案>对的联合表示(joint representation). 2014年Yu等提出了一个基于CNN, 来表示<问题, 答案>对的方法, 主要发掘<问题,答案>对的分布式表示来作为输入.

但是, 这些工作的一个问题是, 他们仅仅是对于<问题, 答案>对之间的语义相关度进行了建模, 忽略了答案之间的相关性信息.

###二. 系统简介

[Hu et al.(2014)](http://www.hangli-hl.com/uploads/3/1/6/8/3168008/hu-etal-nips2014.pdf)文献主要的结构图如下:

![基础系统结构示意图](http://7u2ldb.com1.z0.glb.clouddn.com/B_HU_CNN_MS.png)

本文的主要结构图如下:

![本文系统结构图](http://7u2ldb.com1.z0.glb.clouddn.com/X_zhou_CNNRNN_QA.png)

可以看出, 本文主要结余上述文献, 唯一不同的地方, 在于在经过多次2D-Convollution之后, 不同于采用MLP来得出最后预测结果, 而采用了一个RNN的结构, 多次训练隐层变量, 最终得出预测结果.

####2.1 CNN for <question, answer> Matching

文中这个部分基本基于 Hu et al.(2014)所提出的用于Sentence Match的CNN结构.

主要过程是(参考Hu et al.(2014)):

* 对于原始数据,问题q, 答案c, 来说, 使用word-embedding得到整个句子每个词的向量.
* 第一层1D-Convolution层, 选用窗口为(3x3)大小, 简单拼接得到的向量$\hat{z} _{i,j}^{(0)} = [q _{i:i+k-1}^T, c_{i:i+k-1}^T]^T$之后, 通过卷积, 将这9*N维数据映射为一个数据点
* 在得到所有类别f, 所有类别上的点$i,j$的值$\hat{z} _{i,j} ^{(f)}$的时候, 进行2D-Convolution&Pooling操作. 这次操作本质上没有特别的不同,唯一就是选择用来做卷积的区域不同, 选择了相邻的4个邻接点作为输入, 通过卷积到一个数据点. 也就是 $(i, j) -> (i,j), (i+1,j), (i,j+1), (i+1,j+1)$.
* 多次2D-Convolution&Pooling

####2.2 Recurrent NN for Comment Sequence Labeling

作者选择RNN的原因是, RNN可以通过记录历史的答案信息, 很好的对 不同 答案之间的语义相关度进行建模,从而结合不同答案之间的相关度, 来提高结果.

主要更新方式如下,

$$
x(t) = w _i p(t) + w _h(t-1) + b _h \\
h(t) = \sigma (x(t))  \\
y(t) = g(w _y h(t) + b _y) \\
$$

###三. Experiment

下图是试验所示用的数据示意图

|data|#q|#comment|#avg|%good|
|-|-|-|-|-|
|Train|2600|16541|6.36|48.48|
|Devel|300|1645|5.48|53.19|
|Test|329|1976|6.00|50.46|

在开发集上结果示意图如下:

|Methods|Macro.|Acc.|P|R|F1|
|-|-|-|-|-|-|
|CRF|50.56|59.82|72.41|77.37|74.81|
|CRF+V|52.14|61.03|74.80|76.00|75.40|
|R&CNN|52.10|60.85|75.09|75.09|75.09|

在测试集上结果示意图如下:
|Methods|Macro.|Acc.|P|R|F1|
|-|-|-|-|-|-|
|CRF|40.54|60.12|57.90|95.89|72.21|
|CRF+V|49.50|67.86|65.99|91.68|76.74|
|R&CNN|53.82|73.18|74.39|85.96|79.76|
|SFR|57.29|72.67|80.51|78.03|79.11|

此模型主要有几点好处:

* 通过深度学习, 将短文本(句子)中的词语的深层语义信息提取出来, 相对于CRF等直接从文本里面抽取极为稀疏的语义特征来说, 更为有效.
* 同时,由于CRF是通过细致的特征工程来作为基础, 就会受到数据集质量的影响, 适应性不够.
*对于RNN的使用, 确实能够一定程度上捕捉到答案之间的语义相关性, 从而提高结果.

