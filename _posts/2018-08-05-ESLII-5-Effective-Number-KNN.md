---
title: "ESL II —— 5 —— KNN算法的Effective Number/Degree of Freedom"
description: "整理KNN的一个小细节"
category: [ESL,featured]
comments: true
mathjax: true
excerpt_separator: <!--more-->
tags: [Statistical Learning,Theories,KNN]
key: ESLII_05
---

> ESL II中有提到关于 "Effective Number of $$k$$-Nearest-Neighbors is $$\frac{N}{k}$$, N is Number of Samples"，原文中一笔带过，可能高估了我的统计基础= =！，整理如下：

## 1. 定义

KNN定义不在多讲， KNN算法的Effective Number(又名Degree of Freedom, 或VC-Dimension)：

$$
\begin{equation*}
DF_{knn}(k, N) = \frac{N}{k}
\end{equation*}
$$

<!--more-->

## 2. 预备

我们在提及模型的Effective Number时，一般意义上是说 该模型的“复杂度”、拟合该模型的难度、拟合过程中参数的“自由度”。

一般来说，参数更多的模型更为灵活（因为有更多可以调整学习的“机会（参数）”），但同时也更难训练（因为参数多就意味着需要更多的数据来建模）、更容易过拟合（当数据不够多或者有偏时）。

### 2.1 显式参数模型

一个有显示参数模型的Degree of Freedom(DF)，是指 这个模型有多少个可以*随意取值*的独立参数，本质上也可以定义为模型所有可能的参数向量所张成空间的Rank。

如线性回归$$f(x) = \beta_1^Tx + \beta_0$$中的参数组成向量$$\beta = (\beta_1,\beta_0)$$，具有两个完全任意在$$R$$中取值的独立参数，该向量张成的空间Rank=2，即一般线性回归$$f(x)$$的DF为2。

### 2.2 非显式参数模型

那对于那些没有显式参数的模型，又如何衡量他们的DF呢？

没有参数的模型，如果能找到一个等价且具有显式参数的模型，即可确定其DF。

那么使用KNN算法，基于$$N$$个样本的数据集，对数据进行分类，如果假设N个样本可以自然地分割成x个region，那么对于$$K=k$$的KNN算法，$$x\approx \frac{N}{k}$$, 这个结果也符合我们的直观感受。

这样，如果$$x$$落入某个region，则预测结果就被这个region中的点决定。

基于此，如果我们KNN算法使用的是Majority Vote方式，即：$$k$$个最近邻中，选取占比最高的类别作为分类结果，那么KNN算法的预测结果可以等价地改写为如下公式：

$$
\begin{equation*}
KNN_{NN_k(x)}(x) = \sum_{i=1}^{[\frac{N}{k}]} \arg\max_{b}COUNT(y_b)I(x \in region_r)
\end{equation*}
$$

可知，KNN算法可以等价地改写为一个具有$$\frac{N}{k}$$个参数（$$\arg\max_bCOUNT(Y_b$$)的线性分类模型。

因此，KNN算法的DF为$$\frac{N}{k}$$。

REF:

[1. CrossValidated-Why is N/k the effective number of parameters in k-NN?](https://stats.stackexchange.com/questions/357524/why-is-n-k-the-effective-number-of-parameters-in-k-nn)

[2. CrossValidated-What is meant by effective parameters in machine learning](https://stats.stackexchange.com/questions/114434/what-is-meant-by-effective-parameters-in-machine-learning)
