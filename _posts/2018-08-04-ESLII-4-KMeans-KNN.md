---
title: "ESL II —— 4 —— K-Means 及 K-Nearest-Neighbors算法小结"
description: "整理两种分类模型细节"
category: [ESL]
comments: true
mathjax: true
excerpt_separator: <!--more-->
tags: [Statistical Learning,Theories,KNN,K-Means]
key: ESLII_04
sidebar:
  nav: eslii

---

## 0. Introduction

作为同一类Model-Free的分类方法, K-means和KNN两者可能对于理解特征与标签之间的关系不是太有用, 但是在一些问题中,他们作为Blackbox预测器效果很不错.

[下面](## 1)会就原型方法, 如k-means, LVQ等, 介绍一类基于prototype(原型)点的分类方法.

<!--more-->

## 1. Prototype Methods

### 1.1 K-means聚类算法

对于经典的K-means聚类算法来说, 想法比较直接. 给定无标签数据, 以及期望的聚类个数R, 此算法通过迭代, 移动聚类中心以使得**总的聚类内方差**最小.主要算法步骤如下:

```python
while |centre_old - centre_new| > $\theta$
    for each initial centre:
        find the set of points that is closer to it
        set the mean of the set as the new centre
```

这样, 问题的关键,转到了如何计算*closer*的点集. 主要是在特征空间内使用欧氏距离等, 本次不再赘述.

此外, K-means算法可以用在**分类**问题中:

对于一个有监督K分类问题, 可以有如下流程:

```python
for each class k:
    apply a k-means algorithm to produce R(predefined) prototypes
#produce K*R prototypes
for each new feature x:
    classify x to the class of the closest prototypes
```

但是要注意到, 用于K分类问题的K-means算法有一个弊端, 也就是, 对于某一个聚类来说, 其他不同聚类的点对于这个聚类不提供任何信息. 这种弊端可以导致一些不必要的错误, 如, 如果在分割边界处有聚类中心, 会有潜在的错误分类的风险.


### 1.2 Learning Vector Quantization

正是由于K-means算法中, 关于prototype的无限制, 自由得出, LVQ提出可以通过其他类的数据来限制, 更精细地得出最终每一个类prototype的位置.

主要宗旨是, 通过与prototype同一类的数据点, 来使得prototype靠近同类. 通过与prototype不同类的数据点, 来使得prototype远离不同类.

主要算法如下:

```python
Random choose R prototypes m_r(k)
while \epsilon > 0:
    Sample trainning points x_i
    if g(i) == k:
        m_i(k) += \epsilon(x_i - m_i(k))
    else:
        m_i(k) -= \epsilon(x_i - m_i(k))
    \epsilon -= learning_rate
```


## 2. $$k$$-Nearest-Neighbor Classifiers

$$k$$NN主要思想就是, 无需模型, 对于某个点$$x_i$$, 找到训练集中$$k$$个与之最近的点, 类别占比最多的, 就是$$x_i$$的类别.

可以说, 1-NN就是将每一个训练集点看做一个prototype的k-means分类算法.

主要关键问题, 依然是距离的衡量方法问题.

接下来, 将会介绍三种出于不同原因,改变原始的kNN中的距离衡量方法.

### 2.1 Invariant Metrics and Tangent Distance

在一些问题中, 一些特征看似不同, 但却是经过了一些比较自然的变换之后的结果. 比如, 对于手写数字的识别问题中, 经常出现同样是3, 但是由于书写者的习惯不同,而有一定角度的倾斜的问题. 虽然是经过旋转之后, 在像素上, 有很大不同, 但是其中, 还是应有invariant的信息存在.

而KNN, 可以通过对距离的衡量标准的细细定制, 可以发掘出这种不变的信息,而加以利用.

也就是说, 我们可以通过特殊的kNN距离衡量标准, 把有一定倾斜的手写字体3, 都当做是一个3, 或者说很相似的3.

#### 2.1.1 Tangent Distance

为了解决上述具有不变性质的变换, 以及过大的变换使得结果与原来不应该相同(6旋转180度之后是9)的问题, 选用Tangent Distance.

主要思想就是, 对于一个要判定类别的图像点, 计算他的Tangent Line, 以及其他训练集点的Tangent Line, 找出两两Line之间距离最短的K个, 作为判别类别的邻居集合.

但是, 这里有个问题, 就是Tangent Line的计算问题, 其实, 也就是对于一个图片旋转轨迹的函数, 求得其在某个图像点的Tangent Line的问题.

- [ ] **上述问题有待考究**

### 2.2 Adaptive Nearest-Neighbor Methods

由于KNN不变的衡量标准, 导致在特征向量维度比较高的时候, 会出现一种数据点距离越来越远的情况. 有如下关于中位数的结论:

> 考虑均匀分布在p维空间中的单位矩形$$[-1/2, 1/2]^p$$中, 假设R是位于圆点的数据点的1NN近邻点的半径长度, 则:
$$
median(R)=v_p^{-1/p}\left( 1-\frac{1}{2}^{1/N}\right)^{1/p},\\
其中v_pr^p 是p维空间中半径为r的球的体积大小.
$$

- [ ] **上述问题有待细致考究**

从上式可以看出, 当p越大的时候, R的中位数会越来越趋近于0.5, 也就是单位矩阵的边缘.

上述情况也就造成了当特征维度增大的时候, 通常的欧氏距离对于属于不同类的样本点的**区分度越来越低**(因为越来越多的点与中心的距离贴近边缘), 也就可以说, 是数据点周围的k近邻中, 类分布密度函数不是稳定的, 变化较为剧烈, 而导致相近的点, 通过knn在高维数据集上结果不同.

同时,  近邻分类算法中, 有一个隐含的假设, 就是对于某个点周围的近邻中, 每个类的概率是近似稳定的. 这个假设也使得近邻算法能够取得不错的成绩.

因此, 对于高维度的数据来说, 我们需要一种方法, 可以让我们的近邻中, 类的概率变化不大. 通常来说, 我们从距离衡量方法入手(其实这也是唯一可以入手的把).

#### 2.2.1 Discriminant Adaptive Nearest-Neighbor(DANN)

DANN的距离衡量metric如下:

$$
D(x, x_0)=(x-x_0)^T\Sigma(x-x_0), \\
其中, \Sigma=W^{-1/2}[W^{-1/2}BW^{-1/2}+\epsilon I]W^{-1/2}, \\
W是pooled类内协方差矩阵\Sigma_{k=1}^K\pi_kW_k, \\
B是类间协方差矩阵\Sigma_{k=1}^K\pi_k(\bar{x}_k-\bar{x})(\bar{x}_k-\bar{x})^T
$$

通俗一点说, 这个metric将原来的特征向量, 根据$$W$$映射为一个球体状的数据集, 之后通过求出**球体数据集的类间协方差矩阵**的0特征值对应的方向, stretch, 也就是找到了对于类概率变化不大的方向.

其实这也是一种**Dimension Reduction**的方法, 对于特征的不同维度, 我们通过一些方法, 选出类概率变化不大的维度, 作为衡量距离的数据.

### 2.3 Global Dimension Reduction for Nearest-Neighbor

可以看出, [2.2](### 2.2)中描述的DANN仅仅是对于一个query点, 做一次dimension reduction, 得出分类结果. 但是正体上,我们对于特征空间没有做任何优化操作. 这里的思想就是, 我们从整体考虑, 考虑全部的query点, 来整体地对于特征空间求得最优的子空间, 在子空间上做KNN分类.

对于每一个训练的点, $$x_i$$, 计算the between-centroids sum of squares matrix $$B_i$$, 然后取平均值:

$$
\bar{B}=\frac{1}{N}\sum_{i=1}^NB_i
$$

如果$$e_1, e_2, \ldots, e_p$$是矩阵$$\bar{B}$$的特征向量, 根据特征值$$\theta_k$$从大到小排列, 那么我们取前L个就可以得到最优的rank-L子空间的估计.

也就是说, $$\bar{B}\_{[L]}=\Sigma_{l=1}^L \theta_l e_l e_l^T$$是问题

$$
min_{rank(M)=L}\sum_{i=1}^Ntrace[(B_i-M)^2]
$$

的最优最小二乘解.

- [ ] **上述问题有待细致考究**
