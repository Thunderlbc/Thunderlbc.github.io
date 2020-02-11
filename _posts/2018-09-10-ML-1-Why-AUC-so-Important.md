---
title: "MachineLearning——1——AUC指标为什么如此重要"
description: "整理AUC指标原理、物理意义以及计算方法"
category: [elements]
comments: true
mathjax: true
excerpt_separator: <!--more-->
tags: [Statistical Learning,Theories,AUC,MachineLearning]
key: ML_01
---

## Why AUC metric so Important?

> 之前训练CTR排序模型时，一直使用AUC作为指标。但是对它只有一个感性的认识（衡量排序结果好坏，倾向于把正样本排到负样本之前）；
> 现在将AUC的定义、原理以及物理意义整理如下，并总结常用计算方法；


## 1 定义原理

AUC(Area Under ROC Curve)指标是指基于TPR（True Positive Rate）以及FPR（False Positive Rate）所形成的[ROC][1]（Receiver Operating Characteristic）曲线下方图形（与x轴正方向）的面积。

首先从ROC曲线定义开始。

<!--more-->

### 1.1 ROC曲线

在分类（或者排序）问题中，我们基于数据集$$D=\{(x_{i},y_{i}), \forall i \in [1,...,N], y_i \in {0,1}\}$$，训练得到模型，并为每个新的数据点$$x*,y*$$预测对应0到1之间的概率值（label=1）。

对于M个数据点$$\hat{x}_i, \forall i \in [1,...,M]$$，模型得到对应M个预测概率$$\hat{p}_i, \forall i \in [1,...,M]$$，如果我们选用阈值为$$t$$， 则概率值大于$$t$$的预测结果对应标签为1，否则为0。

基于上述M个(数据点-标签对)集合$$\hat{D}_{t}=\{(\hat{x}_i, \hat{y}_i), \forall i \in [1,...,M], \hat{y}_i \in {0,1}\}$$，我们便可以计算得到[混淆矩阵][2]：

![混淆矩阵](https://i.loli.net/2019/06/27/5d14968708ef923416.png)

上图（从维基百科扒来的）中很清晰地解释了混淆矩阵以及相应各个Rate的计算方法。其中我们主要关注True Positive Rate(TPR)以及False Positive Rate(FPR)：

#### TPR、FPR
TPR又可以称为Recall、Sensitivity、检测概率，FPR又可以称为误报率，他们的计算公式如下：
$$
TPR=\frac{\#True\ Positive}{\#Condition\ Positive},
FPR=\frac{\#False\ Positive}{\#Condition\ Negative}
$$

上述式中TP、FP以及CP、NP在图中都有提及。

#### 曲线绘制

通过上述TPR、FPR在某个阈值$$t$$下得到的比率，我们遍历$$t \in [0,1]$$（实际情况一般是遍历所有$$\hat{p}_i$$）便可以得到多个$$(FPR_i,TPR_i) \forall i \in [1,...,M] $$对。

以FPR为X轴，TPR为Y轴，我们便可以绘制得到ROC曲线了。如下图(来自[维基百科][1])所示：

![ROC Curve](https://upload.wikimedia.org/wikipedia/commons/6/6b/Roccurves.png)



## 2 物理意义

在之前了解AUC的时候，很多文章都提及「它代表了排序算法的性能，AUC指标越高，说明更多正例样本被排序到负例样本的前面」。但并没有详细解释其原因，同时也没有提及其真正的物理意义。

这导致我当时其实并没有理解AUC的本质，5W1H中只知道了一个What以及一个How而已。

### 2.1 统计定义

首先我们给出上述TPR、FPR的统计学定义。假设我们正在评价一个二分类问题，目标值1代表正例，0代表负例。每当模型给出某样本$$(x,y)$$对应的概率值得分后，我们会选定一个阈值$$t \in [0,1]$$，来区分预测结果$$\hat{y}$$的正负，则有如下计算公式：

$$
T(t) = P \{ \hat{p}(x) > t \ | \ y = 1 \}
$$

$$
F(t) = P \{  \hat{p}(x) > t \ | \ y = 0 \}
$$

如此，ROC曲线则可以看做变量$$T(t)$$在变量$$F(t)$$上的函数对应图像。
### 2.2 结论推导

> 本节参考[Alexej Gossman的博客](https://www.alexejgossmann.com/auc/)

那么，根据上节的定义，AUC值即该函数在0到1区间上对于$$F(t)$$的定积分。

$$
AUC = \int_{0}^{1} T_0dF_0
 = \int_{0}^{1} P \{ \hat{p}(x) > F^{-1}(F_0) \ | \ y = 1 \} dF_0
$$

$$
AUC =  \int_{0}^{1} P \{ \hat{p}(x) > t \ | \ y = 1 \} \ \frac{\partial{F_0}}{\partial t} dt \
= \int_{0}^{1} P \{ \hat{p}(x) > \hat{p}(x') \ | \ y = 1 \} ·  P \{  \hat{p}(x') == t \ | \ y' = 0 \} dt \
$$

$$
AUC = \int_{0}^{1} P \{ \hat{p}(x) > \hat{p}(x')\ \& \  \hat{p}(x') = t \ | \ y = 1 \ \&  \ y' = 0 \} dt  \
$$

$$
AUC = P \{  \hat{p}(x) > \hat{p}(x') \ | \ y = 1 \ \& \ y' = 0 \}
$$

从上面公式可以看出，AUC值对应了一个条件概率值，即：**任意给定一个正例与一个负例，模型对于正例的评分大于负例的概率**。这也从统计学上印证了之前的「直观感受」：排序结果的好坏程度。

### 2.3 几何感知

> 本节参考[Scatterplot Smoothers博客](https://madrury.github.io/jekyll/update/statistics/2017/06/21/auc-proof.html)

从直观的几何上，我们又可以如何理解AUC呢？

**TODO： 仍需梳理**


## 3 计算方法

### 3.1 计算方法A

既然有上述关于正、负例分值大小概率的结论，很直观的一个想法就是直接从模型得到的所有样本及其分值中，统计得到AUC值。

给定M个正例、N个负例，以及（M+N）个模型计算得分，我们可以计算AUC如下：
$$
AUC = \frac{\sum I (P(x_+) , P(x_-))  }{M * N}, \
\forall (x_+, x_-) \ pair，其中I(x,y) = \lbrace 1 \ if \ x > y; 0.5 \ if \ x == y; 0 \ if  \ x < y \rbrace
$$

### 3.2 计算方法B

3.1中计算方式不难发现，在实际情况中，遍历所有正、负例性能效率较差，在数据规模较大时，影响模型评估迭代效率。

将所有样本按照得分大小降序排列，排在第一位的rank=1，依次类推。下列公式告诉我们，AUC值的计算，只需要计算所有正例的排列序号之和，与正负例个数M、N相结合即可：

$$
AUC = \frac{\sum_{ins_i \in positive \ class} rank_{ins_i} - \frac{M*(M+1)}{2}}{M * N}
$$

### 3.3 计算方法C

3.2相较于3.1计算复杂度降低不少，但是在数据规模较大时，排序操作会使得这个评估时长急剧增加，同时为系统带来极大负担。当数据规模足够大，无法全部放入内存时，上述3.1、3.2中的方法将不再feasible。

可以采用如下方法：

1. 将模型输出的得分值[0,1]分桶，比如200；
2. 分块地遍历全部数据；
2. 对于每一块数据，计算正例、负例各个得分分桶数量；
3. 待全部数据遍历完成，将各个分桶数量作为sample_weight传入sklearn.metrics.roc_auc_score中；
4. 上述接口中的label，pred分别是：[1,1,1,0,0,0] 以及 [0,0.25,0.5,0,0.25,0.5]

代码如下：
```python

pos = [0. for i in range(partitions+1)]
neg = [0. for i in range(partitions+1)]

...
# incriment the counts at specific index of pos and neg with respect to each score and corresponding true label


label = np.concatenate([np.ones(partitions+1), np.zeros(partitions+1)])
preds = np.concatenate([np.arange(partitions+1)*1.0/(partitions+1),                 np.arange(partitions+1)*1.0/(partitions+1)])
weights = np.concatenate([pos, neg])
auc = metrics.roc_auc_score(label, preds, sample_weight=weights)
```


### 3.4 方法对比

相比而言，3.3在数据规模足够大的情况下，计算简单、资源要求低，同时也能较准确地估计AUC真实值。

3.1、3.2方法在概念上更加直观，但只适用于小数据。

  [1]: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
  [2]: https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Confusion_matrix
