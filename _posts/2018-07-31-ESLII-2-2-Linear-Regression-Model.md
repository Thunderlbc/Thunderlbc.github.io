---
layout: post
title: "ESL II —— 2(2) —— Linear Regression Model"
description: "整理线性回归模型细节"
category: [ESL,featured]
comments: true
mathjax: true
excerpt_separator: <!--more-->
tags: [Statistical Learning,Theories,Linear Regression]
---

<hr>
* TOC
{:toc}
<hr>

继之前所说的一般线性回归模型的定义以及方法：

- 完全理想化的*回归模型*，不考虑观测误差以及$X,Y$之间的依赖性(EPE)
- 把$X,Y$之间的关系直接假设为*线性关系*，演变成**线性回归模型**
- 进一步考虑实际情况，考虑观测误差，亦即：$Y=f(x)+\varepsilon$
- 使用LSE、MSE、EPE等衡量标准，求得最优模型参数

<!--more-->

## 1. Shortcomings of Linear Regression

- Prediction Accuracy. 当使用LSE估计线性回归模型最优参数时，我们也就选择了*low bias*以及*large variance*特性的方式。
- Interpretation. 当特征维数较大的时候，我们更倾向于得出一个较小的Subset，用以涵盖大部分特征信息。

综上，看起来，在线性回归的基础上，通过一些具有*bias*的trick，牺牲一些*bias*，换来更多的预测准确率，似乎是一个不错的选择。

## 2. Best-Subset Selection

很直白的想法就是，我们通过一些衡量手段(*measures*)，遍历所有的特征维度，选取较为符合我们预期的特征子集，作为我们的**Best-Subset**。

> Typically, we choose the smallest model that minimizes an estimate of the Expected Prediction Error

至于选择最优子集的方法，主要分为三种：Forward- Backward- Stepwise Selection 以及双向同时进行。

有QR分解等方法可以使得程序加速运行。

## 3. Shrinkage Methods

上述子集选择方法，直观，但是对于特征来说，是一个离散的0-1选择，要么选，要么就不选。这种现象与我们“**牺牲bias换取low variance**”的初衷相悖。

接下来的Shrinkage类方法，则更加的*连续*，就减少了**variance**。

***

### 3.1 Ridge Regression

> Key Point: 限制在一般线性回归模型中参数$\beta$的大小（绝对值）

初衷：

- 考虑特征之间存在相关性的情形，如果一个特征$x_i$的参数为$\beta_i$，另外一个特征$x_j$的参数为$\beta_j$，同时$\beta_i,\beta_j$异号，并且绝对值都较大。
- 那么，这种情况下，本来这两个特征所提供的信息，有用的起码是一个特征，但是，由于两个比较大的正负参数，相抵，使得这两个特征对于模型完全没有贡献。

岭回归模型如下：

不同于线性回归模型中仅仅对于RSS最优化，岭回归中在RSS的基础上，添加了一个对于参数$\beta$大小的惩罚项。

$$
\hat{\beta}^{ridge}=\mathop{argmin}_{\beta}\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2,
$$

$$
subject \ to \sum_{j=1}^p\beta_j^2\leq t
$$

$$
\Rightarrow \hat{\beta}^{ridge}=\mathop{argmin}_{\beta}\lbrace\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2+\lambda \sum_{j=1}^p\beta_j^2\rbrace
$$

将上面公式表示成矩阵样式，有如下模型：

$$
RSS(\lambda)=(y-X\beta)^T(y-X\beta)+\lambda\beta^T\beta
$$

回忆线性回归模型中，关于$RSS$的最优化过程，即：求导后使导数为0.可以得到：

$$
\hat{\beta}^{ridge}=(X^TX+\lambda I)^{-1}X^Ty
$$

#### 3.1.1 Additional Insight

对于任意矩阵$A$，有$SVD$分解：$X=UDV^T$。其中，$U$为$N\times p$的矩阵，$V$为$p \times p$的正交矩阵。且$U$矩阵的列向量，张成矩阵$X$的列向量空间，$V$矩阵的列向量，张成$X$的行向量空间。$D$则是对角矩阵。对角元素则为奇异值。

那么对于之前所得到的关于线性回归模型中，最优化LSE结果可改写为：

$$
X\hat{\beta}^{ls}= X(X^TX)^{-1}X^Ty= UU^Ty
$$

可以看出，$\hat{y}=X\hat{\beta}=UU^Ty$一式中，$U^Ty$是在对$X$的列空间进行变换，得到相互正交的基$u_i$之后，求得$y$在其中的坐标值$U^Ty$，所以，对于一般线性回归模型中LSE最优化过程来说，本质上，就是将$y$通过**线性变换**到一个正交的坐标系下的过程。

那么，对于岭回归，同样将SVD分解结果带入，可以有：

$$
X\hat{\beta}^{ls}= X(X^TX+\lambda I)^{-1}X^Ty=UD(D^2+\lambda I)^{-1}DU^Ty=\sum_{j=1}^pu_j\frac{d_j^2}{d_j^2+\lambda}u_j^Ty
$$

其中，$d_j^2$为SVD分解中，$D$矩阵的第$i$个对角元素。可以看出，相比于之前的线性回归模型，岭回归是将本来变换后的坐标值，在不同程度上**Shrinking**了一下。

### 3.2 The Lasso

Lasso是与岭回归极其相似，但是却有千差万别的效果的回归模型。具体如下：

$$
\hat{\beta}^{ridge}=\mathop{argmin}_{\beta}\sum_{i=1}^N(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j)^2,
$$

$$
subject \ to \sum_{j=1}^p|\beta_j|\leq t
$$

唯一区别就是，岭回归使用了参数向量的二范数作为限制条件，而lasso则使用了一范数。

解决Lasso问模型的方法更多的，是偏向工程方面，使用各种快速近似方法得到结果。

关于Lasso和岭回归效果的区别，有如下图可表示：

![Lasso_Ridge_Difference](http://7u2ldb.com1.z0.glb.clouddn.com/Lasso_Ridge_diff.jpg "Width:40px;float:right")

![Contour](http://7u2ldb.com1.z0.glb.clouddn.com/Contours_of_Beta.jpg)

***

Done！Thanks

***
