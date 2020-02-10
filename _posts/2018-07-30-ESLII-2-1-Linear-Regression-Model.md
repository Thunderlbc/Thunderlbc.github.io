---
title: "ESL II —— 2(1) —— Linear Regression Model"
description: "整理线性回归模型细节"
category: [ESL,featured]
comments: true
mathjax: true
excerpt_separator: <!--more-->
tags: [Statistical Learning,Theories,Linear Regression]
key: ESLII_02_1
---


线性回归模型 [维基百科](http://zh.wikipedia.org/wiki/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8)

>在统计学中，线性回归是利用称为线性回归方程的最小二乘函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。这种函数是一个或多个称为回归系数的模型参数的线性组合。只有一个自变量的情况称为简单回归，大于一个自变量情况的叫做多元回归。

<!--more-->
------------
## 1. Basic Notations

训练数据由$$(X， Y)$$组成，$$X\in\Re^p$$， $$Y\in\Re$$，两者都是随机变量， 目的则是找到一个$$f$$， 来用$$f(X)$$预测$$Y$$。 同时有，$$X， Y$$的联合分布为$$Pr(X，Y)$$。

### a. Expected Prediction Error(EPE)[IDEAL Frame for Regression]

根据上面的符号，自然有

$$EPE(f)=E(Y-f(X))^2=\int[y-f(x)]^2Pr(dx，dy)$$

对于给定的$$X=x$$，我们可以有

$$EPE(f)=E_XE_{Y|X}([Y-f(X)]^2|X)$$

那么，我们的目标则转变为在给定$$X=x$$的时候，求$$f$$使得$$EPE(f)$$最小。

最小化$$EPE(f)$$一个很自然的想法，就是使$$\frac{\partial EPE(f)}{\partial f}=0$$， 推到过程如下：

$$
\begin{align}
\frac{\partial EPE(f)}{\partial f} & = \frac{\partial \int[y-f(x)]^2Pr(dx，dy)}{\partial f} \\
& = -2\int_{X=x}(y-f)Pr(dy) \\
& = 0 \\
& \Rightarrow f\int_{X=x} Pr(dy)= \int_{X=x} yPr(dy)
\end{align}
$$

由于$$\int_{X=x} Pr(dy)=1$$， 因此，有：

$$
\begin{equation}
f=\int_{X=x} yPr(dy)=E(Y|X=x)
\end{equation}
$$


也就是说，使得$$EPE(f)$$最小的$$f$$应是随机变量$$Y$$对于$$X$$的条件随机分布。

从定义上来讲，$$EPE(f)$$是衡量观测值$$Y$$与预测值$$\hat{f}(X)$$之间的差异，而均方差（[MSE](http://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E5%B7%AE)）则是衡量预测值$$\hat{f}(X)$$与真实值$$f(X)$$之间的差异，有如下公式：若有

$$Y_0=f(x_0)+\varepsilon _0$$

则有，

$$E(Y_0-\hat{f}(x_0))^2=\sigma^2+E(x_0^T\hat{\beta}-f(x_0))^2=\sigma^2+MSE(\hat{f}(x_0))$$

也就是说，$$EPE(f)$$与$$MSE(\hat{f})$$之间的差别，只是一个观测值$$Y_0$$的恒定方差$$\sigma^2$$。

### b. Linear Regression[Closer to Reality]

当然，上面的情况是完全**理想**状态中的模型，它**并没有**考虑：

- 观测值$$Y$$不一定完全等于$$f(X)$$， 更一般的时候，应有$$Y=f(X) + \varepsilon$$， 这就造成了上述的$$EPE(f)$$不能完全表达出真实的情况。同时$$f=E(Y\|X=x)$$一般情况下**难以求出**。

- 上述模型中，有一个条件假设，即：$$X=x$$的情况下进行考虑。

上述两个情况不同的模型有不同的解决方法。

- [KNN模型](http://zh.wikipedia.org/wiki/%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E6%B3%95)， 某点的预测值，只与此点周围$$K$$个临近点有关。此算法对于上述两点的近似为**期望使用Average样本点近似**和**条件假设使用仅与周围$$K$$个点有关近似**。

- 线性回归模型， 直接假设$$f$$对于全部样本点，有一个全局线性关系，即： $$f(x)= E(Y\|X=x)\approx x^T\beta$$。 将此式带入$$EPE(f)$$可得， $$\beta=[E(XX^T)]^{-1}E(XY)$$， 而线性回归模型将其中的期望**使用在训练数据中Average近似**， 如： $$\hat{\beta} = (X^TX)^{-1}X^Ty$$。

## 2. Introduce the Linear Regression

定义与标识符号与之前相同。

线性回归模型有如下形式：

$$f(X)=\beta_0 + \sum^p_{j=1} X_j\beta_j$$

其中，$$X^T=(X_1, X_2, \ldots, X_p)$$。此处，$$X_i$$为标量。

### a. Least Square Estimate(LSE)

> Pick the coefficients $$\beta=(\beta_0, \beta_1, \ldots, \beta_p)^T$$ to minimize the Residual Sum of Squares(RSS)

$$RSS(\beta)=\sum^N_{i=1}(y_i-f(x_i))^2=\sum^N_{i=1}(y_i-\beta_0-\sum^p_{j=1}x_{ij}\beta_j)^2 \tag{2.a.1}$$

Each $x_i = (x_{i1}, x_{i2}, \ldots, x_{ip})$。

接下来，我们开始最小化上式。 记$$X$$为$$N×(p+1)$$矩阵，它的每一行为一个输入向量。$$y$$为$$N$$维向量，作训练集中的输出。 则，公式(2.a.1)可转换为如下：

$$RSS(\beta)=(y-X\beta)^T(y-X\beta)$$

这是一个有$$p+1$$个参数的方程。对于$$\beta$$求偏导数，得：

$$\frac{\partial RSS}{\partial \beta}=-2X^Ty+2X^TX\beta=-2X^T(y-X\beta)$$

$$\frac{\partial^2RSS}{\partial \beta \partial \beta^T}=2X^TX$$

#### a.1 $$X$$是满列秩的

如果， 此时**假设**$$X$$**是满列秩**的， 那么，当我们把一阶偏导数置为0的时候，有

$$X^T(y-X\beta)=0$$

可以得到**唯一解**：

$$\hat{\beta}=(X^TX)^{-1}X^Ty\tag{2.a.a.1}$$

当我们把$$X$$矩阵用它的列向量$$(x_0, x_1,\ldots, x_p), x_0\equiv 1$$表示时，我们可以说，我们是在找一个关于$$X$$的列向量的一个线性表示方法，也就是$$\hat{\beta}$$，来最精确的表示出$$y$$。

从公式(2.a.a.1)可以看出，我们所要找的$$\beta$$应是满足$$y-X\hat{\beta}$$与$$X$$所张成的向量空间正交的$$\hat{\beta}$$。 那么我们之前选取$$\hat{\beta}$$最小化$$RSS(\beta)=\|\|y-X\beta\|\|^2$$的行为，则是使得残差向量$$y-\hat{y}$$与$$X$$**的列向量**所张成的向量空间*正交*。

#### a.2 $$X$$*不是* 满列秩的

当然，所假设的满列秩，更多的，只是理想情况。满列秩当且仅当这$$p$$个特征之间相互独立。因此，一旦有一个或者几个列向量之间线性相关（这通常是很常见的），那么我们就不能按照公式(2.a.a.1)来直接解决这个问题。

从简单说起，当$$X$$只有一维特征的时候，也就是说$$X=(x_1,x_2, \ldots, x_N)^T$$，$$y=(y_1,y_2,\ldots,y_N)^T$$。那么可以由上述$$\hat{\beta}$$得到：

$$\begin{align}
\hat{\beta}&=\frac{\sum^N_1 x_iy_i}{\sum^N_1 x^2_i}，\\
r_i&=y_i-x_i\hat{\beta}.
\end{align}\tag{2.a.a.2}$$

如果记$$<x, y>=\sum^N_{i=1}x_iy_i=x^Ty$$，即内积。那么公式(2.a.a.2)又可表示为：

$$\begin{align}
\hat{\beta}&=\frac{<x,y>}{<x,x>}, \\
r&=y-x\hat{\beta}.
\end{align}$$

当$$X$$有多维特征时候，但是这几维特征都是相互正交的话，即：$$<x_i, x_j> = 0, \forall i \neq j$$， 这就是我们上面讨论的$$X$$是满列秩的情况，对于$$\hat{\beta_j}$$来说，依然有：

$$\hat{\beta_j}=\frac{<x_j, y>}{<x_j,x_j>}$$

那么，对于不满列秩的情况，可以先把$$X$$**转换为满列秩**的，也就是列向量之间正交的，然后即可通过已知方法得出最后所求。

回忆之前提到的残差向量$$y-\hat{y}$$与$$X$$**的列向量**所张成的向量空间*正交*，假设有两个特征向量$$x_1, x_2$$， 我们可以先将$$x_2$$在$$x_1$$上回归，所的残差$$z$$就与$$x_1$$正交，也就是说，此时$$z$$与$$x_1$$无关，如果此时将$$y$$在$$z$$上回归，所得出的唯一参数就应该是多元回归的时候，$$x_2$$的参数。

那么，重复上述步骤，先$$x_2$$后$$x_1$$则可得出$$x_1$$的参数，即可得到所有参数。

上述方法一般化如下：

- Step 1. 初始化$$z_0=x_0=1$$
- Step 2. 对于每一个$$j\in {1,2,\ldots}$$，分别将$$x_j$$在$$z_0, z_1, \ldots, z_{j-1}$$上回归，得出系数$$\hat{\gamma}\_{lj}=\frac{<z_l, x_j>}{<z_l,z_l>}$$，之后可以算出残差向量$$z_j=x_j-\sum^{j-1}\_{k=0}\hat{\gamma}\_{kj}z_k$$
- Step 3. 将$$y$$在最后一个残差向量$$z_p$$上回归，以算出参数$$\hat{\beta}_p$$

但是，上述步骤过于繁杂，需要多次重复。其实，经过证明，可以找到一种只需要遍历一遍$$X$$的列向量即可得出所有参数（而非最后一个向量的参数）的方法。

其实，对于上面的Step 2（其实也叫做[Gram-Schmidt正交化](http://zh.wikipedia.org/wiki/%E6%A0%BC%E6%8B%89%E5%A7%86-%E6%96%BD%E5%AF%86%E7%89%B9%E6%AD%A3%E4%BA%A4%E5%8C%96)），我们可以将这个过程用对矩阵$$X$$的分解表示。$$$$X=Z\Gamma$$$$

其中，$$Z$$的每一列是$$z_j$$（按照顺序）,而$$\Gamma$$的第$$kj$$个元素为$$\hat{\gamma}\_{kj}$$。回想[$$QR$$分解](http://zh.wikipedia.org/wiki/QR%E5%88%86%E8%A7%A3)，构造对角矩阵$$D$$其中$$D_{jj}=\|\|z_j\|\|$$，则有：

$$X=ZD^{-1}D\Gamma=QR$$

其中，$$Q^TQ=I$$，$$R$$是一个上三角矩阵。

那么

$$\hat{\beta}=(X^TX)^{-1}X^Ty=(R^TQ^TQR)^{-1}R^TQ^Ty=(R^TR)^{-1}R^TQ^Ty=R^{-1}Q^Ty$$

$$\hat{y}=X\hat{\beta}=QRR^{-1}Q^Ty=QQ^Ty$$

上述式子是完全可解的，因为$$R$$是上三角矩阵。

## 3. To be Continued

**Subset Selection**, **Shrinkage Methods** and **Comparison** among them.
