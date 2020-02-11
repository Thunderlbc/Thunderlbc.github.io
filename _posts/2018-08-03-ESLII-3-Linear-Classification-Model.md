---
title: "ESL II —— 3 —— Linear Classification Model"
description: "整理线性分类模型细节"
category: [ESL,featured]
comments: true
mathjax: true
excerpt_separator: <!--more-->
tags: [Statistical Learning,Theories,Linear Classification]
key: ESLII_03
sidebar:
  nav: eslii

---

## 1.Linear Decision Boundaries

>**Intuition**
>Instead of making an assumption on the form of the **class densities**, we make an assumption on the form of the **boundaries** that **seperating classes**.

对于离散分类问题来说, 我们**总是**能够把**input space**分成对应不同label的子空间.

这些子空间之间的边界可以是Rough的,也可以是smooth的.其中,有一类比较重要, 常见的,则是线型决策边界([linear decision boundaries](http://en.wikipedia.org/wiki/Decision_boundary))

如,  当我们fit一个线型模型来建模$k,l$的时候,有

$$
\begin{equation}
\hat{f}_k(x)=\hat{\beta}_{k0}+\hat{\beta}_k^Tx
\end{equation}
$$

因此, 对于类别$$k$$以及$$l$$之间的决策边界来说, 有:

$$
\begin{equation}
set \lbrace x:(\hat{\beta}_{k0}-\hat{\beta}_{l0}) + (\hat{\beta}_k - \hat{\beta}_l)^Tx = 0 \rbrace
\end{equation}
$$

这相当于有一个**HyperPlane**作为决策边界,属于$$k$$的点在这个边界上方.

这种方法属于一种**对每一类都建立一个判别函数$$\delta_k(x)$$,并选择最大的函数值作为分类结果**的方法.

当然,我们的主题还是要寻求**线型**决策边界. 事实上, 我们只需要判别函数$$\delta_k(x)$$是线型的,或者此函数的**单调变换**是线型的即可.

<!--more-->

## 2. Logit Tranformation

Logit变换就是上文中"判别函数的**单调变换**是线型" 的情况.

下面解释为何这么说.

如同我们已经知道的,逻辑回归的表达式对于2分类来说,有:

$$
\begin{align}
P(G=1|X=x) &= \frac{e^{\beta_0 + \beta^Tx)}}{1+e^{\beta_0 + \beta_T}} \\
P(G=2|X=x) &= \frac{1}{1+e^{\beta_0 + \beta_T}}.
\end{align} \tag{1}
$$

那么罗辑回归为什么要有这样一种形式呢? 原因在于由于Logit变换$$logit(p) = \frac{p}{1-p}$$对于$$p$$来说,是一个**monotone**的变换.正如前面所说的,如果一个判别函数的monotone的变换结果,是一个关于$$x$$的线型方程, 那么原始的判别函数所得出的决策边界,也应是*线型*的.

那么,对于判别函数$$P(G=1\|X=x)$$来说,不妨设有:

$$\begin{equation}
log \frac{P(G=1|X=x)}{P(G=2|X=x)} = \beta_0 + \beta^Tx
\end{equation}$$

则,根据这个假设倒推得出的判别函数$P(G=1\|X=x)$应具有**线型决策边界**.

接下来, 就很容易推出公式(1).

## 3. Linear Discriminant Analysis

同样对于判别函数$$P(G=k\|X=x)$$来说，可以根据贝叶斯法则变换为：

$$
\begin{equation}
P(G=k|X=x)=\frac{P(G=k,X=x)}{P(X=x)} = \frac{P(X=x|G=k)P(G=k)}{P(X=x)}
\end{equation}
$$

如果，对于两类$$k$$与$$l$$来说，我们假设每个类都的似然函数$$P(X\|G)$$服从 多元正态分布(multivariate Gaussian distribution),即：

$$
\begin{equation}
P(X=x|G=k) = \frac{1}{(2\pi)^{\frac{p}{2}} |\Sigma_k|^{\frac{1}{2}}}e^{-\frac{1}{2}(x - \mu_k)^T\Sigma_k^{-1}(x-\mu_k)}
\end{equation}
$$

结合上述两式，线性判别分析(Linear Discriminant Analysis,LDA)假设所有的类服从的分布，具有相同的协方差矩阵$$\Sigma$$.

可知相应的$$Logit(P)$$应为：

$$
\begin{align}
Logit(P) & = log \frac{P(G=k|X=x)}{P(G=l|X=x)} \\
& = log\frac{p_g}{p_l}-\frac{1}{2}(\mu_k+\mu_l)^T\Sigma^{-1}(\mu_k-\mu_l)+x^T\Sigma^{-1}(\mu_k-\mu_l)  \tag{2}
\end{align}
$$

可以看出，这是一个关于$$x$$的线性方程。也就是说，通过一些**Constraints**,LDA使得判别函数$$P(G\|X)$$的单调变换$$Logit(P)$$是一个关于$$x$$的线性方程。结合上文，我们知道LDA的**决策边界**应是**线性**的。

从公式(2)我们可以看出，对于每一类来说，有线性判别函数：

$$
\begin{equation}
\delta_k(x)=x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+log(p_k)
\end{equation}
$$

我们只需找到$$argmax_k \delta_k(x)$$即可。这就是线性判别分析。

这里,对于$$\delta_k(x)$$来说,我们只需估计他的参数$$\pi_k,\mu_k,\Sigma$$即可. 可采用如下方式进行估计(Full log-likelihood estimates):

$$
\begin{align}
\hat{\pi}_k &=N_k/N, \ where \ N_k \ is \  the \ Number\ of\ class-k\ observations; \\
\hat{\mu}_k&=\sum_{g_i=k}x_i/N_k; \\
\hat{\Sigma} &=\sum_{k=1}^K\sum_{g_i=k}(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T/(N-K).
\end{align}
$$

## 4. Logistic Regression

如同第2部分Logit Transformation中所讲的那样，Logistic Regression仅仅是在判别函数的逻辑变换的基础上，假设变换后满足关于$$x$$的线性关系。

下面主要叙述关于Logistic Regression目标函数的最优化过程。

逻辑回归模型主要通过最大化条件似然$$P(G\|X)$$来最优化.那么我们的目标函数有如下形式:

$$
\begin{equation}
l(\theta)=\sum_{i=1}^Nlogp_{g_i}(x_i;\theta),
\end{equation}
$$

where $p_k(x_i;\theta)=P(G=k\|X=x_i;\theta)$.

那么,对于参数$$\beta$$,二分类情况的条件似然函数如下:

$$
\begin{align}
l(\beta)&=\sum_{i=1}^N\lbrace y_ilogp(x_i;\beta)+(1-y_i)log(1-p(x_i;\beta)) \rbrace \\
 &=\sum_{i=1}^N\lbrace y_i\beta^Tx_i -log(1+e^{\beta^Tx_i}\rbrace
 \end{align}
 $$

 对上式求导可有:

 $$
 \begin{equation}
 \frac{\partial l(\beta)}{\partial \beta}=\sum_{i=1}^Nx_i(y_i-p(x_i;\beta))=0,
 \end{equation}
 $$

 这是一个关于$$\beta$$的非线性方程组.我们可以使用Newton-Raphson算法迭代近似求出:

 $$
 \begin{equation}
 \beta^{new}=\beta^{old}-(\frac{\partial ^2 l(\beta)}{\partial \beta \partial \beta^T})^{-1} \frac{\partial l(\beta)}{\partial \beta}
 \end{equation}
 $$
 where $$\frac{\partial ^2 l(\beta)}{\partial \beta \partial \beta^T}=-\sum_{i=1}^Nx_ix_i^Tp(x_i;\beta)(1-p(x_i;\beta))$$ is [Hessian Matrix](http://en.wikipedia.org/wiki/Hessian_matrix) or Second Derivative.

## 5. Logistic Regression or LDA?

从Logistic Regression 和LDA的出发点以及假设条件我们可以看出,两者最后的形式都有:

$$
\begin{equation}
\log \frac{P(G=k|X=x)}{P(G=K|X=x)} = \beta_0 + \beta^Tx
\end{equation}
$$

的形式,但是,两者得出这个结果的方式不同.

- LR直接对于$$P(G=k\|X=x)$$进行假设,**Directly**有上式的结果.
- LDA则结合贝叶斯法则,通过对$$P(X\|G=k),P(G)$$进行假设,然后得出与上式类似的形式的结果.

也就是说, LR对于模型的**Assumption**更少,适用范围也就更加广泛点.

同时,由于LDA对于$$P(X\|G=k),P(G)$$的假设,模型参数得出的时候(如,MLE)需要结合一些离决策边界比较远的数据, 也就意味着,LDA对于一些**Outliers**是不够鲁棒的.

与此同时,LDA关于上述$$P(X\|G=k)$$的假设(服从多元正态分布),意味着对于输入数据$$X$$中,如果有离散的变量存在,理论上来说,会使得模型精确度下降(事实上没那么大区别),而LR则对于观测数据没有任何假设,更**Safer** 和 **Robust**.

## 6. Seperating HyperPlanes

>Intuition
>本质上来说,无论是LDA或者是LR,都是在估计一个线性的决策边界(hyperplane)来将训练数据分离开来.

主要想法则是,构建一个线性的决策边界,使得尽可能地将数据分为不同的类别.

>预备知识
>Perceptron: 任何将输入的features经过线性变换输出一个sign的分类器,都称之为感知机.

关于Seperating HyperPlane, 有Rosenblatt's Perceptron Learning Algorithm:

>主要思想:
通过最小化错分数据点同决策边界的距离之和,来寻找一个合适的分割平面.
i.e. 最小化:
$$
\begin{equation}
D(\beta,\beta_0)=-\sum_{i \in M}y_i(x_i^T\beta + \beta_0)
\end{equation}
$$

Done!


