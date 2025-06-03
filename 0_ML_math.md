# ML数学基础 <!-- omit in toc -->

> 部分面试会直接考察统计与数学知识。即使不直接考察，在ML环节用数学佐证自己的观点也很有裨益

## 1. 概率统计

### **中心极限定理 / Central Limit Theorem**
给定一个任意分布的总体，每次从总体中随机抽取n个样本，一共抽m次。然后把这m组抽样分别求平均值，这些平均值的分布接近正态分布。

其表示的是样本均值分布的特征。

##### 中心极限定理案例 
假设我们有一家快餐店，想了解顾客的等待时间分布。实际的等待时间可能并不服从正态分布 - 可能是偏斜的、右尾较长的分布（因为有些顾客可能需要等待特别长的时间）。

这些样本是什么： 在这个例子中，"这些样本"指的是每个顾客的等待时间。每个单独的等待时间是一个样本点

我们每天记录30个顾客的等待时间，计算这30个等待时间的平均值。这是一个"样本均值"。
持续做这个实验50天，我们就得到了50个"样本均值"。
根据中心极限定理，这50个样本均值的分布将近似服从正态分布，即使原始的等待时间分布不是正态的。

### **假设检验 / [Hypothesis Testing](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)**

- 通过样本来推测总体是否具备某种性质
- 和最大似然类似？做出某个假设之后，依据其分布计算出，给出在这个分布下观察到这个现象的概率

### **P值 / P-value**

- 在假设原假设H0正确时，出现当前证据或更强的证据的概率
- the p-value represents the probability of obtaining a test value, 
which is as extreme as the one which had been observed originally. 
The underlying condition is that the null hypothesis is true.
- 例子：H0是coin是fair coin， 出现事件是5次toss硬币全部向上，概率0.03125， 没有更extreme的情况会出现，所以p-value是0.03125，足够极端 < 0.0.5


### **[z检验](https://en.wikipedia.org/wiki/Z-test)** 
用于两个总体均值之间是否存在明显差异， 或者样本平均值是否与已知的总体平均值有显著差异

Z = (X̄ - μ₀) / (σ / √n)  for sample mean vs population mean

Z = (X̄₁ - X̄₂) / √(σ₁²/n₁ + σ₂²/n₂) for two population


- 均值对比的假设检验方法主要有Z检验和T检验，Z检验面向总体数据和大样本数据，而T检验适用于小规模抽样样本

### **[t检验/t-test](https://en.wikipedia.org/wiki/Student%27s_t-test)**

- t检验比z检验的普适性更强，z检验要求知道总体标准差，但实际研究中无法获知总体标准差，一般都会用t检验。且当样本量足够大的时候，数据接近正态分布，
t检验几乎成为了z检验，z检验是t检验的一个特例

单样本t-test: t = (X̄ - μ₀) / (s / √n)

两总体t-test: t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)

### **[F-test](https://en.wikipedia.org/wiki/F-test#:~:text=An%20F-test%20is%20any,which%20the%20data%20were%20sampled.)**
比较两个或多个variance，或评估统计模型对数据拟合的sigificance。F检验基于F分布，该分布是两个chi-sq distribution 比值的分布

F = s₁² / s₂² (bigger s1, smaller s2)


**Bayes theorem**

Bayes' Theorem is a fundamental theorem in probability theory that describes the probability of an event based on 
prior knowledge of conditions related to the event. It provides a way to update existing beliefs (prior probabilities) based on new evidence to form new beliefs (posterior probabilities)

**P(A|B) = [P(B|A) × P(A)] / P(B)**

For multiple mutually exclusive events A₁, A₂, ..., Aₙ, the total probability of event B can be expressed as
**P(B) = P(B|A₁)P(A₁) + P(B|A₂)P(A₂) + ... + P(B|Aₙ)P(Aₙ)**

so can derive: **P(Aᵢ|B) = [P(B|Aᵢ) × P(Aᵢ)] / [P(B|A₁)P(A₁) + P(B|A₂)P(A₂) + ... + P(B|Aₙ)P(Aₙ)]**

**VIF**

**ANOVA**

**simpson paradox**

**蒙特卡洛**

## 2. 矩阵论

### **特征值与特征向量 / Eigenvalue & Eigenvector**

for a matrix A if Av = λv, then λ is eigenvalue and v is eigenvector

- 矩阵理解为不同坐标系下的变换。几何上看，特征向量是矩阵变换后方向不变的向量，而特征值则是该向量在变换中被拉伸或压缩的比例。

### **迹 / Trace**

- 主对角线上的元素之和
- 矩阵的迹与特征值之和有关
- 协方差矩阵的迹是样本方差的和

## 3. 微积分

机器学习中使用的微积分主要用于优化和反向传播

### Jacobian Matrix
First-Order Derivatives, for vector-valued functions (multiple outputs) with respect to multiple inputs.

### Hessian Matrix
Second-Order Derivatives, for scalar-valued functions (one output) with respect to multiple inputs — second-order derivatives.

## 4. 问答

- What is p-value? What is confidence interval? Explain them to a product manager or non-technical person.
  - toss coin example for p-value
  - confidence interval is It's a numerical range that we are somewhat "confident" contains the true population value. Not probability

- How do you understand the "Power" of a statistical test?

- If a distribution is right-skewed, what's the relationship between medium, mode, and mean?

- When do you use T-test instead of Z-test? List some differences between these two.

- Dice problem-1: How will you test if a coin is fair or not? How will you design the process(有时会要求编程实现)? what test would you use?

- Dice problem-2: How to simulate a fair coin with one unfair coin?
- 3 door questions.
- Bayes Questions:Tom takes a cancer test and the test is advertised as being 99% accurate: if you have cancer you will test positive 99% of the time, and if you don't have cancer, you will test negative 99% of the time. If 1% of all people have cancer and Tom tests positive, what is the prob that Tom has the disease? (非常经典的cancer screen的题，做会这一道，其他都没问题了)
- How do you calculate the sample size for an A/B testing?
  - 确定显著性水平 α 和统计功效 1−β，常见选择是0.05和0.8
- If after running an A/B testing you find the fact that the desired metric(i.e, Click Through Rate) is going up while another metric is decreasing(i.e., Clicks). How would you make a decision?
- Now assuming you have an A/B testing result reflecting your test result is kind of negative (i.e, p-value ~= 20%). How will you communicate with the product manager? If given the above 20% p-value, the product manager still decides to launch this new feature, how would you claim your suggestions and alerts?
- 给定visitors and conversations，怎么计算significance
- 什么是type I/II error
- 圆周上任取三个点，能组成锐角三角形的概率是多大？
- rejection sampling
- [假设现有一枚均匀硬币，现要投掷硬币，直到其两次出现正面，求投掷的期望次数](https://zhuanlan.zhihu.com/p/64262250)
- Frequentists vs. Bayesians
  - One is called the frequentist interpretation. In this view, probabilities represent long run frequencies of events. For example, the above statement means that, if we flip the coin many times, we expect it to land heads about half the time.
  - The other interpretation is called the Bayesian interpretation of probability. In this view, probability is used to quantify our uncertainty about something; hence it is fundamentally related to information rather than repeated trials. In the Bayesian view, the above statement means we believe the coin is equally likely to land heads or tails on the next toss
  - One big advantage of the Bayesian interpretation is that it can be used to model our uncertainty about events that do not have long term frequencies. For example, we might want to compute the probability that the polar ice cap will melt by 2020 CE. This event will happen zero or one times, but cannot happen repeatedly. Nevertheless, we ought to be able to quantify our uncertainty about this event. To give another machine learning oriented example, we might have observed a “blip” on our radar screen, and want to compute the probability distribution over the location of the corresponding target (be it a bird, plane, or missile). In all these cases, the idea of repeated trials does not make sense, but the Bayesian interpretation is valid and indeed quite natural. We shall therefore adopt the Bayesian interpretation in this book. Fortunately, the basic rules of probability theory are the same, no matter which interpretation is adopted.