# 1. ML Basics

## 1.1. ML small notes

#### Why in gradient boosting literature and many optimization algorithms, the loss function is written as: $L(y, F) = \frac{1}{2}(y - F)^2$ ?

- There is an extra coefficient of 1/2 in front of MSE calculation.

The only reason for this is **mathematical convenience**:

- When differentiating squared error with the 1/2 coefficient: $\frac{\partial}{\partial F}[\frac{1}{2}(y - F)^2] = F - y$
Without the coefficient: $\frac{\partial}{\partial F}[(y - F)^2] = 2(F - y)$
- But this is just a formal difference that doesn't affect how the algorithm works or the final results.

#### Softmax Function

## 2. Gradient (梯度)
对于函数 $f(x_1, x_2, ..., x_n)$，其梯度表示为:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$$

这个向量指向函数值增加最快的方向，其大小表示在该方向上的变化率

梯度的几何意义：

- 它是一个向量，指向函数增长最快的方向,每个component是一个分量
- 向量的大小（模长）表示最大增长率
- 在机器学习中，我们通常沿着梯度的负方向移动，以最快地减小损失函数值

Gradient的计算：

- 对于函数 $f(x_1, x_2, ..., x_n)$，计算每个变量的偏导数： $\frac{\partial f}{\partial x_1}$, $\frac{\partial f}{\partial x_2}$, ..., $\frac{\partial f}{\partial x_n}$
- 形成梯度向量： $\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right)$

例如，对于函数 $f(x,y) = x^2 + 2xy + y^2$：

- $\frac{\partial f}{\partial x} = 2x + 2y$, $\frac{\partial f}{\partial y} = 2x + 2y$
- gradient $\nabla f = (2x+2y, 2x+2y)$

考虑函数 $f(x,y) = x^2 + y^2$：

- 偏导数：$\frac{\partial f}{\partial x} = 2x$ 和 $\frac{\partial f}{\partial y} = 2y$
- 梯度：$\nabla f = (2x, 2y)$
在点 $(3,4)$ 处的梯度是 $(6,8)$
- 这个梯度向量告诉我们，在点 $(3,4)$ 处，函数$f$沿着向量 $(6,8)$ 的方向增长最快。为了最小化函数，我们需要沿着 $(-6,-8)$ 方向移动。

## 2. Cross Entropy

直观上，交叉熵可以理解为：

- 当预测概率接近真实概率时，交叉熵值较小
- 当预测概率偏离真实概率时，交叉熵值较大
- 特别是，当模型对正确类别的预测概率接近1时，交叉熵损失接近0

 当模型对正确类别的预测概率接近0时，交叉熵损失趋向无穷大，这会对模型产生很大的惩罚

用一个"错误"的概率分布Q来表示"真实"概率分布P时所需的平均编码长度: 
$$H(P, Q) = -\sum_{i} P(i) \log Q(i)$$

- P(i)是真实分布的概率
- Q(i)是预测分布的概率
- 求和遍历所有可能的事件i

$$D_{KL}(P||Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)} = \sum_{i} P(i) \log P(i) - \sum_{i} P(i) \log Q(i) = -H(P) + H(P,Q)$$

因此： $$H(P,Q) = H(P) + D_{KL}(P||Q)$$


### Cross Entropy in Binary classification

In binary classification problem, cross entropy is also called log loss or binary cross entropy

$$H(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

- whereas y∈{0,1} is actual label，ŷ is prediction

### Kullback-Leibler Divergence (KL 散度)

Kullback-Leibler散度(也称为KL散度或相对熵)是衡量两个概率分布P和Q差异的非对称度量

- 衡量两个分布的差异
- 非负性：$D_{KL}(P||Q) \geq 0$，当且仅当P=Q时等于0
- 非对称性：通常$D_{KL}(P||Q) \neq D_{KL}(Q||P)$
- 不满足三角不等式：不是一个真正的距离度量

 $$D_{KL}(P||Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}$$

relationship between KL-divergence and cross entropy
$$D_{KL}(P||Q) = H(P,Q) - H(P)$$

Cross entropy = KL divergence  + Informational entropy. 

$H(P)$ is fixed becasue this is actual distribution (As the case of distribution P = distribution Q)

- e.g labels are[0.1,0.2,0.7,0]:

- H(P) = -0.1×log(0.1) - 0.2×log(0.2) - 0.7×log(0.7) - 0×log(0)

Minimize cross entropy $H(P,Q)$ is equivalent to minimizing KL divergence $D_{KL}(P||Q)$

### Softmax function in Nueral Network (NN)

Softmax = $\hat{y}j = \frac{e^{z_j}}{\sum{k=1}^{C} e^{z_k}}$

## 3. Parametric Model and Non-parametric Model

参数模型假设数据来自具有固定数量参数的特定分布或函数形式。
Parametric models assume data comes from a specific distribution or functional form with a fixed number of parameters.

### Parametric Models

- Fixed number of parameters: 不随训练数据量增加而变化
- Assumes specific functional form: 对数据分布有明确假设
- 训练后只保留参数：不需要原始训练数据
- "形状"预先确定: Adapts to data through parameter adjustments

### Non-parametric Models

- Variable parameter count: 随训练数据增加而可能增加
- Fewer prior assumptions: 让数据决定模型结构
- 通常保留训练数据：预测时需要参考原始数据
- Shape determined by data: 模型结构随数据复杂性调整

### Semi-parametric Models

They contain parametric components (with a fixed number of parameters) and non-parametric components (where parameter count or complexity may grow).

- 混合结构：部分假设, 对数据分布的某些方面做出假设，但允许其他方面保持灵活
- 在参数模型的效率和非参数模型的灵活性之间取得平衡

## 4. Online Learning

**Online learning**:   数据流式到达，模型实时更新, Data arrives in streams, model updates in real-time

**Offline learning**: Batch training, periodic retraining

### Production Standard Practice

- 离线预训练: 历史数据训练基础模型 / Pre-train on historical data
- 在线微调: 实时数据增量更新 / Incremental updates with real-time data
- 混合预测: 基础模型 + 在线调整 / Base model + online adjustments

#### Why This Approach

- Cold Start: 需要初始模型 / Need initial model
- Stability: 基础模型保证下限 / Base model ensures lower bound
- Adaptability: 在线学习跟上变化 / Online learning adapts to changes

### Online Learning Gradient Computation Challenges

在offline training当中， gradient calculation based on 完整数据集，梯度估计准确.

For online learning, the gradient is calculated based on single sample, and could vary and noisy.

#### Mathematical Principle of Adaptive Learning Rate 

Adagrad更新规则

$$\begin{align}
g_{t,i} &= \frac{\partial \ell(f_\theta(x_t), y_t)}{\partial \theta_i} \\
G_{t,i} &= G_{t-1,i} + g_{t,i}^2 \\
\theta_{t+1,i} &= \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,i} + \epsilon}} g_{t,i}
\end{align}$$

- $\theta_{t,i}​$:  第t步第i个参数
  - 含义: 模型的第i个参数在第t个时间步的值
  - 与样本关系: 每处理一个新样本，时间步t增加1，所有参数都可能更新
- $g_{t,i}$​ - 第t步第i个参数的梯度
  - 直接由第t个样本$(x_t, y_t)$ 计算得出，每个新样本产生新的梯度
- $G_{t,i}​$ - 累积梯度平方和 / Accumulated squared gradients
  - 含义: $G_{t,i} = \sum_{s=1}^{t} g_{s,i}^2$ (从第1个样本到第t个样本的梯度平方累加)
  - 与sample: 随着样本增加而累积，记录了参数i的"学习历史"
- η - 基础学习率 / Base learning rate
  - 全局学习率参数，通常设为固定值(如0.01)
  - 与具体样本无关，但影响对每个样本的响应强度

自动调节学习步长

由于$G_{t,i}​$ 单调递增，一般会做periodic reset， sliding window， Exponential Decay或者直接用Adam

#### Online learning model storage strategies

- Checkpoint Mechanism, 不是每次更新都存储
  - 按时间间隔: 每5-10分钟保存一次
  - 按样本数量: 每1000-10000个样本保存一次
  - 按性能变化: 模型性能显著变化时保存

- 增量存储 / Incremental Storage: 只存储参数的变化量，不存储完整模型
  - $\theta_{\text base}$​: 基础模型参数
  - $\Delta\theta_i$​: 第i次更新的参数变化量
  - 存储空间: O(更新次数) 而不是 O(模型大小 × 更新次数)

- 分层存储 / Hierarchical Storage: 三层架构
  - 内存层 (Redis): 毫秒级读写，存储热参数
  - SSD层: 秒级读写，存储检查点
  - 冷存储 (HDFS/S3): 分钟级读写，长期备份

#### Disaster Recovery 容灾和恢复

版本回滚 / Version Rollback
- 在线学习后模型性能下降，需要回滚

多副本策略 / Multi-replica Strategy: 冗余存储, 同时在多个位置存储检查点
- 主副本: 用于正常服务
- 备副本: 用于故障恢复
- 跨地域: 防止单点故障

#### Redis Redis 内存存储 Remote Dictionary Server
- 内存数据库: 数据存储在RAM中，不是硬盘
- 键值存储: 类似Python的字典，key-value结构
- 超快速度: 读写速度在微秒级别