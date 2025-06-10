# Deep Learning

## **Universal Approximation Theorem(万能近似定理)**

A feedforward neural network with a single hidden layer containing a sufficient number of neurons can approximate any continuous function on a compact subset of ${R}^n$
 with arbitrary precision.
具有单一隐藏层和足够多神经元的前馈神经网络，能够以任意精度近似任何定义在有界闭集上的连续函数

- it provides the theoretical foundation for neural network capabilities.
- **Advantages of Depth**: Mention that although the theorem primarily applies to single-hidden-layer networks, deep networks can achieve the same approximation effect with fewer parameters.
- **Practical Limitations**: Acknowledge the gap between theory and practice
- **Non-linear activation functions** are key to making this theorem work

### **Convergence criteria**:

- Stable Loss function output
  - If the loss from Loss function is lower than a certain threshold like 0.001 **OR**
  - Loss function changes between several consecutive epochs is smaller than a threshold like 0.001
- Performance in validation set
  - If evaluation set stops to perform better (loss not decreasing)
- L2 norm of the gradient is super small or the update in parameters is small

### Gradient Exploding and Vanishing (梯度爆炸与梯度消失)

Gradient exploding and vanishing are two critical problems that occur during the training of deep neural networks, especially in deep architectures or recurrent networks. 尤其在深层架构或循环网络中更为常见。

当通过反向传播计算梯度时，我们使用链式法则。对于深度网络:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot \frac{\partial a_n}{\partial a_{n-1}} \cdot ... \cdot \frac{\partial a_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial w_1}$$

- $L$ is the loss function
- $a_i$ is the activation value in ith layer
- $w_1$ is the weight in first layer

#### **Gradient Vanishing**

Gradients become extremely small as they propagate backward through the network. 梯度在网络中反向传播时变得极小

if each partial derivative term is < 1, then $$\prod_{i=1}^{n} \frac{\partial a_{i+1}}{\partial a_i} \to 0 \text{ as } n \text{ increases}$$

- Earlier layers train very slowly or not at all
- Model performance plateaus quickly
- Changes to weights in earlier layers are minimal

Common reasons:
- Saturating Activation Functions(饱和激活函数)
  - Sigmoid and tanh saturate (flatten) for extreme input values（变平）
  - Their derivatives approach zero in these regions
- Deep Networks(深度网络)
  - Many layers compound the problem
- Small Weights(权重过小)
  - Weights < 1 shrink gradients with each layer

Practical Impact

- Learning Stagnation学习停滞
  - Earlier layers stop learning meaningful features, 较早的层停止学习有意义的特征, Only later layers adapt to the data
- Shallow Network Behavior(浅层网络行为), Deep network effectively behaves like a shallow one
- Poor Representation Learning
  - Lower layers critical for abstract feature extraction

**Solution**

- ReLU and Variants(ReLU及其变体)
- Proper Weight Initialization(适当的权重初始化)


#### **Gradient Exploding**

Gradients become extremely large as they propagate backward. 梯度在反向传播时变得极大

if each partial derivative term is > 1, then:
$$\prod_{i=1}^{n} \frac{\partial a_{i+1}}{\partial a_i} \to \infty \text{ as } n \text{ increases}$$

- Training becomes unstable with large weight updates. 不稳定
- Loss values become NaN (Not a Number)
- Model weights grow to infinity

Common reasons

- Large Weights
  - Weights > 1 amplify gradients with each layer

- High Learning Rates
  - Can cause extreme weight updates,可能导致极端的权重更新
- Recurrent Networks (RNN)
  - Long sequences multiply the same weights many times

Practical Impact

- Training Instability(训练不稳定), Large, erratic weight updates, Model can diverge completely. 
- Numerical Overflow(数值溢出)
  - Leads to NaN values or training crashes
- Learning Rate Sensitivity(对学习率敏感)
  - Even small learning rates can cause unstable training

**Solution**

- Gradient Clipping(梯度裁剪)
  - Cap gradient values at a maximum threshold(限制最大梯度)
- Weight Regularization
  - L2正则化惩罚大权重， Keeps weights from growing too large
- Learning Rate Reduction（降低学习率）
  - Smaller learning rates for unstable models
- Proper Weight Initialization（适当的权重初始化）
  - Prevents large initial values that amplify.

### **Covariate Shift（协变量偏移）**

Internal Covariate Shift: 当网络中某一层的参数更新导致该层的输出分布变化，从而使得下一层的输入分布发生改变的现象 $P_{train}(X) \neq P_{test}(X)$

- parameter update：当我们更新前一层的weight和bias时，该层的输出分布会改变
- Cascading Structure: Due to the cascading nature of deep networks，较早层的微小变化会对后续所有层产生累积影响

#### **Specific Impacts of Covariate Shift**

- 训练难度增加：
  - 每一层不得不持续适应其输入分布的变化
  - 就像在移动的目标上练习射击
- 学习速度减慢：
  - 需要更小的学习率来避免不稳定性
  - 收敛过程变得更加波动和缓慢
- 梯度问题加剧：
  - 当激活值落入饱和区域（如sigmoid的两端），梯度几乎为零
  - 输入分布的剧烈变化会加剧这一问题
- 优化困难：
  - 损失函数表面变得更加复杂和不平滑
  - 更难找到全局最优解

#### **Solutions**:

Batch Normalization (批量归一化)

这是解决内部协变量偏移最常用的方法

Normalizing across all spatial positions and all samples for each channel

- 整个训练集：

  - 典型的训练集可能包含数千甚至数百万张图像（例如ImageNet有超过120万张图像）
  - 一次性处理整个训练集通常在计算上是不可行的
- Mini-Batch处理：
  - 我们将训练集分成多个小批量(mini-batches)
  - 每次迭代处理一个mini-batch（例如32、64或128张图像）
  - 对每个mini-batch计算梯度并更新模型参数
- 批量归一化的应用：
  - 批量归一化正是在这些mini-batch上执行的
  - 对当前mini-batch中的图像计算统计量（均值和方差）
  - 使用这些统计量对同一mini-batch中的数据进行归一化

在每个epoch结束后会重新洗牌!!!

```
训练集 (10,000张图像)
       |
       ↓
分成mini-batches (每个64张图像)
       |
       ↓
每次迭代处理一个mini-batch
       |
       ↓
对当前mini-batch应用批量归一化:
  - 计算这64张图像的统计量
  - 使用这些统计量归一化特征
  - 更新移动平均值（用于推理）
       |
       ↓
更新模型参数，进入下一个mini-batch (param indeed will be updated frequently based on batch number, but it will help converge faster)
```

## Normalization techniques

为什么需要Normalization？

- 核心问题：Internal Covariate Shift
- 不norm的话 第100个epoch的input or output is very differently distributed from the first epoch

### Batch Normalization (BatchNorm) /  批归一化

Normalizes across the batch dimension, computing mean and variance over all samples in a mini-batch for each feature。

Batch Norm

- 问的是：在这个batch中，特征1的值应该如何分布？
- 对每个feature（即每一列），在一个mini-batch的所有样本上归一化。

对一个batch内每个通道分别归一化（均值/方差），让数据分布固定，加速收敛

#### Understanding

- Batch size是“每次训练看到多少条样本”的意思
- 每次梯度更新用多少行数据，通常固定为32、64、128、256等，和特征数无关

每次将所有data points assigns to different batch, we use one batch at a time during each epoch. 一个batch forward and back propagation走完一遍再开始下一个batch, $w$ is updated multiple times (batch numbers).走完一个epoch，继续走下一个epoch，但会重新把所有dp assign到不同batch

Pros

- Fast convergence / 收敛快速
  - 允许使用更大的学习率
- Reduces initialization sensitivity / 降低初始化敏感性
- Have some regularization effect
- 与CNN架构的语义匹配

Cons
- Poor performance with small batches / 小批次性能差
- Creates batch dependencies / 产生批次依赖

**Applications / 应用: CNNs, computer vision / 卷积神经网络、计算机视觉**

### Layer Normalization (层归一化)

Layer Norm:

- 问的是：对于这个样本，它的所有特征应该如何分布？
- 对每个样本（即每一行），把它的所有feature归一化

对每个样本的所有特征维度进行归一化，不依赖于batch

- 对单个样本的所有特征计算均值和方差
- 标准化：(x - μ) / √(σ² + ε)
- 缩放和平移：γ * x_norm + β
  - γ和β的数量 = feature维度数

#### 特点：

- 不依赖batch size，适用于RNN和小batch场景
- 对序列长度变化不敏感
- 训练和推理时行为一致
- 不会因为batch中样本分布差异大而不稳定
- 特别适合domain adaptation和few-shot learning
- Layer Norm: 跨特征归一化，适合序列模型和小batch

Pros:

- 特别适合RNN/Transformer等序列模型
- 不受batch size影响
- 在NLP任务中效果显著
- parallelzation friendly
  - 每个样本可以独立计算，便于并行
  - 不需要等待整个batch才能计算

适用场景
- 变长序列：每个样本序列长度不同
- 小样本学习：batch size很小(如1-4)
- 在线学习：需要逐个样本处理
- 推理优化：避免维护额外统计量
- Attention机制：保持token表示稳定

### Group Normalization

把一组通道内的所有feature放一起计算统计量 (结合了BN和LN的思想)

- 不依赖batch size：小batch或batch=1时仍然稳定
  - 因为多个channel放在一起
  - Group Norm主要用于batch_size=1-8的场景
- 比Layer Norm更符合CNN语义：不是跨所有通道，而是跨相关通道组
- 计算成本适中：比BN稳定，比LN更符合视觉任务直觉

#### Group Norm: Batch Norm在小batch时的替代品

### Instance Norm

对每个样本的每个通道独立归一化

- 每个归一化单元的数据点相对较少

图像的空间相关性提供了足够的统计信息
风格迁移的目标是"去统计化"而非"精确统计"
在典型的应用场景下尺寸足够大
实际效果证明这种"粗糙"的统计量已经足够

这也解释了为什么Instance Norm主要用于特定任务(风格迁移)

### Other notes

使用频率排序

Batch Norm - CNN的标配

Layer Norm - Transformer的标配

Group Norm - 小batch CV任务的救星

Instance Norm - 风格迁移的专用工具

Q1: BatchNorm和LayerNorm的最大区别？

- BatchNorm归一化的是batch内的每个feature，依赖batch
- LayerNorm归一化的是每个样本的全部feature，独立于batch

Q: 为什么Transformer用LayerNorm？

- 因为序列任务/变长输入/小batch/推理时，LayerNorm更稳定且易于并行

Q: 如果batch size=1怎么办？
- 用GroupNorm或LayerNorm。

Q: BatchNorm在推理阶段用什么？
- 用moving average的均值和方差，不用当前batch的统计量。

## Activation Function

### ReLU (Rectified Linear Unit)

$$\mathrm{ReLU}(x) = \max(0, x)$$

Pros:

- Easy to calculate, converge fast, gradient is always zero
- Sparse activation（负区间直接为0），有一定正则化效果。
- Practically, perform good in Deep NN
- ReLU family (ReLU, Leaky ReLU) are computationally cheapest - just comparisons and multiplications.

Cons：

- Dead neuron problem: 如果权重更新后神经元长期落在负区间，梯度为0无法激活

- 输出非0均值，可能导致网络某些层偏移。(Non-zero mean output may cause certain layers in the network to become biased or shifted)
  - 但如果做batch norm可以很好的解决这个问题
- 梯度问题：
  - $x>0$时梯度为1，$x<0$时为0。
  - 负区间无法更新权重（dead neuron）。

典型场景：

- 现代CNN、MLP等深度网络的默认激活函数。
  - because sparse activations help with feature detection and computational efficiency, plus dead neurons are less problematic with many parallel feature maps
- 基本是所有主流深度学习框架的“默认首选”

由于存在神经元死亡问题，所以现在有改进型ReLU：

#### Leaky ReLU

负区间给一点小斜率$\text{LeakyReLU}(x) = \max(\alpha x, x)$，$\alpha$通常取0.01。

- 这样$x<0$时，梯度是$\alpha$，不会彻底为0

#### GELU (Gaussian Error Linear Unit)
定义与原理: GELU是近年来很流行的激活函数，BERT、GPT等大模型广泛使用

标准GELU激活函数公式
$$\mathrm{GELU}(x) = x \cdot \Phi(x)$$

其中 \Phi(x) 的定义
$$\Phi(x) = \frac{1}{2} \left[ 1 + \operatorname{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$

常用近似公式

$$\mathrm{GELU}(x) \approx 0.5\, x \left( 1 + \tanh \left[ \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right] \right)$$

Pros:

- 输出是smooth and approximately symmetric，正负区间都能“软门控”，不会像ReLU那样在负区间直接砍掉。
  - 在0附近关于origin symmetric (奇函数对称)
- 梯度始终不为0，正负区间都能“学习”
- 在NLP/Transformer/大模型训练中比ReLU效果更好，收敛更平滑

Cons:

- 计算量比ReLU大（有CDF/tanh），但现代框架已高度优化
- 直观上不如ReLU简单

典型场景

- BERT/GPT/ViT等大模型用的GELU
  - because the smooth, probabilistic gating allows better gradient flow through very deep architectures and self-attention mechanisms benefit from the softer activation boundaries

#### PReLU (Parametric ReLU)

定义与原理： PReLU（带参数的ReLU）是在Leaky ReLU基础上的升级版。

公式：$\mathrm{PReLU}(x) = \max(0, x) + a \cdot \min(0, x)$

- 其中$a$是一个可学习的参数，可以随着训练自动调整（而不是像Leaky ReLU那样人为固定为0.01）

Pros:

- Gradient at negative range（$a$），避免神经元“彻底死亡”。
- $a$可以自适应数据和网络，不再需要手动调节负区间斜率，适应性更强。
- 在某些图像识别等任务上，PReLU能略优于ReLU/LeakyReLU。

Cons:

- Add some parameters（每个通道/每个神经元都可以有自己的$a$）。
- 如果$a$学习太大，可能导致负区间输出爆炸，需正则化约束。

典型场景

- 深度CNN/图像模型（如ResNet有论文用过PReLU）。


### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Pros:

- 输出范围$(0, 1)$，适合概率建模（二分类输出层）。
- 早期神经网络和逻辑回归广泛使用

Cons:

- Gradient vanishing：输入远离0时，导数趋近于0，导致深层网络学习缓慢。
- 输出非0均值，容易引入偏移（“非中心化”）。
- Small gradient in back propagation, hard to converge

#### 梯度问题

- 最大值出现在$x=0$时，为$0.25$，两侧梯度迅速变小。
- 深层网络容易导致梯度几乎为0（梯度消失问题）。

典型场景:

- 二分类模型的输出层。
- 早期MLP/RNN中间层（但现在已较少用）

### Tanh

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

Pros:

- 输出范围$(-1, 1)$，均值为0，有助于加快收敛（中心化）。
- 梯度区间比sigmoid稍大，效果通常优于sigmoid

Cons:

- Also suffer from Gradient loss，大输入绝对值时梯度接近0。
- 仍然受限于“激活函数饱和”导致的训练慢

典型场景：

- Classic RNN hidden layers, and hidden layers in early deep learning models
- 需要输出对称分布symmetric (zero-centered)时
  - symmetric distribution适用于需要捕捉正负信息，或者上游数据中心化的场景，尤其在RNN等序列建模、残差建模等领域
- RNNs traditionally used tanh for hidden states because the bounded, zero-centered output prevents exploding activations when processing long sequences, though modern RNNs often use gating mechanisms instead

### Softmax

多分类输出常用softmax，输出归一化为概率分布
一般只用于output layer binary softmax is equivalent to Sigmoid

Computational Cost Trade-offs
ReLU family (ReLU, Leaky ReLU) are computationally cheapest - just comparisons and multiplications. GELU is most expensive due to error function/tanh calculations, but the performance gains in large models justify the cost. Sigmoid/Tanh fall in between with exponential operations. In practice, modern GPUs handle these differences well, so performance gains often outweigh computational overhead.

## Optimizers

### ADAM (Adaptive Moment Estimation) 最重要, 靠北

Adam自带L2 norm $$J_{total} = J_{data} + \frac{\lambda}{2} ||\theta||^2$$

一阶矩估计 (类似Momentum):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J(\theta_t)$$
- 梯度的指数移动平均, 提供"方向"信息
- 类似Momentum的作用
二阶矩估计 (类似RMSprop):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta J(\theta_t))^2$$
- 梯度平方的指数移动平均, 提供"尺度"信息
- 类似RMSprop的作用
偏差修正 (Bias Correction):
$$\hat{m_t} = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v_t} = \frac{v_t}{1-\beta_2^t}$$
参数更新:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} \hat{m_t}$$

$$\text{更新方向} = \frac{\hat{m_t}}{\sqrt{\hat{v_t}}}  \quad \text{（标准化的momentum）}$$

Pros:

- Adaptive learning rate
  - Each parameter has its own learning rate. 
    - 梯度大的参数 → 小步长（精细调节）
    - 梯度小的参数 → 大步长（快速学习）
  - Insensitive to HyperParameters
    - 默认参数就差不多
    - 不像SGD需要仔细调节学习率和momentum
  - 处理稀疏梯度优秀：
    - 对于很少更新的参数（如embedding），保持较高学习率
    - 非常适合NLP任务
  - Fast convergence
    - Normally 5-10 times faster than SGD
  - Memory efficiency
    - only need to store $m_t, v_t$

Cons:
- Generalization 争议
  - Generalization is not as good as SGD in some cases
  - Adam容易在训练集上过拟合
- Learning rate decay
- Memory efficieny 
  - 比SGD需要2倍额外内存（存储 $v_t, m_t$​）
  - 大模型训练时可能是问题

什么时候用Adam：

- 深度学习默认首选
- Transformer、CNN等复杂模型
- 快速原型开发
- 稀疏数据/NLP任务

什么时候考虑SGD：

- 追求最佳泛化性能
- 简单模型（logistic regression等）
- 计算资源极度受限
- 已有成熟的SGD调参经验

导致学习率过小，类似AdaGrad问题

### AdamW (Adam with Weight Decay) 更重要, 靠北

对于普通Adam, loss function 带一个自动L2正则化防止overfit（本来adam generlaization就不好）
$$J_{total} = J_{data} + \frac{\lambda}{2} ||\theta||^2$$

但这么一带 算gradient的时候就会包含L2 norm会有点参数自己调节自己的意思

权重衰减被"自适应化"了
- L2项 $\lambda \theta$被包含在梯度中
- Adam的自适应学习率会对权重衰减项也进行缩放
- 结果：权重衰减的效果变得不可预测

AdamW approach

虽然损失函数的梯度还是带L2 norm的derivative, but we only use the data part gradient!!!
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla J_{data}  \text{  (只用数据梯度)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla J_{data})^2$$
因为gradient中没有包含正则化，所以我们在此处让正则化直接作用于系数
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m_t}}{\sqrt{\hat{v_t}} + \epsilon} - \eta \lambda \theta_t$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} (\nabla J_{data} + \lambda \theta_t)$$

Pros:

- 解决weight decay问题
  -regularization直接作用于参数，不被自适应学习率扭曲(不进入gradient)
  - 所有参数统一衰减比例，真正约束大参数
- better generalization
  - Reduce overfitting
- Keep Adam Pros
  - 自适应学习率，收敛快
  - 对稀疏梯度友好
- 大模型训练表现优秀
  - Transformer等复杂架构的首选
  - 在预训练大模型上效果突出

Cons:

- Extra HyperParameter
  - 需要调节 $\lambda$, 虽然通常0.01就work，但有时需要fine-tune
- 对小模型可能过度正则化：
  - 简单任务可能不需要这么强的正则化
  - 有时Adam甚至SGD就够了

#### AdamW已经成为深度学习（特别是Transformer）的默认优化器选择，基本可以作为第一尝试


### Batch Gradient Descent (BGD) 普通一optimizer

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

- $\theta_t$​：当前时刻（第t步）的参数值
- $\theta_{t+1}$​：下一时刻（第t+1步）的参数值
- η：学习率（步长），控制每次更新的幅度
- ∇θ​J(θt​)：损失函数在当前参数处的梯度
  - 等于所有data points的梯度平均值

Pros:

- Steady convergence
- Accurate gradient direction

Cons:
- Computational inefficiency（每步都要遍历全部数据
- Require a lot of memory 内存需求高
- can't handle huge datasets
- Can't be used in online learning.

### SGD (Stochastic Gradient Descent)

Stochastic = 随机性：

- 每次只用一个随机选择的data point计算梯度
- 不是用全部数据的"真实梯度"，而是用单个样本的"噪声梯度"

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t; x^{(i)}, y^{(i)})$$

Pros:
- Fast, memory friendly 速度快，内存友好
- Noise is helpful to escape from local optimal噪声帮助跳出局部最优
  - 噪声帮助跳出局部最小值
  - 更好的泛化：某种程度的"正则化"效果
- Online learning 支持在线学习

Cons:

- Instable Convergence收敛不稳定
- Severe oscillation震荡严重，可能不收敛到最优解
- 可能在最优点附近"抖动"
  - 可以通过Learning Rate Decay解决。

### MBGD (Mini-Batch Gradient Descent) 全世界人都是讲究折衷的

$$\theta_{t+1} = \theta_t - \eta \frac{1}{m} \sum_{j=1}^{m} \nabla_\theta J(\theta_t; x^{(j)}, y^{(j)})$$

这个公式类似于把一个batch的SGD绑在一起然后求平均

Pros:
- A balence between BGD和SGD，
- GPU parallelization friendly
  - 一般的256size和GPU的核心数很接近，所以可能一下GPU就跑完了。
- Convergence relatively stable

Cons:
- Need to tune batch size，仍有一定噪声
- Batch size选择：32-512常见，太小不稳定，太大类似BGD

### Momemtum Gradient Descent

Momentum：球有惯性，会"记住"之前的运动方向，能冲过小坑

$$v_t = \beta v_{t-1} + \eta \nabla_\theta J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$
- $v_t$: momentum term
- $\beta$ momentum coefficient (mostly 0.9), how much memory remember

Pros:

- speeding convergence
  - When the gradient keeps the same, momentum accumulation
  - normally 2-10 times faster than SGD
- reduce oscillation
  - 平滑优化路径
- Jump out of local minima
  - 惯性帮助冲过small local minima
- 计算简单：
  - 只需要额外存储一个velocity vector

Cons:

- 可能Overshooting：
  - 惯性太大时可能冲过最优点
  - 在最优点附近可能来回震荡
- 额外HyperParameter：
  - 需要调节momentum coefficient β
  - 虽然0.9通常work，但有时需要fine-tune
- Converge slow in some circumstance：
  - 如果梯度方向频繁变化，momentum可能帮倒忙
  - 在noisy gradient情况下有时不如adaptive methods
- Still need learning rate tuning：
  - 不是自适应学习率，还是需要manual tuning
  - 没有解决不同参数需要不同学习率的问题

### AdaGrad (自适应学习率优化算法)

$$G_t = G_{t-1} + (\nabla_\theta J(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \nabla_\theta J(\theta_t)$$

- $G_t$：累积所有历史梯度的平方
- 梯度大的参数 → $G_t$​大 → 学习率小（分母大）
- 梯度小的参数 → $G_t$​小 → 学习率大

直觉理解
- 已经更新很多的参数（梯度累积大）→ 降低学习率，精细调节
- 很少更新的参数（梯度累积小）→ 保持较大学习率，快速学习

Pros

 -self-adjust learning rate：不需要手动decay
- 适合稀疏数据：对rarely updated parameters保持高学习率

Cons
- **Fatal Flaw**  - 学习率单调递减：
  - $G_t$​只会越来越大（累积平方）
- 学习率 $\frac{\eta}{\sqrt{G_t}}$会趋向于0
- 训练后期几乎停止学习

### RMSprop (Root Mean Square Propagation)

$$v_t = \beta v_{t-1} + (1-\beta) (\nabla_\theta J(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \nabla_\theta J(\theta_t)$$

**核心创新：解决AdaGrad的致命缺陷**
指数移动平均 vs 累积求和：

AdaGrad: $G_t = G_{t-1} + (\nabla_\theta J)^2$ 累积所有历史）

RMSprop: $v_t = \beta v_{t-1} + (1-\beta) (\nabla_\theta J)^2$（指数衰减历史）

直觉理解
- 有限记忆"的自适应学习率：
- 只记住最近的梯度信息（通过指数衰减）
- 远古的梯度会被"遗忘"
- 学习率不会单调递减到0

参数 $\beta$的作用：

- $\beta$ = 0.9（常用）：大约记住最近 1/(1-0.9) = 10 步
- $\beta$ 越大：记忆更长，更平滑
- $\beta$ 越小：更像AdaGrad，但仍然有衰减

Pros

- Solve learning rate decay：不会训练后期停止学习 (learning rate will not keep shrinking)
- Keep adaptability：仍然给不同参数不同学习率
- 计算高效：只需额外存储 $v_t$

Cons

- 新增HyperPrameters：需要调节 β
- 仍无momentum：没有利用梯度方向信息
- 可能不稳定：在某些情况下仍有震荡

Optimizor | Best for | Convergence Speed | Memory | Hyperparameter sensitivity
--| -- | -- | -- | -- 
SGD + momemtum | Simple model, best generalization | medium | Lowest | High (lr, momemtum)
Adam | Complex Models, fast prototyping | Fast | 2 * SGD | Low (default works)
AdamW | Transformers, large models | Fast | 2 * SGD | Medium (weight decay)
RMSprop | RNN, non-stationary objectives | Medium fast | 1.5 * SGD | Medium

## Other topics in Optimizers

### Learning Rate Scheduling (学习率调度策略)

直观理解 - 开车比喻, 想象你在开车找停车位：

- 刚开始: 可以开快点探索 (高学习率)
- 接近目标: 要慢下来精确停车 (降低学习率)
- 如果开过头: 需要倒车调整 (learning rate decay)

常用的scheduling策略

- Step Decay (阶梯衰减)
- Exponential Decay
- Cosine Annealing (余弦退火)
  - 先慢再快最后平滑 比较适合LLM

### Convergence Guarantees

绝大部分DNN都是non-convex 问题

现实中的深度学习 (Non-convex):

- SGD: "我只能保证爬到某个山峰，不一定是最高的"
- Adam: "我也只能爬到某个山峰，但我爬得快！"
- 关键: 在深度学习中，找到"足够好的山峰"比找到"理论最高峰"更重要

### Gradient Clipping (梯度裁剪)

Gradient Norm Clipping (范数裁剪)
- Like a 智能限速系统
```
if ||gradient|| > threshold:
    gradient = gradient * (threshold / ||gradient||)
```

## Parameter Initialization

### 不能做

- 全零初始化 (最糟糕)
  - 所有神经元完全相同，梯度也相同 → 对称性破缺失败
- 太大的初始化
  - Activation value explosion → 激活函数饱和 → 梯度消失
- 太小的初始化
  - 激活值接近0 → |z|过小 → 信息传递微弱 → 深层累积后梯度消失。

理想初始化的目标 

- 对称性破缺: 不同神经元有不同的初始化
- 激活方差稳定: 前向传播时激活值不会过大/过小
- 梯度方差稳定: 反向传播时梯度不会消失/爆炸
- 适应网络架构: 针对不同激活函数和网络深度

### Xavier/Glorot Initialization 经典初始化

保持前向和反向传播的方差稳定

- 假设：
  - 输入x的方差为Var(x)
  - 权重W独立同分布
  - 激活函数线性（或接近线性）

```
y = Wx + b
Var(y) = Var(W) × Var(x) × fan_in
```
希望: Var(y) = Var(x) (方差保持不变)
结论: **Var(W) = 1/fan_in**

对于权重矩阵 $\mathbf{W} \in \mathbb{R}^{m \times n}$：

$$\text{fan\_in} = n \quad \text{(输入连接数)}$$
$$\text{fan\_out} = m \quad \text{(输出连接数)}$$

Xavier Normal
$$W \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in} + \text{fan\_out}}\right)$$
Xavier Uniform
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}, \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)$$

fan_in + fan_out 是前向和反向传播方差稳定需求的折中：
- 太关注fan_in：反向传播可能不稳定
- 太关注fan_out：前向传播可能不稳定
- 加起来平均：两边都照顾到

#### Use Case

- 激活函数: tanh, sigmoid, linear
- 网络: 传统全连接、早期CNN

### He Kaiming Initialization (现代标配)

专门为ReLU激活函数设计, 由于ReLU杀死一半神经元：

- 1. 前向传播：需要补偿信息丢失 → 关注fan_in
- 2. 反向传播：梯度自然稀疏 → fan_out影响减小
- 3. 结论：主要考虑fan_in即可

```
Var(ReLU(Wx)) = 0.5 × Var(Wx)  # ReLU杀死一半
要保持方差稳定: Var(Wx) = 2 × Var(x)
因此: Var(W) = 2 / fan_in
```

He Normal
$$W \sim \mathcal{N}\left(0, \frac{2}{\text{fan\_in}}\right)$$
He Uniform
$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{\text{fan\_in}}}, \sqrt{\frac{6}{\text{fan\_in}}}\right)$$

### Small variance Normal - 大模型标准

为什么大模型用小方差初始化？

- 大模型的特殊问题: 参数规模巨大:
  - GPT-4: 估计1.7T parameters
  - 小误差 × 巨大参数量 = 大问题
- 现代大模型标配LayerNorm
  - 无论输入方差多大，输出都被"重置
- LayerNorm is not that perfect
  - The very first layer is not protected by LayerNorm.
- 小方差(0.02)的经验来源
  - BERT论文的经验值

### Variance and propagation

We want to keep the variance between layers similar

方差 = 信号强度 因为：

- 方差衡量数值的"离散程度"
- 在神经网络中，离散程度 = 信息的丰富程度
- 信息丰富 = 表达能力强 = "信号强"

**保持方差稳定 = 保持每层的"信息处理能力"稳定！**

对称性破缺：
- 不是指数值关于0对称
- 而是指神经元之间的功能对称性
- 目标：让每个神经元有独特的"个性"
- 方法：给每个权重不同的随机初始值

## QA

Q1. 为什么不能全零初始化?

- Symmetry breaking failure
  - The weight, activation and gradient is all the same
- 网络无法学到不同的特征

Q2: "Xavier和He初始化的区别

- Xavier适合tanh/sigmoid等饱和激活函数
- He专门为ReLU设计
  - ReLu will kill half of the neurons, so needs bigger initial variance to compensate
Q3. 现代大模型用什么初始化？
- Transformer通常用较小的normal初始化（std=0.02）加上layer normalization
- 这比传统的Xavier/He更保守，因为layer norm已经在稳定激活分布，不需要初始化承担太多责任
Q4: 如何选择初始化方法
- 主要看激活函数：
  - ReLU系列用He，
  - tanh/sigmoid用Xavier，
  - GELU用小方差normal。
- 但现代深度网络通常有batch norm或layer norm，初始化的重要性相对降低了。

Q5. 为什么Adam在理论上不如SGD，但实践中大家都用Adam？

- Majority of deep learning is non-convext optimization
- Theoretically optimal does not mean much
- Adam虽然没有convex收敛保证，但它在non-convex landscape上的经验表现更好 - 收敛更快，对超参数不敏感，特别适合复杂模型like Transformers。
- 在实践中，我们更关心能否找到好的局部最优，而不是理论上的全局最优

Q6. Saddle point (鞍点)

- 在高维空间中，随机点是鞍点的概率 >> 是极值点的概率
- 鞍点处梯度为零但不是最优解，会导致训练停滞。现代优化器如Adam和momentum通过惯性和自适应学习率来帮助逃脱鞍点

## Other notes

### 为什么均值为0（zero-mean）有助于加快收敛？

核心原理：

- 当激活函数输出的均值接近0时，后续层的输入也更可能分布在激活函数的“线性区域”，这样反向传播时梯度不会特别小或特别大，训练会更稳定。

- 均值为0可以减少梯度更新时的“偏移累积”，每一层的输入分布更平衡，有利于参数收敛，不会“拖一边”。

- 反向传播时，如果激活值集中在某一侧（比如Sigmoid输出全在0~1），链式求导后容易一层层越乘越偏向0，容易造成梯度消失。

- 中心化的激活（比如tanh输出在-1到1，均值0）能让梯度分布更均匀，有利于更快、更稳定地收敛。

### Why most Deep Learning is Non-convex optimization problem.

只要有一个隐藏层 + 非线性激活函数，损失函数就变成non-convex了
English: Even one hidden layer + nonlinear activation → non-convex loss landscape

```
# Forward pass
h = σ(W1 @ x + b1)      # 隐藏层
y = W2 @ h + b2         # 输出层
Loss = (y - target)²    # MSE loss

Loss = (W2 @ σ(W1 @ x + b1) + b2 - target)²
我们假设最简单的情况x=1,W1=w,ReLU激活, W2=1, b1=b2=0
Loss(w) = (ReLU(w × 1) - target)²
       = (max(0, w) - target)²
```
非线性激活函数让参数"纠缠"在一起，破坏了损失函数的凸性


















# 



归一化（BatchNorm/LayerNorm/GroupNorm/InstanceNorm）：每种都能简述原理、核心公式、各自优缺点、应用场景，最好写一张表总结。

激活函数+优化器+初始化：整理一页A4纸，什么场景用啥，优缺点两句带过，面试不会让你细抠推导，主打“场景举例+优劣势”。

CNN全流程：能顺口讲出来：输入—卷积—激活—池化—flatten—全连接—输出，知道padding/stride的作用，经典结构VGG/ResNet能各说一句。

RNN/LSTM/GRU结构差异、梯度传播：按“普通RNN->LSTM/GRU怎么解决问题->缺点->和Transformer对比”套路。

Transformer完整口述，重点记忆：QKV+多头+位置编码+Encoder/Decoder区别

归一化机制（BatchNorm/LayerNorm/GroupNorm/InstanceNorm原理与区别，应用场景，优缺点）

 **激活函数：**ReLU, LeakyReLU, PReLU, Sigmoid, Tanh, Softmax，各自适用场景和优缺点

 权重初始化（Xavier/He）原理

 **损失函数设计：**回归与分类、平衡损失

 优化器比较（SGD/Adam/RMSProp等原理/优劣）

 **CNN常考细节：** 卷积、池化、stride、padding、flatten、全连接（你已有不少，但可以整理成独立章节，方便快速背诵）

 RNN/LSTM/GRU：原理/门控结构/优缺点/适用场景/梯度消失爆炸对策/和Transformer对比

 Attention/Transformer：（你的Self-attention、QKV、mask、多头、PE基本都全，但建议整理成“面试八股”+公式+对比总结段，和RNN差异，面试经常问“你能从头讲讲Transformer吗”，要练顺！）

 Encoder/Decoder结构，BERT/GPT/Seq2Seq：

Encoder-only/Decoder-only/Encoder-Decoder异同与应用

BERT的Pre-training/Finetuning流程，Masked LM/NSP/Prompt-tuning

GPT自回归生成、Causal Mask

Seq2Seq（机器翻译/摘要），输入输出格式，典型模型结构

“为什么BERT不能直接生成文本？”、“BART/T5为什么能搞text-to-text?”

 参数量/内存/推理/并行工程化常见面试点，如“如何做大模型推理加速/分布式训练/量化蒸馏/剪枝/混合精度”