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

### Batch Normalization (BatchNorm) /  批归一化

Normalizes across the batch dimension, computing mean and variance over all samples in a mini-batch for each feature。

对一个batch内每个通道分别归一化（均值/方差），让数据分布固定，加速收敛

#### Understanding

- Batch size是“每次训练看到多少条样本”的意思
- 每次梯度更新用多少行数据，通常固定为32、64、128、256等，和特征数无关

Pros

- Fast convergence / 收敛快速
- Reduces initialization sensitivity / 降低初始化敏感性
- Allows higher learning rates / 允许更高学习率

Cons
- Poor performance with small batches / 小批次性能差
- Creates batch dependencies / 产生批次依赖

#### Applications / 应用: CNNs, computer vision / 卷积神经网络、计算机视觉

## Activation Function

### ReLU (Rectified Linear Unit)

$$\mathrm{ReLU}(x) = \max(0, x)$$

Pros:

- Easy to calculate, converge fast, gradient is always zero

- Sparse activation（负区间直接为0），有一定正则化效果。

- Practically, perform good in Deep NN

Cons：

- Dead neuron problem: 如果权重更新后神经元长期落在负区间，梯度为0无法激活

- 输出非0均值，可能导致网络某些层偏移。(Non-zero mean output may cause certain layers in the network to become biased or shifted)
- 梯度问题：
  - $x>0$时梯度为1，$x<0$时为0。
  - 负区间无法更新权重（dead neuron）。

典型场景：

- 现代CNN、MLP等深度网络的默认激活函数。
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








## Other notes

### 为什么均值为0（zero-mean）有助于加快收敛？

核心原理：

- 当激活函数输出的均值接近0时，后续层的输入也更可能分布在激活函数的“线性区域”，这样反向传播时梯度不会特别小或特别大，训练会更稳定。

- 均值为0可以减少梯度更新时的“偏移累积”，每一层的输入分布更平衡，有利于参数收敛，不会“拖一边”。

- 反向传播时，如果激活值集中在某一侧（比如Sigmoid输出全在0~1），链式求导后容易一层层越乘越偏向0，容易造成梯度消失。

- 中心化的激活（比如tanh输出在-1到1，均值0）能让梯度分布更均匀，有利于更快、更稳定地收敛。



















# 


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
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