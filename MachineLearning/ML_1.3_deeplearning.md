# Deep Learning

## Deep Learning Topics

### **Universal Approximation Theorem(万能近似定理)**

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

## Neural Network (MLP)

Neural networks consist of layers of interconnected nodes (neurons) that process information. The three main types of layers are input, hidden, and output layers.

- Input Layer: Receives the initial data features.
- Hidden Layers: Performs transformations on the inputs.
- Output Layer: Produces the final prediction.

### Key Ideas

**Forward Propagation** (前向传播): During forward propagation, data flows from the input layer through the hidden layers to the output layer. Each neuron computes:

- Weighted sum of inputs: z = Σ(weight × input) + bias
- Apply activation function: a = f(z)

**Activation Functions** (激活函数): Activation functions introduce non-linearity into the network, allowing it to learn complex patterns.

如果没有activation function或者用linear activation function会导致不管多少层，都相当于一个单层线性模型 $f(x) = W_n(W_{n-1}(...(W_1x + b_1)...) + b_{n-1}) + b_n = W'x + b'$

- **ReLU Rectified Linear Unit**（修正线性单元）：f(x) = max(0, x)
  - Most common in hidden Layer
  - 有助于解决vanishing gradient problem (梯度消失问题)
  - Sparse activation （许多神经元输出0）
  - 可能出现0梯度
    - Leaky ReLU： $f(x) = \begin{cases}x, & \text{if } x > 0 \\alphax, & \text{if } x \leq 0\end{cases}$
      - 解决了ReLU的"死神经元"问题
      - 对所有负值输入提供一个小的、非零梯度
    - Parametric ReLU (PReLU)： $f(x) = \begin{cases}x, & \text{if } x > 0 \\alpha_i x, & \text{if } x \leq 0\end{cases}$
      - 负值部分的斜率 $\alpha_i$是可学习的参数，而不是固定值
      - 网络可以自动学习最优的负值斜率
      - 通常比Leaky ReLU表现更好
      - 增加了模型的参数量和复杂度
- Sigmoid：$f(x) = \frac{1}{1+e^{-x}}​$
  - Output between 0 and 1
  - For binary classification
  - 对极端值存在梯度消失问题
- Tanh Function: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
  - Outputs between -1 and 1
  - Zero-centered, helps with convergence
  - Still has vanishing gradient issues
- Softmax: $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
  - Used for multi-class classification output
  - Converts values into probabilities that sum to 1

**Loss Functions**

Classification Loss:

- Binary Cross-Entropy: $L = -[y \cdot \log(p) + (1-y) \cdot \log(1-p)]$

- Categorical Cross-Entropy: $L = -\sum_{i} y_i \cdot \log(p_i)$

Regression Loss:

- Mean Squared Error (MSE): $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

- Mean Absolute Error (MAE): $L = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$

- Huber Loss: Combines MSE and MAE, less sensitive to outliers

**Backpropagation and Gradient Descent** (反向传播和梯度下降)

Backpropagation is the algorithm used to calculate gradients, which show how much each weight contributes to the error. 反向传播是用于计算梯度的算法，显示每个权重对错误的贡献程度

Steps:

- Calculate error using loss function
- Compute gradient of error with respect to each weight
- Update weights in the direction that reduces error
- $w_{new} = w_{old} - \alpha \cdot \frac{\partial L}{\partial w}$

**Chain Rule** = $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$

### **Optimization Algorithms** 优化算法：

#### **Stochastic Gradient Descent (SGD)**

与标准梯度下降(GD)不同，SGD每次使用单个样本或小批量(mini-batch)样本来计算梯度，而不是整个数据集
$$\theta = \theta - \alpha \nabla_\theta J(\theta; x^{(i:i+n)}, y^{(i:i+n)})$$

- $\theta$ is model parameter
- $\alpha$ is learning rate
- $\nabla_\theta J$ is the gradient of Loss function wrt parameters

Pros:

- Computationally efficient, only use a small batch of data in each iteration, useful in large-scale data
- Enables online learning (possible to update the model as data arrives) 

Cons：

- Convergence path oscillates 震荡, making training unstable
- May get stuck in saddle points or local minimal 可能陷入鞍点或局部最小值
- Uses the same learning rate for all parameters所有参数使用相同的学习率

想象你在山上寻找最低点, **标准梯度下降**：每一步，你都环顾四周全景，确定最陡峭的下降方向, **SGD**：每一步，你只看几个方向，根据有限视野决定下一步。

#### **SGD with Momentum**

SGD with Momentum is an improvement over standard SGD that accelerates training by accumulating "momentum" from past gradients, especially in regions where gradient direction is consistent. 

它通过累积过去梯度的"动量"来加速训练过程，特别是在梯度方向一致的区域

$$v_t = \gamma v_{t-1} + \alpha \nabla_\theta J(\theta)$$

$$\theta = \theta - v_t$$

- $v_t$ is the velocity vector at the current time step 速度向量
- $\gamma$ momentum coefficient, typically 0.9
- $\alpha$ is the learning rate

Pros:

- Accelerates training, especially in regions with consistent gradient direction
- Reduces oscillations, making convergence smoother (减少震荡，使收敛更平滑)
- Helps overcome local minima (有助于越过局部最小值)
- Better handles ravine-like surfaces and saddle points (更好地处理峡谷状曲面和鞍点)

Cons:
- Adds another hyperparameter (momentum coefficient) that needs tuning (增加了一个需要调整的超参数，即动量系数)
- May overshoot the optimal point in some situations (在某些情况下可能会过冲最优点)

#### SGD with momemtum is used in most cases especially in deeper networks and training needs to be accelerated.

### Regularization Technique in NN

#### **1. Dropout** 随机失活

Randomly "turns off" a percentage of neurons during training (训练过程中随机"关闭"一定比例的神经元), Typical dropout rates: 0.2-0.5 (20%-50% of neurons are randomly deactivated)

Why Important

- Prevents co-adaptation (共适应) between neurons.
- Acts like an ensemble of many different networks
- Simple to implement with significant effects

#### **2. L2 Regularization (Weight Decay)**

More frequently used than L1 in NN

#### 3. 批量归一化 (Batch Normalization)

Normalizes activations within each mini-batch, Rescales the inputs to each layer to have a specific mean and variance. 标准化每个小批量内的激活值, 为每层的输入重新调整均值和方差.

#### 4. 层归一化 (Layer Normalization)

在特征维度而非批量维度上归一化, 每个样本独立归一化

Why important?

- Batch-size independent, suitable for RNN and NLP tasks
- 在序列长度变化的情况下更稳定
- Transformer架构的关键组件

## Convolutional Neural Networks (CNNs) 卷积神经网络

CNNs are applicable to various types of **grid-structured data.** 网格结构的数据

### Convolutional Layer （卷积层）

直观理解：把卷积层想象成一组特殊的"放大镜"。每个放大镜专门寻找图像中的特定模式

#### **Convolutional Kernel / Layer （卷积核）**

A convolutional kernel is precisely a weighted matrix (小型权重矩阵) designed to support specific detection purposes in neural networks. Each kernel is a small matrix of weights (parameters) that the network learns during training. These weights determine what pattern or feature the kernel will respond to. 权重决定了卷积核将响应什么模式或特征


卷积层是CNN的核心组件，主要负责提取（图像的）特征。它通过应用一组可学习的过滤器（卷积核）来检测输入中的局部模式，如边缘、纹理或特定形状

- 每个kernel算出来的feature value = **sum of (kernel value * matrix value)**
- Each kernel is used to caupture ONE feature
- Each kernel generates its own **feature map**
- for example, we have three kernel to detech horizontal / vertical / diagonal lines respectively. 

#### **Stride (步长)**

 Controls how we sample or move through structured data. It determines the step size between consecutive applications of an operation. (一次走几个grid)

- **Controlling information density**: Higher stride means more sparse sampling
- Managing computational efficiency: Higher stride reduces computation

Benefits:

- Reduces computational cost (fewer convolution operations)
- Acts as a form of downsampling (reduces spatial dimensions)
- Increases the effective receptive field of subsequent layers
- Can help prevent overfitting by reducing parameters

Trade-offs:

- Loss of spatial information and fine details
- May miss important features if stride is too large
- Can cause "checkerboard artifacts" in some architectures

#### **Padding (填充)**

Padding is a strategy for handling boundaries in structured data. It determines how operations behave at the edges of your data (是否在边界添加额外的dp)

Purpose

- Keep spatial dimensions:没有padding kernel会减小输出尺寸
- Keep edge information: data (pixel) on the edge will be in less kernel than those more in center
- control the dimension of output

Type of padding

- Valid Padding (No Padding) 不填充
- Same Padding (输出的shape与输入一样):
  - Adds enough padding so that output size equals input size
  - For a k×k kernel, add (k-1)/2 padding on all sides (when stride=1)
- Full Padding
  - 每个输入(像素)参与相同数量的kernel
  - Adds enough padding for each input pixel to participate in the same number of convolutions
  - For a k×k kernel, add (k-1) padding on all sides

Benefits:

- Preserves spatial dimensions (with "same" padding)
- Treats edge pixels more equally in the convolution process (在卷积过程中更平等地处理边缘像素)
- Allows for deeper networks without rapid spatial dimension reduction (允许更深的网络而不会快速减小空间维度)
- Provides control over output feature map size

Trade-offs:

- Adds computational overhead (more pixels to process)
- Zero padding can create artificial edges at the boundaries (人工边缘)

$$\text{Output Size} = \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

#### How kernel works:

- 过滤器的完整维度：
  - 在具有多个输入通道的卷积层中，每个过滤器不是2D的，而是3D的
  - 对于输入形状[14,14,16]，对于 3*3 kernel，每个过滤器的完整形状是[3,3,16]
  - 这意味着每个过滤器包含3×3×16=144个权重参数（加上1个偏置）
- 过滤器如何应用：
  - 每个3D过滤器会考虑所有输入通道
  - 过滤器在所有16个输入通道上同时执行卷积
  - 对所有通道的卷积结果求和，生成单个输出特征图
  - 再加上一个偏置值
- 输出通道数量：
  - 每个完整的3D过滤器生成一个输出特征图（而不是16个）
  - 使用32个这样的过滤器，总共生成32个输出特征图
  - 因此最终输出形状为[14,14,32]
所以下一层的输入通道数等于上一层的kernel数量


### **Pooling Layer (池化层)**

直观理解：把池化层想象成做笔记时的"重点标记"。你不需要记下看到的每一个细节，只需要最重要的部分。

pooling layers primarily serve a dimensionality reduction function in convolutional neural networks

- Reducing spatial dimensions（高度和宽度）
- Creating translation invariance
- Reducing computation and parameters in subsequent layers
- Summarizing features in local regions (总结局部区域的特征)

When your feature map dimensions aren't perfectly divisible by the pooling size, the extra data points (right-most and bottom-most) will be discarded **by default**

**Stride** in pooling window is normally and default set as the length of pooling window.

- non-overlapping windows
- clean dimensionality reduction without redundancy
- Most efficient implementation.

#### **Why Do We Need Pooling Layers?**

- 降低计算复杂度
  - Decreases the spatial dimensions (height and width) of feature maps
  - Significantly reduces parameter count and computation in subsequent layers
  - Makes training larger, deeper networks feasible
- Providing Translation Invariance(平移不变性):
  - Insensitive to small variations in input (like slight object movement)
  - Helps the network focus on "whether a feature exists" rather than "where exactly it is"
- Preventing Overfitting:
  - Controls model complexity by reducing parameter count
  - Provides a simple form of regularization
  - Forces subsequent layers to work with more abstract features
- Increasing Receptive Field:
  - Allows deeper convolutions to see a larger portion of the original input
  - Helps capture larger-scale patterns and features

### **Flatten Layer (扁平化层)**

Why we need Flatten Layer?
- Dimension Transformation:
  - Converts the 3D tensor output from convolutional layers (height × width × channels) into a 1D vector
  - 这是连接卷积结构和全连接层的必要桥梁
- Structural Adaptation:
  - Convolutional layers process structured grid data (preserving spatial information)
  - Fully connected layers require 1D inputs (no spatial preservation)
  - 扁平化实现了这两种不同结构之间的兼容
- Global Feature Integration (全局特征集成)：
  - 扁平化后，特征图中的每个元素都成为单独的特征
  - Allows fully connected layers to access features from all spatial positions
- Establishing Spatial Invariance (建立空间无关性)：
  - Convolutional and pooling layers provide local invariance (within small regions)
  - 扁平化和全连接层组合提供全局视角

### **Fully connected Layer (全连接层)** 

如果卷积层是发现各种"局部特征"的专家，那么全连接层就是把这些特征合理组合的专家

#### **Steps in Fully connected Layer**

- After convolutional and pooling layers, we have a m*m*k feature map (m*m spatial dimensions, k channels)
  - k个chanel在此处可以理解为k个视角
- After flattening, we get a one-dimensional vector with $m*m*k$ elements
- If there are n neurons in current fully connected layer, then the weight matrix is [$m*m*k$, $n$] shaped to convert our one-dimensional vector into 1 * k shaped vector (each one value) for a neuron by 
$$z_i = x * w_i + b_i$$
- Apply activation function (probly ReLU)
- **Repeat the last steps for number of FCL times** until to the output layer in fully connected layer.
- In the output layer, use activation function (sigmoid or softmaxor liear or ReLU) to convert the forecast to probability or value. 

### Back-propagation process in CNN (反向传播)

一旦所有梯度都计算好，**所有层的参数同时更新**

- 1. Use loss function to calculate the loss.
- 2. Calculate the Gradient in the output layer (A vector)
  - 大部分情况Gradient由mini batch来计算(SGD), 而不是全体梯度。

#### **Steps**

- Forward Propagation
  - Input data passes through the network from input to output layer
  - Calculate the output of each layer (including intermediate activations)
  - Store intermediate results at each layer (for use in backpropagation) 保存每层的中间结果（用于反向传播）
- Calculate Output Layer Error
  - Compare model predictions with true labels
  - Compute error using a loss function (e.g., cross-entropy or mean squared error)
  - Calculate gradient of loss with respect to output layer $\frac{\partial L}{\partial z^{(output)}}$
- Backpropagate the Error
  - Propagate error signal backwards from output layer toward input layer
For each layer (从输出层向输入层，逐层传递误差信号):
    - Receive gradient from the subsequent layer
    - Calculate pre-activation gradient through activation function derivative
    - Calculate gradients for the layer's parameters (weights, biases)
    - Calculate gradient to pass to the previous layer
- Parameter Update
  - Use optimization algorithm (SGD, Adam, etc.) and learning rate to adjust parameters
- Repeat Process
  - Repeat above steps for each batch
  - Complete specified number of training epochs or until convergence

### Optimization Algorithms

Adam (Adaptive Moment Estimation):

直观地说，ADAM就像一个聪明的徒步者：

- 记住行进的惯性（动量）
- 适应不同地形调整步伐（自适应学习率）
- 谨慎开始，逐渐变得自信（偏差校正）

Combines momentum and RMSProp
$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L(w_t)$

$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L(w_t))^2$

With bias correction: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$, $\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\hat{m}_t$

Adaptive learning rates for different parameters

传统梯度下降就像总是朝着最陡峭的方向迈出相同大小的步子

- 容易在陡峭区域走得太快，在平缓区域走得太慢
- 可能在狭窄的峡谷中来回弹跳

ADAM则像一个智能徒步者：

- 记得之前走过的路（动量）
- 为不同的地形调整步长（自适应学习率）
- 在平坦区域能走得更快，在陡峭区域会小心谨慎

#### Key features in ADAM

**1. 记忆前进方向 - 动量(Momentum)**
就像一个滚动的球会保持其运动方向一样，ADAM"记住"参数过去的更新方向：

- 有助于穿越平坦区域（梯度接近零的地方）
- 抑制震荡，使训练更平稳
- 帮助逃离局部最小值

**2. 为每个参数定制步长 - 自适应学习率**
不同的参数可能需要不同大小的更新：

- 频繁更新的参数获得较小的步长
- 不常更新的参数获得较大的步长
- 减少学习率设置的困难

**3. 偏差校正 - 冷启动问题的解决方案**
ADAM解决了训练初期统计量不准确的问题

- 确保训练初期不会有异常的大步更新
- 使得算法在任何时间点都表现良好

## Recurrent Neural Networks (RNNs) 循环神经网络

RNNs process sequential data where order matters, maintaining "memory" through hidden states.

RNN: $f(x_t, h_{t-1}) → y_t, h_t$ (current input + previous hidden state)

RNNs were specifically designed for sequential data.

Most RNN only has one hidden layer, but within the layer we have multiple (64, 128, 256, 512 hidden states) 

The reason why the number of hidden states is a power of 2: hardware optimization.

### Pros and Cons

#### Pros

- Parameter Efficiency (same weight)
- Memory Capability (built in memory thry hidden states, by W_hh)
- Abstract Pattern Recognition
  - Excellent for high-level tasks: sentiment analysis, topic classification.

#### Cons

- Short-Term Memory Bias
  - Recency bias: Recent inputs overwrite older information
  - Information decay: Exponential forgetting of distant past
  - Context window: Effective context much shorter than sequence length
  - Poor long-range dependencies: Struggles with syntax, coreference
- Information Bottleneck (BIGGEST LIMITATION)
- Gradient Vanishing/Exploding
  - Training instability: Requires careful initialization and gradient clipping
- Sequential Computation Bottleneck
  - Even other NN models also has to do layer by layer, but they can compute parallely within layer, but RNN needs to compute hidden state sequentially.
- Same Parameters Everywhere Problem

### Components

#### 1. Hidden State (隐藏状态) - The Memory Component

A vector that encodes information from all previous time steps
$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
The notes (hidden state) summarize everything you've seen / being trained so far.

- Dimensionality: Usually 64, 128, 256, or 512 dimensions (means number of neurons)
- Initialization: Typically initialized to zeros: $h_0​=0$
- Information Bottleneck: Fixed size forces compression of all past information

#### 2. Weight Matrices - The Learning Parameters
Input-to-Hidden Weights ($W_{xh}$)
$$W_{xh} \in \mathbb{R}^{d_h \times d_x}$$

Maps input features to hidden dimensions

- Intuitive: "How should current input influence my thinking?"
- Example: In sentiment analysis, learns that "excellent" should strongly activate positive sentiment neurons

#### 3. Hidden-to-Hidden Weights ($W_{hh}$) Most Critical Component

Controls how past information flows to present

- Intuitive: "How should my previous thoughts influence current thinking?"
- Critical Property: Same weights used across all time steps - this enables parameter sharing and generalization

#### 4. Hidden-to-Output Weights ($W_{hy}$)

$$W_{hy} \in \mathbb{R}^{d_y \times d_h}$$

Projects hidden state to output space

- Intuitive: Given my current understanding, what should I predict?

#### 5. Activation Functions - The Non-linearity

**Typical** Hidden Activation: $tanh$ (hyperbolic tangent)
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
$$\frac{d\tanh(z)}{dz} = 1 - \tanh^2(z)$$

Why $tanh$ over ReLU in vanilla RNNs?
- Range: (−1,1) bounded output prevents exploding activations
- Zero-centered: Helps with gradient flow in recurrent connections
- Smooth: Differentiable everywhere (important for BPTT)

### RNN Architecture Types

- One-to-Many: Image captioning (image → sequence of words)
- Many-to-One: Sentiment analysis (sequence of words → sentiment)
- Many-to-Many (synced): Video classification (frame-by-frame)
- Many-to-Many (async): Machine translation (English → Chinese)

### RNN training steps

RNN forward equation:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

- $W_{hh} h_{t-1}$ is inherit from previous layers
- W_{xh} x_t new input


```
x₁ → h₁ → h₂ → h₃ → h₄ → ŷ
     ↑    ↑    ↑    ↑    ↑
   W_hh W_hh W_hh W_hh W_hy
```

### Backpropagation Through Time (BPTT) in RNN

The gradient flowing back to the first hidden state is proportional to the product of the recurrent weight matrix (transposed) multiplied across all time steps from 2 to T
#### Why 
$$\frac{\partial L}{\partial h_1} \propto \prod_{t=2}^{T} W_{hh}^T$$

### RNN other notes

#### $W_{hh}$ and $W_{xh}$ is shared across all hidden states

pros:

- Massive parameter reduction: $O(T \times d_h^2)$ to $O(d_h^2)$
- Generalization: Can handle any sequence length
- Translation invariance: Same pattern processed consistently

Costs:

- Limited expressiveness: Same transformation at every step
- Performance bottleneck: May not be optimal for all positions
- Complex optimization: Single weight matrix must work everywhere.

#### Dimensionality Compression in RNNs

$$W_{hh} h_{t-1} = \begin{bmatrix} w_{11} & w_{12} & \cdots & w_{1d} \\ w_{21} & w_{22} & \cdots & w_{2d} \\ \vdots & \vdots & \ddots & \vdots \\ w_{d1} & w_{d2} & \cdots & w_{dd} \end{bmatrix} \begin{bmatrix} h_{t-1}^{(1)} \\ h_{t-1}^{(2)} \\ \vdots \\ h_{t-1}^{(d)} \end{bmatrix}
$$

Whh​ acts as a "selector matrix"


**Each row** decides how to combine previous information:
- **Large weights**: "This past information is important, keep it"
- **Small weights**: "This past information is less relevant, forget it"
- **Negative weights**: "This past information should reduce current activation"

#### **2. Nonlinear Compression via Tanh**

$$\tanh: \mathbb{R}^d \rightarrow (-1,1)^d$$


$\tanh: \mathbb{R}^d \rightarrow (-1,1)^d$


### LSTM (Long Short-Term Memory)

LSTM was designed to solve the vanishing gradient problem in vanilla RNNs by creating gradient highways.

#### Pros:

- Solves vanishing gradient problem (vs Vanilla RNN)
- Handles long-term dependencies (vs Vanilla RNN)
- Selective memory management
- Real-time/streaming capability (vs Transformers)
- Lower computational requirements (vs Transformers)

#### Cons

- Complex implementation with many hyperparameters (vs Vanilla RNN)
- Sequential processing bottleneck - no parallelization (vs Transformers)
- Still has vanishing gradients for very long sequences (vs Transformers)
- Information bottleneck through fixed-size states (vs Transformers)
- Limited context window compared to Transformers
- Slower training due to sequential nature (vs Transformers)

#### Core Innovation: Dual Memory System

- Vanilla RNN: One type of memory: hidden state $h_t$ (short-term memory)
- LSTM: Two types of memory: Hidden state $h_t$ + Cell state $C_t$ (long-term memory)

``` LSTM
Input: x₁ → x₂ → x₃ → x₄

Cell:    C₁ → C₂ → C₃ → C₄  (long-term memory highway)
         ↓    ↓    ↓    ↓
Hidden:  h₁ → h₂ → h₃ → h₄  (short-term working memory)
```

#### LSTM Architecture: The Three Gates

Key insight: Use gates to control information flow instead of simple matrix multiplication.

1. Forget Gate: "What to throw away"

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

- Range: (0,1) via sigmoid
- Purpose: Decides which information to discard from cell state
- Intuition: "Should I forget the previous subject when I see a new sentence?"

2. Input Gate: What new information to store?

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

- $i_t$​: How much of new information to store
- $\tilde{C}_t$​: New candidate information
- Intuitive: "This word is important, remember it" vs "This word is filler, ignore it"

3. Output Gate: "What to output"

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

- Purpose: Controls what parts of cell state to output
- Intuitive: "Based on my memory, what should I say now?"

#### LSTM State Updates
Cell State Update: The Memory Highway

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Why this solves vanishing gradients**:

- Direct path:   $C_{t-1} \rightarrow C_t$ when $f_t \approx 1$
- No repeated matrix multiplication: Gradient flows directly
- Selective forgetting: Only forget when $f_t \approx 0$

**Hidden State Update**

$$h_t = o_t \odot \tanh(C_t)$$

$$\frac{\partial C_T}{\partial C_1} = \prod_{t=2}^{T} f_t$$

#### Training steps

1. All Gates Compute Their Decisions as below

$$\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}\\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)}\\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate)}\\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)}
\end{align}$$
- where $[h_{t-1}, x_t]$ is the concat of two vectors (into a longer vector)
- $\sigma$ is a sigmoid function

2. Gates Apply Their Decisions (Sequential)

$$\begin{align}
C_t^{forget} &= f_t \odot C_{t-1}\\
C_t^{new} &= i_t \odot \tilde{C}_t\\
C_t &= C_t^{forget} + C_t^{new} = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t\\
h_t &= o_t \odot \tanh(C_t)
\end{align}$$

3. The below two can be parallel as no dependency
$$\begin{align}
C_t^{forget} &= f_t \odot C_{t-1} \\
C_t^{new} &= i_t \odot \tilde{C}_t
\end{align}$$

#### Back propagation 

1. Cell State Gradient highway

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$
the dirivative of $C_t$ wrt $C_{t-1}$ is below
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

And after back-propagation to the first hidden state

$$\frac{\partial C_T}{\partial C_1} = \prod_{t=2}^{T} f_t$$

Compared with vanilla RNN

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W_{hh}^T \cdot \text{diag}(\tanh'(z_t))$$

$$\frac{\partial L}{\partial C_{t-1}} = \frac{\partial L}{\partial C_t} \cdot f_t + \frac{\partial L}{\partial h_t} \cdot o_t \cdot (1 - \tanh^2(C_t)) \cdot f_t$$

The gradient of hidden state and cell state (highway) is additive, then the complete gradient formula is (two hidden states):

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_2} \frac{\partial h_2}{\partial h_1} + \frac{\partial L}{\partial C_2} \frac{\partial C_2}{\partial h_1}$$

or generalized as 

$$\frac{\partial L}{\partial h_1^{(1)}} = \underbrace{\sum_{t=2}^T \frac{\partial L}{\partial h_t^{(2)}} \frac{\partial h_t^{(2)}}{\partial h_t^{(1)}} \prod_{k=2}^{t} \frac{\partial h_k^{(1)}}{\partial h_{k-1}^{(1)}}}_{\text{Through hidden states}} + \underbrace{\sum_{t=2}^T \frac{\partial L}{\partial C_t^{(2)}} \frac{\partial C_t^{(2)}}{\partial h_t^{(1)}} \prod_{k=2}^{t} f_k^{(1)}}_{\text{Through cell states}}$$

```
Loss L
    ↓
   h_T ←──────── C_T
    ↓ ↘          ↓
   h_2 ←──────── C_2
    ↓ ↘ \        ↓ \
    ↓  \ \       ↓  \
    ↓   \ \      ↓   \
    ↓    \ ↘     ↓    ↘
   h_1 ←──────── C_1
       ←─────────────
```

### GRU (Gated Recurrent Unit)

GRU Advantages:
- fewer parameters (typically 25% fewer)
- Faster training and inference
- Less prone to overfitting
- Simpler implementation

LSTM Advantages:

- More expressive (separate cell state)
- Better for very long sequences
- More control over information flow
- Established track record

#### Key difference from LSTM

Merged Cell and Hidden State
- LSTM: Separate $C_t$ and $h_t$
- GRU: Only $h_t$ (no separate cell state)

Combined Forget and Input Gates
- LSTM: Separate $f_t$ and $i_t$ gates
- GRU: Single $z_t$ gate where $(1-z_t)$ replaces forget gate


3. No Output Gate

LSTM: ht=ot⊙tanh⁡(Ct)h_t = o_t \odot \tanh(C_t)
ht​=ot​⊙tanh(Ct​)
GRU: Hidden state directly used as output

two gates instead of three
- Reset Gate
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
- Update Gate
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

GRU equation update:

candidate hidden state:
$$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$
FInal hidden state:
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

#### GRU Training Process
Forward Pass
- Compute both gates: $r_t$​, $z_t$
- Compute candidate: $\tilde{h}_t$ (using reset gate)
- Combine old/new: $h_t$ (using update gate)

Backward Pass
**Gradient through update gate:**
$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

为什么业界有时更爱GRU?
- 实际工业界里，如果序列不是超长、模型对精度不是极致敏感，GRU因为简单、推理快，参数少，往往优于LSTM。如果你能补一句“比如语音识别

在语音识别/小型时序分类用GRU更合适，在复杂NLP/长文本时用LSTM更合适.

GRU有时候过拟合风险小，但调参空间更窄；LSTM收敛慢但能学更复杂的东西；我实际跑过xx比赛，用GRU能更快出baseline


