# Transformers

## Introduction to Transformers

Transformers revolutionized deep learning by introducing the "Attention is All You Need" paradigm, completely eliminating recurrence and convolutions in favor of pure **attention mechanisms**.

**本质的intuitive理解**：每个token给其它token打分，然后加权融合

Transformers解决了RNN和CNN的核心限制：

- Parallelization: Unlike RNNs, all positions can be processed simultaneously 同时处理
- Long-range dependencies: Direct connections between any two positions 任意两个位置之间的直接连接
- Computational efficiency: Better hardware utilization (GPU-friendly) 
- Scalability: Can handle very long sequences effectively

## Self-attention mechanism

让序列中的每个位置都能直接关注到序列中的任何其他位置


### 1. Input representation

given input sequence: $X = [x_1, x_2, ..., x_n] \in \mathbb{R}^{n \times d}$
where
- $n$ = 序列长度 (sequence length)
- $d$ = 词向量维度 (embedding dimension) (每个词用多少维向量表示)
- $x_i \in \mathbb{R}^d$ -> 第i个词的向量表示

like
```
句子: "The cat sat"
n = 3 (三个词)
d = 4 (假设词向量是4维)

X = [
  [0.1, 0.2, 0.3, 0.4],  # "The"的向量
  [0.5, 0.6, 0.7, 0.8],  # "cat"的向量  
  [0.9, 1.0, 1.1, 1.2]   # "sat"的向量
]
```

### 2. Create Query, Key, Value (QKV) matrix

Self-attention的灵感来自信息检索：

- Query (查询): "我在寻找什么信息？"
- Key (键): "我提供什么信息？"
- Value (值): "如果被选中，我贡献什么内容？"

$$Q = XW_Q \in \mathbb{R}^{n \times d_k}$$
$$K = XW_K \in \mathbb{R}^{n \times d_k}$$
$$V = XW_V \in \mathbb{R}^{n \times d_v}$$

whereas 

- $W_Q \in \mathbb{R}^{d \times d_k}$ Query权重矩阵
- $W_K \in \mathbb{R}^{d \times d_k}$ Key权重矩阵
- $W_V \in \mathbb{R}^{d \times d_v}$ Value权重矩阵
- Normally $d_k = d_v = d$ (为了简化)

Query 和 Key 生活在一个 $d_k$ 维的"语义空间"中，用来计算相似度 (normally 64)

### 3. calc 注意力分数 (Attention Scores)

$$\text{Scores} = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}$$

#### 3.1 The meaining in the numerator:

$$\text{Scores}_{i,j} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{i,l} \times k_{j,l}$$

- This computes the similarity (dot product) between position $i$'s query and position $j$'s key. (位置 $i$ 的query与位置 $j$ 的key之间的相似度)

#### 3.2 Why divided by $\sqrt{d_k}$?


假设 $q_i$ 和 $k_j$ 的元素是独立同分布，均值为0，方差为1
$$\text{Var}(q_i \cdot k_j) = \text{Var}\left(\sum_{l=1}^{d_k} q_{i,l}k_{j,l}\right) = d_k$$

proof: 

- 方差 = 平方的期望 - 期望的平方 --> $\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$
- and we have $E[q_{i,l}] = E[k_{j,l}] = 0$ and $\text{Var}(q_{i,l}) = \text{Var}(k_{j,l}) = 1$
- 单个乘积项的方差: 
  - $\text{Var}(q_{i,l} \times k_{j,l})$
  - $\text{Var}(q_{i,l} \times k_{j,l}) = E[(q_{i,l} \times k_{j,l})^2] - (E[q_{i,l} \times k_{j,l}])^2$
- As two components are independent and mean = 0 then $E[q_{i,l} \times k_{j,l}] = E[q_{i,l}] \times E[k_{j,l}] = 0 \times 0 = 0$
- 平方的variance --> $E[(q_{i,l} \times k_{j,l})^2] = E[q_{i,l}^2 \times k_{j,l}^2] = E[q_{i,l}^2] \times E[k_{j,l}^2]$
  - as $\text{Var}(X) = E[X^2] - (E[X])^2 = E[X^2] - 0 = E[X^2]$
  - so $E[q_{i,l}^2] = \text{Var}(q_{i,l}) = 1$
  - so $E[k_{j,l}^2] = \text{Var}(k_{j,l}) = 1$
- 所以单个乘积的variance： --> $\text{Var}(q_{i,l} \times k_{j,l}) = E[q_{i,l}^2] \times E[k_{j,l}^2] - 0 = 1 \times 1 = 1$
- $q_i \cdot k_j = \sum_{l=1}^{d_k} q_{i,l} \times k_{j,l}$
  - 独立随机变量和的方差等于方差的和
- $\text{Var}(q_i \cdot k_j) = \sum_{l=1}^{d_k} 1 = d_k$

**After we divide by $\sqrt{d_k}$**

$$\text{Var}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) = \frac{1}{(\sqrt{d_k})^2} \times \text{Var}(q_i \cdot k_j) = \frac{1}{d_k} \times d_k = 1$$

### 4. 计算注意力权重 (Attention Weights)

$$\text{Weights} = \text{softmax}(\text{Scores}) \in \mathbb{R}^{n \times n}$$

**Apply softmax by row**
$$\text{Weights}_{i,j} = \frac{\exp(\text{Scores}_{i,j})}{\sum_{k=1}^{n} \exp(\text{Scores}_{i,k})}$$

Softmax properties

- 概率性质: $\sum_{j=1}^{n} \text{Weights}_{i,j} = 1$
- 非负性: $\text{Weights}_{i,j} \geq 0$

### 5. Weighted Output 加权输出

$$\text{Output} = \text{Weights} \cdot V \in \mathbb{R}^{n \times d_v}$$

The output for position $i$ is a row-vector like:

$$\text{output}_i = \sum_{j=1}^{n} \text{Weights}_{i,j} \cdot v_j$$

$$\text{output}_i = \sum_{j=1}^{n} \text{Weights}_{i,j} \cdot \begin{bmatrix} v_{j,1} \\ v_{j,2} \\ \vdots \\ v_{j,d_v} \end{bmatrix} = \begin{bmatrix} \sum_{j=1}^{n} \text{Weights}_{i,j} \cdot v_{j,1} \\ \sum_{j=1}^{n} \text{Weights}_{i,j} \cdot v_{j,2} \\ \vdots \\ \sum_{j=1}^{n} \text{Weights}_{i,j} \cdot v_{j,d_v} \end{bmatrix}$$

### Full Self-Attention formula

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Time Complexity: (Total is $O(n^2 \cdot d)$)

$QK^T$ calculation: $O(n^2 \cdot d_k)$

Softmax transformation: $O(n^2)$

Weight times v: $O(n^2 \cdot d_v)$

### Space Complexity: (Total is $O(n^2 + n \cdot d)$)

Attention matrix: $O(n^2)$

Q, K, V matrices: $O(n \cdot d)$

## Multi-head attention mechanism

单头注意力的局限：

只能学会一种关系模式, 但忽略了其他重要信息.

多头注意力的优势：

- Head 1: 关注语法关系（主谓宾）
- Head 2: 关注语义关系（金融概念）
- Head 3: 关注长程依赖（deposits → costs）
- Head 4: 关注位置关系（相邻词）

每个头都是一个"专家"，专门负责某种类型的关系

Single head attention:
$$\text{SingleHead}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-head attention:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### create projection for each head $i$

$$Q_i = XW_i^Q \in \mathbb{R}^{n \times d_k}$$
$$K_i = XW_i^K \in \mathbb{R}^{n \times d_k}$$
$$V_i = XW_i^V \in \mathbb{R}^{n \times d_v}$$

#### weight matrix dimensions

$$W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$$
$$W_i^V \in \mathbb{R}^{d_{model} \times d_v}$$
#### 关键维度关系：
$$d_k = d_v = \frac{d_{model}}{h}$$
- h is the number of heads

为什么这样设计？
- 保持总参数量合理
- 每个头处理更小的子空间
- 最终concat后恢复原始维度

但是在single-head attention中， $d_k$ 很可能是不等于$d_model$的，我们可以用projection map回来

### Each head calculate attention separately

$$\text{head}_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i$$

### Concatenation output from all heads

$$\text{Concat}(\text{head}_1, ..., \text{head}_h) \in \mathbb{R}^{n \times (h \times d_v)}$$
As we have h heads
$$h \times d_v = h \times \frac{d_{model}}{h} = d_{model}$$

$$\text{Concat}(\text{head}_1, ..., \text{head}_h) \in \mathbb{R}^{n \times d_{model}}$$

### Final Linear projection

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
whereas 
$$W^O \in \mathbb{R}^{d_{model} \times d_{model}}$$

$W^O$ 是一个"信息整合专家"

- 输入：来自多个头的拼接特征
- 输出：智能整合后的最终表示

## Positional Encoding (位置编码)

核心问题：Self-Attention 无法理解词序

self-attention 是 permutation invariant (排列不变的)

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- John loves Mary
- Mary loves John
- loves Mary John
会产生几乎相同的注意力模式，因为相对的$\cdot$是一样的

### Solution: Positional Encoding

为每个位置添加独特的"位置标识"：
$$\text{Input} = \text{Word Embedding} + \text{Positional Encoding}$$

- Uniqueness: 每个位置有唯一的编码
- Consistency: 相同位置总是得到相同编码
- relative relationship: 能表达位置间的相对距离
- Generalization: 能处理训练时未见过的序列长度
- Boundedness: 编码值不能无限增长

### Sinusoidal Positional Encoding (正弦位置编码)

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

- pos: 位置索引 (0, 1, 2, ...)
- $i$: 维度索引 (0, 1, 2, ..., $d_{model}/2 - 1$)
- dmodel​: 模型维度 (如512, 768)
- $10000$: 基础频率常数
- 维度指的是位置编码向量的特征维度

#### Why Sin function?

- Periodicity提供唯一编码
- 相对位置可以线性表示
- Generalize to未见过的长度

10000这个常数怎么来的？

- 经验选择，确保合适的频率范围
- 覆盖常见序列长度

1. Periodicity and uniqueness

$$\text{Period}_i = 2\pi \times 10000^{2i/d_{model}}$$

2. relative positional relationship 相对位置关系
关键性质: 位置 $pos+k$的编码可以表示为位置 $pos$ 编码的线性组合.

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

proof:
- Gievn $PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
  - and $PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
- for simplicity, we have: $\omega_i = \frac{1}{10000^{2i/d_{model}}}$
- Then $PE_{(pos,2i)} = \sin(pos \cdot \omega_i)$ and 
  - and $PE_{(pos,2i+1)} = \cos(pos \cdot \omega_i)$
- Calculate $PE{pos+k}$
  - $PE_{(pos+k,2i)} = \sin((pos+k) \cdot \omega_i) = \sin(pos \cdot \omega_i + k \cdot \omega_i)$
  - $PE_{(pos+k,2i+1)} = \cos((pos+k) \cdot \omega_i) = \cos(pos \cdot \omega_i + k \cdot \omega_i)$
- And we have $\sin(A + B) = \sin A \cos B + \cos A \sin B$
  - and $\cos(A + B) = \cos A \cos B - \sin A \sin B$
- Then $PE_{(pos+k,2i)} = \sin(pos \cdot \omega_i)\cos(k \cdot \omega_i) + \cos(pos \cdot \omega_i)\sin(k \cdot \omega_i)$
  - $PE_{(pos+k,2i+1)} = \cos(pos \cdot \omega_i)\cos(k \cdot \omega_i) - \sin(pos \cdot \omega_i)\sin(k \cdot \omega_i)$
- Then we have $$\begin{bmatrix} PE_{(pos+k,2i)} \\ PE_{(pos+k,2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(k \cdot \omega_i) & \sin(k \cdot \omega_i) \\ -\sin(k \cdot \omega_i) & \cos(k \cdot \omega_i) \end{bmatrix} \begin{bmatrix} PE_{(pos,2i)} \\ PE_{(pos,2i+1)} \end{bmatrix}$$

可以自动识别偏移量为k的模式，并推广到任何位置的k为偏移量的关系。

每个位置的编码是所有频率的组合：

- 高频分量区分相邻位置
- 低频分量表达粗粒度关系
- 组合起来每个位置都有独特的"指纹"

#### Compared with Learnable Positional Embedding

Pros

- 完全自适应

Cons

- Needs lots of parameters (max_seq_len * $d_{model}$)
- Limited to max_seq_len, cannot be generalized to longer sequence.

效果确实更适应任务，但推理时不能泛化到比训练更长的文本

#### Rotary Position Embedding

现在大模型主流用RoPE（旋转位置编码），能更好地支持相对位置和超长序列推理，是sin/cos PE的自然进化版

## Transformer Encoder/Decoder 架构

```
Input Tokens
    ↓
Encoder Stack (could be6 layers)
    ↓
Encoder Outputs
    ↓
Decoder Stack (could be6 layers) ← Target Tokens (Training)
    ↓
Output Probabilities
```
原始Transformer是Encoder-Decoder架构，设计用于seq2seq（如机器翻译）

## Encoder structure

```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm (Residual Connection + Layer Normalization)
  ↓
Feed-Forward Network
  ↓
Add & Norm (Residual Connection + Layer Normalization)
  ↓
Output
```

For layer $l$ encoder:

1. Multi-Head Self-Attention
$$Z^{(l)} = \text{MultiHead}(X^{(l-1)}, X^{(l-1)}, X^{(l-1)})$$

2. Add and Norm
$$X^{(l)}_{\text{attn}} = \text{LayerNorm}(X^{(l-1)} + Z^{(l)})$$

3. Feed-Forward Network
$$Y^{(l)} = \text{FFN}(X^{(l)}_{\text{attn}})$$

4. Add & Norm
$$X^{(l)} = \text{LayerNorm}(X^{(l)}_{\text{attn}} + Y^{(l)})$$

### Feed Forward Network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

FNN Purpose:

- Non-linear transformation: ReLU激活函数引入非线性
- Feature transformation: 两次线性变换改变特征表示
- Dimensionality transformation: $d_{model}$ → $d_{ff}$ → $d_{model}$​（先扩展后压缩）
- 位置独立: 对每个位置独立应用相同变换

intuitive understanding:
```
输入特征: [语义, 语法, 位置, ...]  (d_model维)
    ↓ W_1, ReLU
中间表示: [更丰富的特征组合]       (d_ff维, 通常4倍大)
    ↓ W_2
输出特征: [增强的语义, 语法, ...]  (d_model维)
```

### Layer Normalization

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

for $x$ in $\mathbb{R}^{d_{model}}$

- $\mu = \frac{1}{d_{model}}\sum_{i=1}^{d_{model}} x_i$
- $\sigma = \sqrt{\frac{1}{d_{model}}\sum_{i=1}^{d_{model}} (x_i - \mu)^2 + \epsilon}$

learnable parameters:
- $\gamma \in \mathbb{R}^{d_{model}}$​: 缩放参数
- $\beta \in \mathbb{R}^{d_{model}}$​: 偏移参数
- $\epsilon$: 数值稳定性常数（通常 $10^{-6}$）

### Residual Connection (残差连接)

$$\text{Output} = \text{LayerNorm}(\text{Input} + \text{Sublayer}(\text{Input}))$$

Why residual connection?
- 梯度流动: 提供直接的梯度路径，缓解梯度消失
- 训练稳定: 让深层网络更容易训练
- 恒等映射: 允许层学习恒等变换（什么都不改变）
- 特征保持: 保留原始信息的同时添加新信息

## Decoder Structure

```
Target Input
  ↓
Masked Multi-Head Self-Attention  ← 与Encoder的区别
  ↓
Add & Norm
  ↓
Multi-Head Cross-Attention        ← 与Encoder的区别
  ↓
Add & Norm
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output
```
### Decoder的三个注意力机制

"流水线"关系 (有dependency)
- Masked Self-Attention: 防止看到未来信息
- Cross-Attention: 关注编码器输出
- Feed-Forward: 特征变换

### Masked Self-Attention

核心问题: 在训练时，decoder能看到完整的target，但必须模拟推理时的自回归生成。

- 训练时能看到完整目标句子，但推理时只能逐步生成

$$\text{Mask}_{i,j} = \begin{cases} 
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i 
\end{cases}$$

$$\text{MaskedAttention} = \text{softmax}\left(\frac{QK^T + \text{Mask}}{\sqrt{d_k}}\right)V$$

给$QK^T$ 在相应位置加上 负无穷来将未来mask住。

### Cross-Attention

**Decoder的Query关注Encoder的Key和Value**

$$\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}$$

### Feed-Forward Network
```
原始特征空间: 低维线性表示 (d_model维)
         ↓ W_1 线性扩展
高维中间空间: 丰富的特征组合和交互模式 (d_ff维)
         ↓ ReLU选择性激活 + W_2压缩
增强特征空间: 包含非线性特征的紧凑表示 (d_model维)
```

FFN的作用：把线性组合转换为更复杂的非线性表示

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

1. step 1 linear expansion (特征维度扩展)
$$h=xW1​+b1​$$
  - normally 其中 $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$​，通常$d_{ff} = 4 \times d_{model}$
  - 将低维表示映射到高维空间, 创造更多的"特征探测器", 提供学习复杂模式的容量.

2. Non-linear activation

$$f=max(0,h)=ReLU(h)$$
  - 非线性特征生成, ReLU激活创造稀疏性和非线性, 选择性地激活相关特征组合, 产生原始空间无法表达的特征.

3. Linear Compression / Linear Projection (先行压缩)

$$\text output=fW_2​+b_2​$$
- 特征选择和压缩: 从丰富的高维特征中选择最重要的信息, 将复杂特征组合压缩回原始维度，保留增强信息，丢弃冗余信息。

### Modern Variants

#### Encoder only models

- BERT, RoBERTa: 只用Encoder，适合理解任务
- 输出: 分类/标注结果

#### Decoder-Only models

- GPT系列: 只用Decoder，适合生成任务  

#### Encoder-Decoder variants
- T5: 所有任务都转为text-to-text
- BART: 加入去噪预训练
- mT5: 多语言版本














































##########################################################################









如何把位置编码加进输入？

在Transformer中，位置编码的加入方式很简单，就是把每个token的embedding向量和它对应位置的PE向量直接逐元素相加，然后送入Transformer的后续层。这样每个token既包含了内容信息，也融合了位置信息

如果没有位置编码会怎样?

如果没有位置编码，Transformer是完全感知不到顺序的（Permutation Invariant），模型就无法区分‘猫吃鱼’和‘鱼吃猫’。也就是说，任何词的排列组合在Self-Attention看来是一样的，会极大影响序列任务的表达能力和学习效果


“sin/cos positional encoding的数学表达式/直觉意义”

Transformer整体结构

Multi-Head + Add & Norm + FFN + Residual，理解基本架构


为什么用Layer Norm而不是Batch Norm？

序列长度变化: Layer Norm对每个样本独立
训练稳定性: 不依赖batch统计量
推理一致性: 训练和推理完全相同

2. 为什么FFN维度是4倍？

经验最优: 实验发现4倍是效果和效率的平衡点
表达能力: 更大的中间层提供更强的非线性变换
计算考虑: 太大会显著增加计算量

3. 为什么用ReLU而不是其他激活？