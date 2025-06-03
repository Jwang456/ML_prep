# Unsupervised Learning (Minor topic)

## K-means clustering

### 工业面试最易被问：
- 如何选K？
  - 业务驱动（优先), Elbow法则（肘部法）
- 如何做初始化？
  - K-means++（推荐，远离已选中心）and 多次随机初始化取最优
- 如何防止局部最优？
  - 多次初始化, K-means++, 运行多遍取最优结果
- feature需不需要normalize？
  - 必须归一化/标准化（normalize/standardize, 否则大尺度特征主导距离计算

### K means concepts
Pros:

- Easy to implement, calc complexity is $O(nkt)$
- suitable for sphere distributed data
- easy to interpret

Cons:

- Need to define K
- sensitive to initial setup, could fall into local optimum
- Asuume cluster is sphere and similar size


目标：将n个数据点分成k个簇，使簇内距离最小，簇间距离最大

Steps:

- Random Initialization：选择k个初始聚类中心
- Distribute samples：每个点分配给最近的聚类中心
- update center：重新计算每个簇的中心点（均值）
- repeat until convergence：直到收敛（中心点不再变化或变化很小

### K means training and deduction

$$J(\mathbf{C}, \boldsymbol{\mu}) = \sum_{i=1}^{n} \sum_{k=1}^{K} c_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

- $x_i$ ith data point
- $\mu_k$: center of kth cluster
- $c_{ik}$: indicator, 1 if $x_i$ belongs to cluster k else 0

约束条件
$$\sum_{k=1}^{K} c_{ik} = 1, \quad \forall i \in \{1,2,...,n\}$$
$$c_{ik} \in \{0,1\}, \quad \forall i,k$$

- 每个point属于且只属于一个cluster
- indicator只能是0或者1

#### Training process

**E-step 分配步骤**
固定$\mu_k$, 优化$c_{ik}$

- 遍历所有数据点，每个点分配给最近的聚类中心
- 完成所有点的分配后，进入M-step
$$c_{ik} = \begin{cases}
1 & \text{if } k = \arg\min_{j} |\mathbf{x}_i - \boldsymbol{\mu}_j|^2 \
0 & \text{otherwise}
\end{cases}$$

**M-step 更新步骤**
固定$c_{ik}$， 优化$\mu_k$

- 一次性重新计算所有聚类中心
- 基于E-step得到的完整分配结果
$$\boldsymbol{\mu}_k = \frac{\sum_{i=1}^{n} c_{ik} \mathbf{x}_i}{\sum_{i=1}^{n} c_{ik}}$$

通过EM算法迭代优化。E-step分配样本到最近中心，M-step更新中心为簇内均值。目标函数单调递减，必定收敛到局部最优

### Common questions

#### How to choose K
- The most important part is let business intuition drives the K value (业务驱动优先)
  - 如果说advertiser says allocate budget at 5 groups, then we do 5 groups.
- Emperically make K = pow(n / 2, 0.5) or use industry experience. 
- Data driven validation
  - Elbow Method: 随着K增大，簇内平方和(WCSS)必然下降，但下降速度会逐渐放缓，在某个K值处出现"拐点"，这就是最优的K值
    - Within-Cluster Sum of Squares (WCSS)

#### 如何处理不同scale的特征？
- 特征标准化 (Feature Normalization / standardization)
  - 如果不做的话，scale大的data会完全drive我们的optimization equation，导致model不care小scale变量
- Normalization and standardizatio methods
  - Z-score normalization (每个特征变成均值0、标准差1的分布)
  - Min-Max缩放(所有特征缩放到[0,1]区间)
  - $\begin{equation}x_{\text{norm}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}\end{equation}$
  - 或者对于right-skewed data我们做log transformation再做normalization.

#### K-means有什么局限性
- 形状假设限制, 只能发现球形簇
  - 无法处理椭圆形、月牙形等复杂形状
- 异常值敏感
  - 均值容易被极值拉偏, 一个异常点可能严重影响聚类中心
- 初始化敏感
  - 局部最优问题：不同初始化可能得到完全不同结果
  - 解决：K-means++初始化、多次运行取最佳
- 高维数据表现差
  - 维度诅咒：高维空间中距离失去区分度

### K-means ++
- 核心理念：让初始中心尽可能分散，彼此距离远
- 策略：第一个中心随机选，后续中心优先选择离已有中心远的点

#### K-means++算法详解
- 1：随机选择第一个聚类中心 $\mu_1 = x^{(i)}, \quad i \sim \text{Uniform}(1, n)$
- 2：对于每个数据点，计算到最近已选中心的距离 $D(x^{(i)}) = \min_{j=1,\ldots,t} |x^{(i)} - \mu_j|^2$
- 3：按距离平方成比例选择下一个中心
$P(x^{(i)} \text{ 被选中}) = \frac{D(x^{(i)})}{\sum_{j=1}^{n} D(x^{(j)})}$
- 4：重复Step 2-3，直到选出K个中心

#### K-means++的结果不会太差，有$O(\log K)$的近似保证

相比随机初始化，有理论性能保证

#### 搜广推应用

User聚类：

- 把相似用户分组，每组推荐策略不同
- 解决cold start：新用户分配到最相似的簇

Item (ads)聚类：

- 商品分类，相似商品归为一类
- 协同过滤中降维，提升计算效率

#### 高维诅咒：为什么失去区分度？
距离集中现象 (Distance Concentration)
核心问题：高维空间中，所有点对之间的距离趋于相等
数学表达：
$$\lim_{d \to \infty} \frac{\text{Var}[|X-Y|]}{\mathbb{E}^2[|X-Y|]} = 0$$
直观理解：距离的方差趋于0，所有距离都接近均值

为什么会发生距离集中？

数学原因：中心极限定理, 欧几里得距离的平方：
$$|X-Y|^2 = \sum_{i=1}^{d} (X_i - Y_i)^2$$
- 每个$(X_i - Y_i)^2$是独立随机变量
- 根据中心极限定理，距离平方收敛到正态分布
- 方差相对于均值变小 → 距离集中

几何直觉：球体体积分布
- 3维球：体积主要在中心, 100维球：体积主要在表面附近.
  - d=3,ϵ=0.1：外壳占总体积的 $1-(0.9)^3 = 27\%$
  - d=100,ϵ=0.1: 外壳占总体积的$1-(0.9)^{100} \approx 100\%$

## PCA (Principal Component Analysis)

### 常问问题
- 主成分怎么选？
累计解释方差比例（variance explained ratio， 碎石图（scree plot,业务需求优先
-  PCA能不能用于解释变量？
  - 不适合直接解释变量（主成分是线性组合，难以物理/业务解释）
  - 可分析贡献度（通过主成分系数 components_ 分析变量贡献）
- PCA和feature selection的区别？
  - PCA：降维，产生新特征（主成分，不是原始特征）
  - Feature selection：筛选原始特征
- SVD和eig有什么trade-off？
  - SVD：数值稳定，适合高维、秩亏数据
  - eig/eigen decomposition：快但只适用于对称方阵

### 必须要normalization啊，因为找的是variance最大的方向，and variance is related with the data scale.

降维的动机：高维数据的挑战

- 存储和计算成本高
- 维度诅咒影响算法效果
- 特征间存在冗余和相关性
- 难以可视化和理解

PCA的解决方案：找到数据的"主要变化方向"，用更少的维度表示原始数据

Pros:
- computational and storage efficiency的巨大提升
- Implicit regularization
  - the loss of variance explained is able to separate signal and noise
- Fix curse of dimensions
  - 距离计算在高维失效的经典问题

Cons:
- Loss Interpretability
  - 主成分是原始特征的线性组合，很难解释meaning
  - 也可以
    - 保留原始特征重要性：pca.components_ 查看每个主成分的构成
    - 混合策略：关键特征不降维，其他特征PCA
- Limitation of linear assumption
  - PCA只能捕获线性相关性
- 怎么选择主成分个数?
  - look at the variance explained ratio and determine the num of principal components

### Process in PCA

#### 中心化 Centralization

$$\tilde{X} = X - \bar{X}$$
- $\bar{X}$是样本均值

为什么要中心化？

- PCA找的是数据变化的方向
- 如果不中心化，第一主成分可能只是指向数据均值的方向
- 中心化后，主成分才真正反映数据的变异模式

#### Covariance 协方差矩阵计算
协方差矩阵：

$$C = \frac{1}{n-1}\tilde{X}^T \tilde{X}$$
- 因为我们在计算样本均值的时候已经用掉一个约束条件， 为了获得无偏估计，所以除以n-1自由度
- 协方差矩阵的含义：
  - $C_{ij}$表示第i个特征和第j个特征的协方差
  - 对角元素 $C_{ii}$是第i个特征的方差
  - 非对角元素 $C_{ij}$衡量特征i和j的线性相关性

#### Principal Component Decomposition
$$C = P\Lambda P^T$$

- $P = [p_1, p_2, ..., p_d]$ 是特征向量矩阵（主成分方向）
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, ..., \lambda_d)$是特征值矩阵
- $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d \geq 0$
  - eigen value of covariance matrix always $\lambda_i \geq 0$

问题：
- 数值不稳定：当数据维度高时，$\tilde{X}^T \tilde{X}$的条件数很大
- 计算复杂：需要先算出$d \times d$的协方差矩阵，再特征值分解

Alternative (more stable) way is **SVD**

对于任意矩阵 $A \in \mathbb{R}^{m \times n}$，都存在分解：

$$A = U\Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$ 是正交矩阵（左奇异向量）
- $V \in \mathbb{R}^{n \times n}$是正交矩阵（右奇异向量）
- $\Sigma \in \mathbb{R}^{m \times n}$是对角矩阵，对角元素为奇异值$\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0$

在PCA中，我们可以直接对中心化数据矩阵 $\tilde{X}$进行SVD：

$$\tilde{X} = U\Sigma V^T$$

Then covariance matrix is:

$$C = \frac{1}{n-1}\tilde{X}^T \tilde{X} = \frac{1}{n-1}V\Sigma^T U^T U\Sigma V^T = \frac{1}{n-1}V\Sigma^2 V^T$$

SVD好在哪
- 计算 $X^TX$会平方数据的条件数, 导致数值误差放大
- 计算效率优势, 当 $n \ll d$ 时：
  - 传统方法：需要计算 $d \times d$的协方差矩阵
  - SVD方法：只需要计算前$n-1$个奇异值和奇异向量
- 更好的数值精度控制
  - 自动处理秩亏缺情况
    - 对于对称矩阵（如协方差矩阵）：Not full rank ⟺ 存在零特征值
  - 提供更精确的低秩近似
  - 更好地识别真正的零奇异值

#### 降维变换
选择前k个主成分： $P_k = [p_1, p_2, ..., p_k]$

降维变换：$Y = \tilde{X} P_k$  (哪怕用的SVD也得做这一步， $P_k \text or V_i$只是告诉我们方向 没告诉我们在这个方向的具体坐标！)

重构（可选）：$\tilde{X}_{\text{reconstruct}} = Y P_k^T$


### Other knowledge
$$\begin{align}
v^T C v &= v^T \left(\frac{1}{n-1}\tilde{X}^T \tilde{X}\right) v \
&= \frac{1}{n-1} v^T \tilde{X}^T \tilde{X} v \
&= \frac{1}{n-1} (\tilde{X} v)^T (\tilde{X} v) \
&= \frac{1}{n-1} |\tilde{X} v|^2 \geq 0
\end{align}$$

## Other Unsupervised learnings.

### 1. GMM (Gaussian Mixture Model) / Soft Clustering

#### 常问问题
GMM和K-means区别？
- GMM是软聚类（概率分配，每个点属于多个cluster概率）
- K-means是硬聚类（每点只分配给一个簇）
- GMM建模分布（高斯混合），K-means只考虑均值
- Each data point属于多个cluster的概率分布，而不是硬分配到某一个cluster
- 是probability density （可以大于1，积分为1）

什么场景GMM优于K-means？
- 簇分布非均匀/球形（如椭圆、不同size/密度）
- 点可“同时属于多个簇”（软分配需求）
- 数据呈现混合分布（如不同方差的高斯簇）

数学直觉： 假设数据由K个高斯分布混合生成
- 用户画像：用户可能同时属于多个群体（年轻+高消费）
- 推荐系统：商品可能属于多个类别
- 异常检测：低概率点 = 异常 (prob很低 属于outlier,想想normal dist的图!!!)

**K-means其实是GMM的一个特例（协方差趋于0、只取最大概率**

### 2.  DBSCAN (Density-Based Clustering)

核心思想: 基于密度聚类，自动发现异常点

核心参数
- eps ($\epsilon$): 邻域半径
- MinPts: 成为核心点的最小邻居数量

Description
- Core point: 邻域内点数 ≥ MinPts
- Border point: 在core point邻域内
- Noise point: 既不是core也不是border = 异常点

Pros:
- Automatically determine the number of clusters
- Able to identiify different shaped cluster
- naturally noise detection and robust to outlier

Cons
- Parameter is important and hard（eps敏感）
- 密度变化大的数据集效果差
- Poor at high-dimensional data

### 3. t-SNE/UMAP (非线性降维)
#### t-SNE

t-SNE核心思想: 保持局部邻近关系的非线性降维

邻居关系比绝对距离更重要：
- t-SNE：优先保证相邻国家还是相邻，远距离可以牺牲

step1: 高维空间建模邻近关系
```
原始思路：计算所有点之间的欧氏距离
t-SNE思路：把距离转换成"相似性概率"

比如：
- 点A和B很近 → P(A选B作为邻居) = 0.8
- 点A和C很远 → P(A选C作为邻居) = 0.001
```
step2: 低维空间重构邻近关系
```
最常见为2D，比较容易display， intuitive
在lower dimensional画布上随机放置所有点
调整位置，让lower dimension的邻居概率尽可能匹配高维空间
- 如果高维中A、B是邻居，lower dimensional中也要让A、B靠近
- 如果高维中A、C很远，lower dimensional中A、C也要分开
```

Pros:
- Separate clusters in higher-dimensions to lower-dimensions (probly 2D)
- Found the grouping structure
- Especially good for visualization, the border is clear

Cons:
- The distance between cluster is meaningless
- Cannot be used to feature extraction, the distance definition is changed
- Every time the result is slightly different.
- 需要计算所有点对之间的相似性 → O(n²) 10万个点 = 100亿次计算

#### UMAP

核心哲学："拓扑结构比metric更重要"

想象场景：你要把一个复杂的3D雕塑压扁成2D浮雕
- t-SNE：关注局部细节，但整体形状可能变形
- UMAP：既保持局部细节，又尽量保持整体形状

既保持局部邻居又保持全局关系(topology)
用近似最近邻算法 → O(n log n)  10万个点 ≈ 170万次计算, 速度提升几千倍！


#### t-SNE vs UMAP

Choose t-SNE when:
- 数据量中等（<5万点）
- Find clusters is most important
- Need clear cluster borders visualization
- Don't care about the distance between cluster.

Choose UMAP when:

- 数据量大（>5万点）
- Need to keep the global structure
- Will do future analysis (like classification)
- Need the result after dimension-reduction has meaning.

特点 | t-SNE | UMAP
-- | --| --
速度| 慢 O(n²) | 快 O(n log n)
全局结构| 较差 | 更好
参数调节 | perplexity(困惑度) | n_neighbors, min_dist