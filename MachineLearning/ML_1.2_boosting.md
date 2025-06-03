# Boosting Models

**Boost** refers to the process of enhancing overall predictive performance by sequentially combining multiple weak learners.

Core Concepts

- 弱转强 Combining multiple "weak learners" (models that perform moderately alone) into a "strong learner"
- 序列化学习 Sequential Learning: Each new model focuses on correcting errors made by previous models
- 逐步改进 Progressive Improvement: Each added model "boosts" the overall performance

## AdaBoost (Adaptive Boosting) (Not that important)

Start from Binary classification and now expanded to multi-class classification or regression problem.

 It works by sequentially training weak learners (typically shallow decision trees) 浅层决策树 or 决策树桩(Decision Stumps)and combining them into a strong ensemble model, with each learner focusing more on samples that previous models misclassified.

Pros:

- Simple and intuitive algorithm, effectively reduces bias and variance, 
- can achieve high accuracy with simple base models
- Less prone to overfitting than a single complex model
- Automatically handles feature selection (when using decision trees)

Cons:

- Sensitive to noisy data and outliers
- Slower than more modern boosting algorithms
- And typically underperforms compared to gradient boosting method


### Key features

- 自适应权重机制(Adaptive Weighting Mechanism): 为训练样本分配权重, Increases weights for misclassified examples in each iteration, 迫使后续模型更关注困难案例.
- 简单基础模型（通常是决策树桩 - 单层决策树）

- Weighted Voting： Final prediction combines all models through weighted majority voting， 表现更好的模型在最终集成中获得更高权重。
- 无需复杂调参（No Hyperparameter Tuning： 主要参数是估计器（弱学习器）的数量






## Gradient Boosting Decision Trees (GBDT) 梯度提升决策树

The core idea of GBDT is: continuously fitting the residuals (or gradients) of the previous round's model. This approach follows the principle of gradient descent in function space, adding new trees in the direction that reduces the loss function each time. 不断拟合前一轮模型的残差（或梯度）。这种方法遵循函数空间中的梯度下降原理，每次沿着使损失函数减小的方向添加新树

GBDT的训练过程可以想象成一个"**relay race**"，每棵树都试图修正前面所有树犯下的错误

- 第一棵树：预测初始值(mean value or log(odds)) and calculate negative gradient (to see how much fcst differ from actual)
- 下一棵树：专门学习**negative gradient of loss function from the last tree(损失函数的负梯度)**, New tree's predictions need to be multiplied by learning rate (don't fully adopt the new tree's opinion)
- 更新总预测：将调整后的新预测加到之前的预测上, 并 **重复步骤2-5**：不断构建新树来修正剩余错误
- In regression problem, forecast is raw score, and in binary classification problem, it is convert back to probability thru **sigmod function** $P(y=1) = \frac{1}{1+e^{-\text{rawscore}}}$

For classification problem: 我们始终在原始分数空间累积预测，而不是概率空间
计算梯度时必须先将原始分数转换为概率

（Like based on each grown tree, it is used to adjust the residual from the previous tree, and there will be new new fcst from the new grown tree (actually is adjustment), **fcst  += new_fcst * learning rate** to update the forecast value.）

Pros:

- Powerful Predictive Performance (**Usually outperforms random forest**, because the iteratively fitting the residual of previous model)
- Flexible Feature Handling
  - Automatically handles feature interactions (all tree based model can)
  - Naturally handles missing values and categorical features
- Feature Importance Assessment:
  - Provides built-in feature importance measures
  - Helps with model interpretation and feature selection

Cons:

- **Sequential Training Limitations**: cannot be parallelized, training can be lengthy and not suitable for ultra large scale datasets.
- Overfitting Risk:
  - Prone to overfitting without proper regularization
  - Requires careful hyperparameter tuning
  - Sensitive to noise in training data
- Memory consumption: Storing the entire ensemble requires significant memory, need to evaluate all trees when predicting.
- Interpretation Complexity

### GBDT Hyperparameters:

- n_estimators (树的数量): 使用早停或验证曲线确定最佳数量, 注意：**与learning_rate紧密相关**

- learning_rate(控制每棵树的贡献权重): 较小值需要更多树，但通常泛化性能更好
  - High learning rate may make us trust new forecast too much and may **over-react**, 就像开车时大幅度转动方向盘
  - 策略：先固定树数量，调整学习率，再调整树数量

- max_depth: 限制每棵树的深度, 控制模型复杂度的主要参数之一
  - 过拟合指标：过深的树通常导致过拟合

- subsample（子采样率: 控制每次迭代使用的训练数据比例, 增加随机性，减少过拟合
  - 权衡：过低可能导致欠拟合
  - 此处是**无放回抽样**

### Loss Functions:

#### Regression problem loss function

- 均方误差(MSE)： $L(y, F) = \frac{1}{2}(y - F)^2 \text{ 或 } (y - F)^2$
- 平均绝对误差(MAE)： $L(y, F) = |y - F|$

#### Classification problem loss function

- 对数损失(Log Loss/二元交叉熵)： $L(y, F) = -y\log(p) - (1-y)\log(1-p)$ 其中 $p = \frac{1}{1+e^{-F}}$ 是sigmoid函数转换后的概率
  - or multi-class cross entropy
- 指数损失： $L(y, F) = e^{-yF}$ 其中y∈{-1,1}

### Subsample and Gradient Transfer Mechanism in GBDT

**subsample**

- 虽然只有部分数据(subsample)用于训练，但所有数据点的负梯度都会被计算
- 每一轮都重新计算所有点的梯度，不管它们是否参与了前一轮训练

**梯度传递机制**

- 每棵新树训练时使用的目标值是基于当前整体模型计算的负梯度, 这些负梯度反映了"模型还需要在哪些方向调整才能降低损失"
- 新树试图拟合这些方向，从而逐步改进整体模型

## XGBoost (eXtreme Gradient Boosting)

XGBoost is essentially an **enhanced implementation of GBDT**

The core idea of XGBoost remains sequentially building trees with each tree learning from the gradient of previous trees, but it incorporates more advanced regularization techniques and optimization methods.


Key differences:

- Regularization: XGBoost Introduces tree complexity penalty terms, controls overfitting
- Optimization Method: Uses both first and second-order gradients
- Tree Growth Strategy: Supports both greedy algorithm and approximate algorithm, more flexible
- Engineering Implementation: Parallel computing, cache optimization
- XGBoost是单棵树内部的计算可以并行化，显著降低training时间

Pros:

- Predictive Performance: Usually achieves best prediction accuracy on structured data
- Speed and Efficiency
- Missing Value Handling:
  - 对于每个节点的每个特征分裂，XGBoost尝试两种选择： 
    - a) 将所有缺失值的样本送到左子节点 
    - b) 将所有缺失值的样本送到右子节点
  - 选择产生最大增益的方向作为该特征缺失值的"默认方向"
- Flexibility: Supports multiple objective functions and custom objectives
-Regularization: Built-in regularization options to prevent overfitting
- Feature Importance: Provides multiple ways to evaluate feature importance

Cons:
- **Real-time Prediction**: Model size may cause latency in real-time prediction scenarios
- Parameter Complexity: Large number of hyperparameters to tune
- Memory Consumption: Requires substantial memory when processing large datasets
- Interpretability: Though more interpretable than deep learning, less intuitive than single decision trees
- Small Datasets: May tend to overfit on small datasets
 

### Mathematical Foundation of XGBoost

#### Regularized Objective Function

$$\text{Obj} = \sum_{i=1}^n l(y_i, \hat{y}i) + \sum_{k=1}^{K} \Omega(f_k)$$

- $l(y_i, \hat{y}_i)$ 是训练损失函数
- $\Omega(f_k)$ 是对第k棵树的复杂度的惩罚
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2 + \alpha\sum_{j=1}^T |w_j|$
- T是树的叶节点数，w是叶节点分数，γ和λ和$\alpha$是正则化参数

#### Second-order Taylor Expansion Approximation

XGBoost使用损失函数的二阶泰勒展开，比传统GBDT更精确：

$$\text{Obj}^{(t)} \approx \sum_{i=1}^n [l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

#### Detailed Explanation of XGBoost Algorithm

- 初始化模型： $\hat{y}_i^{(0)} = 0$
- 对于每轮 t = **1, 2, ..., T**
  - a. 计算一阶和二阶梯度： $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$

  - b. 构建树结构：找到最优分裂点，最大化： $\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$ 其中 $G$ 和 $H$ 是节点中样本的一阶和二阶梯度和。

  - c. 计算叶节点权重（系数）： $w_j = -\frac{G_j}{H_j + \lambda}$， 用这个乘以eta得到一个标量（加到fcst）

  - d. 更新模型： $\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i)$ 其中 $\eta$ 是学习率。

For example we have a regression problem which loss function is MSE：$L = \frac{1}{2}(y - \hat{y})^2$ 一阶梯度：$g = \frac{\partial L}{\partial \hat{y}} = \hat{y} - y$ 二阶梯度：$h = \frac{\partial^2 L}{\partial \hat{y}^2} = 1$
We will try every potention split for every feature and try to maximize our gain function:
$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$

左叶节点：$w_L = -\frac{G_L}{H_L + \lambda} = -\frac{-25}{3 + 1} = \frac{25}{4} = 6.25$
右叶节点：$w_R = -\frac{G_R}{H_R + \lambda} = -\frac{20}{2 + 1} = -\frac{20}{3} = -6.67$

### Two Split-finding Algorithms in XGBoost

#### Exact Greedy Algorithm:

- 对每个特征的每个可能分裂点精确计算增益
- 适用于中小规模数据, 计算量大，不适合大数据集

#### 基于分位数  quantile-based approach for finding split points (default)

- XGBoost first sorts feature values for each feature, 然后基于百分位数选取候选分割点
- 这种算法保证了分位点之间的样本数量近似相等, 计算准确但可能较为耗时

#### 基于直方图的近似算法(Histogram-based Approximate Algorithm)

- 将连续特征值分组到离散的箱(bin)中创建直方图
- 仅考虑直方图的边界作为候选分裂点, 大幅减少计算量和内存使用

First step is feature value binning, 将每个特征的值域划分为固定数量的箱（如256个）, 每个样本点被映射到对应的箱中.
And then Gradient Statistics Aggregation (梯度统计聚合), 对每个箱，累积落入该箱的所有样本的梯度统计量, 存储每个箱的一阶梯度和(G)和二阶梯度和(H). And lastly Efficient Split Point Finding, 只需扫描直方图的边界而非所有数据点
计算复杂度从O(n)降低到O(bins)


### Regularization Mechanisms in XGBoost

L2 Regularization:

- Penalizes the sum of squares of leaf weights: $\frac{1}{2}\lambda\sum_{j=1}^T w_j^2$

L1 Tree Complexity Penalty:

- Penalizes the number of leaf nodes: $\gamma T$

Implicit Regularization:

- eta (learning rate): Shrinks the contribution of each tree: $\eta f_t(x_i)$, smaller learning rates typically provide better generalization
- Randomness: 
  - Column sampling->Randomly selects feature subsets for each split
  - Row sampling->Randomly selects sample subsets for training each tree
- Early Stopping technique

### Hyperparameters in XGBoost

Tree Structure HPs
- max_depth: Maximum depth of tree, controls model complexity (typically 3-10)
- min_child_weight: Minimum sum of weights needed in a leaf node, prevents overfitting
- gamma (min_split_loss): Minimum loss reduction required for a split, 越大树越保守
- max_leaves: Maximum number of leaf nodes, 适用于grow_policy='lossguide'

Randomness and Sampling Parameters
- subsample：训练样本的抽样比例（0.5-1）
- colsample_bytree：每棵树的特征抽样比例 （决定这棵树可用的特征集）
- colsample_bylevel：每层的特征抽样比例（从当前tree中features中选择每层可用的特征）
- colsample_bynode：每个节点的特征抽样比例 (从每层可用特征中选择每个节点可用的特征)
- 上面这三个是层层嵌套的关系，

Learning Parameters
- eta (learning rate): 每棵树的贡献缩减系数
- num_boost_round (n_estimators)：总迭代次数（树的数量）

Regularization Parameters
- lambda (reg_lambda): L2 regularization term
- alpha (reg_alpha): L1 regularization term

Objective Function Parameters
- objective: Learning task and objective function
- eval_metric: Evaluation metrics

## LightGBM (Light Gradient Boosting Machine)

Pros

- **High Efficiency**: Histogram-based algorithm significantly reduces memory usage and computation time, Training speed typically 5-10x faster than XGBoost
- Capability to Handle Large-Scale Data: GOSS technique, EFB technique
- Categorical Feature Support
- Parallel and Distributed Support
  - Supports feature parallel, data parallel, and voting parallel
  - Well-optimized distributed training

Cons:

- Overfitting Risk
  - Leaf-wise growth strategy more prone to overfitting
  - May not perform as stably as XGBoost on small datasets
- Hyperparameter Sensitivity
  - More sensitive to parameters like num_leaves and max_depth
  - Requires more hyperparameter tuning to achieve optimal performance
- Unbalanced Tree Structure
- Interpretability Challenges
  - Unbalanced tree structure makes feature importance interpretation more complex
  - Tree visualization and explanation more difficult than with balanced trees
- Not Ideal for Small Datasets

### Key Features

- Histogram-based Algorithm (无需排序)
  - 将连续特征值分组到离散的箱(bin)中创建直方图
  - 仅考虑直方图的边界作为候选分裂点, 大幅减少计算量和内存使用
- Leaf-wise Tree Growth
  - 每次只选择一个具有最大增益的叶子节点进行分裂， 不考虑节点所在层级，只考虑分裂增益
  - May produce very unbalanced trees
- Specialized Techniques for Large Datasets
  - see **Goss** and **EFB** below
- Native Categorical Feature Support

#### GOSS (Gradient-based One-Side Sampling) 基于梯度的单侧采样: 

**其核心思想是：在样本空间中，具有较大梯度的样本对模型训练更加重要，应该优先保留。**
工作原理

- 样本排序：根据每个样本的梯度绝对值大小进行排序， 梯度大的样本表示模型对这些样本预测得不好，需要更多关注。
- 样本筛选：保留梯度绝对值最大的前a%样本（如前20%）， 从剩余的小梯度样本中随机采样b%（如从剩余80%中采样10%）
- 权重调整：对保留的小梯度样本增加权重系数：(1-a)/b， 这确保了采样后信息增益的统计一致性

效果：
- 降低训练复杂度：使用更少的样本进行训练
- 保留关键信息：确保模型能学习到最具挑战性的样本
- 提高效率：比随机采样更高效，因为它有目的地保留信息量大的样本

#### EFB (互斥特征捆绑)
其核心思想是：**许多特征在稀疏空间中互斥（几乎不会同时取非零值），可以安全地捆绑成一个特征，减少特征维度**
工作原理：
- 构建特征图：将每个特征视为图中的一个节点， 如果两个特征不能安全捆绑（冲突较多），则连接它们
- 图着色：使用图着色算法将特征分组， 同一颜色的特征可以捆绑在一起
- 特征合并：将同组特征合并为一个单一特征， 通过偏移值避免原始值冲突

效果：
- 降低特征维度：减少特征数量，加速训练
- 保留信息：几乎无损地合并特征
- 特别适合稀疏数据：如文本数据、点击数据等高维稀疏场景
- 加速高达8倍：在高维稀疏数据上可获得显著加速

#### **Distributed training in LightGBM**

**核心概念**: 基本思想是通过只交换每台机器最有希望的分割候选点，而不是共享完整直方图或全局评估所有可能的分割点，从而减少通信开销.
Steps
- 本地Top-K选择：每台机器为其本地数据中的特征构建直方图, 本地评估所有可能的分割点, 每台机器选择具有最高本地增益的前K个分割点（例如，K=10）

- 候选点收集：所有机器的本地top-K候选点都发送到主节点, 每个候选点包括：特征索引、分割值/桶和本地增益
- 全局评估：
  - 对于收集集合中的每个唯一候选点：
    - 向所有机器请求必要的统计信息
    - 每台机器提供该分割点的梯度和和二阶导数和
    - 聚合这些统计信息以计算全局增益
- 最终选择：
  - 选择具有最高全局增益的候选点作为分割点

## CatBoost (Categorical Boosting)

### Key Features

- Advanced Categorical Feature Handling
  - Instead of one-hot encoding, CatBoost uses an advanced encoding scheme.
  - For each categorical feature, it computes target statistics.
- 有序提升Ordered Boosting
  - Uses ordered target statistics to prevent target leakage
  - 创建数据集的多个排列
- Fighting Overfitting
  - Built-in regularization techniques
  - Symmetric trees with oblivious splits 使用平衡分裂的对称树
  - Combination of multiple permutation-based models 结合多个基于排列的模型
- Built-in Missing value handling：
  - CatBoost能原生处理缺失值，no pre-processing needed
  - for each potential split，自动寻找缺失值的最优方向
  - In categorical features, treats missing values as a separate category
- Ordered Gradient estimation in CatBoost


### Prediction Shift Problem in Traditional Gradient Boosting and CatBoost Solution

Prediction shift is a fundamental challenge in traditional gradient boosting methods that creates inconsistency between training and prediction distributions, affecting model generalization.

#### Why Prediction Shift would happen in traditional Gradient Boosting method?
- 迭代训练机制：
  - 每次迭代都基于当前模型的预测结果计算梯度 and Hessian
  - 当前模型包含了对整个训练集的"记忆"
- 自我依赖循环：
  - 第t轮的梯度依赖于第t-1轮的模型预测
  - 这个预测本身是在同一数据上学习得到的, 形成了一种自我强化的循环依赖.

这就导致了两个关键问题：

**1. 目标泄露(Target Leakage)**：

第t-1轮模型已经"看过"样本i及其真实标签*yi*, 模型的预测, ​已经包含了关于yi​的信息,所以计算的梯度​会有偏差

**2. 条件偏移(Conditional Shift)**：

- 训练时：梯度计算依赖于在同一数据上训练的模型
- 预测时：模型应用于全新数据，没有这种依赖

这种分布差异导致了训练和预测之间的偏移

**Example**:
Traditional Gradient Boosting：

- 第10轮迭代时，计算样本5的梯度
- 使用了前9轮在所有100个样本(包括样本5)上训练的模型
- 这个模型已经"看过"样本5的真实标签
- 计算的梯度有偏差，不代表真实的预测误差

CatBoost方法：

- 创建数据的随机排列
- 如果样本5在排列中位于位置60
- 只使用前59个样本训练的模型(assistant model)来计算样本5的梯度
- 这个模型从未"看过"样本5，提供无偏估计

将所有training做完permutation以后分成几个连续的组，第一组（或前m组）是用前k-1（k-m）轮完整模型计算梯度（traditional gradient），对于后面的组，利用前面组上训练的model去计算gradient，来尽可能减少计算复杂度与减少leakage的影响。

#### Ordered target statistics for encoding in CatBoost: Prevent encoding value leakage in categorical variables.

首先创建数据的随机排列（permutation, 对于each row，仅使用排列中在它之前的样本来计算统计量 (like an ordered target encoding).
$$\text{Value} = \frac{\sum_{i \in \text{category}} y_i + \text{prior}}{\text{count}(\text{category}) + \text{prior weight}}$$

### CatBoost的对称树(Oblivious Trees)

也称为平衡决策树，是CatBoost使用的一种特殊树结构，其特点是**同一层的所有节点使用完全相同的分裂规则**。这种设计与传统决策树有显著不同 (All node in the same level uses the same split rule like x2 <= 3)

- At the beginning of each leveol, considering all possible features and split points, and choose the optimal splitting rule for the entire level(单一最佳分裂规则)

Pros of Oblivious Trees

- Against overfitting
  - 强制所有节点使用相同规则是一种结构化正则化
  - 限制了模型的复杂度，减少了过拟合风险
- Feature Interaction
  - 层级决策路径自然捕获特征交互
  - 每一层可以视为一个自然的特征交叉
- Ensemble effect
  - CatBoost构建多棵对称树
  - 单树的"限制"被整体集成效果抵消, ensemble的多样性可以弥补单树的局限性





Feature | XGBoost | LightGBM | CatBoost | AdaBoost
-- | -- | -- | -- | --
Release Year | 2014 | 2017 | 2017 | 1997
Optimization Target | Second-order Taylor expansion | Histogram optimization | Ordered boosting | Sample weights
Tree Growth Method | Level-wise | Leaf-wise | Symmetric tree | Level-wise
Missing Value Handling | Built-in optimization | Zero value processing | Built-in optimization | Preprocessing needed
Categorical Feature Handling | Requires preprocessing | Automatic handling | Special optimization | Requires preprocessing
Training Speed | Fast | Very fast | Medium | Slower
Memory Efficiency | Medium | High | Medium | High
Feature Importance | Multiple methods | Multiple methods | Multiple methods | Simple weighting
Suitable Data Scale | Medium to large | Large to very large | Medium to large | Small to medium
Parameter Sensitivity | Medium | High | Low | Low
Characteristics | All-rounder | Speed priority | Categorical feature friendly | Simple implementation
