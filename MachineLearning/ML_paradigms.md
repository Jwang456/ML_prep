# 1. Supervised Learning (Mostly used in industry)

## 1.1 Classification Models

### 1.1.1 Logistics Regression

#### Pros:

- Interpretable and explainable method
- Less prone to overfitting when using regularization
- Directly applicable for multi-class predictions
- Robustness to Feature Noise
- Computationally efficient and scales well to large datasets

#### Cons

- Assuming linearity between input and output (Linear Separability)
- Can overfit with small, high dimensional data
- Multicollinearity impact on interpretation

In a binary classification problem, y_hat is the probability of 1, and the probability of 0 is 1- y_hat

**Sigmod Function** (logistics function) is

$$
y = \frac{1}{1+e^{-(w^{T} x + b)}} 
$$ 

Interpreted as P(Y=1 | X), given set of x, the probability of y = 1

derivative of sigmode function is $\sigma'(z) = \sigma(z)(1-\sigma(z))$

#### Cross Entropy (Loss Function in binary classification) is:

$$H(y, \hat{y}) = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

for each observation, and for all observations, just calculate the average of all individual cross entropy.

$$
\mathcal{L}(f(x), y) =
\begin{cases}
-\log(f(x)) & \text{if } y = 1 \\
-\log(1 - f(x)) & \text{if } y = 0
\end{cases}
$$

#### Minimize cross entropy in a binary classfication problem is equivalent to maximize log softmax probability

- In binary classification problem, cross entropy is $H(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i) = -y_j \log(\hat{y}_j) = -\log(\hat{y}_j)$
- We want to minimize $\min H(y, \hat{y}) = \min(-\log(\hat{y}_j))$
- Softmax function = $\hat{y}j = \frac{e^{z_j}}{\sum{k=1}^{C} e^{z_k}}$
- Thus，maximize $\log(\hat{y}_j)$ 等价于最大化对正确类别j的对数softmax概率。

#### Discretize continuous features

Pros:

- Enhanced Robustness to Outliers (极端值被限制在最高/最低分箱内，影响有限)
- Facilitates Feature Crossing (如"年龄段_x_职业"可直接编码为新特征)
- More Convenient Handling of Missing Values (单独的缺失值分箱：可将缺失值作为独立分箱)
- Introducing Non-linear Expressiveness (每个分箱作为独立特征，允许不同区间有不同权重)
- Reducing Overfitting Risk (减少了对训练数据中噪声的敏感性)
- Improved Computational Efficiency(稀疏计算优化：x[i]==0时可跳过乘法，大幅提升计算速度)

#### Gradient calculation in Logistics Regression

$\hat{y} = \sigma(z) = \sigma(w^Tx + b)$，$\sigma$ 是sigmoid函数。

calculation for parameter $w$ is 
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$
 - $w$ is parameters for independent variables
 - $w = (w_1, w_2, ..., w_n)$

损失函数对 $\hat{y}$ 的gradient: $$\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$

$\hat{y}$ 对 $z$ 的gradient (sigmoid函数的导数): $$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$$

$z$ 对 $w$ 的gradient: $$\frac{\partial z}{\partial w} = x$$

将上述三部分组合: $$\frac{\partial L}{\partial w} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y}) \cdot x$$

化简后得到: $$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x$$

在线性回归中，使用均方误差损失函数: $$L(y, \hat{y}) = \frac{1}{2}(\hat{y} - y)^2$$

其中 $\hat{y} = w^Tx + b$

参数 $w$ 的gradient计算为: $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$$

损失函数对 $\hat{y}$ 的gradient: $$\frac{\partial L}{\partial \hat{y}} = \hat{y} - y$$

$\hat{y}$ 对 $w$ 的gradient: $$\frac{\partial \hat{y}}{\partial w} = x$$

组合后得到: $$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x$$
 - $x$ 实际上是一个 feature vector $x = [x_1, x_2, ..., x_n]$
 - $\frac{\partial L}{\partial w_i} = (\hat{y} - y) \cdot x_i$

 $$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot [x_1, x_2, ..., x_n] = [(\hat{y} - y)x_1, (\hat{y} - y)x_2, ..., (\hat{y} - y)x_n]$$

 - $(\hat{y} - y)$ 是一个标量（预测误差）

误差项的共通性：$(\hat{y} - y)$ 这一项在两种回归中都可以解释为"预测误差"

假设我们有一个样本，特征向量 $x = [100, 0.5, 3]$，预测误差 $(\hat{y} - y) = 0.2$：

梯度向量计算为： $$\nabla_w L = 0.2 \times [100, 0.5, 3] = [20, 0.1, 0.6]$$

这说明：

我们得到的是一个整体的梯度向量 $[20, 0.1, 0.6]$
第一个权重 $w_1$ 的梯度分量最大(20)，因为对应特征值最大(100)
所有梯度分量都由相同的预测误差(0.2)计算得到，但被各自的特征值放大/缩小
这种理解更准确地反映了梯度计算的整体性，同时解释了为什么不同参数会以不同速率更新。

#### 默认梯度下降方法
小批量梯度下降 (Mini-batch Gradient Descent)

计算方式：

- 每次用一个小批量(如32或64行)数据
- 公式：对批量B中的样本计算$\nabla_w L = \frac{1}{|B|}\sum_{i\in B}(\hat{y}_i - y_i)x_i$

### 1.1.2 Decision Trees and Random Forest

### Decision Tree

Decision trees **don't use methods like gradient descent** to optimize parameters, but instead use **impurity measures** as splitting criteria.

If feature $X_j$ is used at multiple nodes, the decision rule might look like: $$\text{If } X_j \leq t_1 \text{ AND } X_j > t_2 \text{ AND } X_j \leq t_3 \text{ THEN } y = c_k$$

Where $t_1, t_2, t_3$ are different thresholds, and $c_k$ is the prediction value.

Pros:

- 直观可解释: Decision process is clear and similar to human thinking
- No Feature Preprocessing Required and can process both numerical and categorical features simultaneously
- 自动feature selection: Algorithm naturally performs feature selection, prioritizing important features
- 处理非线性关系: Can model complex non-linear patterns and feature interactions
- 计算效率高: Training and prediction usually fast

Cons:

- Tendency to Overfit: Easily constructs overly complex trees, capturing noise in training data
- Instability: Small changes in data can result in completely different tree structures
- Greedy Algorithm Limitations: Selects locally optimal splits at each step, may miss globally optimal tree


Gini Impurity (CART algorithm) = Gini(t) = $\text{Gini}(t) = 1 - \sum_{i} [p(i|t)^2]$

- Lower the Gini Impurity, purer the group
- A completely pure node has a Gini value of 0

Information Entropy 信息熵 = $\text{Entropy}(t) = -\sum_{i} [p(i|t) \cdot \log_2(p(i|t))]$

- Lower entropy indicates a purer node, with entropy of 0 meaning the node contains only one class

**The decision tree is growed based on Impurity Improvement**

$$\text{Gain} = I(t_{\text{parent}}) - \sum_{j \in \text{children}} \frac{N_j}{N} I(t_j)$$

- $I(t)$ 是节点 $t$ 的不纯度（基尼或熵）
- $N$ 是父节点中的样本数
- $N_j$ 是子节点 $j$ 中的样本数

Decision Tree Growth Algorithm

- Evaluate All Possible Splits:
  - Try each possible split point for each feature
  - Calculate information gain for each candidate split

- Select Best Split:

  - Choose the feature and split point that produces the maximum information gain
  - Split the data and create child nodes

- Build Recursively:

- Repeat steps 1 and 2 for each child node
- Continue until stopping conditions are met (high node purity, few samples, maximum depth reached)

### Random Forest

Pros:

- Excellent Predictive Performance: Significantly improves prediction accuracy compared to single decision trees
- **Robustness and Resistance to Overfitting**: Ensemble method reduces variance and risk of overfitting and insensitive to noise and outliers
- Versatility: Applicable to both classification and regression tasks and handles numerical and categorical features.
- Parallelizability: Tree construction and evaluation can be parallelized independently

Cons:

- Reduced Interpretability: More difficult to interpret compared to a single decision tree.
- Intensive Computational Resource and Large Model Size: 
  - Longer prediction time compared to single models
  - Training can be slow on large datasets
  - Deployment may be limited in resource-constrained environments
- Sensitive to changes in Hyper-parameters

Random Forest is a powerful machine learning algorithm based on ensemble learning principles (集成学习原理). It improves performance, stability, and generalization ability by constructing multiple independent decision trees and combining their predictions

分类问题：
$$\hat{y} = \text{mode}(h_1(x), h_2(x), ..., h_B(x))$$

回归问题：
$$\hat{y} = \frac{1}{B}\sum_{b=1}^{B}h_b(x)$$
whereas 
- $h_b(x)$ 是第 $b$ 棵树的预测
- $B$ 是森林中树的总数
- mode 表示取众数（最常见的类别）

#### Building process of random forest

* 1.Bootstrap Sampling: For each tree $b = 1, 2, ..., B$:

  - Randomly sample with replacement $n$ samples from the training set of size $n$, forming subset $D_b$
  - Approximately 63.2% of original samples will appear in each bootstrap sample (as bootstrap has replacement, so one row could be selected for multiple times.)

* 2.Building Randomized Decision Trees: For each bootstrap sample $D_b$

  - Build tree $T_b$ using a modified decision tree algorithm
  - At each node split, consider only $m$ randomly selected features (typically $m \approx \sqrt{p}$, where $p$ is the total number of features)
* 3.Ensemble Prediction
  - For classification: Use majority voting to combine predictions
  - For regression: Calculate the average of all tree predictions

* 4.Out-of-Bag Error Estimation
  - Use OOB samples for each tree to calculate performance metrics, providing a built-in validation mechanism: $\text{OOB Error} = \frac{1}{n}\sum_{i=1}^{n}I(\hat{y}_i^{oob} \neq y_i)$
  -  Where $\hat{y}_i^{oob}$ is the prediction made using only trees that did not include sample $i$.

Dual Sources of Randomness

1. Sample Randomness (Bagging)

- Each tree is trained on a different bootstrap sample (bagging is ensemble learning using bootstraping sampling technique)
- Ensures diversity and independence between trees

2. Feature Randomness

- Each node split considers only a randomly selected subset of features
- Further increases diversity between trees

Key Parameters:

- n_estimators: control the number of trees in a forest
- max_features: Controls how many random features to consider at each node. default $\sqrt{p}$ for classfication and $p/3$ for 
  - Even within the same decision tree, we still need to resample at each node
  - The most important HP in RF, can use grid search in cross validation to find a best value of it. 
- Max_depth and min_sample: Controls complexity of individual trees,if depth is reached or in current node the number of obs is smaller than min sample.
  - typically use unrestricted trees, but constraints can improve speed
- Bootstrap Sample Size (max_samples): Controls proportion of training samples used for each tree. default is 100%

Characteristic| Decision Tree| Random Forest
-- | -- | --
Predictive Performance | Prone to overfitting, high variance| Reduced variance, strong generalization
Stability |Sensitive to data changes, unstable| Robust to data changes, stable results
Interpretability | Highly interpretable, visualizable decision paths|Reduced interpretability, only feature importance
Computational Complexity|Fast training and prediction |Slower training and prediction, parallelizable
Memory Requirements |Low to moderate| Higher (proportional to number of trees)
Feature Selection | May favor certain features |More balanced and reliable feature importance
Parameter Sensitivity | Highly sensitive to parameter settings | Relatively insensitive to parameter settings
Outlier Handling | Easily influenced by outliers | Robust to outliers
Randomness | Deterministic results (fixed seed) | Randomness introduces diversity
Bias-Variance Trade-off | Low bias, high variance | Slightly increased bias, significantly reduced variance
