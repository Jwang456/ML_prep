- Explain core concepts simply first - Begin with high-level explanations
- Dive deeper when prompted - Have details ready if the interviewer wants to explore
- Connect to practical applications - How/when you'd use each algorithm

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

### 1.1.2 SVM Support Vector Machines (支持向量机)

**Basic Principles**：支持向量机是一种强大的监督学习算法，专为分类和回归任务设计。SVM的核心思想是找到一个**最优超平面**，使其能够最大化分离不同类别的数据点。

for regression problem: 找一个函数，使得尽可能多的数据点位于距离函数ε范围内.

Pros:

- Effective Decision Boundaries 高效的边界决策
  - 最大间隔分类
  - Strong generalization: Performs well on unseen data.
  - Ability to handle complex decision boundaries: Can capture complex non-linear relationships using the kernel trick (核技巧可以捕捉复杂的非线性关系)
- Advantages for High-Dimensional Data 针对高维数据的优势
  - 在特征数量大于样本数量的情况下依然有效
  - Resistance to overfitting: Especially in high-dimension, low-sample-size scenarios.
- Computational and Memory Efficiency
  - 模型紧凑性：只依赖支持向量，而非全部训练数据
  - 预测速度快：仅使用少数支持向量计算预测, 不需要存储全部训练数据.
- Solid Theoretical Foundation
  - 凸优化问题：保证找到全局最优解而非局部最优
  - 明确的几何解释 (Clear geometric interpretation): Easy to understand and explain

Cons:

- Computational Complexity Challenges
  - 标准SVM在大数据集上训练缓慢（复杂度在O(n²)到O(n³)之间）
  - 难以扩展 (Difficult to scale): Not suitable for datasets with millions of samples
- Parameter Tuning Complexity (参数调优复杂性)
  - Need to pick the proper kernel function
  - sensitive to hyperparameters
    - Need a lot of CVs to find the best parameters.
- Data Preprocessing Requirements
  - Strict feature scaling requirements: Unnormalized features can severely impact performance
  - 噪声敏感：对噪声和异常值较为敏感
  - Sensitive to feature redundancy: 不会自动处理共线性
- 多类分类需要额外策略：本质上是二分类算法

对于二分类问题，SVM的目标函数为: 
$$\min_{w, b} \frac{1}{2} ||w||^2 \\
\text{subject to } y_i(w^T x_i + b) \geq 1, \forall i$$

#### Support Vectors

Support vectors are the training data points that lie closest to the decision boundary

支持向量：分类正确且距离边界最小的样本

- 决定边界位置和方向的关键点
- 对模型有实质性影响的唯一点
- 通常仅占训练数据的一小部分

#### Hyperparameters:

**C parameter**: 
In practical applications, data is rarely perfectly linearly separable. Soft-margin SVM introduces slack variables, allowing some data points to violate the margin constraints (允许一些数据点违反间隔约束)

The C parameter controls regularization strength
- **High C values**: Emphasize correctly classifying each training sample, potentially leading to overfitting
- **Low C values**: Allow more misclassifications, pursuing simpler decision boundaries for better generalization

**gamma $\gamma$**

Intuitive understanding: Defines the "influence radius" of a single training example: 定义了单个训练样本的"影响范围"

- High Gamma: 每个样本影响范围小，决策边界更复杂，更关注局部模式，可能过拟合
- Low Gamma: 每个样本影响范围大，更平滑的决策边界，捕捉更全局的模式

#### Kernel Method

$$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

Common Kernel Functions
- Radial Basis Function (RBF)/Gaussian Kernel $K(xi,xj)=exp⁡(−γ∣∣xi−xj∣∣^2)$
  - 径向基函数(RBF)核/高斯核 
  - 通用性最强，默认首选
  - γ控制决策边界的"波动性", 较小的γ产生较平滑的边界
- Linear Kernel $K(xi​,xj​)=xi^T​xj​$
- Polynomial Kernel $K(xi,xj)=(γxi^Txj+r)^d$
- Sigmoid Kernel $K(xi,xj)= \tanh(\gamma x_i^T x_j + r)$



### 1.1.3 Decision Trees and Random Forest

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
Predictive Performance | Prone to overfitting, high variance	| Reduced variance, strong generalization
Stability |	Sensitive to data changes, unstable	| Robust to data changes, stable results
Interpretability | Highly interpretable, visualizable decision paths|Reduced interpretability, only feature importance
Computational Complexity|	Fast training and prediction	|Slower training and prediction, parallelizable
Memory Requirements	|Low to moderate| Higher (proportional to number of trees)
Feature Selection|	May favor certain features	|More balanced and reliable feature importance
Parameter Sensitivity	|Highly sensitive to parameter settings	|Relatively insensitive to parameter settings
Outlier Handling	|Easily influenced by outliers	|Robust to outliers
Randomness|	Deterministic results (fixed seed)	|Randomness introduces diversity
Bias-Variance Trade-off|Low bias, high variance	|Slightly increased bias, significantly reduced variance

## Regression Models

### 2.1.1 Linear regression

Pros:

- Interpretability: Clear feature importance and Easy to explain
- Computational Efficiency: Training speed and Low computational requirements.
-  Robustness and Stability: 不易过拟合, 预测稳定

Cons:

- assumes linearity between inputs and outputs
- sensitive to outliers
- can under fit with small, high-dimensional data

#### **Assumptions**:

- **Linearity**: There is a linear relationship between the independent variables(X) and the dependent variables (y)
- **Independence of Errors**: Independence assumes that there is no relationship or correlation between the errors (residuals) of different observations.
- **Normality of Errors**: The residuals of the linear regression model are assumed to be normally distributed.
- **Homoscedasticity (Constant Variance of Errors)**: Homoscedasticity assumes that the variability of the errors (residuals) is constant across all levels of the independent variables.
  - 同方差性假设要求误差项的方差在所有自变量水平上保持恒定。如果违反此假设(出现异方差性)，最小二乘估计虽然仍然无偏，但不再是最有效的，且标准误差的估计会不准确，影响统计推断的有效性。同方差性确保了模型对所有数据点的预测精度一致
- **No Multicollinearity**: No Multi-collinearity between features

#### Ordinary Least Square

$$y = w^T x + b$$

targeting on 

$$w = \arg\min_w Loss(w)$$

where as Loss function (MSE / n) as $L(w) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - w^T x_i)^2$

From Gradient-wise, the gradient of loss function wrt weight is (for each row i):

$$\frac{\partial L(w)}{\partial w_j} = -2(y_i - w^T x_i)x_{ij}$$
$$\frac{\partial L(w)}{\partial w_j} = -2(y - \hat{y})x_j$$

For all training data, it is:

$$\frac{\partial L(w)}{\partial w_j} = -\frac{2}{n}\sum_{i=1}^{n}(y_i - w^T x_i)x_{ij}$$

- $w_j$ is the coefficient (weight) for the j-th feature
- $i$ is the i-th training sample (row)

##### Matrix Representation (因为loss function是凸函数)

$$L(w) = (y - Xw)^T(y - Xw)$$ 

- 你也可以加上1/n, 没影响的

展开后

$$L(w) = y^Ty - y^TXw - w^TX^Ty + w^TX^TXw$$
- 注意到 $y^TXw = w^TX^Ty$
- 因此$L(w) = y^Ty - 2w^TX^Ty + w^TX^TXw$

损失函数求导并令其等于零

$$\frac{\partial L(w)}{\partial w} = -2X^Ty + 2X^TXw = 0$$

$$X^TXw = X^Ty$$

$$w = (X^TX)^{-1}X^Ty$$

##### Maximum Likelihood Estimation representation

寻找参数值，使得观测到的数据出现的概率最大。用数学形式表示为：$w_{MLE} = \arg\max_w P(X|w)$

- $P(X|w)$ 是似然函数，表示在参数 $w$ 条件下观察到数据$X$的概率

线性回归中的误差假设

- 在线性回归中，我们假设误差项 $\varepsilon$ 服从正态分布：$\varepsilon \sim N(0, \sigma^2)$

由于 $y = w^Tx + \varepsilon$，这意味着给定 $X$和 $w$，$y$ 的条件分布也是正态的

- $P(y|x,w) \sim N(w^Tx, \sigma^2)$
- $P(y|x,w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y-w^Tx)^2}{2\sigma^2}\right)$

假设我们有 $n$ 个独立观测样本 $(x_1,y_1),...,(x_n,y_n)$，样本之间相互独立，则联合似然函数为：

$L(w) = P(D|w) = \prod_{i=1}^n P(y_i|x_i,w) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\right)$

为了简化计算，我们通常最大化对数似然函数

$\ln L(w) = \sum_{i=1}^n \ln P(y_i|x_i,w) = \sum_{i=1}^n \ln \frac{1}{\sqrt{2\pi\sigma^2}} - \sum_{i=1}^n \frac{(y_i-w^Tx_i)^2}{2\sigma^2}$

$\ln L(w) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i-w^Tx_i)^2$

在这个表达式中，第一项是常数，不依赖于 $w$。因此，最大化对数似然等价于最小化第二项中的平方和

$w_{MLE} = \arg\max_w \ln L(w) = \arg\min_w \sum_{i=1}^n (y_i-w^Tx_i)^2$

##### Maximum a posterior estimation (To be done)
