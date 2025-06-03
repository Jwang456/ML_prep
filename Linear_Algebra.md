## 0 Generals

### Eigenvalues and Eigenvectors

eigenvector是非零向量v，当矩阵A作用于它时，只改变其大小而不改变方向： Av = λv

- Eigenvectors show the principal axes of transformation

Eigenvalue是标量λ，表示特征向量在变换中的缩放比例。

- Eigenvalues indicate the degree of stretching or compression along these directions

Properties

- Eigenvectors define directions that don't rotate during the transformation
- 特征值的乘积等于矩阵的行列式. The product of eigenvalues equals the determinant of the matrix 
- 特征值之和等于矩阵的迹（对角元素之和）The sum of eigenvalues equals the trace of the matrix (sum of diagonal elements)
- If a matrix has a zero eigenvalue, it is singular


### TODO: SVD

## Singular Matrix 奇异矩阵

奇异矩阵是一个方阵，其行列式等于零。这意味着矩阵不可逆（没有逆矩阵）。从代数角度看，奇异矩阵将某些非零向量映射到零向量，导致信息丢失，无法恢复原始输入。

English： A singular matrix is a square matrix whose determinant equals zero. This means the matrix is not invertible (has no inverse). Algebraically, a singular matrix maps some non-zero vectors to zero vectors, causing information loss that makes it impossible to recover the original input.

- 零行列式 Zero determinant: det(A) = 0
- 无唯一逆矩阵 No unique inverse: There is no matrix B such that AB = BA = I
- 线性相关性 Linear dependence: The rows (or columns) of the matrix are linearly - dependent
- 非满秩 Not full rank: The rank of the matrix is less than its dimension
- 至少一个特征值为零 At least one eigenvalue is zero: The characteristic polynomial has a zero root

### Near-Singular Matrices
中文： 近奇异矩阵的行列式接近但不等于零。它们技术上可逆，但逆矩阵极不稳定。微小的数据变化会导致逆矩阵发生巨大变化。近奇异矩阵的条件数（最大特征值与最小特征值的比率）非常大，表明矩阵是"病态的"。

English： Near-singular matrices have determinants close to but not equal to zero. They are technically invertible, but their inverses are highly unstable. Small changes in the data cause enormous changes in the inverse. Near-singular matrices have very large condition numbers (ratio of largest to smallest eigenvalue), indicating the matrix is "ill-conditioned."

Issues with Matrix Inversion

- 对于奇异矩阵：逆不存在，无法计算β = (X'X)^(-1)X'y
- 对于近奇异矩阵：逆矩阵元素可能非常大，导致系数估计极不稳定
- 数值问题：浮点算术中微小的舍入误差可能导致巨大的计算误差

## Maximum Likelihood Estimation (最大似然估计)

**似然(Likelihood)** 是给定模型参数条件下，观测到特定数据的概率。与通常的概率不同，似然函数将观测数据视为固定的，而将模型参数视为变量。

形式上，如果我们用 $\theta$, 表示模型参数，用 $X$表示观测数据，则似然函数可以表示为： 

$$L(\theta|X) = P(X|\theta)$$

- 在概率$P(X|\theta)$中， $\theta$被视为固定的， $X$是变量
- 在似然$L(\theta|X)$中, $X$ 被视为已经观察到的固定值  $\theta$是变量

例子：以硬币投掷为例：
- 对于一枚硬币，其正面朝上的概率为 $\theta$
- 如果我们投掷10次，观察到1次正面，9次反面
- 概率角度：如果硬币正面概率是 $\theta$，则观察到这一特定序列的概率是 $\theta({1 - \theta})^9$
- 似然角度：已经观察到1次正面9次反面，不同 $\theta$ 值对应不同的似然值，例如：
  - 若 $\theta = 0.5$, 似然≈0.001
  - 若 $\theta = 0.1$, 似然≈0.035
  - 显然，$\theta = 0.1$ 的似然值更大，意味着观测数据更支持"硬币偏向反面"的假设

### 最大似然估计的核心思想非常直观：选择能使观测数据出现概率最大的参数值作为估计值

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta|X) = \arg\max_{\theta} P(X|\theta)$$

例子：
正态分布的参数估计假设数据 $X = \{x_1, x_2, ..., x_n\}$， 来自均值为 $\mu$，方差$\sigma^2$ 的正态分布：

正态分布的概率密度函数: $P(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

Likelihood function: $L(\mu,\sigma^2|X) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$

Log likelihood function: $\ell(\mu,\sigma^2|X) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2$

对$\mu$求导并令一阶导数=0： $\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i-\mu) = 0$

$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n}x_i$

## Matrices

### Positive-definite Matrix and Semi-positive definite matrix

正定矩阵 (Positive Definite)：
对称矩阵$A$是正定的，当且仅当对所有非零向量$x$：

$$x^T A x > 0$$

半正定矩阵 (Positive Semi-definite)：
对称矩阵$A$是半正定的，当且仅当对所有向量$x$：

$$x^T A x \geq 0$$

等价判定条件

矩阵$A$正定 ⟺ 所有特征值 $\lambda_i > 0$

矩阵$A$半正定 ⟺ 所有特征值 $\lambda_i \geq 0$

### Orthogonal matrix

$Q^TQ = QQ^T=I$