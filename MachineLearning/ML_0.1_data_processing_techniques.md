## 1. Imbalanced Data and Sampling

If the size of major class / size of minor class > 3 (or 10) is considered as imbalanced data (extreme imbalanced)
or the size of minor class is too small.

### why imbalanced data affect model?

- Algorithm Optimization Mechanism: Most classification algorithms optimize for overall accuracy by default. With imbalanced datasets, simply predicting the majority class can achieve high accuracy.
- Learning Process Bias: Models encounter majority class samples more frequently, learning their features more thoroughly
- Sizes of minor classess may be not enough to build model. 

Like we have a credit card transaction dataset:

- 99,000 legitimate transactions (99%)
- 1,000 fraudulent transactions (1%)

Model always predicts "not fraud", resulting in:

- All 99,000 legitimate transactions correctly predicted
- All 1,000 fraudulent transactions incorrectly predicted

**Practical impact**: A bank using such a model would allow all fraudulent transactions to pass through, causing enormous financial losses, despite the model's 99% accuracy.

### Undersampling and Oversampling

Under-sampling (Reduces total sample count)

- Reduces majority class samples to balance the dataset.
- Actual Working Process:
  - Identify: Determine which majority class samples can be removed
  - Delete: Remove selected majority class samples from the training set
  - Balance: Make the final class ratio more even

Undersmapling gives up some of data points in the major class, undermine the effect from major class, may cause a **highly-biased model**. So we can do undersampling repetitively and combine the resuly. But repetitively training costs and also has a risk of **overfitting** because of minor classes data points are used repetitively.

Oversampling (Increases total sample count)

- Increases minority class samples to balance the dataset.
- Actual Working Process:

  - Selection: Determine how to create new minority classsamples
  - Generation: Create new minority class samples (duplication or synthesis)
  - Addition: Add new samples to the training set

Over-sampling simply repeat the minor class, thus it will over-emphasize the existing minor data point(s), if some of minor data points are labeled incorrectly or noise, then the error is amplified. So the risk of oversampling is **Overfitting**

#### SMOTE is a special technique of oversampling

Instead of repeating existing minor classes data points, it uses KNN to generate new minor classes.

- reduce the risk of overfitting
- more resistant to noise data
- May generate some suspicious data points

## 2. Multicollinearity 多重共线性

多重共线性是指预测模型中的自变量（特征）之间存在高度相关性的情况。当两个或多个预测变量高度相关时，它们提供了重叠的信息，使得模型难以确定每个变量的独立影响。

Multicollinearity refers to a situation where independent variables (features) in a predictive model are highly correlated with each other. When two or more predictor variables are highly correlated, they provide redundant information, making it difficult for the model to determine the independent effect of each variable.

Multicollinearity is essentially an **information redundancy problem**:

- When features are highly correlated, they contain overlapping information
- This redundancy makes it difficult for models to distinguish each feature's independent contribution
- Equivalent to signals being duplicated while noise remains independent, reducing signal-to-noise ratio

Perfect Multicollinearity

- Definition: One variable can be perfectly represented by a linear combination of others
- Example: Including both Celsius and Fahrenheit temperature variables

Near Multicollinearity
- Definition: Strong correlation exists between variables, but not a perfect linear relationship
- Example: Age and years of work experience, income and education level

### Cons:

Unstable Coefficient Estimates

- 想象两个几乎相同的变量X₁和X₂：模型难以确定影响Y的"功劳"应归于X₁还是X₂.可能今天认为X₁很重要，明天又认为X₂更重要, 类似于两个几乎相同的工人，难以确定谁的贡献更大.
- example:In linear regression, coefficient estimates are: $\boldsymbol{\beta} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$, when variables are highly correlated, $\mathbf{X}'\mathbf{X}$ matrix becomes nearly singular, making its inverse highly unstable.

Reduce Statistical Significance

- 系数的方差公式为 $\text{Var}(\boldsymbol{\beta}) = \sigma^2(\mathbf{X}'\mathbf{X})^{-1}$

- High multicollinearity makes standard errors larger, which makes t-values smaller (not larger), since: $t = \frac{\beta}{\text{SE}(\beta)}$

- This leads to higher p-values and reduced statistical significance.

### Detection

**Condition Number and Eigenvalue Analysis**
- Condition number: Ratio of the largest eigenvalue to the smallest eigenvalue
- Criterion: Condition number >30 indicates moderate collinearity; >100 indicates severe collinearity
- Advantage: Can detect collinearity among multiple variables

Correlation Matrix
- Method: Calculate correlation coefficient matrix between features
- Criterion: Absolute correlation coefficient >0.7 usually indicates potential issues
- Limitation: Can only detect pairwise linear relationships

### Solution

Variable Selection and Removal
 - Correlation screening: Keep only one variable from highly correlated groups
 - Stepwise regression: Use forward, backward, or stepwise methods to select variables
 - **Domain knowledge based**: Choose more important variables based on business understanding

Feature Engineering Methods
- Principal Component Analysis (PCA): Transform correlated variables into orthogonal principal components
- Factor Analysis: Identify underlying common factors to reduce dimensionality
- Variable combination: Create composite indices or ratio variables

Regularization Techniques
 - Ridge Regression (L2 regularization): Adds penalty to all coefficients, especially effective against multicollinearity
 - Lasso Regression (L1 regularization): Can compress some coefficients to zero, achieving feature selection
- Elastic Net: Combines advantages of L1 and L2 regularization

## 3.High Cardinality variables

高基数指的是一个分类变量有大量独特的类别值。例如，"国家"、"城市"、"产品代码"或"客户ID"等字段可能有数百、数千甚至数百万个不同的值。

High cardinality refers to categorical variables with a large number of unique category values. For example, fields like "country", "city", "product codes", or "customer IDs" might have hundreds, thousands, or even millions of different values.

### Problems caused by high-cardinality variables

- Curse of dimensionality: One-hot encoding produces too many features
- Overfitting risk: Rare categories may cause the model to memorize rather than learn
- Computational burden: Increases memory and processing time requirements
- Poor generalization: Difficult to handle unseen categories in new data.

### Methods for Handling High Cardinality

Frequency-based Grouping

- Keep top N most frequent categories, group others as "Other"
- Implementation: Keep categories appearing above a threshold, classify the rest as "Other"
- Pros:
  - Simple implementation, Intuitive and transparent
  - Preserves informative categories
  - Good stability: Does not depend on target variable, no data leakage
  - Compatible with various models: Applicable to almost all types of models

- Cons:
  - Information loss: Valuable information in rare categories may be ignored
  - Homogeneous treatment: All rare categories are treated equally, ignoring differences between them
  - Single grouping logic: Relies solely on frequency, **lacks business insights**

2. Target Encoding

- Replace categories with the mean of the target variable
  - Pros: Captures relationship with the target
  - Cons: Prone to overfitting, needs regularization
- Pros
  - Captures predictive relationships: Directly utilizes the relationship between categories and target variable
  - Good dimensionality reduction: Converts categorical variables into single numerical features
  - Smooth processing: Can handle rare categories through techniques like Bayesian smoothing
- Cons:
  - Overfitting risk: Easily leads to overfitting on training data
  - Target leakage: Improper implementation leads to target information leakage
  - Sensitive to noise: Performance decreases significantly when labels have noise
  - Requires complex validation: Usually needs cross-validation or special encoding strategies
  - New category handling: Unseen categories in test set require special treatment


3. Feature Hashing

- Feature Hashing (also known as "The Hashing Trick") is a technique that maps high-cardinality categorical variables to a fixed-dimensional vector space without storing a category-to-index mapping table

  - For a category value $c$ in a categorical variable $x$:

  - 计算哈希值：$h(c) = \text{hash}(c)$

  - 确定索引位置：$i = h(c) \mod n$，其中 $n$ 是目标维度 (Set position $i$ in the feature vector to 1 (or other value))
- Hash Collisions and Handling
  - Collisions occur when two different category values map to the same index position.
  - Increase the target dimension size to reduce collision probability (usually set to 2-10 times the number of categories).
  - When number of unique values is super big, we will use 5 - 10% of the number of unique values in hashing approach, we will of course need to endure some collision, but will save 95% of memory, and training speed significantly improved.
