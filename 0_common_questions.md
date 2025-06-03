## Bias-variance tradeoff

The bias-variance tradeoff is one of the fundamental concepts in machine learning that helps us understand model performance and generalization

Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias means the model makes strong assumptions about the data and consistently misses relevant patterns - it's systematically wrong. For example, trying to fit a linear model to inherently non-linear data creates high bias. This leads to underfitting.

- how far off the model's average prediction is from the true value

Variance refers to the model's sensitivity to small fluctuations in the training data. High variance means the model captures noise and random patterns that don't generalize to new data. For instance, a decision tree with no depth limit might perfectly memorize training data but fail on test data. This leads to overfitting.

The tradeoff exists because these two sources of error are inversely related:

- Simple models (like linear regression) typically have high bias but low variance
- Complex models (like deep neural networks) typically have low bias but high variance

Total Error = Bias² + Variance + Irreducible Error

Optimize the tradeoff by

- Model selection: Choose complexity appropriate for the data - use cross-validation to find the sweet spot where total error is minimized
- Regularization: Add penalties (L1/L2) to constrain model complexity, reducing variance at the cost of slightly higher bias
- More data: Generally reduces variance without affecting bias, which is why it's often the best solution when feasible

## 你训练了两个模型：

- 模型 A：准确率 95%，AUC 为 0.82
- 模型 B：准确率 92%，AUC 为 0.91

你会怎么选？请说明你的理由。

## 你训练好了一个表现良好的模型并成功上线。上线之后，业务团队反馈模型效果逐渐变差了。请问你会如何排查问题？有哪些可能原因？你会如何监控一个生产环境中的 ML 模型？

- Confirm the drop is real
- Investigate Data Issues (most common root cause)
  - Data drift
  - Concept drift: Has the relationship between features and labels changed
  - missing labels
- Check Model-Specific Issues
- Monitoring & Prevention
  - Data drift monitors.
  - Prediction distribution tracking (e.g., is prediction score mean shifting?)
  - Performance metrics dashboards

## 请你解释一下在构建机器学习模型时，为什么要进行特征选择？你常用哪些方法来做特征选择？能不能给个具体的例子

## 你上线一个CTR预估模型后，发现AUC在生产环境明显下降，你会怎么系统性定位和解决这个问题？

明确问题定义（Clarification）
- AUC下降是短期还是长期？
- 是线上实时AUC、日志回放AUC，还是有label延迟？
- 是否只在某些渠道/流量/人群下降？

When I observe a significant AUC drop after deploying a CTR prediction model in production, I approach the diagnosis systematically:

First, I clarify whether the drop is global, or only in certain segments, and check for issues with label delay or AB testing configuration.

Next, I investigate five key areas:

Data Distribution Shift: Production traffic may differ from training data. I’d compare key feature distributions (e.g. user/device/ad slot) between train/test and online logs.

Feature/ETL Consistency: I’d check that online and offline preprocessing (feature engineering, normalization, categorical mapping) are 100% aligned. Feature drift or missing features are common causes.

Label/Data Pipeline Delay: Are online metrics calculated on delayed or partial labels? Is all feedback being logged and joined correctly?

Model Serving Consistency: I’d confirm that the exact model weights and inference code are being loaded in production, and match offline experiments.

Business/System Changes: Any new AB buckets, traffic shifts, or recommendation logic changes?

In practice, I usually start by comparing the online and offline feature values for sampled requests, visualize feature histograms, and review logs. If features match, I check model versioning, label delay, and business changes. This structured approach has helped me catch issues like mapping bugs, data delay, and traffic split mistakes in previous roles.