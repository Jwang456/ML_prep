# Taro Course

## Essential ML concepts

### Foundation Ideas

#### ML paradigms

- supervised vs unsupervised vs reinforcement

#### Feature engineering techniques

- numerical
- categorical
- text

feature selection and transformation techniques.

#### Time series analysis

lag features, rolling avg, trend, seasonality

Random forest accumulate gini impurity decrease by feature and normalize to calc the feature importance. 

### Clear Framework

C for Conceptualize

L for Link: connect with other concepts

E for example: real world use-case

A for advantages and disadvantages

R for result, how would I measure success. 

### Red Flags

Math without context (得说出数学后面的实际用法)

Complex explanation (讲的简单点 说人话 说的intuitive)

Misapplied Approaches (用合适的model)

Corrlaton vs Causation (没懂？？？)

## ML coding (很多func我记不住啊 我都是现拷打LLM？？？)

use standard ML library yo solve a real world problem. (类似Ford面试)

### EDA exploratory data analysis

Mediocre Performance

- Basic checks for null values
- quick data summary

Stellar Performance

- thorough analysis of distributions
- Feature relationship exploration
- Potential outlier identification
- data quality assessment.

### Feature engineering

Mediocre Performance

- Normalization
- Basic one- hot encoding 

Stellar Performance (Text data怎么弄啊？？？)

- Creative feature generation
- Thoughtful feature selection
- Complex data type handling

### Model training

Mediocre Performance

- standard model with default

Stellar Performance

- Thoughtful model selection
- Hyperparameter tuning
- Ensemble approaches

### Evaluation

Mediocre Performance

- Basic accuracy metrics

Stellar Performance

- Problem appropriate metrics
- Tradeoff analysis
- Business objective alignment


### Code quality

Mediocre Performance

- Functional but messy code
- difficult to follow logic

Stellar Performance

- Clean, well organized code
- Proper documentation
- Efficient implementation

### Common follow up questions

- Model improvement
  - more features
  - more advanced architecture
- Online Evaluation
  - Latency, drift monitoring
- Production issue
  - distribution shift
  - data pipeline failures
  - How the model handle edge cases

## System Design

Problem Navigation

High-level design

Data Strategy and Feature Engineering

Modelling approach

Evaluation and monitoring.