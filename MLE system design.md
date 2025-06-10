# System Design

latency, noisy data, evolving products requirements.

## High level summary

复述/clarify问题

画block/数据流

讲每个block怎么实现，trade-off怎么选

讨论监控、扩展、上线、故障、A/B test

应对follow-up

```
problem navigation

High level design

Data Strategy and Feature Engineering

Modeling Approach

Evaluation Deployment and Monitoring
```

## Q1 CTR model

Suppose we want to build a production-ready CTR prediction system for Samsung Ads. When a user sees an ad, we want to estimate the probability they will click on it, in real-time, at high scale.

### Offline model

1. Data Pipeline

1.1 Collecting data.

- impression logs
- Click logs
- conversion logs

1.2 Feature engineering and consistency

- Do one-hot, target and embedding for variables
- Numerical normalization and handling missing values
- Interaction
- Consistency Guarantee
  - use same code to generate the feature online / offline

1.3 Model Training 模型

- Two models: Two towers and CDN model
- 超参数调优（HPO）贝叶斯
- 分别做training，都需要negative sampling

1.4 Evaluation 评估（AUC/Calibration/AB实验/业务指标多维offline evaluation）

- Retrieval模型主要看召回率/覆盖率（Recall@k、MRR等
- Ranking模型则看最终排序质量（AUC、NDCG、CTR等

1.5 模型artifact/权重、特征字典/embedding存储（GCS/S3/特征库/embedding库）

- 存GCS / S3 里面，存savedModel and parquet
- 线上服务启动时从对象存储加载到内存/本地磁盘，或者embedding定期同步到embedding服务

1.6 模型上线部署/灰度发布/回滚机制

- 模型服务部署（Model Serving）：使用TensorFlow Serving, FastAPI, feature and embedding saved in redis or ANN libarary like Faiss
- 模型本地化部署、开启多线程/批量推理，GPU/TPU加速。
- 特征工程提前预处理，线上只做最小化数据拼接
- Canary Release 灰度发布
  - load balancer let some of customer sees the new model to validate the new model.
- Monitoring Alert
  - business metrics: CTR/CVR/GMV等核心指标实时监控，异常自动告警
  - Drift: 预测分布漂移、特征重要性变化
- Disaster Recovery & High Availability
  - 多机房部署 Multi-datacenter Deployment
    - 地域智能路由：可以提及基于用户地理位置的智能DNS
  - Data Backup & Recovery
  - Rollback Mechanism



2.inverted index filtering (from business rules)

- inverted index意思是给属性 找出相关广告
  - forward index是给出广告找出属性
  - 一般advertiser来build这个
- 初筛优先是为了高效缩小范围，后续ML和ANN检索才可落地
- 倒排交集的工程优化点是排序、bitset、缓存和并行化
- depends on how strict the targeting rules are 可能会剩下几千-几十万
- Stored in Redis (本地环境) sub-ms level retrival

3.Two-Tower retrival Model.

- Get user feature to do inferencing.
  - General features (性别、年龄、注册地、设备类型、兴趣标签等)
  - Dynmaic features (最近点击/浏览、当前上下文（时间、位置、频道等），变化快)
- Inferencing using user feature to user embedding, and then calculation similarity (cosine similarity with the pre-stored ads embedding stored in **Vector Database could be Faiss**) and retrieve the top k ads (probably 500).
  - Use ANN supported by faiss libabry to retreive top K ads

4. Fine-ranking model (精排模型)

- DCN model (explicitly feature crossing) Deep + Cross Network
  - Cross network
    - concat [user_features, item features and contextual features]
    - interaction with x_0 by multiplying new result with x_0
    - odot also introduce non-linearity and it does not need activation function.
    - TV ads的behavior 没有那么丰富,DCN就挺好的。
  - Deep network (标准MLP)
    - have activation function, parameter数量大
        - 最后在combination layer 把D and C network合起来
- Wide and deep model is also an option
  - The interaction is captured by manual feature crossing
    - 无法所有都cross,太多，太稀疏
    - 只手工交叉“已知业务相关”的强特征
- MMOE（Multi-gate Mixture-of-Experts
  - 多目标/多任务排序场景首选，能同时学多个输出（如CTR/CVR/GMV等）
  - 核心能力在于多任务建模，单目标排序下未必优于DCN
- DIN (Deep interest Network) 适合用户行为密度大的时候，很显然电视不算
  - 用attention (不同于QKV那个attention mechanism,把历史行为和当前广告动态匹配、聚焦最相关的ads by attention然后feed进MLP, see the relevancy score?)

5. calibrition

- Platt Scaling（最常用，适合logistic/二分类）
把模型输出分数（logit/概率）再送进一个简单的Logistic Regression（单层sigmoid），映射成更“可靠”的概率

7. Log collection (Async)

- 通用处理
  - 写异步队列(如Kafka) log queue里, 结构化（structured）或半结构化（semi-structured）为主
  - ingest完存BQ里，后续可以用sql query查询分析

- Request log, 一收到request我直接async处理了
- Impression log 在广告以展示出来就collect
- Click log, 用户点击广告瞬间 (带impression_id),返回backend API
  - async to kafka and BQ
- Conversion log,用户产生转化（如下单/注册）时(带impression_id and click_id)，返回backend API
  - async to kafka and BQ
- Response log， 系统给出广告推荐后, 记录推荐的广告list、预测分数、实际展示情况等
- Monitoring Logs
  - Serving log	模型推理细节	可以log模型输入输出、打分等（调优、回溯、线上A/B）
  - Error/异常log	发生异常/超时/异常流量时

8. Online learning / Incremental Learning

- 让模型能快速适应新用户/新广告/业务场景变化/数据分布漂移，减少滞后
- 在线训练样本要求标签准确、特征一致、去除噪声/作弊/异常流量，并通过采样和加权等方式保证分布与离线训练集相近，防止模型失衡和漂移
- 我们所有模型都会上传到GCS（或S3/对象存储），线上服务在启动或切换模型时自动从对象存储拉取指定版本的模型文件。Canary//Ensemble方案下，新旧模型可共存并行部署，流量通过服务层灵活分配，随时支持版本切换和回滚
  - 用SGD做gradient descent.


9. 竞价拍卖机制 (Bidding/Auction Mechanisms)

- 第一价格拍卖 (First Price Auction)
  - 最高出价者获胜，按出价付费
  - 需要考虑其他竞价者行为，通常出价低于真实估值
- 第二价格拍卖 (Second Price Auction)
  - 最高出价者获胜，按第二高价付费
  - 诚实出价是最优策略，出价等于真实估值
- 广义第二价格 General Second Price (GSP)
  - 多个商品(广告位)，每个获胜者按"下一个竞争者"的有效出价付费
- 收入优化：
  - 相比纯第二价格，GSP能产生更高收入
  - 每个位置的定价更精确反映其价值
- Incentive Mechanism
  - 鼓励广告主提升广告质量(质量分)
  - 平衡出价和相关性，提升用户体验
- 策略复杂性
  - 不像第二价格拍卖有明显的诚实出价策略
  - 需要考虑位置价值、竞争环境等多因素
- eCPM (effective Cost Per Mille)
  - 有效每千次展示成本，用来综合评估广告价值
  - 计算公式：eCPM = 出价 × 质量分

$$eCPM = Bid × CTR × CVR × Value$$

我们fine ranking完事以后，可能剩下20个广告，但我们可能只有5个广告位，并且不是所有advertiser都想投到所有广告位的(for example Nike likes big ads spot not small ones),我们应该是对每个广告位qualified的广告算eCPM然后按照eCPM排序(for example, if there are two identical ads spots only one of the left and one on the right, we will pick top 2 eCPMs) to display, and then 按照下一个排序位置的哥们的eCPM / 他自己的quality得到他应该付多少钱for this impression

## Common Questions

### 1. Cold Start

#### New User Cold Start 怎么办

- 先根据demographics, device, location etc 在历史上的user做clustering，求平均embedding进model
- cold user gets warm
  - gradually do Weighted sum/fusion, 把他自己的matrix和average embedding weighted average一下.

#### New Ads Cold Start 怎么办 (harder than user cold start)

- 收集新广告的structured+unstructured内容特征
  - unstructured: 广告文案、商品标题、图片、视频
  - bert 提取text embedding，cnn提取image embedding, 用Meta embedding generator to get the new ID embedding.
  - Meta-Learning approach，核心思想是'学会如何学习embedding
    - 在历史广告上模拟冷启动场景，训练一个embedding generator，输入是广告的内容特征(文案、类别、创意等)，输出是初始ID embedding.
    - 同时优化冷启动表现和快速适应能力，使用加权loss: α * cold_start_loss + (1-α) * warm_up_loss

### 2. Feature / Data sparsity 特征稀疏性”在推荐/广告系统中的挑战

- Embedding Lookup
  - 把每个稀疏的类别（如广告ID、tag、词语、user ID等）用一个小的稠密向量（如64维、128维embedding）表示
  - 模型训练时自动学习这些向量
- Feature Hashing 热门用户/广告单独建embedding，长尾走hash桶
  - hash函数把每个类别的原始取值（如字符串/数字）映射到固定数量的桶（buckets），比如100万（1M）个桶
    - hash to int32 and modul0 it  
  - 然后每个桶分配一个embedding，模型训练时只学这些桶的embedding
- Feature Crossing + Dimensionality Reduction
  - 特征交叉：组合多个类别特征（如“region+device”），对高阶组合再做embedding。
  - 降维：对于极高维稀疏特征，可以做PCA/SVD等降维，或者先做hashing再embedding

### 3. Feature Consistency

- Single source of truth - 特征定义统一 (代码共享)
- 用Feature Store进行特征管理
  - 离线、实时数据都从同一个feature源自动生成
  - 特征版本管理和回溯
  - integrated feature center（集成特征中心），不仅仅是“存储和管理特征”，很多场景下特征的计算/加工流程也会在Feature Store中完成或托管

### 4. 特征/数据漂移与模型feature /data drip and model robustness

- real-time distribution detection and alert
  - 对核心特征（如年龄、地域、曝光数）、模型分数（比如CTR打分）、业务标签（比如用户是否点击/购买）做实时分布统计和历史对比，如果分布“偏离历史”就报警
    - mean and variance (two sample t-test)
    - KS检验（Kolmogorov-Smirnov Test）
      - 算distribution的cdf
      - 用来判断当前分布和历史分布是不是同一个分布
    - PSI（Population Stability Index）
      - 专门用来衡量两个分布（当前vs历史）的差异程度。
      - PSI>0.25通常报警
        - 分桶，按比重差距*ln(比重之比)再加和。
- Anomaly detection trigger model roll back or retrain




## TODO:
Bidding & Auction Mechanism (完全缺失)
这是ads system的核心business logic:
- Second-price auction vs First-price
- Real-time bidding (RTB) pipeline  
- Bid optimization (maximize ROI)
- Budget pacing and delivery
- eCPM = CTR × CVR × Bid calculation

Business Metrics & Constraints (缺失)
- Revenue optimization vs User experience
- Advertiser ROI, Budget constraints
- Frequency capping, Brand safety
- Attribution modeling (view-through vs click-through)
- Lift measurement and incrementality

Scale & Performance (部分缺失)
- QPS requirements (millions of ad requests/sec)
- Latency budgets (<100ms for real-time)
- Storage scaling (billions of ads)
- Multi-datacenter deployment
- Cost optimization strategies














-----------------------------------
-----------------------------------
-----------------------------------
-----------------------------------

### Q1. Suppose we need to design a real-time ad recommendation system for Samsung TVs. When a user starts an app or channel, we need to select and deliver the most relevant ad to the user within 100ms. How would you design the system end-to-end.

```
   +---------------------+           +--------------------------+  
   |   Samsung TV (User) |  --->     |  API Gateway / Frontend  |  
   +---------------------+           +--------------------------+  
                                                |  
                                                v  
                                    +------------------------------+  
                                    |  Real-time Recommendation    |  
                                    |  Service (Prediction Layer)  |  
                                    +------------------------------+  
                                    /          |         \  
                 +--------------+  +--------------+  +-----------------+  
                 |  User Profile|  |   Ad Catalog |  | Targeting Rules |  
                 +--------------+  +--------------+  +-----------------+  
                          |  
                          v  
             +-----------------------+  
             |  Feedback Logging     | <---(async event bus)---+  
             |  & Model Retraining   |                        |  
             +-----------------------+                        |  
                                                              |  
                        +------------------------------+      |  
                        | Monitoring & AB Testing      | <----+  
                        +------------------------------+  

```