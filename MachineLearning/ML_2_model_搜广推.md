# ML (NN) structures

## 负采样 / Negative Sampling

实际也是down sampling的一种，我们在为training选择合适的negative

downsample 的  (user, item) pairs，具体是item部分

### 为什么需要负采样？/ Why Negative Sampling?

- 在推荐系统中，我们只有正样本（用户实际交互的物品），但缺少明确的负样本。
- 隐式反馈特点:
  - 用户点击了某个视频 → 正样本 ✅
  - 用户没点击某个视频 → 不知道是负样本还是未曝光 ❓

### method 1: take all non-interaction as negative samples

Negative samples = All items - Interacted items

- negative samples >> positive samples (extremely imbalanced), massive calculation.

### Method 2: Random Negative Sampling 随机负采样

$$P(j|u) = \frac{1}{|I \setminus I_u^+|} \quad \forall j \notin I_u^+$$

- $I_u^+$: 用户u的正样本集合
- $I \setminus I_u^+$: 所有物品减去正样本集合

从用户不喜欢的（没interact）中随机采样，全标成0， 虽然会错杀用户可能感兴趣的内容，但根据big number theory，整体正确。
- 单个样本可能标错，但千万个样本的平均效果趋于正确

### Method 3: Popularity-based Negative Sampling / 基于流行度的负采样

$$P(j|u) = \frac{(\text{count}_j)^\alpha}{\sum_{k \in I \setminus I_u^+} (\text{count}_k)^\alpha}$$

- $\alpha = 0$: $P(j|u) = \frac{1}{|I \setminus I_u^+|}$
- $\alpha = 1$: $P(j|u) = \frac{\text{count}_j}{\sum_k \text{count}_k}$
- $\alpha = 0.75$: $P(j|u) = \frac{(\text{count}_j)^{0.75}}{\sum_k (\text{count}_k)^{0.75}}$

这更接近真实场景：用户需要在"真正感兴趣的内容"和"大众热门内容"之间做选择

### Method 4: Hard Negative Mining / 困难负采样

$$j^* = \arg\max_{j \notin I_u^+} \text{score}(u,j)$$

选择模型当前预测分数最高，但用户实际未交互的物品

就像"针对性训练"：专门挑选模型最容易判断错误的案例来训练

一般先用Random + Popularity sampling做出forecast，再用hard negative mining做调整

### Method 5: Hybrid Strategies / 混合策略

$$P_{\text{hybrid}}(j|u) = \lambda_1 \cdot P_{\text{random}}(j|u) + \lambda_2 \cdot P_{\text{popular}}(j|u) + \lambda_3 \cdot P_{\text{hard}}(j|u)$$

$$\lambda_1 + \lambda_2 + \lambda_3 = 1, \quad \lambda_i \geq 0$$

随机负采样 + 基于流行度的负采样 + 困难负采样的组合，可以将$\lambda_3$ shrink to 0 for cold start problem in hard negative mining

### Evaluating Negative Sampling Quality / 评估负采样质量

好的负采样应该满足

- Challenging但不impossible: 不能太容易，也不能太难
- Representative: 反映真实的混淆场景
- Diverse: 保持多样性，避免过度偏向某类物品

#### Average Negative Score 负样本平均预测分数

$$\bar{s}_{\text{neg}} = \frac{1}{|N|} \sum_{j \in N} \text{score}(u,j)$$

- 太高 like > 0.7 --> negative sample 质量差
- 正好 like between 0.3 and 0.5 --> negative sample 适中挑战性

#### Separation / 正负样本分离度

$$\text{Separation} = \bar{s}_{\text{pos}} - \bar{s}_{\text{neg}} = \frac{1}{|P|}\sum_{i \in P} \text{score}(u,i) - \frac{1}{|N|}\sum_{j \in N} \text{score}(u,j)$$

or express as

$$\Delta s = \mathbb{E}[s_{\text{pos}}] - \mathbb{E}[s_{\text{neg}}]$$

- 分离度太大太小都不行，太大(> 0.4), 信号太强可能太简单，太小(< 0.1)，模型学习困难
- 理想区间 0.2 - 0.4

#### Diversity / 负样本类别多样性

$$\text{Diversity} = |\{c(j) : j \in N\}|$$
$$\text{Diversity}_{\text{norm}} = \frac{|\{c(j) : j \in N\}|}{|C|}$$

#### 好的负样本 = 适度挑战 + 清晰区分 + 多样覆盖 

## Two-Tower Architecture

Normally 用binned groups当作feature, 重点是速度快和recall好就可以，fine-ranking后面再做。

Two-Tower模型把推荐问题分解为用户表示学习和物品表示学习两个独立的子问题。

**Core Concept**: Two-Tower decomposes recommendation into two independent subproblems: user representation learning and item representation learning.

``` two tower
User_Embedding = f_user(x_user)
Item_Embedding = f_item(x_item) 
score(u,i) = sim(User_Embedding_u, Item_Embedding_i)
```

Core Advantages:

- Scalability / 可扩展性:
  - Item embeddings can be pre-computed offline
  - User embeddings computed in real-time / 用户embedding实时计算
  - Supports millions of items in catalog / 支持百万级物品库

- Engineering-Friendly:
  - Simple deployment, two models can be updated independently / 部署简单，两个模型可独立更新
  - Online serving only requires ANN search / 在线服务只需ANN搜索
  - Low latency (<10ms) / 延迟低 (<10ms)

- Good Performance:
  - Excellent performance in recall stage / 在召回阶段表现优秀
  - Learns meaningful user/item representations / 能学到有意义的用户/物品表示


**User Tower**
$$h_{user}^{(l)} = \sigma(W_{user}^{(l)} \cdot h_{user}^{(l-1)} + b_{user}^{(l)})$$
**Item Tower**
$$h_{item}^{(l)} = \sigma(W_{item}^{(l)} \cdot h_{item}^{(l-1)} + b_{item}^{(l)})$$

- Normally both towers are MLPs (Multi-Layer Perceptrons)

**Final Embeddings**
$$\text{User\_Embedding} = h_{user}^{(L)}$$
$$\text{Item\_Embedding} = h_{item}^{(L)}$$

**Similarity (Cosine)**
$$\text{sim}(u,i) = \frac{\text{User\_Embedding}_u \cdot \text{Item\_Embedding}_i}{||\text{User\_Embedding}_u|| \times ||\text{Item\_Embedding}_i||}$$

- 因为cosine function归一化了embedding的magnitude (because of the division of norm)

**Pairwise Loss**

$$L = \sum_{(u,i,j)} \max(0, \text{score}(u,j) - \text{score}(u,i) + \text{margin})$$

### Two-tower details

- Use User tower to understand user behavior
- Use Item tower to understand the items

#### Feature Engineering

- User Tower Features:
  - Demographics (年龄、性别、地理位置)
  - Behavioral features (点击、购买、浏览历史)
  - Contextual features (时间、设备、场景)

``` user tower
Input → Embedding Layer → Dense Layer 1 → Dense Layer 2 → ... → User Embedding (128/256 dim)
```

- Item Tower Features:
  - Content features (类别、标签、描述文本)
  - Statistical features (热度、CTR、转化率)
  - Temporal features (发布时间、趋势度)

``` item tower
Input → Embedding Layer → Dense Layer 1 → Dense Layer 2 → ... → Item Embedding (128/256 dim)
```

想象两座独立的塔:

- 左塔: 专门处理用户信息，输出用户向量
- 右塔: 专门处理物品信息，输出物品向量
- 桥梁: 通过相似度计算连接两塔

这种分离设计的好处是可以独立计算和缓存embeddings

### Loss Function 

✅ Ranking tasks: learning relative preference order

#### Bayesian Personalized Ranking Loss

$$L_{\text{BPR}} = -\sum_{u} \sum_{i \in I_u^+} \sum_{j \in \text{Neg}(u)} \log \sigma(\text{score}(u,i) - \text{score}(u,j))$$

#### Pairwise Loss

$$L_{\text{pairwise}} = \sum_{u} \sum_{i \in I_u^+} \sum_{j \in \text{Neg}(u)} \max(0, \text{score}(u,j) - \text{score}(u,i) + \text{margin})$$

- Negative sampling: $\text{Neg}(u)$ = sampled negative items
- Full negatives: $\text{Neg}(u)$  = all non-interacted items
- 第一个sum u是sum by all users
- 每一个L大于1实际上是惩罚 for positive没有明显区分于negative

#### Binary Cross Entropy

$$L_{\text{BCE}} = -\sum_{(u,i) \in D^+} \log \sigma(\text{score}(u,i)) - \sum_{(u,j) \in D^-} \log(1-\sigma(\text{score}(u,j)))$$

Two-Tower的通用训练目标

### Training process

Two-Tower是联合训练的，User Tower和Item Tower是相互依赖的！

- User embedding的好坏依赖于与Item embedding的配合
- 无法完全独立评估User Tower

**设计理念**: 两个独立的tower

**训练现实**: 必须联合优化，无法独立评估

这是架构简洁性 vs 可解释性的trade-off。工业界选择了前者，因为：

- 工程部署简单
- 效果足够好
- 可扩展性强

Key Understanding:

- Item信息和flag联合构成User Tower的"结构化目标"
- Item info + flag jointly serve as "structured target" for User Tower。
  - User Tower视角: 我要学会与这个item相似
  - Item Tower视角: 我要学会与这个user相似

#### Forecasting

- User triggered the inferencing (by opening the page, refresh etc)
- Collecting user features
  - Static features: like age, gender, city etc (can be cached)
  - dynamic features: like recent k clicks, recent ignores, device, location, current session dehavior
- Feed user feature thru user tower to get the latest user embedding
- Utilize **ANN**: Use the user embedding to find the top k (maybe 500) items from all item embeddings
- Feed the top k embeddings to a deep-cross network for fine-ranking
- send ads based on the DCN ranking.


### Questions

1. Q1. Two-Tower相比传统协同过滤有什么优势？

- Scalability: 支持百万级用户和物品
- cold start: 可以处理新用户/新物品
- various features: 可以融入多种特征类型
- real time: 用户embedding可实时计算

2. Q2. 为什么不用一个大模型直接建模user-item交互？

- Computational complexity: 100万用户×100万物品=10¹²组合，不现实
- storage issue: 无法存储所有组合的参数
- real time requirement: 无法在毫秒级完成inferencing

3. 两个Tower的网络结构通常是什么样的?
```
User Tower: Embedding → MLP(512→256→128) → Vector Normalization
Item Tower: Embedding → MLP(512→256→128) → Vector Normalization
```

4. Why 用cosine similarity而不是dot product

- Normalization: cosine不受embedding magnitude影响
- stability: 避免某些item因为magnitude大而总是被推荐
- Interpretability: 纯粹的方向相似度，更有意义

5. Two-Tower如何训练？损失函数是什么？

6. 负采样策略有哪些？各自优缺点

- 随机采样: 简单，但负样本可能太容易
- 热门采样: 更realistic，但可能bias向热门item
- 困难采样: 效果好，但计算成本高

7. 如何判断Two-Tower训练得好不好？

- 离线指标: Recall@K, NDCG@K, AUC
- Embedding质量: 相似用户embedding相似
- A/B测试: 最终的线上效果

8. 线上服务流程是怎样的？

 - 离线: 预计算所有item embedding，构建ANN索引
- 在线: 实时计算user embedding → ANN搜索 → 返回TopK

9. 如何保证低延迟？

- 预计算: item embedding离线计算
- 缓存: 静态特征缓存

10. Two-Tower的核心限制是什么？

- 交互建模弱: 无法学习复杂的user-item交互模式
- 特征交叉受限: 两个tower间信息交流有限
- 联合训练: 虽然架构独立，但训练必须联合

11. 如何解决Two-Tower的交互建模问题？

- 两阶段: Two-Tower做召回，复杂模型做精排
- Cross-Network: 在两个tower间添加交叉连接
- Multi-Tower: 增加cross tower建模交互

12. 冷启动问题如何解决？

- 新用户: 基于人口学特征、热门推荐、规则引导
- 新物品: 基于内容特征、相似物品、专家标注
- Warm-up策略: 快速收集初始反馈

## Approximate Nearest Neighbors

Help me recall the top K items / Ads.

ANN的核心思想: **Trade-off: 精度 vs 速度**

- Exact Search:     100% 精度, O(n) 时间
- Approximate:      95-99% 精度, O(log n) 时间

在推荐系统中，用户不会感知到第50名和第52名的差别, 但会感知到100ms vs 10ms的延迟差别.

### 空间分割 (Space Partitioning)

#### FAISS IVF

核心思想：空间分区 + 倒排索引

- 把100万个向量聚类成1000个区域（每个区域约1000个向量）
- 查询时只搜索最近的10个区域（1万个向量 vs 100万个向量）
- 速度提升100倍，精度损失5%

为什么受欢迎？

- 平衡性好: 速度、精度、内存都不错
- 支持更新: 可以增量添加新向量
- 参数简单: 主要就两个参数调优
- 工程成熟: Facebook开源，生态完善

典型性能

-构建时间: 5-15分钟
-查询延迟: 2-5ms
-召回率: 95-98%
-内存占用: 1.5倍原始数据

#### FAISS HNSW

核心思想：分层导航图

- 构建多层图结构：上层稀疏导航，下层密集搜索
- 类似高速公路系统：先走高速到附近，再走小路到精确位置
- 贪婪搜索：每一步都走向最近的邻居

为什么性能最好？

- 查询极快: 0.5-2ms，比IVF快2-3倍
- 精度最高: 98-99.5%召回率
- 算法先进: 基于小世界网络理论
- 适合高QPS: 并发性能好

典型性能

- 构建时间: 20-60分钟（较慢）
- 查询延迟: 0.5-2ms（最快）
- 召回率: 98-99.5%（最高）
- 内存占用: 3-5倍原始数据（较大）

## DCN (Deep & Cross Network)

**Deep & Cross Network，是一种精排模型,ranking model, 专门用于学习特征交叉**
在ANN搜索后选出500个候选，用DCN精排 (fine-ranking)

Why we need DCN model?

- The limitation of Two-Tower
  - similarity = user_vector · item_vector(只能学习到这种简单交互)
- DCN可以显式学习这些交叉特征
  - 子哦对那个学习所有重要features的combinations (interaction)

```
场景: 从500个候选中精确排序出20个
输入: 用户详细特征 + 商品详细特征 + 上下文
输出: 精确的相关性分数
要求: 精度高, 特征交叉丰富
```

### DCN components

When to use DCN: 精排阶段，不是召回阶段

pros:

- Explicit feature crossing: good Interpretability.
- coefficient efficiency: Cross Network参数量线性增长
- 工程友好: 结构清晰，易于实现

Cons:

- Computational complexity: 比Two-Tower慢很多
  - DCN需要process Top K个candidates, 假设其他一切都一样，也慢50倍。
  - 也无法pre-calc出来
- Features dependent: Need sophisticated feature engineering.
- storage spending: 需要存储更多特征信息

#### Cross Network 交叉网络

这是DCN的核心创新，不是call任何其他model的API, 专门用于显式学习特征交叉

input vector = x_0 = [user_features, item_features, context_features]

ALL features concatenated into one vector,所有特征拼接成一个向量作为input

Explicit Feature Crossing: 每一层都与原始输入x_0进行交叉

- $x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l$
- User features cross with other user features / 用户特征与其他用户特征交叉
- Item features cross with other item features / 物品特征与其他物品特征交叉
- User features cross with item features / 用户特征与物品特征交叉
- Context features cross with everything / 上下文特征与所有特征交叉
- $w_l$ is a weight vector and $b_l$ is the bias, they sum as a number
- **This element level multiplication is non-linear.**

Bounded Complexity: 参数量是O(d)而不是O(d²)，d是特征维度

High-Order Feature Interactions: L层Cross Network可以学习到L+1阶的特征交叉

权重值不可解释: 具体权重值0.1, 0.3的业务含义仍然模糊,但相比于下面这个牢弟可解释性强很多。

#### Deep Network (深度网络)

标准的MLP结构，用于学习隐式特征交互

$$h_{l+1} = σ(W_l · h_l + b_l)$$

- Good at learning complicated non-linear relationship
- 参数量大，表达能力强 (几千倍于cross network)
- inplicit crossing, bad interpretability.

#### Combination Layer (组合层)

$$y = σ(w_{cross}^T · x_{cross} + w_{deep}^T · h_{deep} + b)$$

#### DCN result

DCN is a scorer or called ranker, each iteration if will return the score.但在整体流程里，当iterate完top K items, 会返回一个tuple of (index, score), 我们返回top m by score，然后我们可以根据index定位到最终的top items (ads, videos)

### MMOE（Multi-gate Mixture-of-Experts，多任务学习主流架构）

1. 推荐/广告排序里的多目标学习（比如CTR/CVR/GMV等同时优化）


2. 多个共享的专家网络（Experts），每个任务有独立门控（Gate），决定“本任务”从每个专家网络里取多少信息

3. 比传统多任务“硬分叉”结构表达能力强（既能共享也能差异化）

4. 输入特征→共享底层MLP专家→每个任务有独立Gate（softmax输出权重）→加权sum后各自MLP→输出多目标

5. 广告排序：CTR（点击率）、CVR（转化率）、GMV（成交额）一起学，不容易互相“掣肘”又能信息共享

- Conversion Rate
- Gross Merchandise Value / Volume

6. “门控机制解决了多任务间负迁移，提升各目标效果”

#### Details

什么是负迁移 (negative transfer)

传统Multi-Task Learning, 不同task share bottom MLP layers and will make less flexible and may not be any optimal direction.

```
输入特征 → 共享MLP底层 (几层) → 分叉点 → Task1 MLP → CTR输出
                                    → Task2 MLP → CVR输出  
                                    → Task3 MLP → GMV输出
```

- 不同任务之间互相"拖后腿"，导致多任务效果比单任务还差
- 具体例子 - 广告排序:
  - Task 1: CTR预测 (点击率) - 需要学习"吸引眼球"的特征
  - Task 2: CVR预测 (转化率) - 需要学习"真实需求"的特征
- 这两个目标可能冲突：

  - 标题党内容: CTR高但CVR低
  - 实用内容: CVR高但CTR一般
- Mathmatically speaking: $∇L = α·∇L_{CTR} + β·∇L_{CVR}$
  - 如果这俩方向相反 → 参数更新方向不是任何一个任务的最优方向

#### MMOE process

MMOE 用K个MLPs (normally 4-8)

```
Input Features
       ↓
   ┌─Expert1─┐  ┌─Expert2─┐  ┌─Expert3─┐
   │  MLP1   │  │  MLP2   │  │  MLP3   │
   └─────────┘  └─────────┘  └─────────┘
         ↓           ↓           ↓
     Gate1    Gate2         Gate3
    (Task1)  (Task2)       (Task3)
         ↓           ↓           ↓
   [w11,w12,w13] [w21,w22,w23] [w31,w32,w33]
         ↓           ↓           ↓
    Weighted Sum   Weighted Sum  Weighted Sum
         ↓           ↓           ↓
    Task1_Tower   Task2_Tower   Task3_Tower
         ↓           ↓           ↓
     CTR_Loss     CVR_Loss     GMV_Loss
```

每个Expert会收到来自**所有使用它的任务的梯度**，而不是所有MLP的梯度

每个任务到底用哪几个MLPs其实是动态变化的

- 最开始对于每一个task，每一个MLP的weight (we call it gate here) is actually close to uniform distribution
- 在back propagation过程中根据weight回收到的gradient做optimization
  - 不光是每个MLP的optimization
  - 也是每个gate的optimization

### Deep Interest Network

DIN主要用在**精排阶段 (Fine-ranking)**, after retrieving top K most related items from two-tower model.

核心问题：静态兴趣表示的局限

用户兴趣的多样性：

- 用户同时对多个品类感兴趣：数码、美食、服装...
- 当前推荐item只与部分历史行为相关
- 需要动态激活相关的兴趣，抑制无关的兴趣

**DIN的解决方案：Local Activation**

- 核心思想：根据候选item，动态计算历史行为的重要性权重

```
用户历史行为序列 + 当前候选item → Attention权重 → 加权历史表示 → 拼接特征 → 精排MLP
```

- history items list通常之选最近N个, like 50-200
- Embed history list AND current items
  - history list after embedding could be 200*128 dimentional matrix
  - 当前item after embedding is 1 * 128
- feature interaction (crossing) combinations
  - Historical items feature (一个hist embedding，我们enumerate hist hist挨个来attention)
  - current item features
  - hist items features $\odot$ current item features
  - Historical items feature - current item features
  - 都是 1*128 (1 * m)维的, 拼接成 1 *4m dimension的
- 上面这一步相当于从4各方面describe我们现在正在enumerate的historical item，用MLP算一个attention的数
- enumerate完全部hist items，我们对n个hist items有了n个attention score wrt current item
- 对每个原始history embedding (每个都是 1 * m 维) * 自己的attention score
- 将所有history的embedding accross items加起来（还是1*m维）
  - 因为 m 维embedding中每个维度都代表一个语义空间，所以在此处我们实际将过去的items转换到对于m维不同category（语义）的attention（兴趣）强度
  - 不standardize是因为这里也没有gradient的问题，我们更关心绝对magnitude（数字大小代表attention大小）
- 这样我们对于K个candidates都已经有了一个m维的vector代表在不同维度的强度
- apply another MLP to convert this vector into a scalar which denotes the strength (attention, interest) that customer to this specific item.
- 根据上面的MLP output把K个candidates重新排序，然后选最开始的几个feed给customer

#### Other points

DIN能显著提升CTR/GMV等指标，尤其在用户行为密集的场景下效果突出

DIN的attention score是非归一化的，保留幅值可直接表达‘激活强度’，工程上实践证明更适合推荐场景

在DIN中我们对user history items的选择也是需要负采样的，不能全positive，这样学不到customer不喜欢啥。
采样比例的选择:

- 冷启动: 1:1 (避免负样本过多)
- 稳定期: 1:3 (标准比例)
- 精调期: 1:5 (提高区分度)

New User Cold Start / 新用户冷启动

- 新用户没有历史行为，无法计算个性化embedding
  - Based on user profile and demographics do initialization
  - Use similar user average embedding

New Item Cold Start / 新物品冷启动

- 新物品没有用户交互历史，无法学习item embedding
  - Use category combinations
  - text description embedding
  - similar item average embedding

随着interaction的增多，逐渐增加behavior weight (cold start以context / property为主)

## 搜广推 evaluation metrics

AUC (Area under Curve):非常直观的代表模型排序能力

- 衡量模型区分正负样本的能力
- 取值：0.5-1.0，越高越好
- 工业界标准：=>0.75优秀
- 优点：不受样本不平衡影响，易于理解

NDCG@K (Normalized Discounted Cumulative Gain)

- 考虑位置权重的排序质量
- 排在前面的相关结果权重更高
- 实际应用：评估推荐列表的排序效果
- 为什么重要：用户只看前几个推荐

Precision@K / Recall@K

- Precision@K：前K个推荐中正确的比例
- Recall@K：所有正确答案中被推荐的比例
- trade-off：精度和召回的平衡
- 实际意义：Precision关注推荐准确性，Recall关注覆盖完整性

#
#
#
#
#
#
#
#
#
#
#
##
#
#
#
#
##
#
#
#
#
##
#
#
#
#
##
#
#
#
#
##
#
#
#
#
#








#######################

案例/场景补充（可以作为后续追问/主动加分）
比如举例“在广告检索中如何做负采样（如曝光日志作为弱负样本、用流行广告或hard mining sample平衡曝光bias）”、“在冷启动场景怎么调整negative mining”等。

冷启动问题/Embedding泛化

适当补充：“对于新用户/新物品，可以用内容特征（profile/category embedding）初始化，后续随行为更新。”

工业A/B、线上部署经验（如果面试官关心，可以加）

比如你如何做embedding离线全量+在线增量同步、如何评估线上性能/召回率等。


Q12: 百万级物品如何做ANN搜索？
Answer:

FAISS: Facebook开源，支持GPU加速
索引类型: IVF、HNSW、PQ等
精度vs速度: 可配置approximate程度
