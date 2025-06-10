
## Tell me about a time when you disagreed with a teammate

- NL2SQL project, 最困难的地方其实就是语义理解，比如F150，或者atchinson的sales，很难去和value连接起来。
- my idea is explicitly dictionary + embedding. While another one of our technical experts says we should only do explicitly.
  - There is no perfect answer here. Explicit only is easier to launch but not super compatible, and the embedding is more difficult.
- What we did is to discuss with the business partner and collects the potential scope of this proj, which is so big. If all explicit, is not gonna work or too hard.
- It took 2 extra weeks to implement but make it more scalable. 

nothing personal，dont feel being challenged

no level superiority, 正常交流，只是idea的交流

## Describe a challenging project you worked on

- Trade-in appraisal model, 需要准，而且最开始不知道怎么做，end to end都得靠自己，e-Commerce 协作
  - data选哪个, data乱, strict accuracy requirements, cold start for new models, business/tech gap.
  - clarify requirements and demonstrate model impact to non-technical stakeholders.
  - Model complexity vs interpretability: Customized XGBoost vs 更简单但可解释的linear model
  - Data quality vs coverage
- e2e development and test, from data collection to final development.
- Built custom data cleaning & feature engineering pipeline; designed and deployed a customized XGBoost API on GCP.
  - And we also have the customized XGBoost API.
- Result in launched in all Ford dealers in 10 regions and batch forecast in FMCC app.
- Globally RMSE is ~ 2000 USD, and MAE is ~1400, for Ford is even smaller ~1700 Ford,  ~1200. 

## How do you handle tight deadlines and competing priorities?

- evaluate priorities
  - 我用impact vs urgency matrix来评估所有任务
  - 区分哪些是hard deadline vs soft deadline，哪些是业务critical
  - 识别dependencies - 什么必须先完成才能unblock其他工作
- 拆分大项目into small pieces
- 主动communicate tradeoffs， prioritize以后另一个会delay多久
  - team strength去分配任务

例子

- Impact analysis: NL2SQL虽然uncertain，但potential impact更大，可能带来新的business value
- Dependencies: SF project是enhancement，可以稍微delay；但NL2SQL的opportunity window很短
- Resource allocation: 我把80%时间投入NL2SQL POC，20%时间maintain SF project progress
- 把NL2SQL分解成MVP components，focus在最能demonstrate value的parts
- 主动和SF stakeholders沟通，解释trade-off和新的timeline
- 快速build了demo pipeline来prove concept feasibility"

## Tell me about a time you failed and what you learned

- 这个说SF，2023年做的SF sales forecast，我主要在做sales units forecast并deliver to executive management,但是我们offline结果其实不错，但launch了以后就很一般，很多时候都是我们field ops team手动改
- I realize now that I focused too much on historical accuracy and didn't adequately plan for distribution shifts in live data. 
- field ops、executive沟通反馈
- 后面发现是target shift，对于历史上的shifting我们加了一些flag去表示，但对于未来的flag我们其实是不知道的。这个东西我们没法oversee，让我意识到了这个我们monitoring的重要性，所以我们开始monitor sales dist的KS test.一直报alert，意识到了我们没有normal month，我们之前做的更多的是对于test，但test和未来其实也是不一样的。
- 所以我们又开始加上lag term + trend variable + 做了一个新的tempo model
- 现在performance挺好的，executive现在信任我们

## Describe a situation where you had to learn something new quickly

- 在trade-in appraisal项目中，我们发现Ford之前的XGBoost系统没有migrate到GCP，整个team对GCP也比较new。同时我们希望对API有更好的control和customization能力
  - xgboost API customization: confident index, stratified validation sampling, monotone const in training, flexible HPO (given yaml, what algorithm you want, hist or GPU_hist)

- 我需要快速学习GCP ecosystem并build我们自己的XGBoost API，确保既能满足Ford的accuracy requirements，又能给我们flexibility for future enhancements.

- Vertex AI, Cloud Run, API Gateway等services, how XGBoost deployed in GCp, how HPO in vertex AI

- Learning while practicing

## Tell me about a time you mentored someone / helped a teammate grow

- 在trade-in appraisal项目中，我们来了一个new grad junior engineer。我发现她很talented但缺乏confidence，不太敢ask questions，这让我想起了自己刚入职时的经历

- onboard到这个complex project，同时build up her confidence.
  - Knowledge transfer (meeting告诉她e2e steps)
  - Accessibility (office hour)
  - Progressive complexity (simple to hard)
  - 强调没有stupid questions
- SDE background so XGBoost and Automotive domain Knowledge.
- GCP knowledge.

## Influence without authority, Took initiative

### Took initiative

- 主动识别opportunity，自己决定invest time做POC
- 没有先去ask permission，而是直接开始build initial version
- 用working demo来prove value，rather than trying to convince with words

### Influence without authority

- The SF report is not that hurry as our current stock level is pretty healthy
- 通过concrete results而不是talking来influence decisions
- 用initial version demonstrate feasibility和business value
- Let the work speak for itself

## Why samsung 啊？ （我有的选么，明知故问）

- traditional industry into the tech sector
  - I currently more like saving costs and optimizing operational efficiency
  - 更希望直接干create business value的role

- 有庞大的数据规模和真实线上场景
  - 在tradein，我非常意识到数据的重要性
  - large scale 有了，更像了解real time 来solve更complex的problems
  - leverage Samsung的ecosystem data来optimize ad targeting
- building scalable ML solutions, collaborating cross-functionally, and driving real business outcomes gives me the right foundation to contribute and grow at Samsung Ads.
- 我相信我在building production ML systems的experience，加上对Samsung ecosystem data的excitement，能帮我在这个role里create real impact

## explain a complex technical concept to non-technical stakeholders

- pick the understandable metrics, 
- In Tradein program,给business解释model的performance and limitations
  - Modelyr limitation, RMSE and MAE and the potential risk and opportunity.
- smart calculator that learns from data
  - The model we are using is like a chain of experts, each one learns a small piece of data. And we learn the historical pattern using lots of them. 每个专家都从前一个专家的'错误'中学习. Boosting is boosting of the previous sub-model, try to correct the error. And extreme is the optimuization techniques make it fast and accurate.
- Still one of the SOTA in grid data

##  balance technical debt vs new features

- In SaleForesight, when I realized the target shifting problem and want to fix it, there is an emerging request to build an arrival model to make the report more complete.
- What I need to do is to determine to fix the reliability issue first or continue on the new deliverables to make the entire report better.
- I was evaluating the risk and impact of two pathways.
  - 先fix reliability - safer但delay executive deliverable
  - 先build new feature - 满足business需求但继续rely on manual fixes.
  - The technical debt is there, problematic but we have field ops to handle it.
    - Discuss the gap, and give them a guardrail to shorten their manual work real work.
  - The new feature has more exposure, more impact as it is part of the report sent to all sales sector managements including CEO, benefit all our team members.
- So we deliver the stock forecast in 4 weeks and goes back to tech debt  