
# Tradein

54% of Ford/Lincoln new vehicle buyers will trade in their used vehicles.

25% of new vehicle purchases are targeted to go through Ford eCommerce portal.

2,733 dealers who have activated their trade-in preferences on eComm.

2019-2020 Fusion battery and power loss recall (plus a 15% premium).

Staging / Prod in API logging service??

monitoring dashboard.

API Gateway, Apigee-> victorOps->On-call support.

## Questions: 

1. Can you briefly describe the business problem your Trade-in vehicle valuation or forecasting platform is designed to solve? What were the key pain points for the business?
    - Problem
      - lack of transparency and consistency in vehicle trade-in pricing
      - manual pricing was highly subjective and varied across dealerships.
    - Solution
      - deliver fast, objective, and data-driven valuations, improving customer trust.
      - Consistency and guaranteed, the customer is guaranteed to sold the vehicle at our appraisal price.
    - Impact
      - Now has been launched to National Ford Dealers and Ford Credit App to all financed user.

2. Can you describe the end-to-end architecture and workflow for your automated Trade-in valuation platform? Please highlight how data flows from ingestion to model deployment and online serving.
    - Data Ingestion:
      - fully automated from raw auction data extraction to feature store ingestion, with all steps version-controlled and logged for traceability
      - We had Dealer recorded tradein vehicle information, but the information is biased (depends on the dealer negotiation skills)
      - So we decided to use the data from action site, auto auction is competitive, it is more stable and volume is a lot more. We firstly use JDPower and now autoIMS.

    - ETL / Feature Engineering:
      - Extract auction data from MongoDB, do data filteration and feature engineering including
        - remove no appraise make, exact trim match for ford vehicles, and msrp string embedding match for competitor vehicles, remove msrp / residual / mileage outliers, segment level market index and missing value filling.
        - Fully automated with logs and error or warnings.
          - Failure is an error
          - If too much missing or the missing portion within a vin9 is too big also warning.
    - Training Pipeline:
      - Cloud function containerize everything
      - Model is trained on a weekly basis
      - Use a customized XGBoost API we built to enable
        - confident index, stratified validation sampling, monotone const in training
        - Automatically monthly HPO in Vertex AI
      - Model is saved in GCS buckets, fallback mechanism.

Model Validation & Selection:

- There is no **Ground Truth**
- Most evaluation is offline by acution price
- And our online traffic is not significant to do A/B test
- We do have a small portion of holdout to serve as a complete out of sample test to compare models before we switch.
  - 还有various week over week 的file, if change too much we will send alert
  - for example vin9_id average adjusted price, if deviated more than 2% over week, we will monitor.
    - Partially switch model thru cloud run.

Model Deployment:

Online Prediction Service:

- 我们不用太worry，这个没有那么复杂，一般也就是200-300ms
- pipeline, we receive the request (already standardized) --> 直接进cloud function做process,我们cloud function是active的，存着和model相关的feature generation code,直接call end point in vertex AI，得到forecast， we do the post training forecast, for example the most important several are: color region model comparison, carfax score adjustment and residual cap.
HTTP POST request

Batch Forecast

- VIN number --> 用vin9 dictionary 给他map上，再和internal database的结果cross check一下。
- Batch forecast through our deployed model
- Upload to BigQuery and email sent.

Monitoring & Alerting:

- Monitoring Dashboard to monitor the latency, success rate, queries.
  - 展示历史趋势
- We also have Terraform on call message sent to out webex or teams chanel
  - Failed to send values
- Save our received log and sent log semi-structured into GCS and structured BQ


User Integration:

- 和eCommerce平台团队一起定义API接口规范. json to json