# Credit Scoring Business Understanding

## Basel II and the Need for Interpretable Models

The Basel II Accord emphasizes accurate risk measurement and regulatory transparency in financial institutions. Banks are required to justify how credit risk is measured and demonstrate that their models are reliable, stable, and understandable. As a result, credit scoring models must be interpretable and well-documented so that regulators, auditors, and business stakeholders can understand how predictions are generated.

Interpretable models improve trust, support regulatory compliance, and make it easier to explain lending decisions to customers and management. Proper documentation also enables reproducibility, governance, monitoring, and validation of the model over time. In regulated financial environments, explainability is often as important as predictive performance.


## Importance of Proxy Variables in Credit Risk Modeling

In many real-world datasets, a direct “default” label may not exist. In such cases, a proxy variable is necessary to approximate customer credit risk behavior. Examples of proxy indicators include missed payments, late repayments, low account balances, high utilization rates, or prolonged inactivity.

Proxy-based prediction introduces several business risks. First, the proxy may not perfectly represent actual default behavior, leading to incorrect risk estimation. Second, biased or poorly designed proxies can unfairly classify customers, resulting in discriminatory lending decisions. Third, if the proxy relationship changes over time, model performance may deteriorate, causing financial losses and regulatory concerns.

Therefore, proxy variables must be carefully selected, validated, and monitored to ensure that they reasonably represent true credit risk behavior.



## Trade-offs Between Interpretable and High-Performance Models

Simple and interpretable models such as Logistic Regression with Weight of Evidence (WoE) are widely used in financial institutions because they are transparent, easy to explain, and aligned with regulatory expectations. These models allow analysts and regulators to understand the contribution of each variable to the final prediction.

However, simpler models may have lower predictive accuracy when compared to advanced machine learning techniques such as Gradient Boosting or XGBoost. High-performance models can capture complex nonlinear relationships and interactions in the data, often improving prediction accuracy and risk segmentation.

Despite their strong performance, complex models are more difficult to interpret, validate, and document. This creates challenges in regulated environments where explainability, fairness, and accountability are critical. Financial institutions must therefore balance predictive power with transparency, governance, operational simplicity, and regulatory compliance when selecting a credit scoring model.
