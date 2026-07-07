# Business Recommendations

## Executive Summary

Using a dataset of 30 000 historical credit‑card customers, we built models to predict whether a customer will default on their payment in the following month.  Advanced models such as Random Forest and XGBoost significantly outperform a logistic regression baseline.  The Random Forest achieved an accuracy of 94 % and captured more than 80 % of actual defaults with high precision.  Key drivers of default include past payment behaviour (particularly recent delays), high outstanding bill amounts relative to the credit limit and low ratios of payments to bills.  The following recommendations translate these findings into actionable strategies for the business.

## Key Findings

1. **Payment history is the strongest predictor of default.**  Customers who were late in paying in recent months (high `PAY_*` values) have a significantly increased risk of future default.  In particular, delays in the most recent month (`PAY_0`) carry the highest weight.
2. **High utilisation and low payment ratios indicate risk.**  Clients with large outstanding balances relative to their credit limits (`LIMIT_BAL`) and those who pay only a small fraction of their bills (`payment_ratio`) are more likely to default.  Conversely, customers who consistently pay more than the minimum due have a lower risk.
3. **Demographic factors matter less.**  Variables such as age, education, gender and marital status show weaker correlations with default compared with behavioural variables.  Thus, focusing on behavioural signals yields better predictive power while minimising reliance on sensitive demographic attributes.
4. **Class imbalance is manageable.**  Although only ~22 % of customers defaulted, tree‑based models still achieved strong recall without extensive re‑sampling.  Monitoring precision and recall remains important when operationalising the model.

## Business Recommendations

1. **Implement a risk‑based monitoring system.**  Deploy the Random Forest model as a decision‑support tool to assign a default probability to each active customer.  Customers with high predicted probabilities should enter a proactive monitoring queue for personalised outreach.
2. **Offer early‑intervention programs.**  For high‑risk customers, provide payment reminders, flexible repayment plans or financial counselling.  Intervening in the month before a potential default could reduce charge‑offs and customer churn.
3. **Adjust credit limits prudently.**  When the model identifies clients with high utilisation and repeated delays, consider temporarily reducing credit limits or freezing increases until repayment behaviour improves.  Conversely, clients with low utilisation and strong payment ratios can be candidates for limit increases.
4. **Enhance customer communication.**  Use predictive scores to prioritise communications such as SMS reminders or app notifications.  Clear messaging that outlines upcoming due dates, minimum payments and potential penalties can encourage on‑time payments.
5. **Continuous model monitoring and retraining.**  Periodically retrain the model with recent data to capture shifts in consumer behaviour or economic conditions.  Monitor metrics such as recall and false‑positive rates to ensure that performance does not degrade over time.

## Risk Assessment

Implementing a predictive model introduces operational and regulatory considerations:

- **Fair‑lending compliance:** Although demographic variables were included for analysis, final decisioning should avoid direct use of sensitive attributes (e.g. gender, marital status) to comply with fair‑lending regulations.  Tree‑based models can pick up proxies for these attributes, so periodic bias audits are recommended.
- **Model risk management:** Document the model’s design, performance, data sources and limitations.  Set up an approval process for changes and regularly test the model against back‑testing and stress scenarios.
- **Data privacy:** Ensure that customer data used for modelling is handled securely and anonymised appropriately when sharing insights outside the credit‑risk team.

## Future Opportunities

1. **Incorporate additional behavioural data.**  Transaction‑level information, credit‑bureau scores or digital‑footprint data could enhance predictive power and provide earlier signals of distress.
2. **Explore alternative algorithms.**  Gradient‑boosting frameworks (LightGBM, CatBoost) or deep‑learning models may yield incremental gains, especially when additional data is available.
3. **Develop explainability tools.**  Use SHAP values or partial dependence plots to interpret model predictions for individual customers and to support transparent decision‑making.
4. **Deploy a customer‑facing dashboard.**  Integrate the model into a dashboard that visualises customer risk segments, enabling credit managers to drill down into the factors driving an individual’s score and to monitor portfolio‑level trends.

By acting on these insights, the credit‑card business can reduce default losses, allocate resources more efficiently and improve customer satisfaction through tailored engagement.