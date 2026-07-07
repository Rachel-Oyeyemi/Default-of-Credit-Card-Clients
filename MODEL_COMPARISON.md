# Model Comparison

This document compares the performance of the baseline model (Logistic Regression) against more advanced algorithms (Random Forest and XGBoost) for predicting credit el‑card defaults.  All models were trained on the engineered dataset with a 70/30 train–test split stratified by the target.  Hyper‑parameters were kept at reasonable defaults; further tuning could yield additional improvements.

## Baseline Model – Logistic Regression

Logistic regression provides an interpretable baseline and is often used in credit‑risk modelling.  It assumes a linear relationship between the log‑odds of default and the input features.  In our evaluation, the logistic regression model achieved **77.9 % accuracy** but exhibited **zero precision and recall** because it failed to classify any observations as defaults.  This occurs when the default class is under‑represented and the model’s threshold (0.5) is too high.  Lowering the decision threshold or applying class weights could improve recall.  Nevertheless, logistic regression’s **ROC–AUC of 0.653** indicates the model has some ability to rank‑order risk, but its classification at the default threshold is poor.

## Advanced Model – Random Forest

A Random Forest is an ensemble of decision trees that reduces variance through bagging and random feature selection.  It handles non‑linear relationships and skewed distributions without requiring feature scaling.  The Random Forest achieved **94.4 % accuracy**, **92.7 % precision** and **80.9 % recall**, with an **F1‑score of 0.864** and **ROC–AUC of 0.961**.  These results indicate that the model effectively identifies defaulting customers while maintaining a low false‑positive rate.  The ensemble nature of the forest provides robustness to outliers and multicollinearity.

## Advanced Model – XGBoost

XGBoost is a gradient‑boosting algorithm that builds trees sequentially to correct the errors of previous trees.  It offers excellent performance on structured tabular data.  Our XGBoost model delivered **87.8 % accuracy**, **84.4 % precision** and **55.0 % recall**, with an **F1‑score of 0.666** and **ROC–AUC of 0.911**.  Compared with Random Forest, XGBoost has slightly lower accuracy but can be tuned to improve recall (e.g. by adjusting the learning rate, maximum depth, class weights or evaluation metric).  Its moderate recall indicates that while it catches more defaults than logistic regression, it misses some that Random Forest captures.

## Discussion

| Model                 | Accuracy | Precision | Recall | F1‑Score | ROC–AUC | Pros | Cons |
|-----------------------|---------:|----------:|------:|---------:|--------:|------|------|
| **Logistic Regression** | 0.779   | 0.000    | 0.000 | 0.000    | 0.653   | Highly interpretable; fast to train | Poor classification at default threshold; assumes linear relationships; sensitive to class imbalance |
| **Random Forest**      | 0.944   | 0.927    | 0.809 | 0.864    | 0.961   | Captures non‑linear interactions; robust to outliers; high recall and precision | Larger model size; less interpretable; training can be slower |
| **XGBoost**            | 0.878   | 0.844    | 0.550 | 0.666    | 0.911   | Powerful gradient‑boosting algorithm; handles complex patterns | Requires careful hyper‑parameter tuning; slightly lower recall and accuracy than Random Forest |

**Recommendation:** Based on the above results, the Random Forest model offers the best balance of precision and recall and achieves the highest ROC–AUC.  Its ability to capture complex patterns without extensive tuning makes it suitable as the primary model for deployment.  Logistic regression can still be used to provide interpretable insights into feature importance and baseline risk scores, while XGBoost may yield improvements with further tuning (e.g. adjusting the evaluation metric to optimise recall for the positive class).