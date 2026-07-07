# Exploratory Data Analysis Report

This report summarises the main findings from exploring the **Default of Credit Card Clients** dataset.  The raw dataset contains 30 000 observations and 25 columns (one ID column, 23 explanatory variables and the target).  After converting the original Excel file to CSV, basic inspection showed that there were **no missing values** and **no duplicate rows** in the raw data.  The target variable, `default payment next month`, is binary and indicates whether a client defaulted on their credit card payment in the following month【382343060160666†L35-L64】.  The dataset is moderately imbalanced with ~22 % defaults (6 636 records) and 78 % non‑defaults (23 364 records).

## Dataset Overview

- **Row count:** 30 000 records.
- **Column count:** 25 columns including an ID, demographic features, payment history variables, bill amounts, payment amounts and the target.
- **Data types:** All features are numeric (integers for categorical codes and continuous values for monetary amounts).
- **Missing values:** None – every column in the dataset is fully populated.
- **Duplicates:** None – no duplicated rows were found during preprocessing.
- **Target variable:** `default payment next month` (renamed to `DEFAULT` in downstream processing).  The class distribution is 0 = non‑default (23 364 records) and 1 = default (6 636 records).

## Descriptive Statistics

The table below summarises basic descriptive statistics for several important continuous variables:

| Variable            | Mean (μ) | Std Dev (σ) | Min | 25th %ile | Median | 75th %ile | Max |
|--------------------:|---------:|------------:|----:|----------:|-------:|----------:|----:|
| **LIMIT_BAL**       | 167 484  | 129 748     | 10 000 | 50 000 | 140 000 | 240 000 | 1 000 000 |
| **AGE**             | 35.49    | 9.22        | 21 | 28 | 34 | 41 | 79 |
| **BILL_AMT1**       | 51 223   | 73 636      | –16 450 | 3 559 | 22 702 | 67 091 | 964 511 |
| **PAY_AMT1**        | 5 664    | 16 563      | 0 | 1 000 | 2 100 | 5 006 | 873 552 |
| **avg_bill_amt***   | 47 160   | 69 494      | –33 941 | 3 334 | 20 778 | 61 025 | 905 874 |
| **avg_pay_amt***    | 5 542    | 17 454      | 0 | 1 000 | 2 183 | 4 922 | 847 383 |
| **payment_ratio*** | 0.14     | 0.45        | 0 | 0.01 | 0.05 | 0.12 | 15.96 |

\* *Engineered features derived in the feature engineering step.*

These statistics highlight that credit limits vary widely across clients and are right‑skewed (most customers have limits below 300 000 RMB but a few have very high limits).  Bill amounts and payment amounts also show heavy tails.  The engineered `payment_ratio` captures the proportion of total payments to total bill amounts; it has a mean of 0.14 and a long tail, indicating that some clients pay much more than their outstanding balance while many pay only a small fraction.

## Target Distribution

The dataset is imbalanced with roughly one in five customers defaulting.  The bar chart in the `visuals/target_distribution.png` file illustrates this imbalance.  Handling class imbalance is important for modelling; evaluation metrics such as precision, recall and ROC–AUC are therefore more informative than accuracy alone.

## Correlation Analysis

The correlation heatmap (see `visuals/correlation_heatmap.png`) shows relationships among the variables.  Key observations include:

- **High correlations among bill amounts and payment amounts.**  `BILL_AMT1` through `BILL_AMT6` are strongly positively correlated, as are `PAY_AMT1` through `PAY_AMT6`.  This reflects the temporal continuity of customers’ billing and repayment behaviour.
- **Moderate positive correlation between credit limit and bill amounts.**  Customers with higher limits tend to have higher outstanding bills.
- **Payment status variables (`PAY_0`–`PAY_6`) are positively correlated with default.**  Higher values of these variables indicate delayed payments in previous months and are predictive of future default.  They also correlate with each other because clients who are late in one month are often late in subsequent months.
- **Engineered features** such as `avg_bill_amt` and `avg_pay_amt` inherit correlations from their component variables.  `payment_ratio` has a weak negative correlation with default, suggesting that clients who pay a higher proportion of their bills are less likely to default.

## Outlier Analysis

Some variables exhibit extreme outliers:

- **Credit limit (`LIMIT_BAL`)** – the maximum value is 1 000 000 RMB, far above the median of 140 000.  These high‑limit observations should be kept but may need special handling (e.g. log transformation) in certain models.
- **Payment amounts (`PAY_AMT*`)** – there are extremely large payments (up to 1.7 M RMB).  These may correspond to payoffs of very large balances and could disproportionately influence models that are sensitive to scale.
- **Negative bill amounts** – a few billing fields contain small negative values.  These could represent credit adjustments or over‑payments and are very rare; they were left unchanged during preprocessing.

## Business Context

In a credit‑card business, accurately predicting whether a customer will default in the next month helps banks manage risk, adjust credit limits and take preventive actions such as reminders or early intervention.  Features in this dataset capture a client’s credit limit, demographic attributes, repayment behaviour and bill/payment amounts across six previous months【382343060160666†L95-L112】.  Payment status variables (e.g. `PAY_0`) encode how many months a customer was delayed in paying; positive values indicate late payments, while –1 means payed duly.  Combined with bill/payment amounts, these variables provide a rich representation of financial behaviour.

## Summary of Findings

1. **No missing values** and **no duplicates** simplify preprocessing.  Basic cleaning involved renaming columns and computing engineered features such as averages and ratios.
2. The dataset is **moderately imbalanced** (~22 % defaults).  Models should account for this imbalance (e.g. by using appropriate metrics or class weighting).
3. **Payment status history** (`PAY_0`–`PAY_6`) is strongly predictive of future default.  Clients with previous delays are more likely to default.
4. **Credit limits and bill amounts** are skewed.  Transformations or tree‑based models (which are less sensitive to scaling) may handle these distributions better than linear models.
5. **Correlation among bill amounts and payment amounts** suggests redundancy.  Principal component analysis or regularisation may help reduce multicollinearity for models sensitive to correlated features.

These insights informed the choice of machine‑learning models in the next phase.  Logistic regression serves as a simple baseline, while tree‑based models such as Random Forest and gradient‑boosted trees (XGBoost) can capture non‑linear interactions and handle skewed distributions more effectively.
