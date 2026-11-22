# Commercial Construction Insurance Claim Frequency Modeling (Poisson GLM)

A statistical modeling and actuarial pricing project developed in Python to predict claim frequencies for commercial construction insurance (*entreprenadförsäkringar*), establishing a data-driven baseline for risk-based premium pricing.

## Overview
Pricing commercial insurance requires accurate estimation of expected claim frequencies across heterogeneous risk profiles. Using historical exposure and claims data from **2022–2024** provided by **Länsförsäkringar AB**, we trained and evaluated Generalized Linear Models (GLMs) to forecast portfolio claims for **2025**.

### Key Results (Test Set Evaluation)
Because 98.1% of companies in the portfolio recorded zero claims during the observation period, a Poisson GLM with a log-link function and policy `duration` as an exposure offset was selected over standard OLS regression.

| Metric | Base Model (`Log_Revenue`) | Extended Model (+ Deductible & Insured Amount) |
| :--- | :---: | :---: |
| **Predicted Total Claims** (Actual: 5,520) | 5,581 | 5,582 |
| **95% Confidence Interval** | [5,503, 5,659] | [5,504, 5,660] |
| **Predicted / Actual Ratio** | 1.0110 (+1.10%) | 1.0112 (+1.12%) |
| **Mean Absolute Error (MAE)** | 0.0371 | 0.0370 |
| **Root Mean Squared Error (RMSE)** | 0.1374 | 0.1373 |
| **Mean Poisson Deviance** | 0.1436 | 0.1425 |

### Analytical Takeaways
* **Multicollinearity vs. Robustness:** Revenue (*Omsättning*), insured amount (*Försäkringsbelopp*), and deductible (*Självrisk*) showed strong pairwise correlations ($r = 0.43 - 0.46$). While the extended model yielded marginally lower deviance, the base model was prioritized to prevent coefficient instability and overfitting.
* **Feature Significance:** Business sector (e.g., HVAC/VVS, roofing, painters, electricians) and geographic region (metropolitan vs. rural) proved statistically significant risk drivers, whereas calendar year (2022–2024) was statistically insignificant and excluded to improve generalization.
* **Tail Risk & Pricing Governance:** Lift-chart diagnostics across revenue deciles showed strong calibration for core mid-market firms, while highlighting higher variance at extreme upper-tail revenue levels where automated pricing should be paired with qualitative underwriting.

## Methodology
1. **Exploratory Data Analysis:** Inspected zero-inflation (98.1% zero-claim policies), exposure distributions, and numerical correlation matrices.
2. **Preprocessing:** Filtered non-positive policy durations (`duration > 0`) and applied a `+1` shift prior to log-transforming monetary variables.
3. **Model Specification:** Fitted Poisson GLMs (`ln(μ/t) = β₀ + β₁X₁ + ...`) using policy duration `t` as offset, and benchmarked against Negative Binomial regression.
4. **Validation:** Evaluated aggregate portfolio accuracy, 95% confidence intervals (using the Delta method), lift charts across revenue bins, and sector/region breakdowns.

## Repository Structure
```text
├── poisson_regression_final.py   # Data preprocessing, Poisson GLM training & lift-chart diagnostics
├── graphs.py                     # Exploratory distribution plot (empirical vs. theoretical Poisson)
├── requirements.txt              # Python dependencies
└── README.md
## Authors
* **Per Pettersson Ihse**
* Carl Björkvall, Erik Brunnberg, Carl Gustafson *(KTH Royal Institute of Technology, May 2026)*
