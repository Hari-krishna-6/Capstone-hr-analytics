# Part 2 — Supervised Machine Learning Model: Build, Train, and Evaluate

## Label Definitions
* **Regression Target (`y_reg`):** `MonthlyIncome` — A continuous numeric target variable representing the employee's monthly salary in dollars.
* **Classification Target (`y_clf`):** `Attrition` — A natural binary target variable mapping if an employee left the company (`1` for Yes) or stayed (`0` for No).

---

## Preprocessing & Data Leakage Prevention

### Categorical Encoding Strategies
1. **Ordinal Encoding (`BusinessTravel`):** Features like travel frequency contain an inherent logical progression. We applied custom integer mapping (`Non-Travel` = 0, `Travel_Rarely` = 1, `Travel_Frequently` = 2) to preserve this ordinal distance, allowing the model to naturally interpret increasing travel intensities.
2. **One-Hot Encoding:** For nominal categories with no hierarchy (e.g., `Department`, `EducationField`, `JobRole`, `MaritalStatus`, `Gender`, `OverTime`), we utilized `pd.get_dummies()`. Label encoding here would introduce a **false ordinal relationship** (e.g., implying Department 3 is "greater than" Department 1), corrupting distance-based or structural model calculations. 
3. **Dummy Variable Trap Elimination:** The first dummy column for each one-hot encoded category was explicitly dropped (`drop_first=True`) to avoid perfect multicollinearity, which ensures the linear matrices remain stable during regression fitting.

### Data Leakage Prevention in Scaling
To preserve complete evaluation purity, the `StandardScaler` was fit **strictly on the training features matrix** (`scaler.fit(X_train)`), and then used to transform both `X_train` and `X_test`. 

> **Crucial Leakage Note:** Fitting the scaler on the full dataset before splitting would constitute severe data leakage. Doing so mathematically infuses test-set statistics (mean, variance, global boundaries) directly into the training pipeline, yielding over-optimistic evaluation metrics that fail to reflect real-world generalization.

---

## Regression Model Performance (Task 4)

### Model Comparison Table
| Model | Mean Squared Error (MSE) | $R^2$ Score |
| :--- | :--- | :--- |
| **OLS Linear Regression** | 2,845,102.4109 | 0.8124 |
| **Ridge Regression ($\alpha=1.0$)** | 2,845,614.9082 | 0.8123 |

### Coefficient Interpretation
* **Large Positive Coefficients:** A one-unit increase in a scaled positive feature is directly associated with an increase in the predicted `MonthlyIncome` equal to that coefficient's dollar value, assuming all other variables remain static.
* **Large Negative Coefficients:** A one-unit increase in a scaled negative feature corresponds to a drop in the predicted `MonthlyIncome` equal to the absolute value of that coefficient, holding all other features constant.

### OLS vs. Ridge Regularization Profile
Ridge Regression introduces an L2 regularization penalty ($\alpha \sum \beta_i^2$) directly into the loss function, shrinking the magnitude of feature weights toward zero. The `alpha` parameter controls the intensity of this penalty. In this dataset, OLS and Ridge perform almost identically, indicating that the feature spaces are highly informative and not heavily suffering from extreme overfitting or collinear explosions that require massive coefficient suppression.

---

## Classification Model Performance (Task 5)

### Class Imbalance Handling
The training set revealed a severe class imbalance with an Attrition rate of **16.16%** (well below the 35% rubric threshold). To prevent the Logistic Regression model from ignoring the minority class, **SMOTE-NC** was applied exclusively to the training portion.

* **Before SMOTE-NC Class Counts:** Class 0 (Stay): 986 | Class 1 (Left): 190
* **After SMOTE-NC Class Counts:** Class 0 (Stay): 986 | Class 1 (Left): 493 *(Brought to a robust, stable 1:2 minority-to-majority ratio)*

### Baseline Evaluation Results (C=1.0)
* **Overall Accuracy:** 0.8605
* **Baseline Confusion Matrix:**
    ```text
    [[232  15]
     [ 26  21]]
    ```

### Area Under the ROC Curve (AUC) Interpretation
The baseline model achieves an **AUC score of 0.7910**. This value represents the global probability that the model will rank a randomly chosen individual who actually left the company (`Class 1`) with a higher attrition risk score than a randomly chosen individual who stayed (`Class 0`). An AUC of ~0.79 demonstrates a strong, highly capable ability to separate the two populations cleanly.

---

## Decision-Threshold Sensitivity (Task 5b)

### Metric Progress Table
| Threshold | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| **0.30** | 0.4306 | 0.6596 | **0.5210** |
| **0.40** | 0.4717 | 0.5319 | 0.5000 |
| **0.50** | 0.5641 | 0.4681 | 0.5116 |
| **0.60** | 0.6452 | 0.4255 | 0.5128 |
| **0.70** | 0.7778 | 0.2979 | 0.4308 |

### Precision vs. Recall Equations
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

### Domain-Appropriate Optimization
1. **F1-Score Maximizer:** The **0.30 threshold** maximizes the F1-score on this dataset at **0.5210**.
2. **Domain Importance:** For corporate employee retention, **Recall is the more critical metric**. Missing a high-value flight risk employee (**False Negative**) costs a business immense disruption, onboarding delays, and thousands of replacement dollars. A **False Positive** merely prompts HR to conduct a harmless, supportive stay-interview with an employee who wasn't planning to leave anyway.
3. **Threshold Action:** To prioritize employee protection, we should **lower the threshold** to 0.30. The explicit cost of doing so is a drop in Precision, meaning HR will spend operational hours addressing "false alarms" on stable employees.

---

## Regularization Experiment (Task 6)

### L2 Regularization Parameter Comparison Table
| Model Configuration | Precision (Class 1) | Recall (Class 1) | AUC Score |
| :--- | :--- | :--- | :--- |
| **Baseline (C=1.0)** | 0.5641 | 0.4681 | 0.7910 |
| **Strong Regularization (C=0.01)** | 0.5122 | 0.4468 | 0.7931 |

### The Role of Parameter C
The parameter `C` is the **inverse of regularization strength** ($C = 1/\lambda$). Reducing `C` to `0.01` commands a heavy L2 penalty that aggressively dampens feature coefficients. On this dataset, this strong penalty slightly worsened the explicit Precision and Recall scores at the default threshold because it over-flattened some informative features. However, it marginally improved the global class separation capability, boosting the baseline AUC from `0.7910` to `0.7931`.

---

## Bootstrap Confidence Interval for AUC Difference (Task 7)

To determine if the baseline model's performance advantage over the regularized configuration is statistically meaningful, we executed a 500-iteration bootstrap resampling process with replacement on the test data.

* **Mean AUC Difference ($\text{AUC}_{C=1.0} - \text{AUC}_{C=0.01}$):** -0.0028
* **2.5th Percentile (Lower Bound):** -0.0462
* **97.5th Percentile (Upper Bound):** 0.0359
* **Excludes Zero?** **False**

### Statistical Conclusion
Because the 95% confidence interval spanning from **-0.0462 to 0.0359** crosses directly through zero, **the performance difference between the baseline and strongly regularized model is not statistically significant**. The slight variations are purely driven by data sampling distribution nuances, proving that neither model configuration holds a reliable operational advantage over the other on this dataset population.