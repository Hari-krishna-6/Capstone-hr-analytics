# Part 2 – Supervised Machine Learning Model: Build, Train and Evaluate

## Objective

The objective of this part is to build both regression and classification models using the cleaned HR Analytics dataset. Appropriate preprocessing techniques were applied before training to avoid data leakage and improve model performance. The trained models were then evaluated using standard regression and classification metrics.

---

# Dataset Preparation

The cleaned dataset generated in Part 1 (`cleaned_data.csv`) was loaded into a Pandas DataFrame.

The feature matrix **X** was created by selecting all predictor columns except the target variable.

Two target variables were defined:

* **Regression Target (y_reg):** `MonthlyIncome`
* **Classification Target (y_clf):** A binary label created by comparing `MonthlyIncome` with its median.

```python
y_clf = (MonthlyIncome > MonthlyIncome.median()).astype(int)
```

Employees earning above the median income were assigned class **1**, while those earning at or below the median income were assigned class **0**.

---

# Feature Encoding

The dataset contains both numerical and categorical variables. Before training the models, categorical variables were encoded.

* Ordinal categorical variables were encoded using label encoding to preserve their natural order.
* Nominal categorical variables such as department and job role were converted using One-Hot Encoding.

The first dummy column was dropped to avoid multicollinearity.

### Why One-Hot Encoding?

One-Hot Encoding represents each category as an independent binary feature. Unlike Label Encoding, it does not introduce a false numerical order between categories that have no natural ranking.

---

# Train-Test Split and Feature Scaling

The dataset was divided into training and testing sets using:

* Training Data: 80%
* Testing Data: 20%

```python
train_test_split(test_size=0.2, random_state=42)
```

After splitting, feature scaling was performed using **StandardScaler**.

The scaler was fitted only on the training data and then applied to both training and testing datasets.

### Why fit only on training data?

Fitting the scaler on the entire dataset would introduce **data leakage**, because information from the test data would influence the scaling parameters. This would make the model evaluation overly optimistic. By fitting only on the training data, the test set remains completely unseen during training.

---

# Linear Regression Model

A Linear Regression model was trained using the scaled training data.

The model performance was evaluated using Mean Squared Error (MSE) and R² Score.

### Results

| Metric                   |            Value |
| ------------------------ | ---------------: |
| Mean Squared Error (MSE) | **6294571.9051** |
| R² Score                 |       **0.6440** |

The R² score of **0.6440** indicates that approximately **64.4%** of the variation in Monthly Income is explained by the selected features.

The Mean Squared Error represents the average squared prediction error between the actual and predicted salary values.

---

# Regression Coefficients

The coefficients of the Linear Regression model were printed along with the corresponding feature names.

The three features with the largest absolute coefficients were identified from the model output.

### Interpretation

A **positive coefficient** indicates that increasing the corresponding feature increases the predicted monthly income while keeping all other features constant.

A **negative coefficient** indicates that increasing the corresponding feature decreases the predicted monthly income.

Features with larger absolute coefficients have a stronger influence on the prediction compared to features with smaller coefficients.

---

# Ridge Regression

To reduce overfitting caused by large regression coefficients, Ridge Regression was trained using:

```python
Ridge(alpha=1.0)
```

### Results

| Model             |          MSE | R² Score |
| ----------------- | -----------: | -------: |
| Linear Regression | 6294571.9051 |   0.6440 |
| Ridge Regression  | 6274915.7363 |   0.6451 |

### Interpretation

Ridge Regression produced a slightly lower Mean Squared Error and a slightly higher R² score than ordinary Linear Regression.

Ridge Regression applies L2 regularization, which penalizes very large coefficient values. This helps reduce model variance and improves generalization without significantly increasing bias.

The parameter **alpha** controls the strength of the regularization. Larger values of alpha produce stronger regularization by shrinking the regression coefficients more aggressively.

---

# Logistic Regression

For the classification task, Logistic Regression was used to classify employees into two income groups.

Before training, the class distribution was checked.

### Class Distribution

Before balancing:

| Class | Count |
| ----: | ----: |
|     0 |   986 |
|     1 |   190 |

The minority class represented less than 35% of the training data. Therefore, class imbalance handling was required.

SMOTENC was applied only to the training data to generate synthetic minority class samples while preserving the original testing dataset.

After balancing:

| Class | Count |
| ----: | ----: |
|     0 |   986 |
|     1 |   493 |

Balancing the dataset helps the classifier learn decision boundaries for both classes instead of becoming biased toward the majority class.

---

# Logistic Regression Evaluation

After training, predictions were generated on the test dataset.

The resulting confusion matrix was:

```text
[[230 17]
 [25 22]]
```

### Classification Report

| Metric    | Class 0 | Class 1 |
| --------- | ------: | ------: |
| Precision |    0.90 |    0.56 |
| Recall    |    0.93 |    0.47 |
| F1 Score  |    0.92 |    0.51 |

Overall Classification Accuracy:

**86%**

The model performs very well for the majority class while showing comparatively lower recall for the minority class, indicating that predicting higher-income employees remains more challenging.

# ROC Curve and AUC

The performance of the Logistic Regression model was further evaluated using the Receiver Operating Characteristic (ROC) curve. Instead of using only predicted class labels, the ROC curve evaluates the predicted probabilities generated by the model.

The Area Under the ROC Curve (AUC) was also calculated.

### Interpretation

The ROC curve shows the trade-off between the True Positive Rate (Recall) and the False Positive Rate at different classification thresholds.

A higher AUC value indicates that the model has a better ability to distinguish between employees belonging to different income groups. An AUC value close to 1 represents excellent classification performance, whereas an AUC close to 0.5 indicates random guessing.

---

# Precision and Recall

Precision and Recall were used to evaluate the classification model.

### Formula for Precision

[
Precision=\frac{TP}{TP+FP}
]

Precision measures the proportion of predicted positive samples that are actually positive.

### Formula for Recall

[
Recall=\frac{TP}{TP+FN}
]

Recall measures the proportion of actual positive samples that are correctly identified by the model.

### Which metric is more important?

For this dataset, **Recall** is more important because failing to identify employees belonging to the higher income category (False Negatives) may reduce the usefulness of the model. A higher recall ensures that more actual positive cases are detected.

The calculated classification metrics are:

| Metric              |    Value |
| ------------------- | -------: |
| Accuracy            | **0.86** |
| Precision (Class 1) | **0.56** |
| Recall (Class 1)    | **0.47** |
| F1 Score (Class 1)  | **0.51** |

---

# Decision Threshold Sensitivity

The default classification threshold of 0.50 was varied from **0.30** to **0.70** to observe its effect on Precision, Recall and F1-score.

| Threshold | Precision | Recall | F1 Score |
| --------: | --------: | -----: | -------: |
|      0.30 |    0.4306 | 0.6596 |   0.5210 |
|      0.40 |    0.4717 | 0.5319 |   0.5000 |
|      0.50 |    0.5641 | 0.4681 |   0.5116 |
|      0.60 |    0.6452 | 0.4255 |   0.5128 |
|      0.70 |    0.7778 | 0.2979 |   0.4308 |

### Interpretation

As the threshold increases:

* Precision increases because fewer samples are predicted as positive.
* Recall decreases because more positive samples are missed.
* F1-score balances both Precision and Recall.

The highest F1-score was obtained at approximately **0.30**, indicating that this threshold provides the best balance between identifying positive samples and avoiding false predictions.

If the application requires identifying as many positive employees as possible, lowering the threshold is preferred because Recall increases. The trade-off is an increase in False Positives.

---

# Logistic Regression Regularization

A second Logistic Regression model was trained with stronger L2 regularization using:

```python id="ewv75e"
C = 0.01
```

The baseline model used:

```python id="jq6vwy"
C = 1.0
```

### Comparison

The two models were compared using Precision, Recall and ROC-AUC.

The strongly regularized model produced performance very similar to the baseline model.

### Interpretation

The parameter **C** controls the strength of regularization.

* Large **C** → weaker regularization.
* Small **C** → stronger regularization.

Reducing the value of **C** shrinks the model coefficients more aggressively, helping reduce overfitting. In this dataset, stronger regularization did not significantly improve the classification performance.

---

# Bootstrap Confidence Interval

To compare the reliability of the two Logistic Regression models, 500 bootstrap samples were generated from the test set.

For every bootstrap sample, the difference in ROC-AUC between the baseline model and the strongly regularized model was calculated.

### Results

| Statistic               |                 Value |
| ----------------------- | --------------------: |
| Mean AUC Difference     |           **-0.0028** |
| 95% Confidence Interval | **[-0.0462, 0.0359]** |

### Interpretation

The confidence interval includes **zero**, indicating that the observed difference between the two Logistic Regression models is not statistically reliable. Therefore, neither model consistently outperforms the other across different bootstrap samples.

---

# Summary

In this part of the project, both regression and classification models were developed using the cleaned HR Analytics dataset.

Linear Regression achieved an **R² score of 0.6440**, while Ridge Regression slightly improved the performance with an **R² score of 0.6451**, showing that L2 regularization helped reduce prediction error.

For the classification task, class imbalance was handled using **SMOTENC**, and Logistic Regression achieved an overall **accuracy of 86%**. The ROC curve, AUC, confusion matrix and threshold sensitivity analysis provided a detailed evaluation of the classifier. Changing the decision threshold showed the trade-off between Precision and Recall, with the best F1-score obtained around a threshold of **0.30**.

Finally, a stronger regularized Logistic Regression model was evaluated using bootstrap sampling. Since the confidence interval for the AUC difference included zero, the performance difference between the two models was not statistically significant.

Overall, the preprocessing techniques, regression models and classification models developed in this part provide a strong foundation for the advanced ensemble methods implemented in Part 3.