# Part 3 – Advanced Modeling: Ensembles, Hyperparameter Tuning and Machine Learning Pipeline

## Objective

The objective of this part is to compare multiple machine learning models, evaluate their performance using cross-validation, perform hyperparameter tuning, analyse feature importance, and build a reusable machine learning pipeline. The best-performing model is then saved for future predictions.

---

# 1. Decision Tree (Baseline Model)

The first model trained was a Decision Tree Classifier using the default parameters.

### Results

| Metric            |      Value |
| ----------------- | ---------: |
| Training Accuracy | **1.0000** |
| Testing Accuracy  | **0.7347** |

### Interpretation

The model achieved perfect accuracy on the training dataset but considerably lower accuracy on the testing dataset. This indicates that the tree has memorized the training data instead of learning general patterns.

Therefore, the default Decision Tree clearly shows **overfitting**.

Decision Trees are considered **high variance models** because they greedily split the dataset at every node and do not reconsider earlier decisions. As a result, even small changes in the training data can produce a very different tree.

---

# 2. Controlled Decision Tree

To reduce overfitting, a second Decision Tree was trained using:

```python
max_depth = 5
min_samples_split = 20
```

### Results

| Metric            |      Value |
| ----------------- | ---------: |
| Training Accuracy | **0.8472** |
| Testing Accuracy  | **0.8061** |

### Interpretation

Restricting the tree depth reduced the gap between training and testing accuracy.

The parameter **max_depth** limits the maximum number of levels in the tree. A smaller depth prevents the tree from learning unnecessary details present only in the training data.

The parameter **min_samples_split** prevents further splitting when a node contains only a small number of samples. This reduces the effect of noisy observations and improves the model's ability to generalize.

Compared with the unconstrained tree, the controlled tree provides better testing performance while significantly reducing overfitting.

---

# 3. Gini vs Entropy

Two Decision Tree models were trained using different splitting criteria.

### Results

| Criterion | Test Accuracy |
| --------- | ------------: |
| Gini      |    **0.8061** |
| Entropy   |    **0.8095** |

### Formula for Gini Impurity

[
Gini = 1-\sum p_i^2
]

### Formula for Entropy

[
Entropy=-\sum p_i\log_2(p_i)
]

A node with **Gini = 0** means that every sample in that node belongs to the same class, making it a perfectly pure node.

### Interpretation

Both criteria produced similar performance. However, Entropy achieved slightly higher testing accuracy on this dataset.

---

# 4. Random Forest

A Random Forest classifier was trained using the following parameters:

* n_estimators = 100
* max_depth = 10
* random_state = 42

### Results

| Metric            |      Value |
| ----------------- | ---------: |
| Training Accuracy | **0.9926** |
| Testing Accuracy  | **0.8299** |
| ROC-AUC           | **0.8046** |

The Random Forest significantly improved the testing performance compared to both Decision Tree models.

### Feature Importance

The Random Forest calculates feature importance by measuring the average reduction in Gini Impurity produced by each feature across all trees in the forest.

Unlike Linear Regression coefficients, feature importance values do not indicate whether the relationship is positive or negative. Instead, they measure how useful each feature is when making decisions inside the ensemble.

The five most important features identified by the model were printed in the notebook and used later for the feature ablation study.

---

## Bagging in Random Forest

Random Forest is an ensemble learning algorithm that combines many Decision Trees.

Each tree is trained using **Bootstrap Sampling**, where the training samples are selected randomly with replacement.

During every split, only a random subset of available features is considered.

Because each tree sees a different subset of both samples and features, the individual trees become less correlated. Their predictions are then averaged, reducing variance and improving generalization compared to a single Decision Tree.

---

# 5. Gradient Boosting

Gradient Boosting was trained using:

* n_estimators = 100
* learning_rate = 0.1
* max_depth = 3

### Results

| Metric            |      Value |
| ----------------- | ---------: |
| Training Accuracy | **0.9581** |
| Testing Accuracy  | **0.8605** |
| ROC-AUC           | **0.7569** |

### Interpretation

Gradient Boosting achieved the highest testing accuracy among the individual tree-based models.

Unlike Random Forest, which builds trees independently, Gradient Boosting trains trees sequentially. Each new tree attempts to correct the mistakes made by the previous trees, allowing the ensemble to learn increasingly accurate decision boundaries.

The Gradient Boosting model is included in the cross-validation comparison performed later in this project.

# 6. Feature Ablation Study

To analyse the contribution of less important features, the five features with the lowest importance scores from the Random Forest model were removed. A second Random Forest model was then trained using the remaining features.

### Results

| Model                 |    ROC-AUC |
| --------------------- | ---------: |
| Full Random Forest    | **0.8046** |
| Reduced Random Forest | **0.7963** |

### Interpretation

Removing the five least important features resulted in only a small decrease in ROC-AUC.

This indicates that these features contributed relatively little to the model's predictive performance. Therefore, removing them simplifies the model with only a minor reduction in accuracy.

A simpler model reduces computational cost, memory usage and maintenance effort during deployment. However, feature removal should only be considered when the reduction in performance is within an acceptable limit.

---

# 7. Cross Validation

To obtain a more reliable estimate of model performance, five-fold Stratified Cross Validation was performed using ROC-AUC as the evaluation metric.

### Results

| Model                    | Mean ROC-AUC | Standard Deviation |
| ------------------------ | -----------: | -----------------: |
| Logistic Regression      |   **0.8401** |         **0.0257** |
| Controlled Decision Tree |   **0.6840** |         **0.0481** |
| Random Forest            |   **0.8006** |         **0.0369** |
| Gradient Boosting        |   **0.8084** |         **0.0190** |

### Interpretation

Cross-validation evaluates the model on multiple train-test splits instead of relying on a single split. This provides a more reliable estimate of how the model is expected to perform on unseen data.

Among all models, **Logistic Regression** achieved the highest average ROC-AUC, while **Gradient Boosting** showed the lowest variation across folds, indicating more consistent performance.

---

# 8. Hyperparameter Tuning using GridSearchCV

A machine learning pipeline was created consisting of:

* SimpleImputer (Median)
* StandardScaler
* RandomForestClassifier

GridSearchCV was used to search for the best combination of hyperparameters.

### Parameter Grid

* n_estimators = 50, 100, 200
* max_depth = 5, 10, None
* min_samples_leaf = 1, 5

A total of:

**3 × 3 × 2 = 18**

parameter combinations were evaluated.

Using five-fold cross validation,

**18 × 5 = 90**

models were trained.

### Best Parameters

| Parameter        | Value    |
| ---------------- | -------- |
| n_estimators     | **200**  |
| max_depth        | **None** |
| min_samples_leaf | **1**    |

Best Cross-Validated ROC-AUC:

**0.8136**

### Interpretation

Grid Search evaluates every possible parameter combination and therefore provides the best-performing configuration from the specified search space.

Although Grid Search is computationally expensive, it guarantees that every candidate model is evaluated.

Randomized Search is generally faster because it evaluates only a random subset of parameter combinations, making it more suitable for larger search spaces.

---

# 9. Manual Learning Curve

The best pipeline obtained from GridSearchCV was trained using progressively larger portions of the training data.

### Results

| Training Fraction | Training ROC-AUC | Testing ROC-AUC |
| ----------------: | ---------------: | --------------: |
|               20% |           1.0000 |          0.7038 |
|               40% |           1.0000 |          0.7238 |
|               60% |           1.0000 |          0.7790 |
|               80% |           1.0000 |          0.7792 |
|              100% |           1.0000 |          0.7857 |

### Interpretation

The training ROC-AUC remained very high throughout the experiment, indicating that the model is able to fit the training data extremely well.

The testing ROC-AUC gradually increased as more training data became available.

This suggests that collecting additional training data is likely to improve model performance further. The model therefore appears to be **data-limited** rather than limited by model capacity.

---

# 10. Model Serialization

The best-performing machine learning pipeline was saved using:

```python
joblib.dump(best_pipeline, "best_model.pkl")
```

The saved model was then reloaded using:

```python
model = joblib.load("best_model.pkl")
```

Predictions were successfully generated for hand-crafted test samples, confirming that the serialized model can be reused without retraining.

---

# 11. Final Model Comparison

| Model                    | Test ROC-AUC / CV ROC-AUC |
| ------------------------ | ------------------------: |
| Logistic Regression      |      **0.8401 (CV Mean)** |
| Controlled Decision Tree |      **0.6840 (CV Mean)** |
| Random Forest            |      **0.8006 (CV Mean)** |
| Gradient Boosting        |      **0.8084 (CV Mean)** |

### Recommended Model

Although Gradient Boosting achieved the highest testing accuracy, **Logistic Regression** achieved the highest average ROC-AUC during cross-validation (**0.8401**) and demonstrated more reliable generalization across multiple folds.

Considering predictive performance, simplicity and computational efficiency, Logistic Regression is recommended as the final model for this HR Analytics dataset.

---

# Conclusion

In this part of the project, multiple ensemble learning techniques and model optimization methods were explored. A baseline Decision Tree initially showed severe overfitting, which was reduced by limiting the tree depth and minimum samples required for splitting.

Random Forest and Gradient Boosting further improved predictive performance through ensemble learning. Feature importance analysis and feature ablation showed that several low-importance features could be removed with only a small reduction in performance.

GridSearchCV was used to identify the best Random Forest hyperparameters, while the learning curve demonstrated that increasing the amount of training data could further improve model performance. Finally, the best pipeline was serialized as **best_model.pkl**, allowing the trained model to be reused without repeating the training process.

Overall, this part demonstrates a complete machine learning workflow, including ensemble methods, hyperparameter tuning, cross-validation, feature analysis and model deployment.