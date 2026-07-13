# Part 1 – Data Acquisition, Cleaning and Exploratory Analysis

## Objective

The objective of this part is to clean the raw HR Analytics dataset, understand its characteristics through exploratory data analysis, and prepare a clean dataset that can be used for machine learning models in the next parts.

---

# Dataset Information

The dataset was loaded into a Pandas DataFrame using `pd.read_csv()`.

Initial inspection was performed using:

* `df.head()`
* `df.dtypes`
* `df.shape`

### Dataset Details

* Number of Rows: **1485**
* Number of Columns: **35**

The first five rows and data types were checked to understand the dataset before performing any preprocessing.

---

# 1. Missing Value Analysis

Missing values were calculated using:

```python
df.isnull().sum()
(df.isnull().sum()/df.shape[0])*100
```

The analysis showed the following important missing values:

| Column         | Missing Values | Missing % |
| -------------- | -------------: | --------: |
| MonthlyIncome  |             74 |     4.98% |
| YearsAtCompany |            327 |    22.02% |

Since **YearsAtCompany** has more than **20%** missing values, it was identified separately as required in the assignment.

For numerical columns having less than 20% missing values, the missing values were filled using the **median**.

### Why Median?

I used the median instead of the mean because the dataset contains skewed numerical features and outliers. Extreme values can significantly affect the mean, whereas the median represents the middle value and is therefore more suitable for imputing missing values in skewed data.

---

# 2. Duplicate Detection and Removal

Duplicate records were identified using:

```python
df.duplicated().sum()
```

### Result

* Duplicate rows found: **15**
* Duplicate rows removed: **15**

Dataset size after removing duplicates:

* **1470 rows × 35 columns**

After removing duplicates, the missing value percentages were checked again and no significant change was observed.

---

# 3. Data Type Correction

Some columns were converted into more suitable data types.

Repeated string columns were converted into the **category** datatype to reduce memory usage.

Memory usage was compared before and after conversion.

| Stage             | Memory Usage |
| ----------------- | -----------: |
| Before Conversion |   1112.18 KB |
| After Conversion  |    959.52 KB |

Memory saved:

**152.66 KB**

Using the category datatype reduced the memory consumption without changing the information stored in the dataset.

---

# 4. Descriptive Statistics and Skewness

Descriptive statistics were generated using:

```python
df.describe()
```

Skewness of every numerical column was calculated using:

```python
df[col].skew()
```

The two highest skewed columns were:

| Column                  | Skewness |
| ----------------------- | -------: |
| YearsSinceLastPromotion |   1.9843 |
| PerformanceRating       |   1.9219 |

### Interpretation

Both columns have positive skewness.

Positive skew means that most observations are concentrated at lower values while a few large values create a long right tail.

Since the mean is affected by these extreme values, the median provides a better estimate of the centre of the distribution for missing value imputation.

---

# 5. Outlier Detection using IQR

Outliers were detected using the Interquartile Range (IQR) method.

The following columns contained outliers.

| Column                  | Number of Outliers |
| ----------------------- | -----------------: |
| MonthlyIncome           |                123 |
| NumCompaniesWorked      |                 52 |
| PerformanceRating       |                226 |
| StockOptionLevel        |                 85 |
| TotalWorkingYears       |                 63 |
| TrainingTimesLastYear   |                238 |
| YearsAtCompany          |                 53 |
| YearsInCurrentRole      |                 21 |
| YearsSinceLastPromotion |                107 |
| YearsWithCurrManager    |                 14 |

### Interpretation

The detected outliers were **not removed** because they may represent genuine employee information such as employees with higher salaries, longer work experience or higher performance ratings.

Removing these records could reduce the model's ability to learn meaningful real-world patterns. Therefore, the outliers were retained and will be handled appropriately during model training if required.

---

# 6. Data Visualisation

Five different visualisations were created as required.

### Line Plot

A line plot was used to observe the variation of a numerical feature across the dataset.

**Interpretation**

The plot shows how the selected numerical feature changes across observations and helps identify any unusual fluctuations.

---

### Bar Chart

A bar chart was created to compare the average value of a numerical feature across different categories.

**Interpretation**

The comparison shows that different categories have different average values, indicating that the categorical feature may contribute useful information for prediction.

---

### Histogram

A histogram was plotted for the most skewed numerical feature.

**Interpretation**

The histogram shows a positively skewed distribution with most observations concentrated near smaller values and a long right tail caused by a few larger observations.

---

### Scatter Plot

A scatter plot was created between two numerical variables expected to be related.

**Interpretation**

The scatter plot shows a positive relationship between the selected variables, indicating that one variable generally increases as the other increases.

---

### Box Plot

A box plot was created by splitting a numerical variable across categories.

**Interpretation**

The box plot shows differences in median values and variation between different categories. It also highlights several outliers, indicating that some categories contain employees with unusually high or low values.

---

# 7. Pearson Correlation

Pearson correlation was calculated for all numerical columns using:

```python
df.corr()
```

A correlation heatmap was generated to visualise the relationships between numerical features.

### Interpretation

Highly correlated variables indicate strong linear relationships.

However, correlation does not necessarily imply causation. A third factor such as employee experience, job role or department may influence both variables and create the observed correlation.

---

# 8(a). Mean vs Median Imputation

Before imputing the two highest skewed columns, both the mean and median were calculated.

Because both columns are positively skewed, the median was selected for imputation.

Positive skew causes the mean to be pulled upward by large values, while the median remains comparatively stable and better represents the central tendency.

After imputation,

```python
df.isnull().sum()
```

confirmed that no missing values remained in the imputed columns.

---

# 8(b). Spearman Rank Correlation

Spearman correlation was computed using:

```python
df.corr(method='spearman')
```

The Spearman correlation matrix was compared with the Pearson correlation matrix.

For feature selection in Part 2, Spearman correlation will be preferred whenever a monotonic but non-linear relationship exists because it measures rank-based association instead of only linear association.

---

# 8(c). Grouped Aggregation

Grouped aggregation was performed using:

```python
df.groupby(categorical_column)[numeric_column].agg(['mean','std','count'])
```

The ratio between the highest and lowest group mean was:

**1.09**

### Interpretation

A ratio of **1.09** indicates that there is some difference between the groups, although the variation is not extremely large. This suggests that the selected categorical feature contains predictive information but should be combined with other features to improve model performance.

High within-group standard deviation indicates that the categorical feature alone is not sufficient for accurate prediction.

---

# Clean Dataset

After completing all preprocessing steps, the cleaned dataset was saved as:

```python
cleaned_data.csv
```

using

```python
df.to_csv('cleaned_data.csv', index=False)
```

This cleaned dataset will be used in Part 2 and Part 3 for building machine learning models.

---

# Conclusion

In this part, the HR Analytics dataset was successfully cleaned and explored. Missing values were analysed, duplicate records were removed, data types were optimized, and outliers were identified using the IQR method. Descriptive statistics, correlation analysis, grouped aggregation and multiple visualisations were performed to understand the dataset better.

The cleaned dataset was exported as **cleaned_data.csv**, making it ready for supervised machine learning tasks in the next part of the project.