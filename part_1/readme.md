# Lab Notes & Analysis (Part 1)

### 1. Dataset & Purpose
* Cleaned a dataset with 1485 rows and 35 columns.
* It tracks employee details (age, income, role, etc.) to figure out why people quit (Attrition).
* Finding these patterns helps HR step in before people leave, saving the company money on rehiring and training.

### 2. Data Check & Missing Values
* Found missing data in MonthlyIncome (~5%) and YearsAtCompany (~22%).
* Since YearsAtCompany had more than 20% missing values, I left it alone at first so I wouldn't mess up the data distribution by guessing too much.

---

### 3. Imputation Strategy (Task 5 & Sub-task a)
* Looked at the two most skewed columns before filling any blanks:
  * Positive skew (right tail): A few high earners pull the mean way up.
  * Negative skew (left tail): Low values pull the mean down.
* **My Choice:** I used the Median to fill the missing spaces instead of the mean. The median sits right in the middle and doesn't get ruined by crazy outliers or high salaries, keeping the data realistic. 
* Checked `isnull().sum()` after and confirmed all nulls are gone (0).

---

### 4. Outliers (Task 6)
* Used the IQR rule (Q3 + 1.5 * IQR) and found a bunch of outliers:
  * MonthlyIncome -> 123 outliers
  * TotalWorkingYears -> 63 outliers
* **What I'm doing with them:** Keeping them! In HR data, high salaries and long tenures usually just mean you are looking at executives or senior managers. If I delete them, the model won't learn why bosses quit.

---

### 5. Plots I Made
1. **Line Plot:** Shows the sequence of data points across the rows.
2. **Bar Chart:** Compares average values across categories (like income per department).
3. **Histogram:** Shows the shape of the most skewed column (has a long tail on the right).
4. **Scatter Plot:** Tracks how TotalWorkingYears and MonthlyIncome go up together (strong positive relationship).
5. **Box Plot:** Shows the spread and median of incomes split by job roles (lots of dots showing the high-income outliers).

---

### 6. Pearson vs. Spearman (Task 8 & Sub-task b)
* Ran both correlation types to see how variables interact:
  * **If Spearman > Pearson:** It means the relationship is monotonic but non-linear. The variables move together consistently, but they follow a curve rather than a straight line.
  * **If Pearson >= Spearman:** The relationship is linear (moves in a straight line).
* **Decision for Part 2:** I'm going to rely on Spearman for picking features. It catches curved patterns that Pearson misses, so I won't accidentally drop important features just because they aren't perfectly linear.

---

### 7. Grouped Stats (Sub-task c)
* Grouped Department by MonthlyIncome:
  * **High Standard Deviation:** Means there's a huge spread of salaries within the same department. Just knowing someone's department isn't enough to guess their exact salary.
  * **Mean Ratio:** Looked at the highest average department income divided by the lowest. The difference is big enough to prove that Department gives a strong signal for predicting income.