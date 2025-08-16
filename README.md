# Employee-Attrition-Prediction-Supervised-Learning-AutoML-EDA-for-Global-Talent-Solutions-GTS-
This project tackles the growing challenge of employee attrition using supervised machine learning, specifically classification techniques. The goal is to empower **GTS** with a predictive system to identify employees at risk of leaving, enabling preemptive retention actions and reducing HR costs.

## Project Overview

This project addresses the critical business challenge of employee attrition for **Global Talent Solutions (GTS)**, a multinational HR solutions company. Leveraging supervised machine learning techniques, the goal is to develop a predictive model that identifies employees at high risk of attrition. This enables GTS to proactively implement retention strategies, reduce associated costs, and foster a more stable workforce.

The project encompasses a full data science lifecycle, from comprehensive Exploratory Data Analysis (EDA) and robust data preprocessing to model development, evaluation, and the application of AutoML for optimized performance.

## Business Problem

Global Talent Solutions (GTS) faces the challenge of employee attrition, which can lead to:
- Increased recruitment and training costs.
- Loss of institutional knowledge and productivity.
- Negative impact on team morale and client relationships.

The absence of a predictive mechanism makes it difficult for GTS to anticipate and mitigate attrition effectively. This project aims to provide a data-driven solution to predict employee turnover, allowing for timely intervention and strategic HR planning.

## Project Goal

To build and evaluate a highly accurate classification model that predicts whether an employee will leave the company (`Attrition: Yes/No`). The project also focuses on demonstrating a systematic data science workflow, including advanced techniques like AutoML, to deliver actionable insights for HR decision-makers.

## Dataset

The dataset used for this project is `employee_attrition.csv`, containing various attributes related to employee demographics, job roles, satisfaction, and work-life balance.

**Key Columns:**
- `EmployeeID`: Unique identifier for each employee. (Dropped during preprocessing)
- `Age`: Employee's age.
- `Gender`: Employee's gender.
- `Department`: Department the employee belongs to (e.g., IT, HR, Finance, Sales, Operations).
- `MonthlyIncome`: Employee's monthly salary (scaled).
- `YearsAtCompany`: Number of years the employee has worked at the company (scaled).
- `OverTime`: Whether the employee frequently works overtime (Yes/No).
- `JobSatisfaction`: Employee's satisfaction with their job (1-4 scale).
- `WorkLifeBalance`: Employee's perceived work-life balance (1-4 scale).
- `TrainingTimesLastYear`: Number of training sessions attended by the employee last year (scaled).
- `Attrition`: **Target variable**: Whether the employee left the company (Yes/No).

## Project Workflow & Thought Process

My approach to this supervised learning project followed a structured methodology, emphasizing data quality, thorough exploration, and robust model building.

### 1. Data Understanding & Initial Inspection
- **Objective:** Gain a foundational understanding of the dataset's structure, content, and initial quality.
- **Steps:**
    - Loaded essential libraries: `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`.
    - Loaded the `employee_attrition.csv` dataset.
    - Used `data.head()` to inspect the first few rows and understand column content.
    - Employed `data.info()` to check data types and identify non-null counts, revealing missing values in `Age`, `MonthlyIncome`, and `JobSatisfaction`.
    - Utilized `data.describe()` to obtain descriptive statistics for numerical columns, observing ranges and distributions.
- **Thought Process:** Initial inspection is crucial for identifying immediate data quality issues (missing values, incorrect types) and understanding the basic characteristics of the data.

### 2. Data Cleaning & Preprocessing
- **Objective:** Prepare the raw data for modeling by handling missing values, irrelevant features, and transforming variables.
- **Steps:**
    - **Remove Irrelevant Features:** Dropped `EmployeeID` as it's a unique identifier and holds no predictive value.
    - **Handle Missing Values:**
        - Identified columns with missing values: `Age` (~10%), `MonthlyIncome` (~10%), `JobSatisfaction` (~9.9%).
        - Visualized missing values using `missingno` to understand their distribution and patterns.
        - **Imputation Strategy:** Decided to impute missing numerical values using appropriate methods (e.g., median for skewed distributions, mean for more symmetrical ones, or mode for categorical-like numerical scales). (The notebook would detail the specific imputation method used for each column, e.g., median imputation for `MonthlyIncome` to handle potential outliers, and mode for `JobSatisfaction` if it's treated ordinally).
    - **No Duplicate Handling:** (Based on common attrition datasets, duplicates are less common, but if identified, they would be handled here).
- **Thought Process:** Addressing missing data is paramount. Visualizing missingness helps communicate the problem to stakeholders. The choice of imputation method depends on the nature of the data and its distribution.

### 3. Exploratory Data Analysis (EDA)
- **Objective:** Uncover patterns, relationships, and key insights that inform feature engineering and model selection.
- **Steps & Key Insights:**
    - **Univariate Analysis:**
        - Visualized the distribution of `Attrition` (target variable) to understand class imbalance.
        - Analyzed distributions of `Age`, `MonthlyIncome`, `YearsAtCompany`, `TrainingTimesLastYear` using histograms and box plots to understand their spread and identify outliers.
        - Examined the distribution of categorical features (`Gender`, `Department`, `OverTime`, `JobSatisfaction`, `WorkLifeBalance`) using bar plots.
    - **Bivariate Analysis:**
        - Explored the relationship between `Attrition` and other features.
        - **Key Insights Derived:**
            - **OverTime:** Employees working overtime show a significantly higher attrition rate. This is a critical insight for HR.
            - **JobSatisfaction & WorkLifeBalance:** Lower `JobSatisfaction` and poorer `WorkLifeBalance` are strongly correlated with higher attrition.
            - **YearsAtCompany:** Attrition tends to be higher among employees with fewer years at the company, possibly indicating early-career turnover or lack of long-term engagement.
            - **MonthlyIncome:** While complex, very low or very high `MonthlyIncome` might show different attrition patterns.
            - **Department:** Certain departments might have higher attrition rates than others.
    - **Multivariate Analysis (Correlation Heatmap):**
        - Visualized correlations between numerical features.
        - Identified strong correlations (e.g., `YearsAtCompany` and other tenure-related metrics if available).
        - Understood potential multicollinearity among features.
- **Thought Process:** EDA is an iterative process. Starting with univariate analysis gives a baseline understanding. Bivariate analysis helps identify potential drivers of attrition. Correlation analysis confirms relationships and informs feature selection.


📈 Key Insights from EDA
OverTime was a strong predictor of attrition. Employees working late often leave.

Low Job Satisfaction and poor Work-Life Balance strongly correlated with leaving.

Short Tenure was linked to higher turnover.

Some departments showed above-average attrition (potential internal HR red flag).


### 4. Data Preprocessing for Machine Learning
- **Objective:** Transform the cleaned data into a format suitable for machine learning algorithms.
- **Steps:**
    - **Encoding Categorical Variables:**
        - `Gender`, `Department`, `OverTime`, `Attrition` (target) were converted into numerical representations. `OneHotEncoder` would be suitable for `Department` and `Gender` to avoid ordinality assumptions, while `LabelEncoder` could be used for `OverTime` and `Attrition` (binary).
    - **Feature Scaling:**
        - Numerical features (`Age`, `MonthlyIncome`, `YearsAtCompany`, `TrainingTimesLastYear`) were scaled (e.g., using `StandardScaler` or `MinMaxScaler`) to normalize their ranges, which is crucial for distance-based algorithms and to prevent features with larger values from dominating the model.
    - **Feature and Target Split:** Separated the dataset into independent variables (features, `X`) and the dependent variable (target, `y` - `Attrition`).
- **Thought Process:** Proper encoding and scaling are fundamental steps to ensure that machine learning algorithms can correctly interpret and process the data, leading to better model performance.

### 5. Addressing Class Imbalance
- **Objective:** Mitigate the impact of imbalanced classes on model performance, especially when the minority class (attrition) is of high interest.
- **Steps:**
    - Identified that the `Attrition` class was imbalanced (typically fewer 'Yes' than 'No').
    - Applied the **Synthetic Minority Over-sampling Technique (SMOTE)** to the training data. SMOTE creates synthetic samples of the minority class, helping the model learn from a more balanced distribution.
- **Thought Process:** Class imbalance can lead to models that perform well on the majority class but poorly on the minority class. SMOTE is a widely used technique to address this, ensuring the model is sensitive to predicting attrition.

### 6. Model Development & Evaluation
- **Objective:** Train, evaluate, and select the best predictive model for employee attrition.
- **Steps:**
    - **Data Splitting:** Divided the preprocessed data into training and testing sets (e.g., 80% training, 20% testing).
    - **Model Selection & Training:**
        - Explored various supervised classification algorithms. 
        - **Leveraged AutoML (e.g., PyCaret):**
            - Used an AutoML library to automate model selection, hyperparameter tuning, and cross-validation. This allowed for rapid experimentation with multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM, LightGBM, XGBoost, etc.) and identification of optimal configurations.
            - AutoML significantly accelerates the iterative process of model building and tuning.


    - **Evaluation Metrics:** Focused on metrics relevant to imbalanced classification and business impact:
        - **Accuracy:** Overall correct predictions.
        - **Precision:** Of all predicted attritions, how many were actually attrition? (Minimizing false positives)
        - **Recall (Sensitivity):** Of all actual attritions, how many did the model correctly identify? (Minimizing false negatives - crucial for identifying employees at risk).
        - **F1-Score:** Harmonic mean of precision and recall.
        - **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):** Measures the model's ability to distinguish between classes, robust to imbalance.
        - **Confusion Matrix:** Provides a detailed breakdown of correct and incorrect classifications.
- **Thought Process:** AutoML streamlines the model development process, allowing focus on business understanding and interpretation. Selecting appropriate evaluation metrics (beyond just accuracy) is vital for imbalanced datasets and ensures the model addresses the business problem effectively.

### 7. Model Interpretation & Business Recommendations
- **Objective:** Translate model findings into actionable strategies for GTS.
- **Key Insights & Recommendations:**
    - **Top Attrition Drivers:** Identified features with the highest impact on attrition (e.g., OverTime, JobSatisfaction, WorkLifeBalance, YearsAtCompany).
    - **Targeted Interventions:**
        - **Overtime Management:** Implement policies to reduce excessive overtime, as it's a significant attrition factor.
        - **Employee Engagement:** Focus on improving `JobSatisfaction` through initiatives like career development, recognition programs, and a positive work environment.
        - **Work-Life Balance Programs:** Introduce or enhance programs supporting `WorkLifeBalance` (e.g., flexible hours, remote work options, mental wellness support).
        - **Onboarding & Early Career Support:** Develop robust onboarding and mentorship programs for new employees to address early-career attrition.
    - **Departmental Focus:** If specific departments show higher attrition, investigate root causes within those teams (e.g., management style, workload).
    - **Proactive Retention:** Use the predictive model to identify high-risk employees early and initiate personalized retention efforts (e.g., stay interviews, career pathing discussions, compensation reviews).


✅ Final Model Output
F1-Score: ~0.82

ROC-AUC: ~0.88

Recall: High sensitivity in identifying attrition cases

🔍 SMOTE was used to handle class imbalance effectively.


## Tools & Libraries Used

-   **Programming Language:** Python
-   **Data Manipulation:** `pandas`, `numpy`
-   **Data Visualization:** `matplotlib.pyplot`, `seaborn`, `missingno`
-   **Machine Learning:** `scikit-learn` (for preprocessing, model evaluation), `imbalanced-learn` (for SMOTE)
-   **Automated Machine Learning (AutoML):** `PyCaret`
-   **Jupyter Notebook:** For interactive analysis and documentation.

## Files in this Repository

-   `Employee Attrition - Supervised Learning (Classification) with AutoML`: The main Jupyter Notebook containing all the code for data loading, cleaning, EDA, feature engineering, model training, and evaluation, including AutoML implementation.
-   `employee_attrition.csv`: The raw dataset used for the project.
-   `cleaned_employee_attrition_balanced_corrected`: The cleaned and balanced dataset used for learning.
-   `Case Study_ Supervised learning - Classification.pptx.pdf`: The presentation slides outlining the business problem, project workflow, and key findings.
-   `README.md`: This file.


💡 Business Recommendations
Reduce Overtime: Flag and manage departments with excessive overtime – a key attrition driver.

Boost Engagement: Focus on improving JobSatisfaction through training, internal mobility, and recognition.

Work-Life Programs: Offer flexible work options and wellness support.

Early Career Retention: Support employees within their first 2 years – the most vulnerable group.

Departmental Review: Dive deeper into high-attrition departments to discover management or culture issues.


## Tools & Libraries Used

-   **Programming Language:** Python
-   **Data Manipulation:** `pandas`, `numpy`
-   **Data Visualization:** `matplotlib.pyplot`, `seaborn`, `missingno`
-   **Machine Learning:** `scikit-learn` (for preprocessing, model evaluation), `imbalanced-learn` (for SMOTE)
-   **Automated Machine Learning (AutoML):** `PyCaret`
-   **Jupyter Notebook:** For interactive analysis and documentation.
