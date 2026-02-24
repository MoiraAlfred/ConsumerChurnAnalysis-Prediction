# ConsumerChurnAnalysisPrediction
Power BI, SQL and Machine Learning Portfolio Project.

An end-to-end customer churn analytics solution integrating SQL Server ETL, Power BI visualization, and a Python Random Forest machine learning model to analyze historical churn behavior and predict future churners.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Business Context and Objectives](#-business-context-and-objectives)
- [Data Description](#-data-description)
- [Methodology](#-methodology)
- [KPI Framework](#-kpi-framework)
- [Dashboard Design and Data Storytelling Logic](#-dashboard-design-and-data-storytelling-logic)
- [Machine Learning Model — Random Forest](#-machine-learning-model--random-forest)
- [Technical Implementation](#-technical-implementation)
- [Skills Demonstrated](#-skills-demonstrated)
- [Key Insights and Findings](#-key-insights-and-findings)
- [Conclusion](#-conclusion)

---

## Project Overview

This project presents an end-to-end customer churn analysis solution for a telecommunications company, built across three integrated layers:

- **SQL Server** — data ingestion, null cleaning, and ETL into staging and production tables with views
- **Power BI** — two-page interactive dashboard covering executive summary and predictive churn profiling
- **Python (Random Forest)** — machine learning model trained on historical churn data, applied to newly joined customers to predict future churners

The solution is both **retrospectively analytical** — identifying who churned, under what conditions, and for what reasons — and **prospectively predictive** — flagging newly joined customers most likely to churn before the event occurs.

---

## Business Context and Objectives

Customer churn is one of the most commercially significant metrics in subscription-based industries. Acquiring a new customer typically costs five to seven times more than retaining an existing one, making even modest improvements in churn rate financially impactful.

This project addresses two distinct business problems:

1. **Retrospective analysis** — Understanding historical churn patterns across demographics, geographies, account types, and services to inform targeted marketing and retention campaigns
2. **Prospective prediction** — Identifying, from among recently joined customers, those most likely to churn so that proactive intervention can occur before the churn event takes place

**Project Goals:**
- Analyze customer data across all available dimensions
- Identify the churn profile and surface areas for marketing campaign design
- Build a method to predict future churners from new customer data

---

## Data Description

The dataset is a publicly available telecom customer churn dataset loaded into SQL Server as the primary data source. It captures a cross-sectional view of customer status including demographic, geographic, service, payment, and churn outcome information.

| Attribute | Detail |
|---|---|
| **Source** | Publicly available telecom churn dataset (CSV) |
| **Volume** | 6,418 rows |
| **Grain** | One row per customer — Customer ID is unique |
| **Demographic Fields** | Gender, Age, Married |
| **Geographic Fields** | State |
| **Account Fields** | Contract, Payment Method, Tenure in Months, Monthly Charge, Total Revenue, Total Refunds, Number of Referrals |
| **Service Fields** | Phone Service, Multiple Lines, Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data |
| **Churn Fields** | Customer Status (Churned / Stayed / Joined), Churn Category, Churn Reason |
| **Null Handling** | Null audit run across all columns via SQL; replaced with contextual defaults in production table |
| **Staging Table** | `dbo.stg_churn` — raw imported data, no transformations |
| **Production Table** | `dbo.prod_churn` — null-cleaned source for all downstream steps |
| **Views** | `vw_ChurnData` (Churned + Stayed); `vw_JoinData` (Joined) |

---

## Methodology

The project was executed across five sequential stages:

### Stage 1 — SQL Server ETL and Data Preparation
- Created `db_Churn` database and loaded raw CSV into staging table `stg_churn` via Import Flat File wizard
- Ran null audit query across all columns using `ISNULL` conditional sums
- Created production table `prod_churn` by selecting all columns with null values replaced using `ISNULL`
- Created two views: `vw_ChurnData` (Churned + Stayed) for model training, `vw_JoinData` (Joined) for prediction
- Explored data distributions using `GROUP BY` with percentage calculations across gender, contract, status, and state columns

### Stage 2 — Power Query Transformation and Data Modelling
- Imported `prod_churn` into Power BI via SQL Server connector (Import mode)
- Created `Churn Status` binary column (1 = Churned, 0 = otherwise) and `Monthly Charge Range` bucket column
- Built `mapping_AgeGrp` and `mapping_TenorGrp` reference tables with group label and integer sort columns
- Confirmed automatic relationships between mapping tables and production table in model view
- Created `prod_Services` unpivoted table from all service columns for matrix visualization

### Stage 3 — DAX Measure Development
- Created dedicated measures table `tbl_Measures` to keep measures separate from data tables
- Wrote four core explicit DAX measures: Total Customers, New Joiners, Total Churn, Churn Rate
- Added prediction page measures: Count Predicted Churn, Title Predicted Churn (with zero-default to prevent blank card states)

### Stage 4 — Power BI Dashboard Design
- Built Summary page covering KPI cards, demographic, account, geographic, churn distribution, and service sections
- Created Churn Reason tooltip page for drill-through on the Churn Category chart
- Applied PowerPoint-designed PNG background to both pages
- Configured Sort by Column on Age Group and Tenure Group using integer sort columns
- Configured edit interactions to filter mode on selected visuals
- Added page navigation buttons to both report pages

### Stage 5 — Machine Learning Model and Prediction
- Exported SQL Server views to Excel; loaded into Jupyter Notebook using pandas
- Preprocessed training data: dropped irrelevant columns, encoded categoricals with `LabelEncoder`, manually mapped target variable (Stayed: 0, Churned: 1)
- Split data 80/20 with `random_state=42`; trained `RandomForestClassifier` with 100 estimators
- Evaluated model using confusion matrix, classification report, and feature importance chart
- Applied trained model to `vw_JoinData` prediction dataset; exported predicted churners (class 1) to CSV for Power BI import

---

## KPI Framework

### Total Customers
> Count of all unique customer records — establishes the denominator for all rate-based calculations.

```dax
Total Customers = COUNT(prod_churn[Customer ID])
```

### New Joiners
> Count of recently acquired customers — enables net growth monitoring alongside churn volume.

```dax
New Joiners = CALCULATE(COUNT(prod_churn[Customer ID]), prod_churn[Customer Status] = "Joined")
```

### Total Churn
> Sum of binary Churn Status column — provides the raw magnitude of customer loss.

```dax
Total Churn = SUM(prod_churn[Churn Status])
```

### Churn Rate
> Proportion of all customers who have churned — the normalized efficiency metric enabling fair comparison across segments and geographies.

```dax
Churn Rate = [Total Churn] / [Total Customers]
```

---

## Dashboard Design and Data Storytelling Logic

### Page 1 — Executive Summary

The summary page follows a structured top-to-bottom information hierarchy:

- **KPI Cards (top row)** — Total Customers, New Joiners, Total Churn, and Churn Rate for immediate at-a-glance context
- **Demographic Section** — Donut chart (churn by gender) with gender image overlay; combo chart (total customers + churn rate) by age group bucket
- **Account Section** — Churn rate by payment method and contract type; tenure group combo chart; Monthly Charge Range and Marital Status slicers as global cross-page filters
- **Geographic Section** — Top 5 states by churn rate using Top N visual-level filter
- **Churn Distribution** — Bar chart by churn category; hover-triggered tooltip page surfaces churn reason drill-through without consuming canvas real estate
- **Services Section** — Matrix using unpivoted `prod_Services` table with Percent of Row Total and data bars — consolidates all service columns into a single visual footprint

### Page 2 — Churn Prediction Profile

Populated entirely from the Random Forest prediction output:

- **Revenue Grid** — Customer IDs with Total Revenue, Total Refunds, and Number of Referrals
- **Gender Cards** — Male and female churner counts with `+ 0` zero-default measure to prevent blank card states on filtering
- **Profile Charts** — Age group, marital status, tenure, payment method, contract, and state breakdowns scoped to predicted churner population
- **Navigation** — Page navigation buttons on both pages for user-driven switching between summary and prediction views

---

## Machine Learning Model — Random Forest

### Why Random Forest?

Random Forest was selected for its strong empirical track record in binary classification on tabular data, robustness to irrelevant features, and interpretability through feature importance scoring. The ensemble structure — training multiple decision trees on random data and feature subsets, then aggregating by majority vote — reduces overfitting risk without requiring extensive hyperparameter tuning.

### Preprocessing

```python
# Drop irrelevant and outcome-adjacent columns
df.drop(columns=['Customer ID', 'Churn Category', 'Churn Reason'], inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

# Manually map target variable (Stayed = 0, Churned = 1)
df['Customer Status'] = df['Customer Status'].map({'Stayed': 0, 'Churned': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
```

### Model Evaluation

| Metric | Class 0 — Stayed | Class 1 — Churned |
|---|---|---|
| Precision | 86% | 78% |
| Recall | 92% | 65% |
| F1 Score | 89% | 71% |
| Support | 847 | 355 |
| **Overall Accuracy** | **84%** | |

**Confusion Matrix:**

|  | Predicted: Stayed | Predicted: Churned |
|---|---|---|
| **Actual: Stayed** | 783 (True Negative) | 64 (False Positive) |
| **Actual: Churned** | 126 (False Negative) | 229 (True Positive) |

> **Note:** The model performs more strongly on the Stayed class due to class imbalance (2.4× more Stayed instances in the test set). Oversampling techniques such as SMOTE are identified as a future enhancement to improve recall on the Churned class.

### Prediction Output

The trained model was applied to `vw_JoinData` (newly joined customers) following identical preprocessing. The model predicted **378 customers** as likely future churners — exported to CSV and imported into Power BI as the data source for the prediction page.

---

## Technical Implementation

| Component | Detail |
|---|---|
| **Database** | Microsoft SQL Server |
| **SQL Client** | SQL Server Management Studio (SSMS) |
| **Staging Table** | `dbo.stg_churn` |
| **Production Table** | `dbo.prod_churn` |
| **SQL Views** | `vw_ChurnData`, `vw_JoinData` |
| **BI Tool** | Microsoft Power BI Desktop |
| **Data Connectivity** | SQL Server — Import mode |
| **Power Query** | Binary Churn Status column; Monthly Charge Range buckets; Age Group and Tenure Group mapping tables with integer sort columns; Unpivoted Services table |
| **Data Model** | `prod_churn` (fact); `mapping_AgeGrp`, `mapping_TenorGrp`, `prod_Services` (dimensions) |
| **DAX Measures** | Total Customers, New Joiners, Total Churn, Churn Rate, Count Predicted Churn, Title Predicted Churn |
| **Axis Sorting** | Sort by Column — Age Group sorted by `AgeGrpSort`; Tenure Group sorted by `TenorGrpSort` |
| **Tooltip Page** | Churn Reason page — Canvas type: Tooltip; linked via General > Tooltip on Churn Category chart |
| **Background** | PowerPoint-designed PNG at 0% transparency via Canvas Background |
| **Services Matrix** | Unpivoted `prod_Services`; Churn Status as column; Percent of Row Total; Data Bars enabled |
| **Interactions** | Filter mode configured on selected visuals |
| **Navigation** | Rounded rectangle buttons with Page Navigation action |
| **ML Environment** | Jupyter Notebook — Python 3 |
| **ML Libraries** | `pandas`, `scikit-learn`, `matplotlib` |
| **Model** | `RandomForestClassifier` — 100 estimators, `random_state=42` |
| **Train/Test Split** | 80/20, `random_state=42` |
| **Encoding** | `LabelEncoder` for categoricals; manual binary map for target variable |
| **Prediction Output** | CSV export — predicted churners (class 1) only |

---

## Skills Demonstrated

| Category | Skills |
|---|---|
| **Database Engineering** | SQL Server database and table creation, null auditing, ETL staging/production conventions, view design |
| **Power BI** | Power Query transformation, reference table design, dimensional data modelling, DAX measure development, Sort by Column, tooltip pages, matrix visualization with unpivoted data, data bar formatting, format painter, edit interactions, background design |
| **Python / ML** | Binary classification problem framing, feature encoding, target variable handling, Random Forest training and evaluation, confusion matrix and classification report interpretation, feature importance analysis, prediction output generation |
| **Analytics** | Churn analysis methodology, customer segmentation, cohort-based predictive profiling, insight communication for stakeholders |

---

## Key Insights and Findings

### Demographics
- Female customers represent ~64% of the total churner population
- Within this group, the **50+ age bracket** accounts for the largest share of churners — a high-priority segment for targeted retention campaigns

### Competitive Churn
- Competitor-related churn is the **single largest churn driver** in the dataset
- Tooltip drill-through reveals two dominant sub-reasons: **competitor had better devices** and **competitor made better offers** — direct signals for product and pricing teams

### Service Risk Patterns
Using a 60% churn rate threshold as a flag:

- **High churn among non-subscribers:** Customers not subscribed to Device Protection, Online Backup, Online Security, Premium Support, and Streaming Music are churning at >60% — indicating low product attachment among unenrolled customers
- **High churn among subscribers:** Customers subscribed to Internet Service, Paperless Billing, Phone Service, and Unlimited Data are also churning at elevated rates — suggesting perceived quality or value gaps in these core services

### Predictive Output
- The Random Forest model identified **378 newly joined customers** as likely future churners
- The churn prediction page enables analysts to profile this cohort across all dimensions and prioritize proactive outreach before churn events occur

---

## Conclusion

This project demonstrates a complete customer churn analytics solution across the full analytics stack — from relational database ETL through business intelligence visualization to applied machine learning. The solution addresses both retrospective analysis and prospective prediction within a single Power BI report, providing stakeholders with both the historical context to understand past churn and the forward-looking intelligence to prevent future churn.

Each technical decision reflects analytical best practice: staging and production table separation for data integrity, reference mapping tables for dimensional axis control, explicit DAX measures for transparency, an unpivoted service table for compact multi-attribute visualization, and a reproducible Random Forest classifier with class-label conventions aligned to standard binary classification practice.

---

*Developed as part of a data analytics portfolio. The telecom customer churn dataset is publicly available and used for educational and portfolio demonstration purposes.*
