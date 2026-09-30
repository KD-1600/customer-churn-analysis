# 📊 Telecom Customer Churn Analysis

## 📌 Project Overview

This project analyzes customer churn for a telecommunications company using Python and the IBM Telco Customer Churn dataset.

The objective is to identify customer characteristics and service patterns associated with churn and generate business insights that could support customer retention analysis.

The project covers the complete data-analysis workflow:

* Data loading
* Data cleaning
* Exploratory Data Analysis (EDA)
* Customer segmentation
* Data visualization
* Churn-rate analysis
* Correlation analysis
* Logistic Regression classification
* Model evaluation
* Business insights

---

## 🎯 Business Problem

Customer churn is an important business problem for subscription-based companies.

The purpose of this analysis is to understand:

* How common customer churn is
* Which contract types are associated with different churn rates
* Whether customer tenure is related to churn
* How monthly charges relate to churn
* Whether payment method is associated with churn
* Whether internet service type is associated with churn
* Whether additional services are associated with different churn rates

The analysis is descriptive and exploratory; relationships identified in the dataset should not automatically be interpreted as causal effects.

---

## 📂 Dataset

The project uses the IBM Telco Customer Churn sample dataset.

The dataset contains 7,043 customers and 21 columns.

Important fields include:

* `customerID`
* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`
* `tenure`
* `PhoneService`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `MonthlyCharges`
* `TotalCharges`
* `Churn`

### Dataset source

IBM Telco Customer Churn sample dataset:

https://github.com/IBM/telco-customer-churn-on-icp4d

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
customer-churn-analysis/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── outputs/
│   └── charts/
│
├── src/
│   └── analysis.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project:

```bash
cd customer-churn-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

Run the main Python analysis:

```bash
python src/analysis.py
```

The analysis generates charts inside:

```text
outputs/charts/
```

---

## 📊 Analysis Performed

### 1. Churn Distribution

Analyzes the proportion of customers who churned versus those who remained.

### 2. Contract Analysis

Compares churn rates across:

* Month-to-month
* One year
* Two year

### 3. Internet Service Analysis

Compares churn rates among customers using different internet service types.

### 4. Payment Method Analysis

Examines churn patterns across different payment methods.

### 5. Tenure Analysis

Customers are divided into tenure groups:

* 0–12 months
* 13–24 months
* 25–48 months
* 49–72 months

### 6. Monthly Charges Analysis

Examines differences in monthly charges between customers who churned and those who stayed.

### 7. Additional Services

Analyzes churn in relation to:

* Online Security
* Online Backup
* Device Protection
* Tech Support

### 8. Correlation Analysis

Examines relationships between numerical variables such as:

* Tenure
* Monthly Charges
* Total Charges

---

## 🤖 Machine Learning

A Logistic Regression classification model is included as an additional component.

The workflow includes:

1. Train/test split
2. Numerical preprocessing
3. Categorical encoding
4. Model training
5. Churn prediction
6. Accuracy evaluation
7. Classification report
8. ROC-AUC evaluation
9. Confusion matrix
10. ROC curve

The model is included for learning and demonstration purposes rather than as a production churn prediction system.

---

## 📈 Key Insights

The exact numerical findings are generated when the analysis is run.

The analysis investigates whether churn differs across:

* Contract types
* Tenure groups
* Internet service types
* Payment methods
* Monthly charge levels
* Additional services

These relationships should be interpreted as associations in this dataset rather than proof that a particular factor directly causes churn.

---

## ⚠️ Limitations

* The dataset is a sample dataset rather than a live production customer database.
* It represents a customer snapshot rather than a full longitudinal history.
* The analysis does not establish causal relationships.
* The dataset does not contain enough temporal information for a true future churn forecasting evaluation.
* Model performance may differ on a real-world dataset.

---

## 👨‍💻 Skills Demonstrated

This project demonstrates:

* Python programming
* Pandas
* NumPy
* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Statistical comparison
* Feature preprocessing
* Logistic Regression
* Model evaluation
* Git
* GitHub
* Business-oriented data interpretation

---

## 📌 Future Improvements

Possible extensions include:

* SQL-based analysis
* Power BI dashboard
* Random Forest model
* Hyperparameter tuning
* Feature importance analysis
* Customer churn-risk dashboard
* Automated reporting

---

## 👤 Author

kapil dagar

B.Tech Student | Aspiring Data Analyst

GitHub: https://github.com/KD-1600

LinkedIn: https://www.linkedin.com/in/kapil-dagar-6bb794314?utm_source=share_via&utm_content=profile&utm_medium=member_android
