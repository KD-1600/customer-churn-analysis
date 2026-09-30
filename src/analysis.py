# ============================================================
# TELECOM CUSTOMER CHURN ANALYSIS
# Author: Your Name
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "data/Telco-Customer-Churn.csv"
OUTPUT_DIR = "outputs/charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("CUSTOMER CHURN ANALYSIS")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


# ============================================================
# 3. INITIAL DATA EXPLORATION
# ============================================================

print("\n" + "=" * 60)
print("FIRST FIVE ROWS")
print("=" * 60)

print(df.head())


print("\n" + "=" * 60)
print("COLUMN INFORMATION")
print("=" * 60)

print(df.info())


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())


print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print(df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\nCleaning data...")

# TotalCharges is sometimes stored as text.
# Convert it to numeric.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing TotalCharges values
print("\nMissing TotalCharges after conversion:")
print(df["TotalCharges"].isnull().sum())

# Fill missing TotalCharges using 0.
# These records correspond to customers with very low/zero tenure.
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Remove duplicate customer IDs if any
df = df.drop_duplicates(subset="customerID")

print("\nData cleaning completed.")


# ============================================================
# 5. BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. CHURN DISTRIBUTION
# ============================================================

churn_counts = df["Churn"].value_counts()

print("\n" + "=" * 60)
print("CHURN COUNTS")
print("=" * 60)

print(churn_counts)


churn_percentage = (
    df["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nChurn percentage:")
print(churn_percentage)


# Chart 1: Churn Distribution
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/01_churn_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 7. CHURN BY CONTRACT TYPE
# ============================================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\n" + "=" * 60)
print("CHURN RATE BY CONTRACT")
print("=" * 60)

print(contract_churn.round(2))


plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="Contract",
    y=df["Churn"].eq("Yes").astype(int)
)

plt.title("Average Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/02_churn_by_contract.png",
    dpi=300
)

plt.show()


# ============================================================
# 8. CHURN BY INTERNET SERVICE
# ============================================================

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\n" + "=" * 60)
print("CHURN RATE BY INTERNET SERVICE")
print("=" * 60)

print(internet_churn.round(2))


plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="InternetService",
    y=df["Churn"].eq("Yes").astype(int)
)

plt.title("Average Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/03_churn_by_internet_service.png",
    dpi=300
)

plt.show()


# ============================================================
# 9. CHURN BY PAYMENT METHOD
# ============================================================

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\n" + "=" * 60)
print("CHURN RATE BY PAYMENT METHOD")
print("=" * 60)

print(payment_churn.round(2))


plt.figure(figsize=(11, 6))

sns.barplot(
    data=df,
    x="PaymentMethod",
    y=df["Churn"].eq("Yes").astype(int)
)

plt.title("Average Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/04_churn_by_payment_method.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. TENURE ANALYSIS
# ============================================================

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ],
    include_lowest=True
)

tenure_churn = pd.crosstab(
    df["TenureGroup"],
    df["Churn"],
    normalize="index"
) * 100

print("\n" + "=" * 60)
print("CHURN RATE BY TENURE GROUP")
print("=" * 60)

print(tenure_churn.round(2))


plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="TenureGroup",
    y=df["Churn"].eq("Yes").astype(int)
)

plt.title("Average Churn Rate by Tenure Group")
plt.xlabel("Tenure")
plt.ylabel("Churn Rate")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/05_churn_by_tenure.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. MONTHLY CHARGES VS CHURN
# ============================================================

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/06_monthly_charges_vs_churn.png",
    dpi=300
)

plt.show()


# ============================================================
# 12. TENURE VS MONTHLY CHARGES
# ============================================================

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="Churn",
    alpha=0.6
)

plt.title("Tenure vs Monthly Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/07_tenure_vs_monthly_charges.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. ADDITIONAL SERVICES AND CHURN
# ============================================================

service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport"
]

service_results = []

for column in service_columns:

    churn_rate = (
        df.groupby(column)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )

    for category, rate in churn_rate.items():

        service_results.append({
            "Service": column,
            "Category": category,
            "ChurnRate": rate
        })


service_df = pd.DataFrame(service_results)

print("\n" + "=" * 60)
print("SERVICE CHURN ANALYSIS")
print("=" * 60)

print(service_df.round(2))


plt.figure(figsize=(12, 7))

sns.barplot(
    data=service_df,
    x="Service",
    y="ChurnRate",
    hue="Category"
)

plt.title("Churn Rate by Additional Services")
plt.xlabel("Service")
plt.ylabel("Churn Rate (%)")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/08_service_churn_analysis.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. CORRELATION ANALYSIS
# ============================================================

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

correlation = df[numeric_columns].corr()

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

print(correlation.round(2))


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/09_correlation_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. CREATE A SIMPLE CHURN TARGET
# ============================================================

df["ChurnFlag"] = (
    df["Churn"]
    .map({"Yes": 1, "No": 0})
)


# ============================================================
# 16. MACHINE LEARNING MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION MODEL")
print("=" * 60)

# Remove columns that should not be used as predictive features
X = df.drop(
    columns=[
        "customerID",
        "Churn",
        "ChurnFlag",
        "TenureGroup"
    ]
)

y = df["ChurnFlag"]


# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_columns
        ),
        (
            "cat",
            categorical_transformer,
            categorical_columns
        )
    ]
)


# Create model pipeline
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# Predictions
y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# Model metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\nModel Results")
print("-" * 40)

print(f"Accuracy: {accuracy:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/10_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Logistic Regression ROC Curve")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/11_roc_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 19. FINAL BUSINESS SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

overall_churn = (
    df["ChurnFlag"].mean() * 100
)

print(
    f"\nOverall churn rate: "
    f"{overall_churn:.2f}%"
)

print(
    "\nHighest churn contract category:"
)

contract_rates = (
    df.groupby("Contract")["ChurnFlag"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print(contract_rates.round(2))

print(
    "\nHighest churn internet service category:"
)

internet_rates = (
    df.groupby("InternetService")["ChurnFlag"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print(internet_rates.round(2))


print("\nAnalysis completed successfully!")

print(
    f"\nCharts have been saved in: "
    f"{OUTPUT_DIR}"
)