# Predicting Customer Churn using **Apache Spark** + **Orange Data Mining**

![Project Banner](images/banner_spark_orange.png)

> **Goal:** Predict whether a customer will churn using a clean, reproducible pipeline.
> 
> **Pipeline:** **Spark** handles preprocessing (ETL) ➜ **Orange** handles exploration, modeling, evaluation, and ranking.

---

## Table of Contents
1. [Project Summary](#project-summary)
2. [Dataset](#dataset)
3. [Methodology (Rubric-Ready)](#methodology-rubric-ready)
4. [Apache Spark Preprocessing (Code + Explanation)](#apache-spark-preprocessing-code--explanation)
5. [Orange Workflow (Detailed)](#orange-workflow-detailed)
6. [Results (Metrics Table)](#results-metrics-table)
7. [Model Selection & Justification](#model-selection--justification)
8. [How to Reproduce](#how-to-reproduce)
9. [Deliverables](#deliverables)
10. [References](#references)

---

## Project Summary
Customer churn prediction is a key ML task in service industries (telecom / credit card services). This repository demonstrates a **complete ML pipeline**:

- ✅ **Preprocessing with Apache Spark (PySpark)** to clean and standardize the raw dataset.
- ✅ **Exploration + Modeling with Orange** to build multiple models and compare them using standard metrics.
- ✅ **Evaluation + Ranking** to select the best model in an objective way.

---

## Dataset
- **Dataset type:** Tabular CSV
- **Target (Label):** `Churn`
- **Input features:** Customer/service attributes (categorical + numeric)

> **Note:** The raw dataset is read by Spark, cleaned, then exported as a clean CSV that Orange can load.

---

## Methodology (Rubric-Ready)
This section is written to match common academic rubrics (Problem → Data → Preprocessing → Modeling → Evaluation → Results → Conclusion).

### 1) Problem Statement
Build and evaluate ML models that predict `Churn` (Yes/No) from customer attributes.

### 2) Data Preprocessing (Spark)
- Load raw CSV with schema inference.
- Standardize categorical text (trim spaces).
- Handle missing values (drop missing target, fill remaining nulls).
- Export **single clean CSV** for Orange.

### 3) Data Exploration (Orange)
Use visualization widgets (Feature Statistics, Distributions, Box Plot) to understand feature patterns and detect issues.

### 4) Model Training (Orange)
Train and compare multiple algorithms:
- Random Forest
- Decision Tree
- Logistic Regression
- kNN

### 5) Evaluation & Ranking
Evaluate using:
- **CA (Accuracy)**
- **AUC**
- **F1**
- **Precision**
- **Recall**
- **MCC**

Then apply **Rank** to identify the overall best performer based on multiple metrics.

---

## Apache Spark Preprocessing (Code + Explanation)

### Why Spark here?
Spark provides a **scalable ETL layer**. Even when datasets are not extremely large, Spark demonstrates a professional pipeline design:

> **ETL (Spark)** ➜ **ML (Orange)**

### Preprocessing Script
File: `spark_preprocessing.py`

Below is a **representative code snippet** (key parts) from the Spark preprocessing script:

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim

# 1) Start Spark
spark = SparkSession.builder.appName("TelcoCustomerChurn_Preprocess").getOrCreate()

# 2) Read CSV + infer data types
input_csv = r"C:\\Users\\Lenovo\\Desktop\\archive\\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = (spark.read.option("header", True)
              .option("inferSchema", True)
              .csv(input_csv))

# 3) Trim spaces in all string columns
for c, t in df.dtypes:
    if t == "string":
        df = df.withColumn(c, trim(col(c)))

# 4) Remove rows missing the target label
df = df.na.drop(subset=["Churn"])

# 5) Fill remaining null values
df = df.na.fill("Unknown")

# 6) Export a single CSV file for Orange
output_dir = r"C:\\Users\\Lenovo\\Desktop\\archive\\spark_output"
(df.coalesce(1)
   .write.mode("overwrite")
   .option("header", True)
   .csv(output_dir))

spark.stop()
```

### What each step does (fast, exam-style)
1. **SparkSession**: starts the Spark engine.
2. **Read CSV + inferSchema**: loads dataset and auto-detects column data types.
3. **Trim strings**: avoids inconsistent categories (e.g., `"Yes"` vs `" Yes "`).
4. **Drop missing Churn**: supervised learning requires a valid label.
5. **Fill nulls**: prevents training/evaluation errors due to missing values.
6. **coalesce(1) export**: produces **one** CSV file compatible with Orange.

---

## Orange Workflow (Detailed)

### Workflow Diagram
> Place your Orange workflow screenshot here:

![Orange Workflow](images/orange_workflow.png)

✅ The workflow is organized into **three stages**:

### Stage A — Data Exploration
**Widgets (as shown in the workflow):**
- **Feature Statistics**: summarize features and basic distributions.
- **Distributions**: visualize categorical/numeric distributions.
- **Box Plot**: detect outliers and understand spread.

**Why:** to validate the dataset quality before modeling.

### Stage B — Data Cleaning / Preparation
**Widgets (as shown):**
- **Edit Domain**: ensures correct data types and sets the target label (`Churn`).
- **Select Columns**: selects predictors and removes irrelevant fields.
- **Nominal / Ordinal**: manages categorical/ordered features.
- **Continuize**: converts categorical variables into numeric representations.
- **Merge Data**: combines prepared branches into one modeling-ready dataset.

**Why:** many ML models require numeric input and consistent feature encoding.

### Stage C — Modeling, Evaluation, and Prediction
**Data split:**
- **80% training** + **20% test** using the **Data Sampler** widgets.

**Learners used:**
- Random Forest
- Decision Tree
- Logistic Regression
- kNN

**Evaluation:**
- **Test & Score** computes metrics.
- **Rank** gives an overall comparison across metrics.

**Prediction output:**
- **Predictions** widget generates predicted labels.
- **Output** shows selected predictions.

---

## Results (Metrics Table)
The following table summarizes the evaluation metrics produced by the Orange **Test & Score** widget.

> Metrics are shown as **percentages** for readability.

| Model | AUC | Accuracy (CA) | F1 | Precision | Recall | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest** | **96.5%** | **94.0%** | **93.9%** | **93.8%** | **94.0%** | **76.9%** |
| Decision Tree | 78.3% | 91.9% | 91.6% | 91.6% | 91.9% | 68.3% |
| Logistic Regression | 90.8% | 89.2% | 88.4% | 88.4% | 89.2% | 55.8% |
| kNN | 86.4% | 88.6% | 88.1% | 87.8% | 88.6% | 54.5% |

---

## Model Selection & Justification
**Selected model:** **Random Forest**

**Why (short & strong):**
- Achieved the **best overall performance** across key metrics.
- Highest **AUC** (best discrimination), highest **Accuracy**, and strongest **MCC** (balanced performance).
- Consistently strong Precision/Recall/F1, indicating reliable churn detection.

---

## How to Reproduce

### A) Run Spark preprocessing
1. Install Spark + PySpark.
2. Update the `input_csv` and `output_dir` paths in `spark_preprocessing.py`.
3. Run:

```bash
python spark_preprocessing.py
```

This produces a cleaned CSV inside `spark_output/`.

### B) Run Orange workflow
1. Open Orange.
2. Load the workflow file: `orange_workflow.ows`.
3. In the **File** widget, point to the cleaned CSV created by Spark.
4. Run the workflow to reproduce metrics and ranking.

---

## Deliverables
- `spark_preprocessing.py` — Spark ETL preprocessing script
- `spark_output/` — cleaned dataset exported by Spark
- `orange_workflow.ows` — Orange workflow
- `images/orange_workflow.png` — workflow screenshot shown above
- `README.md` — this documentation

---

## References
- Apache Spark Documentation: https://spark.apache.org/docs/latest/
- Orange Data Mining Documentation: https://orangedatamining.com/docs/

---

### ✅ Setup Note (Images)
To enable the images in this README:
- Add your workflow screenshot as: `images/orange_workflow.png`
- (Optional) Add a banner image as: `images/banner_spark_orange.png`

