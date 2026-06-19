# Credit Risk Modelling and Loan Classification System

An end-to-end Machine Learning pipeline engineered in Python to evaluate financial records and predict loan applicant default risks. The system processes a high-dimensional consumer credit dataset, executes feature engineering, and benchmarks four predictive classification models to deliver optimized risk metrics.

---

## Table of Contents
1. [Dataset Overview and Shape](#dataset-overview-and-shape)
2. [Exploratory Data Analysis and Visualizations](#exploratory-data-analysis-and-visualizations)
3. [Dataset Partitioning and Train Test Split](#dataset-partitioning-and-train-test-split)
4. [Core Machine Learning Classifiers](#core-machine-learning-classifiers)
5. [Model Tuning and Optimization - Training, Prediction and Evaluation](#model-tuning-and-optimization---training-prediction-and-evaluation)
6. [Project Summary and Gained Core Skills](#project-summary-and-gained-core-skills)
7. [Local Installation Guide](#local-installation-guide)

---

## Dataset Overview and Shape

The application ingests a structured financial tracking dataset stored as `loan_detection.csv`. Initial preprocessing audits are executed to evaluate data types, identify missing records, and establish data constraints:
- **Total Dataset Dimensions:** 41,188 row entries across 60 dense feature columns.
---

## Exploratory Data Analysis and Visualizations

Exploratory Data Analysis (EDA) was performed to map target label distributions and identify key risk indicator dependencies:
- **Class Imbalance Distribution:** Audited the `Loan_Status_Label` target column, identifying 36,548 rejected applications (Class 0) and 4,640 approved applications (Class 1).
- **Data Visualization:** Generated localized distribution bar plots utilizing Matplotlib and Seaborn to contrast approved vs. rejected records.
- **Feature Dependencies:** Evaluated a correlation matrix against the `Loan_Status_Label` target vector to identify heavily weighted linear indicators of borrower default.

---

## Dataset Partitioning and Train Test Split

- **X_train Shape:** 32,950 samples across 59 predictive feature attributes.
- **X_test Shape:** 8,238 evaluation samples across 59 predictive feature attributes.
- **y_train / y_test Split:** Structured target arrays segmented into 32,950 training labels and 8,238 validation test labels respectively.

---

## Core Machine Learning Classifiers

Four distinct predictive classifiers were trained and benchmarked to evaluate risk prediction variance:

### 1. Logistic Regression
Implemented as a baseline linear classifier (`LogisticRegression()`).

### 2. SVM Classifier
Deployed a Support Vector Machine (`SVC()`) classifier.

### 3. KNN Classifier
Utilized a k-Nearest Neighbors model configured at `n_neighbors=5` to predict loan outcomes.

### 4. Decision Tree
Constructed an initial `DecisionTreeClassifier(max_depth=4)` to map out explicit structural flowcharts mapping feature thresholds.

---

## Model Tuning and Optimization - Training, Prediction and Evaluation

### Model Training
To maximize model performance and counter overfitting, an advanced optimization pipeline was designed.
- **Parameter Grid Inspected:** Evaluated split criteria parameters (`gini`, `entropy`, `log_loss`), selection splitters (`best`, `random`), maximum tree depths ranging from `1` to `5`, and feature options (`sqrt`, `auto`, `log2`).
- **Optimal Architecture Selected:** Selected the best configuration parameter outputs (`'criterion': 'entropy', 'max_depth': 3, 'max_features': 'log2', 'splitter': 'best'`) to build the optimized prediction engine.

### Model Prediction
The fine-tuned, cross-validated model architecture was deployed to generate array matrix default forecasts against both the training and the unseen evaluation validation sets.

### Model Evaluation
Evaluated macro-averaged and weighted F1-Scores to assess performance across heavily imbalanced target loan groups.

---

## Project Summary and Gained Core Skills

This project helped me gain comprehensive experience in data preprocessing, feature engineering, model training, evaluation, and interpretation within a financial risk assessment domain. 

---

## Local Installation Guide

Follow these steps to set up the data science environment and run the pipeline locally on your machine.

### 1. Clone the Repository
```bash
git clone 
cd your-credit-risk-repo
```

### 2. Install Machine Learning Libraries
Ensure your packet installers are up-to-date, then pull down the required data science packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the Machine Learning Pipeline
Execute the master model script file for EDA, model training, and diagnostic plot outputs:
```bash
python pipeline.py
```
