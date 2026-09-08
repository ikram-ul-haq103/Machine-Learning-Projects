# 🔧 Predictive Maintenance System for Industrial Machines

An end-to-end Machine Learning project that predicts industrial machine failures using sensor data. The project focuses on **imbalanced classification, SMOTE, model comparison, and hyperparameter tuning**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

The goal is to detect potential machine failures before they occur, helping reduce:

* Unplanned downtime
* Maintenance costs
* Equipment damage
* Safety risks

The project follows a complete ML workflow:

```text
Data → EDA → Cleaning → Feature Engineering
     → Train/Test Split → Scaling → SMOTE
     → Model Training → Evaluation → Tuning
```

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

* **Samples:** 10,000
* **Target:** `Machine failure`
* **Type:** Binary Classification
* **Failure Class:** ~3%
* **Non-Failure Class:** ~97%

**Source:** UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset

### Features Used

* `Type`
* `Air temperature`
* `Process temperature`
* `Rotational speed`
* `Torque`
* `Tool wear`

Identifier and failure-mode columns were removed during preprocessing.

---

## 🤖 Machine Learning Models

The following classifiers were compared:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

> **Recall is prioritized** because missing an actual machine failure can be more costly than generating a false alarm.

---

## ⚖️ Handling Class Imbalance

The dataset contains approximately **97% non-failure and 3% failure** records.

To address this imbalance, **SMOTE** was applied only to the training data.

This helps the models learn patterns from the minority failure class more effectively.

---

## 🧪 Hyperparameter Tuning

Random Forest was further optimized using:

* `GridSearchCV`
* `RandomizedSearchCV`
* 4-Fold Cross-Validation
* F1 Score as the optimization metric

---

## 🛠️ Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

---

## 📁 Project Structure

```text
Predictive-Maintenance/
│
├── Predictive_Maintenance_System_for_Industrial_Machines.ipynb
├── ai4i2020.csv
└── README.md
```

---

## ▶️ Run the Project

```bash
git clone https://github.com/<your-username>/predictive-maintenance.git
cd predictive-maintenance
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn notebook
jupyter notebook
```

Open:

```text
Predictive_Maintenance_System_for_Industrial_Machines.ipynb
```

Run all cells in order.

---

## 💡 Key Takeaways

* Class imbalance is a major challenge in predictive maintenance.
* SMOTE improves learning of the minority failure class.
* Recall is more important than accuracy for detecting failures.
* Random Forest showed strong overall performance.
* Hyperparameter tuning was used to further improve the model.

---

## 👤 Author

**Ikram Ul Haq**

BS Software Engineering — University of Sargodha

Interested in Machine Learning, AI Engineering

---

## 📄 License

This project is for educational and portfolio purposes.

Dataset: **AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository**

