***Cardiovascular-Disease-Prediction-Using-Machine-Learning-and-XAI***

#Overview:

This project focuses on predicting cardiovascular diseases (CVD), specifically heart attacks, using machine learning (ML) models and explainable AI (XAI) techniques. Cardiovascular diseases remain the leading cause of death globally, emphasizing the need for accurate, reliable, and efficient prediction models.

Using advanced data mining and ML techniques, we aim to predict the likelihood of heart disease based on a patient's medical features. Additionally, we incorporate XAI methods to interpret model predictions, providing insights into critical features influencing the outcomes.

---
***Key Features***

Machine Learning Models: Evaluation of several algorithms such as Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, Naive Bayes, XGBoost, and K-Nearest Neighbors (KNN).
Performance Metrics: Achieved a 95.60% accuracy using the KNN algorithm.
Explainable AI (XAI): Used LIME (Local Interpretable Model-agnostic Explanations) to identify and explain the importance of features contributing to predictions.
Feature Importance Analysis: Highlighted the most impactful medical features, such as chest pain (cp), old peak (oldpeak), and others, for predicting heart disease.

---
***Data and Methodology***

#Dataset

The dataset used includes critical medical parameters such as:

->cp (chest pain type): Strongly linked to heart attack prediction.
->oldpeak: Reflects ST Depression in ECG, indicating myocardial ischemia.
->ca (number of major vessels): Related to blood flow conditions.
->Other parameters include age, cholesterol levels, blood pressure, and more.

---
***Machine Learning Models***

The following models were trained and evaluated:

->Logistic Regression

->Decision Tree

->Random Forest

->Support Vector Classifier

->Naive Bayes

->XGBoost

->K-Nearest Neighbors (KNN)

---
***Explainable AI (XAI)***

We used LIME to:

->Visualize feature importance for individual predictions.

->Provide interpretable insights for both positive and negative predictions.

---
***Results***

Performance Metrics (KNN)

->Accuracy: 95.60%

->Precision (Positive): 0.97

->Recall (Positive): 0.94

->F1-Score (Positive): 0.95

Feature Importance

Using LIME, we identified the most impactful features for heart disease prediction:

->Orange Features: Indicate higher importance and likelihood of heart disease.

->Blue Features: Indicate lower importance and likelihood of heart disease.

->Important Features: cp, oldpeak, ca, etc.

->Value Counts: Highlighting the proportion of patients at risk.
