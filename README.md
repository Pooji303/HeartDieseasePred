# 🏥 Heart Disease Prediction
## 📥 Dataset: [Download Dataset](https://d3ilbtxij3aepc.cloudfront.net/projects/CDS-Capstone-Projects/PRCP-1016-HeartDieseasePred.zip)

This project applies machine learning techniques to predict heart disease based on patient health records. It involves data preprocessing, feature engineering, model training, and evaluation.

## 📌 Project Overview

Heart disease remains one of the leading causes of mortality worldwide. The goal of this project is to build a machine learning model to predict the likelihood of heart disease, aiding early diagnosis and intervention.

## 📚 Directory Structure

```
├── data
│   ├── raw/         # Original dataset
│   ├── processed/   # Cleaned and transformed data
│  
├── notebooks
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── feature_engineering.ipynb  # Feature selection and engineering
│   ├── model_training.ipynb   # Model selection and training
│   ├── model_evaluation.ipynb # Performance evaluation
│  
├── src
│   ├── preprocessing.py  # Data preprocessing functions
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction script
│  
├── reports
│   ├── results.md        # Analysis and model evaluation results
│  
└── README.md             # Project Documentation
```

## 🚀 Technologies Used

- **Python**
- **Scikit-learn**
- **XGBoost**
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **Seaborn & Matplotlib (for visualization)**

## 📊 Data Processing

- Standard scaling applied to normalize features.
- SMOTE used to handle class imbalance.
- Features selected based on exploratory data analysis.

## 🔍 Model Training

- **Model:** ML Models (
- **Hyperparameter tuning** performed using RandomizedSearchCV.
- **Performance metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.

## 📝 Results & Insights

- Support Vector Classifier (SVC) appears to be the best model overall for the dataset.With a Test Accuracy = 86.27%, Train Accuracy = 92.37%.It performs consistently well across all the key metrics.

- **Data Cleaning:**  
  - Outliers were removed using the **Interquartile Range (IQR) method** for features such as:
    - Resting blood pressure  
    - Serum cholesterol  
    - Oldpeak (ST depression)  
    - Age  
    - Maximum heart rate achieved  
  - Removing extreme values instead of replacing them helped maintain data integrity and reduced overfitting risk.

- **Feature Engineering:**  
  - The most influential categorical features include:
    - **Slope of peak exercise ST segment**
    - **Chest pain type**
    - **Number of major vessels**
    - **Fasting blood sugar level**
    - **Resting ECG results**
    - **Sex**
    - **Exercise-induced angina**

- **Model Performance:**  
  - The model was evaluated using standard metrics:
    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1-score**
    - **Confusion Matrix**
  - Hyperparameter tuning was performed using **RandomizedSearchCV** to improve performance.




