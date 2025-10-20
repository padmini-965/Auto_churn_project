# 🧠 AutoChurn — Insurance Customer Churn Prediction

AutoChurn is a machine learning project that predicts whether an insurance customer is likely to churn (leave the company).  
It uses data-driven insights to help insurers identify at-risk customers and take proactive retention actions.

---

## 🚀 Project Overview

This project applies **machine learning techniques** — including data cleaning, feature engineering, and advanced modeling — to build an accurate churn prediction model.  
The final model is deployed using **Streamlit** for an interactive, easy-to-use web interface.

---

## 🧩 Key Features

✅ Data Cleaning and Preprocessing  
✅ Feature Engineering and Outlier Handling  
✅ Handling Missing Data and Categorical Encoding  
✅ SMOTE Balancing for Class Imbalance  
✅ Model Comparison — Logistic Regression, Random Forest, XGBoost  
✅ Final Tuned **XGBoost Model (~0.70 AUC-ROC)**  
✅ Streamlit-based Deployment for Real-Time Predictions  

---

## 🧠 Machine Learning Models Used

| Model                | AUC-ROC | Accuracy | Remarks |
|----------------------|----------|-----------|----------|
| Logistic Regression  | 0.6848  | 0.84 | Baseline linear model |
| Random Forest        | 0.6935  | 0.88 | Captures non-linearities |
| XGBoost (Tuned)      | **0.6964** | 0.88 | Best performer |

**Final Model Used:** Tuned XGBoost

---

## 🧰 Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, imbalanced-learn  
- **Deployment:** Streamlit  
- **Version Control:** Git + GitHub  

---

## 📁 Project Structure

Autochurn_project/
│
├── app.py # Streamlit web app
├── final_churn_model.pkl # Trained XGBoost model
├── scaler.pkl # StandardScaler for preprocessing
├── feature_names.pkl # Feature order for model input
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── .gitignore # Ignore unnecessary files
│
├── data/
│ ├── autoinsurance_churn.csv # Raw dataset (optional)
│
└── notebooks/
├── churn_model_training.ipynb # Jupyter notebook with training process

**Install Dependencies**
pip install -r requirements.txt

**Run the Streamlit App**
streamlit run app.py

**Results Summary**
Best Model: Tuned XGBoost
AUC-ROC: 0.6964
Accuracy: 88%
Precision (Churn): 0.49
Recall (Churn): 0.42
Deployed Streamlit app predicts churn probability for new customer inputs.

## 📊 Usage
```bash
streamlit run app.py

This project is released under the MIT License— feel free to use and modify it.
