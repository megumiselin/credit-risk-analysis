# Credit Risk Analysis with Machine Learning

## Project Overview
This project analyzes **German Credit Data** to identify patterns of creditworthiness and predict the likelihood of loan default.  
The primary goal is to support **financial institutions** in minimizing default risk by leveraging machine learning models and deriving actionable business insights.

---

## 1) Dataset Summary
- **Source**: German Credit Data  
- **Size**: 1000 observations, 20 features  
- **Target Variable**: `Risk` (good = 0, bad = 1)  
- **Class Distribution**: 70% good loans, 30% bad loans (imbalanced dataset)

---

## 2) Exploratory Data Analysis (EDA)
- **Credit Amount & Duration**: Higher loan amounts and longer repayment durations correlate with higher risk of default.  
- **Age Factor**: Younger applicants show a greater likelihood of default; risk generally decreases with age.  
- **Housing & Account Balances**: Borrowers who own homes or have stronger financial reserves (savings/checking accounts) are less likely to default.  
- **Purpose of Credit**: Consumer loans (radio/TV, education) carry more risk than business-related loans.  
- **Demographics (Gender, Job)**: Minor differences, but financial and behavioral features dominate risk prediction.

---

## 3) Imbalance Handling
- Applied **SMOTE oversampling**  
- Used `class_weight="balanced"` for Logistic Regression and Random Forest  
- Emphasized **recall (bad loans)** to reduce missed risky customers

---

## 4) Model Performance Benchmark

| Model               | Accuracy | Precision (bad) | Recall (bad) | F1 (bad) | ROC_AUC |
|---------------------|----------|-----------------|--------------|----------|---------|
| Logistic Regression | 0.62     | 0.40            | 0.57         | 0.47     | 0.64    |
| Random Forest       | 0.72     | 0.58            | 0.25         | 0.35     | 0.64    |
| Gradient Boosting   | 0.75     | 0.69            | 0.33         | 0.45     | 0.68    |
| LightGBM            | 0.69     | 0.63            | 0.46         | 0.53     | 0.66    |
| XGBoost             | 0.66     | 0.61            | 0.39         | 0.48     | 0.65    |

**Best baseline**: Gradient Boosting  
**Balanced option**: LightGBM  

---

## 5) Threshold Tuning
- Default threshold = 0.5  
- Lowered to 0.3 for GradientBoosting:  
  - Accuracy ↓ ~0.53  
  - Recall (bad loans) ↑ ~0.75  

**Business Note**: In banking, recall is more important than accuracy. Missing a risky loan is costlier than rejecting a safe one.

---

## 6) Feature Importance & SHAP
- **Top Predictors**:  
  - Credit amount  
  - Loan duration  
  - Age  
- **Additional Influences**: Housing status, account balances, loan purpose  
- **SHAP analysis** confirmed that **financial profile > demographics** in predicting credit risk.

---

## 7) Business Insights  

### Credit amount and loan duration as strongest risk drivers  
Higher loan amounts combined with longer repayment periods significantly increase the likelihood of default. These customer segments should be subject to stricter credit policies, including additional collateral requirements or differentiated interest rates.  

### Impact of demographic factors  
The analysis indicates that risk decreases with age. Younger applicants represent a higher-risk group and therefore require more careful assessment during the credit approval process.  

### Customer financial profile as an indicator of reliability  
Homeowners are generally more reliable borrowers, while customers with limited savings or low account balances are more likely to default. For these segments, additional checks or documentation should be requested before granting credit.  

### Purpose of credit differentiates risk levels  
Consumer loans such as those for education or radio/TV purchases are associated with higher default probabilities, while business-related loans tend to be safer. Banks should adapt credit products and conditions accordingly.  

### Role of the model in decision-making  
Although the model’s predictive accuracy is moderate, the derived insights are strong enough to inform business strategy. The model should serve as a **decision-support tool rather than a final approval mechanism**, allowing risk managers to identify vulnerable segments early and take proactive measures.  


---

## 8) Conclusion
- **Gradient Boosting (threshold 0.3)** → Best when **recall is priority** (catch more bad loans).  
- **LightGBM** → Best when a **balance between accuracy and recall** is needed.  
- Business rules should complement ML predictions to build a robust risk strategy.  

This project shows how **machine learning + business analysis** can help credit risk management by combining predictive modeling with actionable insights.
