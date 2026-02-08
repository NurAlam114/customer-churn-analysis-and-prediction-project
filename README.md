# customer-churn-analysis-and-prediction-project


📌 Project Overview
This project focuses on predicting **customer churn** using machine learning techniques.  
Customer churn refers to customers who are likely to stop using a service.  
The goal of this project is to identify churn-prone customers based on their demographic, account, and service-related information.

This is an end-to-end **Data Science project**, covering data analysis, preprocessing, model implementation, and evaluation.

---

📊 Dataset
- The dataset contains customer information such as:
  - Gender
  - Tenure
  - Contract type
  - Payment method
  - Monthly & Total charges
- Target variable: **Churn** (Yes / No)

---

🔍 Exploratory Data Analysis (EDA)
Key insights from EDA:
- Customers with **month-to-month contracts** have higher churn rates
- **Early-tenure customers** are more likely to churn
- Contract type and payment method strongly influence churn behavior

EDA visualizations include churn distribution, tenure analysis, contract-wise churn, and payment method comparison.

---

## ⚙️ Data Preprocessing
- Handled missing and invalid values
- Encoded categorical variables using one-hot encoding
- Split data into training and testing sets
- Prepared data for machine learning models

---

🤖 Model Implementation
- Built a **binary classification model** to predict customer churn
- Trained the model on the processed dataset
- Evaluated performance using:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - Accuracy

Model Accuracy: ~72%

This result serves as a strong baseline with clear scope for further improvement.

---

📈 Model Evaluation
- The model performs well in identifying non-churn customers
- Churn prediction remains challenging due to class imbalance
- Evaluation metrics were chosen to reflect real-world business scenarios

---

🚀 Future Improvements
- Handle class imbalance using SMOTE or class weighting
- Hyperparameter tuning
- Try advanced models like Random Forest or XGBoost
- ROC-AUC based evaluation and threshold tuning

---

🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

📌 Conclusion
This project helped me understand how data science techniques can be applied to real-world business problems.  
It demonstrates the importance of data preprocessing, exploratory analysis, and proper model evaluation.

---

## 📬 Contact
If you have suggestions or feedback, feel free to connect with me on LinkedIn.
