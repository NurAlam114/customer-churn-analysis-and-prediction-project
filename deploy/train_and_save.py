import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1) load data
df = pd.read_csv("churn.csv")

# 2) target
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

# 3) one-hot encode (same as your notebook)
X = pd.get_dummies(X, drop_first=True)

# 4) split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6) save model + feature columns (VERY IMPORTANT)
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")

print("Saved: model.pkl and model_columns.pkl")
