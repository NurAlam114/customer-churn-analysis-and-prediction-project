import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Risk Checker")

model = joblib.load("deploy/model.pkl")
model_cols = joblib.load("deploy/model_columns.pkl")


st.write("Choose input method:")

tab1, tab2 = st.tabs(["Single Customer Form", "Upload CSV"])

def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    # one-hot encode exactly like training
    X = pd.get_dummies(raw_df, drop_first=True)

    # align columns with training
    X = X.reindex(columns=model_cols, fill_value=0)
    return X

with tab1:
    st.subheader("Enter new customer info")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=200, value=1)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=80.0)


    raw = pd.DataFrame([{
        "Tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly,
    }])

    if st.button("Predict Risk"):
        X = prepare_features(raw)
        pred = model.predict(X)[0]
        risk = "High Risk (May Churn)" if pred == 1 else "Low Risk (Likely Stay)"
        st.success(f"Prediction: {risk}")

with tab2:
    st.subheader("Upload CSV of new customers (same columns as churn.csv except 'Churn')")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        raw_df = pd.read_csv(file)

        if "Churn" in raw_df.columns:
            raw_df = raw_df.drop(columns=["Churn"])

        X = prepare_features(raw_df)
        preds = model.predict(X)

        out = raw_df.copy()
        out["Churn_Risk"] = ["High Risk" if p == 1 else "Low Risk" for p in preds]

        st.write("Preview:")
        st.dataframe(out.head(20))

        st.write("High Risk customers:")
        st.dataframe(out[out["Churn_Risk"] == "High Risk"].head(50))
