import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

try:
    health = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=2
    )

    if health.status_code == 200:
        st.sidebar.success(
            "API Status: Connected"
        )
    else:
        st.sidebar.error(
            "API Status: Offline"
        )

except:
    st.sidebar.error(
        "API Status: Offline"
    )
    
st.markdown("""
Upload a transaction dataset and identify
potentially fraudulent transactions using a
machine learning model trained on credit card
fraud data.
""")

st.sidebar.info(
    """
    Model: Random Forest

    Dataset:
    Credit Card Fraud Detection

    Features:
    Time, V1-V28, Amount

    Threshold:
    Adjustable
    """
)

threshold = st.slider(
    "Fraud Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)
uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    required_columns = [
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
        ]

    if missing:
        st.error(
            f"Missing columns: {missing}"
        )
        st.stop()
    st.write("Preview")

    st.dataframe(df.head())

    if st.button("Run Detection"):

        payload = {
            "transactions": df.to_dict(
                orient="records"
            ),
            "threshold": threshold
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=30
            )

            response.raise_for_status()

            result = response.json()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.success(
            f"Frauds detected: "
            f"{result['fraud_count']}"
        )

        df["Prediction"] = result["predictions"]

        df["Fraud_Probability"] = result[
            "probabilities"
        ]
        def risk_level(prob):
            if prob >= 0.90:
                return "High"
            elif prob >= 0.70:
                return "Medium"
            else:
                return "Low"

        df["Risk_Level"] = df[
            "Fraud_Probability"
        ].apply(risk_level)
        high_risk = df[
            df["Prediction"] == 1
        ]

        if len(high_risk) == 0:
            st.info("No high-risk transactions detected.")
        else:
            st.dataframe(
                high_risk.sort_values(
                    "Fraud_Probability",
                    ascending=False
                )
            )
        st.subheader(
            "High Risk Transactions"
        )

        if len(high_risk) == 0:

            st.info(
                "No high-risk transactions detected."
            )

        else:

            st.dataframe(
                high_risk.sort_values(
                    "Fraud_Probability",
                    ascending=False
                )
            )
        st.subheader(
            "Top 10 Risk Transactions"
        )

        top_risk = df.sort_values(
            "Fraud_Probability",
            ascending=False
        ).head(10)

        st.dataframe(
            top_risk[
                [
                    "Time",
                    "Amount",
                    "Fraud_Probability",
                    "Risk_Level",
                    "Prediction"
                ]
            ]
        )
        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            df[
                [
                    "Time",
                    "Amount",
                    "Fraud_Probability",
                    "Risk_Level",
                    "Prediction"
                ]
            ]
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )
        fraud_rate = (
            result["fraud_count"] / len(df)
        ) * 100
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Transactions", len(df))

        with col2:
            st.metric("Frauds", result["fraud_count"])

        with col3:
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%"
            )


        frauds = result["fraud_count"]
        normal = len(df) - frauds

        fig, ax = plt.subplots()

        ax.pie(
            [normal, frauds],
            labels=["Normal", "Fraud"],
            autopct="%1.1f%%"
        )
        st.subheader(
            "Fraud Probability Distribution"
        )

        st.bar_chart(
            df["Fraud_Probability"]
        )

        ax.set_title(
            "Fraud vs Genuine Transactions"
        )

        st.pyplot(fig)
        