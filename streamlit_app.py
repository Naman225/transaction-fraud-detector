import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.pipeline.prediction_pipeline import PredictionPipeline

@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

st.sidebar.success(
    "Model Loaded Successfully"
)
st.title("Credit Card Fraud Detector")
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
    value=float(pipeline.default_threshold),
    step=0.05,
    help=f"Optimal threshold from training: {pipeline.default_threshold:.3f}"
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

        try:

            predictions, probabilities = pipeline.predict(
                df,
                threshold=threshold
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )
            st.stop()

        fraud_count = int(predictions.sum())

        st.success(
            f"Frauds detected: {fraud_count}"
        )

        df["Prediction"] = predictions

        df["Fraud_Probability"] = probabilities
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
            fraud_count / len(df)
        ) * 100
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Transactions", len(df))

        with col2:
            st.metric("Frauds", fraud_count)

        with col3:
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%"
            )


        frauds = fraud_count
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
        