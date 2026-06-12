import pandas as pd
from src.utils.save_object import load_object
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self, model_path="artifacts/model/model.pkl",
                 scaler_path="artifacts/model/scaler.pkl"):
        logger.info("Loading model and scaler for prediction pipeline...")
        self.model = load_object(model_path)
        self.scaler = load_object(scaler_path)
        self.feature_scale = ['Time', 'Amount']
        logger.info("Prediction pipeline initialized successfully.")

    def predict(self, df: pd.DataFrame, threshold=0.5):
        """
        Takes a DataFrame with the same features as training data
        (V1-V28, Time, Amount — no 'Class' column) and returns predictions + probabilities.
        """
        df_scaled = df.copy()
        df_scaled[self.feature_scale] = self.scaler.transform(df[self.feature_scale])

        probabilities = self.model.predict_proba(df_scaled)[:, 1]
        logger.debug(
            f"Max fraud probability: "
            f"{probabilities.max():.4f}"
        )
        predictions = (probabilities >= threshold).astype(int)

        logger.info(f"Predictions complete — {int(predictions.sum())} fraud(s) detected out of {len(predictions)} transactions.")
        
        return predictions, probabilities


if __name__ == "__main__":

    pipeline = PredictionPipeline()
    df = pd.read_csv("data/creditcard.csv")

    frauds = df[df["Class"] == 1].head(20)

    labels = frauds["Class"].values
    frauds = frauds.drop(columns=["Class"])
    preds, probs = pipeline.predict(frauds)

    for i in range(len(preds)):
        print(
            f"Row {i}: pred={preds[i]}, "
            f"prob={probs[i]:.4f}, "
            f"actual={labels[i]}"
        )
        