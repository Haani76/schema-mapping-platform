import os
import torch
import numpy as np
from transformers import BertTokenizerFast
from src.models.ner_model import BertNERModel
from configs.config import config


class SchemaPredictor:
    """
    Inference engine for semantic column type prediction.
    Includes confidence scoring and auto-map vs human review logic.
    """

    def __init__(self, model_path: str = None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use HuggingFace Hub model if local model doesn't exist
        local_path = os.path.join(config.BASE_DIR, "models", "best_model")
        if model_path:
            self.model_path = model_path
        elif os.path.exists(local_path):
            self.model_path = local_path
        else:
            self.model_path = "Haani76/schema-mapping-ner"
        self.tokenizer = BertTokenizerFast.from_pretrained(config.MODEL_NAME)
        self.model = self._load_model()
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        print(f"Predictor ready on device: {self.device}")
        print(f"Confidence threshold: {self.confidence_threshold}")

    def _load_model(self):
        """Load the trained model from disk."""
        model = BertNERModel.from_pretrained(
            self.model_path,
            ignore_mismatched_sizes=True,
        )
        model.to(self.device)
        model.eval()
        print(f"Model loaded from: {self.model_path}")
        return model

    def predict_column(self, column_name: str, sample_value: str = "") -> dict:
        """
        Predict semantic type for a single column.

        Returns:
            dict with predicted label, confidence score,
            and routing decision (auto_map or human_review)
        """
        # Tokenize
        tokens = column_name.replace("_", " ").replace("-", " ").split()
        if not tokens:
            tokens = [column_name]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=config.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding["token_type_ids"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        logits = outputs["logits"]  # (1, seq_len, num_labels)

        # Get probabilities via softmax
        probs = torch.softmax(logits, dim=-1)  # (1, seq_len, num_labels)

        # Focus on the first real token (index 1, after [CLS])
        first_token_probs = probs[0, 1, :].cpu().numpy()

        # Get top prediction
        predicted_id = int(np.argmax(first_token_probs))
        confidence = float(np.max(first_token_probs))
        predicted_label = config.ID2LABEL.get(predicted_id, "O")

        # Strip B- prefix for cleaner output
        clean_label = predicted_label.replace("B-", "").replace("I-", "")

        # Get top 3 predictions
        top3_indices = np.argsort(first_token_probs)[::-1][:3]
        top3 = [
            {
                "label": config.ID2LABEL.get(int(i), "O").replace("B-", ""),
                "confidence": float(first_token_probs[i]),
            }
            for i in top3_indices
        ]

        # Routing decision
        if clean_label == "O" or confidence < self.confidence_threshold:
            routing = "human_review"
        else:
            routing = "auto_map"

        return {
            "column_name": column_name,
            "sample_value": sample_value,
            "predicted_label": clean_label,
            "confidence": round(confidence, 4),
            "routing": routing,
            "top3_predictions": top3,
            "threshold_used": self.confidence_threshold,
        }

    def predict_batch(self, columns: list) -> list:
        """
        Predict semantic types for a list of columns.

        Args:
            columns: list of dicts with 'column_name' and optionally 'sample_value'

        Returns:
            list of prediction results
        """
        results = []
        for col in columns:
            column_name = col.get("column_name", col) if isinstance(col, dict) else col
            sample_value = col.get("sample_value", "") if isinstance(col, dict) else ""
            result = self.predict_column(column_name, sample_value)
            results.append(result)
        return results

    def predict_dataframe_columns(self, columns: list) -> dict:
        """
        Predict and summarize results for an entire set of dataframe columns.
        Splits results into auto_mapped and needs_review buckets.
        """
        predictions = self.predict_batch(columns)

        auto_mapped = [p for p in predictions if p["routing"] == "auto_map"]
        needs_review = [p for p in predictions if p["routing"] == "human_review"]

        return {
            "total_columns": len(predictions),
            "auto_mapped_count": len(auto_mapped),
            "needs_review_count": len(needs_review),
            "auto_map_rate": round(len(auto_mapped) / len(predictions), 4) if predictions else 0,
            "auto_mapped": auto_mapped,
            "needs_review": needs_review,
        }


if __name__ == "__main__":
    predictor = SchemaPredictor()

    # Test single predictions
    print("\n--- Single Column Predictions ---")
    test_columns = [
        ("cust_id", "C-1234"),
        ("revenue_q3", "15000.00"),
        ("order_date", "2023-01-15"),
        ("email_address", "john@example.com"),
        ("qty_sold", "250"),
        ("unknown_col_xyz", "???"),
    ]

    for col_name, sample_val in test_columns:
        result = predictor.predict_column(col_name, sample_val)
        print(f"\nColumn: '{col_name}' | Value: '{sample_val}'")
        print(f"  Predicted: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Routing: {result['routing']}")

    # Test batch prediction
    print("\n--- Batch Prediction Summary ---")
    batch_columns = [
        "customer_id", "product_code", "sale_amount",
        "purchase_date", "units_sold", "store_location",
        "contact_email", "phone_no", "rep_name",
        "order_status", "item_category", "weird_col_abc"
    ]
    summary = predictor.predict_dataframe_columns(batch_columns)
    print(f"Total columns:    {summary['total_columns']}")
    print(f"Auto mapped:      {summary['auto_mapped_count']}")
    print(f"Needs review:     {summary['needs_review_count']}")
    print(f"Auto map rate:    {summary['auto_map_rate'] * 100:.1f}%")