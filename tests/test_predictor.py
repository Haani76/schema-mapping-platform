import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predictor import SchemaPredictor
from configs.config import config


@pytest.fixture(scope="module")
def predictor():
    """Load predictor once for all tests."""
    return SchemaPredictor()


class TestSinglePrediction:

    def test_customer_id_prediction(self, predictor):
        result = predictor.predict_column("customer_id", "C-1234")
        assert result["predicted_label"] == "CUSTOMER_ID"
        assert result["confidence"] > 0.85
        assert result["routing"] == "auto_map"

    def test_revenue_prediction(self, predictor):
        result = predictor.predict_column("revenue", "15000.00")
        assert result["predicted_label"] == "REVENUE"
        assert result["confidence"] > 0.85

    def test_date_prediction(self, predictor):
        result = predictor.predict_column("order_date", "2023-01-15")
        assert result["predicted_label"] == "DATE"
        assert result["confidence"] > 0.85

    def test_email_prediction(self, predictor):
        result = predictor.predict_column("email_address", "john@example.com")
        assert result["predicted_label"] == "EMAIL"
        assert result["confidence"] > 0.85

    def test_quantity_prediction(self, predictor):
        result = predictor.predict_column("qty_sold", "250")
        assert result["predicted_label"] == "QUANTITY"
        assert result["confidence"] > 0.85

    def test_unknown_column_routes_to_human_review(self, predictor):
        result = predictor.predict_column("unknown_xyz_abc", "???")
        assert result["routing"] == "human_review"

    def test_response_has_required_keys(self, predictor):
        result = predictor.predict_column("customer_id")
        required_keys = [
            "column_name", "predicted_label", "confidence",
            "routing", "top3_predictions", "threshold_used"
        ]
        for key in required_keys:
            assert key in result

    def test_confidence_is_between_0_and_1(self, predictor):
        result = predictor.predict_column("customer_id")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top3_predictions_length(self, predictor):
        result = predictor.predict_column("customer_id")
        assert len(result["top3_predictions"]) == 3

    def test_threshold_used_matches_config(self, predictor):
        result = predictor.predict_column("customer_id")
        assert result["threshold_used"] == config.CONFIDENCE_THRESHOLD


class TestBatchPrediction:

    def test_batch_prediction_returns_summary(self, predictor):
        columns = ["customer_id", "revenue", "order_date"]
        result = predictor.predict_dataframe_columns(columns)
        assert "total_columns" in result
        assert "auto_mapped_count" in result
        assert "needs_review_count" in result
        assert "auto_map_rate" in result

    def test_batch_total_matches_input(self, predictor):
        columns = ["customer_id", "revenue", "order_date", "unknown_xyz"]
        result = predictor.predict_dataframe_columns(columns)
        assert result["total_columns"] == 4

    def test_batch_auto_map_plus_review_equals_total(self, predictor):
        columns = ["customer_id", "revenue", "order_date", "unknown_xyz"]
        result = predictor.predict_dataframe_columns(columns)
        assert result["auto_mapped_count"] + result["needs_review_count"] == result["total_columns"]

    def test_auto_map_rate_between_0_and_1(self, predictor):
        columns = ["customer_id", "revenue", "order_date"]
        result = predictor.predict_dataframe_columns(columns)
        assert 0.0 <= result["auto_map_rate"] <= 1.0


class TestConfig:

    def test_num_labels(self):
        assert config.NUM_LABELS == 12

    def test_confidence_threshold(self):
        assert config.CONFIDENCE_THRESHOLD == 0.85

    def test_label2id_and_id2label_consistency(self):
        for label, idx in config.LABEL2ID.items():
            assert config.ID2LABEL[idx] == label

    def test_model_name(self):
        assert config.MODEL_NAME == "bert-base-uncased"