import os
import json
import datetime
import numpy as np
from collections import defaultdict, Counter
from configs.config import config


class PredictionMonitor:
    """
    Monitors model predictions over time.
    Tracks confidence distributions, label distributions,
    routing rates, and flags potential data drift.
    """

    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(config.BASE_DIR, "logs", "monitoring")
        os.makedirs(self.log_dir, exist_ok=True)
        self.predictions_log = []
        self.session_start = datetime.datetime.utcnow().isoformat()

    def log_prediction(self, prediction: dict):
        """Log a single prediction result."""
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "column_name": prediction.get("column_name"),
            "predicted_label": prediction.get("predicted_label"),
            "confidence": prediction.get("confidence"),
            "routing": prediction.get("routing"),
        }
        self.predictions_log.append(record)

    def log_batch(self, predictions: list):
        """Log a batch of predictions."""
        for pred in predictions:
            self.log_prediction(pred)

    def get_statistics(self) -> dict:
        """Compute statistics over all logged predictions."""
        if not self.predictions_log:
            return {"error": "No predictions logged yet"}

        confidences = [p["confidence"] for p in self.predictions_log]
        labels = [p["predicted_label"] for p in self.predictions_log]
        routings = [p["routing"] for p in self.predictions_log]

        auto_map_count = routings.count("auto_map")
        human_review_count = routings.count("human_review")
        total = len(self.predictions_log)

        # Confidence statistics
        conf_array = np.array(confidences)
        confidence_stats = {
            "mean": round(float(np.mean(conf_array)), 4),
            "median": round(float(np.median(conf_array)), 4),
            "std": round(float(np.std(conf_array)), 4),
            "min": round(float(np.min(conf_array)), 4),
            "max": round(float(np.max(conf_array)), 4),
            "below_threshold": int(np.sum(conf_array < config.CONFIDENCE_THRESHOLD)),
        }

        # Label distribution
        label_distribution = dict(Counter(labels))

        # Drift detection — flag if avg confidence drops below 0.75
        drift_detected = confidence_stats["mean"] < 0.75

        return {
            "session_start": self.session_start,
            "total_predictions": total,
            "auto_mapped": auto_map_count,
            "needs_review": human_review_count,
            "auto_map_rate": round(auto_map_count / total, 4),
            "confidence_stats": confidence_stats,
            "label_distribution": label_distribution,
            "drift_detected": drift_detected,
            "drift_warning": "Average confidence below 0.75 - possible data drift" if drift_detected else None,
        }

    def save_log(self, filename: str = None):
        """Save prediction log to disk."""
        filename = filename or f"predictions_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump({
                "session_start": self.session_start,
                "predictions": self.predictions_log,
                "statistics": self.get_statistics(),
            }, f, indent=2)
        print(f"Log saved to: {path}")
        return path

    def print_report(self):
        """Print a summary monitoring report."""
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("  MODEL MONITORING REPORT")
        print("=" * 50)
        print(f"  Total Predictions:  {stats['total_predictions']}")
        print(f"  Auto Mapped:        {stats['auto_mapped']}")
        print(f"  Needs Review:       {stats['needs_review']}")
        print(f"  Auto Map Rate:      {stats['auto_map_rate'] * 100:.1f}%")
        print(f"\n  Confidence Stats:")
        print(f"    Mean:    {stats['confidence_stats']['mean']}")
        print(f"    Median:  {stats['confidence_stats']['median']}")
        print(f"    Std Dev: {stats['confidence_stats']['std']}")
        print(f"    Min:     {stats['confidence_stats']['min']}")
        print(f"    Max:     {stats['confidence_stats']['max']}")
        print(f"\n  Label Distribution:")
        for label, count in sorted(stats['label_distribution'].items()):
            print(f"    {label:<20} {count}")
        if stats["drift_detected"]:
            print(f"\n  ⚠️  DRIFT WARNING: {stats['drift_warning']}")
        else:
            print(f"\n  ✅ No drift detected")
        print("=" * 50)


if __name__ == "__main__":
    from src.inference.predictor import SchemaPredictor

    predictor = SchemaPredictor()
    monitor = PredictionMonitor()

    # Simulate predictions
    test_columns = [
        "customer_id", "product_code", "sale_amount",
        "purchase_date", "units_sold", "store_location",
        "contact_email", "phone_no", "rep_name",
        "order_status", "item_category", "unknown_col"
    ]

    print("Running predictions and logging to monitor...")
    for col in test_columns:
        result = predictor.predict_column(col)
        monitor.log_prediction(result)

    monitor.print_report()
    monitor.save_log()