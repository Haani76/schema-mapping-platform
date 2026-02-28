import os
import time
import random
import json
import datetime
from collections import defaultdict
from configs.config import config


class ABTestingFramework:
    """
    A/B testing framework for comparing two model versions.
    Routes traffic between Model A and Model B based on split ratio.
    Tracks performance metrics for each version.
    """

    def __init__(
        self,
        model_a_path: str = None,
        model_b_path: str = None,
        split_ratio: float = 0.5,
        experiment_name: str = "default",
    ):
        self.split_ratio = split_ratio
        self.experiment_name = experiment_name
        self.results = {"A": [], "B": []}
        self.start_time = datetime.datetime.utcnow().isoformat()

        print(f"Initializing A/B test: {experiment_name}")
        print(f"Traffic split: {split_ratio*100:.0f}% Model A / {(1-split_ratio)*100:.0f}% Model B")

        # Load both models
        from src.inference.predictor import SchemaPredictor
        print("\nLoading Model A...")
        self.model_a = SchemaPredictor(model_path=model_a_path)
        print("Loading Model B...")
        self.model_b = SchemaPredictor(model_path=model_b_path)

        print("\nA/B test ready.")

    def route_request(self) -> str:
        """Determine which model to use for this request."""
        return "A" if random.random() < self.split_ratio else "B"

    def predict(self, column_name: str, sample_value: str = "") -> dict:
        """
        Route prediction to either Model A or B based on split ratio.
        Logs result for analysis.
        """
        variant = self.route_request()
        model = self.model_a if variant == "A" else self.model_b

        start = time.time()
        result = model.predict_column(column_name, sample_value)
        latency_ms = round((time.time() - start) * 1000, 2)

        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "variant": variant,
            "column_name": column_name,
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "routing": result["routing"],
            "latency_ms": latency_ms,
        }
        self.results[variant].append(log_entry)

        return {**result, "variant": variant, "latency_ms": latency_ms}

    def get_metrics(self) -> dict:
        """Compute and compare metrics for both variants."""
        metrics = {}

        for variant in ["A", "B"]:
            results = self.results[variant]
            if not results:
                metrics[variant] = {"error": "No data"}
                continue

            confidences = [r["confidence"] for r in results]
            latencies = [r["latency_ms"] for r in results]
            auto_map_count = sum(1 for r in results if r["routing"] == "auto_map")

            metrics[variant] = {
                "total_predictions": len(results),
                "auto_map_rate": round(auto_map_count / len(results), 4),
                "avg_confidence": round(sum(confidences) / len(confidences), 4),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
            }

        # Winner determination
        if "A" in metrics and "B" in metrics:
            if isinstance(metrics["A"], dict) and "avg_confidence" in metrics["A"]:
                a_score = metrics["A"]["avg_confidence"] * metrics["A"]["auto_map_rate"]
                b_score = metrics["B"]["avg_confidence"] * metrics["B"]["auto_map_rate"]
                winner = "A" if a_score >= b_score else "B"
                metrics["winner"] = winner
                metrics["recommendation"] = f"Model {winner} performs better — promote to production"

        return metrics

    def print_report(self):
        """Print a formatted A/B test report."""
        metrics = self.get_metrics()
        print("\n" + "=" * 55)
        print(f"  A/B TEST REPORT: {self.experiment_name}")
        print("=" * 55)
        print(f"  Started: {self.start_time}")

        for variant in ["A", "B"]:
            if variant not in metrics:
                continue
            m = metrics[variant]
            print(f"\n  Model {variant}:")
            print(f"    Total Predictions: {m.get('total_predictions', 0)}")
            print(f"    Auto Map Rate:     {m.get('auto_map_rate', 0)*100:.1f}%")
            print(f"    Avg Confidence:    {m.get('avg_confidence', 0)*100:.1f}%")
            print(f"    Avg Latency:       {m.get('avg_latency_ms', 0):.2f} ms")

        if "winner" in metrics:
            print(f"\n  Winner:         Model {metrics['winner']}")
            print(f"  Recommendation: {metrics['recommendation']}")
        print("=" * 55)

    def save_results(self):
        """Save A/B test results to disk."""
        log_dir = os.path.join(config.BASE_DIR, "logs", "ab_testing")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"ab_test_{self.experiment_name}_{timestamp}.json")
        with open(path, "w") as f:
            json.dump({
                "experiment": self.experiment_name,
                "start_time": self.start_time,
                "split_ratio": self.split_ratio,
                "metrics": self.get_metrics(),
                "results": self.results,
            }, f, indent=2)
        print(f"\nResults saved to: {path}")
        return path


if __name__ == "__main__":
    # Simulate A/B test with same model on both sides
    # In production, Model B would be a newly trained version
    model_path = os.path.join(config.BASE_DIR, "models", "best_model")

    ab_test = ABTestingFramework(
        model_a_path=model_path,
        model_b_path=model_path,
        split_ratio=0.5,
        experiment_name="bert-base-vs-bert-base-v2",
    )

    # Simulate 20 predictions
    test_columns = [
        "customer_id", "revenue", "order_date", "email",
        "quantity", "location", "phone", "product_id",
        "customer_name", "status", "category", "unknown_col",
        "cust_no", "sale_amount", "purchase_date", "contact_email",
        "units_sold", "store_location", "rep_name", "order_status",
    ]

    print("\nRunning A/B test predictions...")
    for col in test_columns:
        result = ab_test.predict(col)
        print(f"  [{result['variant']}] {col} → {result['predicted_label']} ({result['confidence']*100:.1f}%)")

    ab_test.print_report()
    ab_test.save_results()