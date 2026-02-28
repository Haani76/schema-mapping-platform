import os
import json
import datetime
import mlflow
import torch
from configs.config import config


class RetrainingPipeline:
    """
    Continuous retraining pipeline.
    Monitors model performance, detects drift,
    and triggers retraining when needed.
    """

    def __init__(
        self,
        drift_threshold: float = 0.75,
        min_samples_for_retrain: int = 100,
    ):
        self.drift_threshold = drift_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        self.log_dir = os.path.join(config.BASE_DIR, "logs", "retraining")
        os.makedirs(self.log_dir, exist_ok=True)

    def check_drift(self, monitoring_log_path: str) -> dict:
        """
        Check if drift has been detected in monitoring logs.
        Returns drift status and recommendation.
        """
        with open(monitoring_log_path, "r") as f:
            log_data = json.load(f)

        stats = log_data.get("statistics", {})
        avg_confidence = stats.get("confidence_stats", {}).get("mean", 1.0)
        total_predictions = stats.get("total_predictions", 0)
        auto_map_rate = stats.get("auto_map_rate", 1.0)

        drift_detected = avg_confidence < self.drift_threshold
        enough_samples = total_predictions >= self.min_samples_for_retrain
        should_retrain = drift_detected and enough_samples

        return {
            "avg_confidence": avg_confidence,
            "auto_map_rate": auto_map_rate,
            "total_predictions": total_predictions,
            "drift_detected": drift_detected,
            "enough_samples": enough_samples,
            "should_retrain": should_retrain,
            "reason": self._get_reason(drift_detected, enough_samples, avg_confidence),
        }

    def _get_reason(self, drift_detected, enough_samples, avg_confidence):
        if drift_detected and enough_samples:
            return f"Drift detected (avg confidence {avg_confidence:.2f} < threshold {self.drift_threshold}). Retraining recommended."
        elif drift_detected and not enough_samples:
            return f"Drift detected but insufficient samples for retraining. Collecting more data."
        else:
            return f"No drift detected. Model performing well (avg confidence {avg_confidence:.2f})."

    def prepare_retraining_data(self, new_samples: list) -> str:
        """
        Merge new labeled samples with existing training data.
        Returns path to updated training data.
        """
        existing_path = os.path.join(config.TRAINING_DATA_DIR, "train.json")
        with open(existing_path, "r") as f:
            existing_data = json.load(f)

        # Add new samples
        combined_data = existing_data + new_samples
        print(f"Original training samples: {len(existing_data)}")
        print(f"New samples added:         {len(new_samples)}")
        print(f"Combined training samples: {len(combined_data)}")

        # Save updated training data
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        updated_path = os.path.join(config.TRAINING_DATA_DIR, f"train_updated_{timestamp}.json")
        with open(updated_path, "w") as f:
            json.dump(combined_data, f, indent=2)

        return updated_path

    def run_retraining(self, training_data_path: str = None) -> str:
        """
        Trigger a retraining run with MLflow tracking.
        Returns path to new best model.
        """
        from src.data.dataset_loader import get_dataloaders
        from src.models.ner_model import get_model
        from src.training.trainer import NERTrainer

        print("\n" + "=" * 50)
        print("  RETRAINING PIPELINE STARTED")
        print("=" * 50)
        print(f"  Timestamp: {datetime.datetime.utcnow().isoformat()}")

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        print("\nLoading datasets...")
        train_loader, val_loader, test_loader = get_dataloaders()

        # Load fresh model
        print("Loading model...")
        model = get_model()

        # Train
        trainer = NERTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )

        with mlflow.start_run(run_name=f"retrain_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("trigger", "drift_detected")
            mlflow.log_param("training_data", training_data_path or "default")

            best_model_path = trainer.train(num_epochs=3, learning_rate=2e-5)

            mlflow.log_param("retrain_completed", True)

        print(f"\nRetraining complete. New model saved to: {best_model_path}")
        return best_model_path

    def run_full_pipeline(self, monitoring_log_path: str = None, force: bool = False):
        """
        Run the full retraining pipeline:
        1. Check drift
        2. Decide whether to retrain
        3. Retrain if needed
        4. Log results
        """
        print("\n" + "=" * 50)
        print("  CONTINUOUS RETRAINING PIPELINE")
        print("=" * 50)

        # Use latest monitoring log if not specified
        if monitoring_log_path is None:
            monitor_dir = os.path.join(config.BASE_DIR, "logs", "monitoring")
            logs = sorted(os.listdir(monitor_dir)) if os.path.exists(monitor_dir) else []
            if logs:
                monitoring_log_path = os.path.join(monitor_dir, logs[-1])
                print(f"Using latest monitoring log: {logs[-1]}")
            else:
                print("No monitoring logs found. Running monitor first...")
                from src.monitoring.monitor import PredictionMonitor
                from src.inference.predictor import SchemaPredictor
                predictor = SchemaPredictor()
                monitor = PredictionMonitor()
                test_cols = ["customer_id", "revenue", "order_date", "email", "unknown_xyz"]
                for col in test_cols:
                    monitor.log_prediction(predictor.predict_column(col))
                monitoring_log_path = monitor.save_log()

        # Check drift
        print("\nChecking for drift...")
        drift_status = self.check_drift(monitoring_log_path)

        print(f"  Avg Confidence:  {drift_status['avg_confidence']*100:.1f}%")
        print(f"  Auto Map Rate:   {drift_status['auto_map_rate']*100:.1f}%")
        print(f"  Total Samples:   {drift_status['total_predictions']}")
        print(f"  Drift Detected:  {drift_status['drift_detected']}")
        print(f"  Reason:          {drift_status['reason']}")

        # Decide whether to retrain
        if drift_status["should_retrain"] or force:
            if force:
                print("\nForced retraining triggered.")
            else:
                print("\nDrift threshold exceeded. Triggering retraining...")
            new_model_path = self.run_retraining()
            return {"status": "retrained", "new_model_path": new_model_path}
        else:
            print("\nNo retraining needed. Model is healthy.")
            return {"status": "healthy", "new_model_path": None}


if __name__ == "__main__":
    pipeline = RetrainingPipeline(
        drift_threshold=0.75,
        min_samples_for_retrain=5,
    )

    # Run pipeline — will use latest monitoring log
    result = pipeline.run_full_pipeline()
    print(f"\nPipeline result: {result['status']}")