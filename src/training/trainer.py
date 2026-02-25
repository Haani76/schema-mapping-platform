import os
import torch
import mlflow
import mlflow.pytorch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score
from configs.config import config


class NERTrainer:
    """
    Training engine for BERT NER model with MLflow tracking.
    """

    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Training on device: {self.device}")

    def get_optimizer_and_scheduler(self, num_epochs: int, learning_rate: float = 2e-5):
        """Set up optimizer with weight decay and linear warmup scheduler."""
        # Separate parameters - don't apply weight decay to biases and LayerNorm
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

        total_steps = len(self.train_loader) * num_epochs
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return optimizer, scheduler

    def decode_labels(self, label_ids: list) -> list:
        """Convert label IDs back to label strings, ignoring -100."""
        decoded = []
        for label_id in label_ids:
            if label_id == -100:
                continue
            decoded.append(config.ID2LABEL.get(label_id, "O"))
        return decoded

    def evaluate(self) -> dict:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )

                total_loss += outputs["loss"].item()

                preds = torch.argmax(outputs["logits"], dim=-1).cpu().numpy()
                label_ids = labels.cpu().numpy()

                for pred_seq, label_seq in zip(preds, label_ids):
                    pred_tags = []
                    true_tags = []
                    for p, l in zip(pred_seq, label_seq):
                        if l == -100:
                            continue
                        pred_tags.append(config.ID2LABEL.get(int(p), "O"))
                        true_tags.append(config.ID2LABEL.get(int(l), "O"))
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)

        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds)

        return {"val_loss": avg_loss, "val_f1": f1}

    def train(self, num_epochs: int = 5, learning_rate: float = 2e-5):
        """Full training loop with MLflow tracking."""

        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                "model_name": config.MODEL_NAME,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": config.BATCH_SIZE,
                "max_length": config.MAX_LENGTH,
                "num_labels": config.NUM_LABELS,
            })

            optimizer, scheduler = self.get_optimizer_and_scheduler(num_epochs, learning_rate)
            best_val_f1 = 0.0
            best_model_path = os.path.join(config.BASE_DIR, "models", "best_model")
            os.makedirs(best_model_path, exist_ok=True)

            for epoch in range(num_epochs):
                # --- Training phase ---
                self.model.train()
                total_train_loss = 0
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 40)

                for step, batch in enumerate(self.train_loader):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch["token_type_ids"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )

                    loss = outputs["loss"]
                    loss.backward()

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()

                    total_train_loss += loss.item()

                    if (step + 1) % 10 == 0:
                        print(f"  Step {step + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

                avg_train_loss = total_train_loss / len(self.train_loader)

                # --- Validation phase ---
                val_metrics = self.evaluate()

                print(f"\n  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
                print(f"  Val F1:     {val_metrics['val_f1']:.4f}")

                # Log metrics to MLflow
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["val_loss"],
                    "val_f1": val_metrics["val_f1"],
                }, step=epoch)

                # Save best model
                if val_metrics["val_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["val_f1"]
                    self.model.save_pretrained(best_model_path)
                    print(f"  New best model saved (F1: {best_val_f1:.4f})")
                    mlflow.log_metric("best_val_f1", best_val_f1)

            print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")
            mlflow.pytorch.log_model(self.model, "model")

        return best_model_path