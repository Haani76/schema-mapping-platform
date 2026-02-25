import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from src.models.ner_model import get_model
from src.data.dataset_loader import get_dataloaders
from src.training.trainer import NERTrainer
from configs.config import config


def main():
    print("=" * 50)
    print("  Schema Mapping NER - Training Pipeline")
    print("=" * 50)

    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, training on CPU (will be slower)")

    # Load data
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Load model
    print("\nLoading model...")
    model = get_model()

    # Initialize trainer
    trainer = NERTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    # Train
    print("\nStarting training...")
    best_model_path = trainer.train(
        num_epochs=5,
        learning_rate=2e-5,
    )

    print(f"\nBest model saved to: {best_model_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()