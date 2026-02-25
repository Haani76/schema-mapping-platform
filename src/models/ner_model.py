import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from configs.config import config


class BertNERModel(BertPreTrainedModel):
    """
    BERT-based NER model for semantic column inference.
    Built on top of HuggingFace BertPreTrainedModel for full
    compatibility with the HuggingFace ecosystem.
    """

    def __init__(self, bert_config):
        super().__init__(bert_config)
        self.num_labels = bert_config.num_labels

        # BERT backbone
        self.bert = BertModel(bert_config, add_pooling_layer=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # Classification head — maps BERT hidden states to label logits
        self.classifier = nn.Linear(bert_config.hidden_size, bert_config.num_labels)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)

        # Get logits for each token
        logits = self.classifier(sequence_output)  # (batch, seq_len, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return {"loss": loss, "logits": logits}


def get_model():
    """Load and return a fresh BertNERModel."""
    from transformers import BertConfig

    bert_config = BertConfig.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )

    model = BertNERModel.from_pretrained(
        config.MODEL_NAME,
        config=bert_config,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model loaded: {config.MODEL_NAME}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of labels:     {config.NUM_LABELS}")

    return model


if __name__ == "__main__":
    model = get_model()
    print(f"\nModel architecture:\n{model}")