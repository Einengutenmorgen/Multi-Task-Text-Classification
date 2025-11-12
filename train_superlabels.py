# train_superlabels.py
import json
import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import f1_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput

# ------------------------------------------------------------------------
# ✅ Anpassung: Absolute Pfade setzen (sehr robust im HPC/Server Umfeld)
# ------------------------------------------------------------------------
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
DATASET_PATH = "/home/s2chhauu/superlabel/dataset"
OUTPUT_DIR = "/home/s2chhauu/superlabel/tuned_models/superlabel_finetune"

NUM_SUPERLABELS = 8
TRANSFORMER_DIM = 1024
MLP_HIDDEN_DIM = 768

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling für Transformer Output"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class SuperlabelModel(PreTrainedModel):
    """Gefrorener Transformer + MLP Kopf"""

    config_class = AutoConfig

    def __init__(
        self,
        config,
        model_name: str,
        num_superlabels: int,
        mlp_hidden_dim: int,
        transformer_dim: int = TRANSFORMER_DIM,
    ):
        super().__init__(config)
        self.model_name = model_name
        self.num_superlabels = num_superlabels

        self.transformer = AutoModel.from_pretrained(model_name)

        # Freeze Base Model
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(transformer_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.head = nn.Linear(mlp_hidden_dim, num_superlabels)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutput:
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = (
            transformer_output.last_hidden_state
            if hasattr(transformer_output, "last_hidden_state")
            else transformer_output[0]
        )

        pooled_output = mean_pool(last_hidden_state, attention_mask)
        mlp_features = self.mlp(pooled_output)
        logits = self.head(mlp_features)

        loss = None
        if labels is not None:
            labels = labels.to(dtype=logits.dtype, device=logits.device)
            mask = (labels != -100.0).float()
            safe_labels = torch.where(mask.bool(), labels, torch.zeros_like(labels))

            per_elem_loss = self.loss_fct(logits, safe_labels)
            masked_loss = per_elem_loss * mask
            num_relevant = mask.sum()

            loss = masked_loss.sum() / num_relevant if num_relevant > 0 else masked_loss.sum()

        return ModelOutput(loss=loss, logits=logits)


def compute_metrics(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    mask = labels != -100.0
    sig = 1 / (1 + np.exp(-logits))
    preds = sig > 0.5

    valid_preds = preds[mask].astype(int)
    valid_labels = labels[mask].astype(int)

    if valid_labels.size == 0:
        return {"f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0}

    f1_macro = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(valid_labels, valid_preds, average="macro", zero_division=0)

    return {"f1_macro": float(f1_macro), "precision_macro": float(p), "recall_macro": float(r)}


def main():
    logger.info(f"--- Lade Dataset von: {DATASET_PATH} ---")
    full_dataset = load_from_disk(DATASET_PATH)

    if isinstance(full_dataset, DatasetDict):
        train_dataset = full_dataset.get("train") or full_dataset[list(full_dataset.keys())[0]]
        val_dataset = full_dataset.get("validation") or train_dataset.train_test_split(test_size=0.1)["test"]
    else:
        split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]

    logger.info(f"Train Samples: {len(train_dataset)}")
    logger.info(f"Val   Samples: {len(val_dataset)}")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = SuperlabelModel(config, MODEL_NAME, NUM_SUPERLABELS, MLP_HIDDEN_DIM)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb",
        num_train_epochs=3,
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("--- Starte Training ---")
    trainer.train()

    logger.info("--- Speichere bestes Modell ---")
    final_model_path = f"{OUTPUT_DIR}/best_model"
    trainer.save_model(final_model_path)
    trainer.save_state()

    eval_results = trainer.evaluate(val_dataset)
    with open(f"{OUTPUT_DIR}/final_eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    logger.info("✅ Training abgeschlossen.")
    logger.info(eval_results)


if __name__ == "__main__":
    main()
