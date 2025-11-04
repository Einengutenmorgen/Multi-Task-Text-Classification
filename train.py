# train.py

import torch
import torch.nn as nn
import torch.optim as optim
# Import Subset for manual splitting
from torch.utils.data import DataLoader, SequentialSampler, Subset
# Import sklearn's train_test_split
from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import json

# Import our custom modules
import data_loading
from data_loading import (
    UnifiedDataset,
    TaskSampler,
    SCHEMA,
    JIGSAW_LABEL_COLS,
    DAVIDSON_LABEL_MAP,
    OLID_LABEL_MAP,
    RUMOUR_LABEL_MAP
)
from model import MultiTaskModel
from loss import MultiTaskLoss

# --- 1. Configuration ---
CONFIG = {
    "MODEL_NAME": "mixedbread-ai/mxbai-embed-large-v1",
    "EPOCHS": 3,
    "BATCH_SIZE": 8,  # Adjust based on your VRAM
    "MAX_LENGTH": 256,
    "BASE_LR": None,   # For the mxbai trunk (Set to None to freeze)
    "HEAD_LR": 1e-4,   # For our new classification heads
    "WEIGHT_DECAY": 0.01,
    "VALIDATION_SPLIT": 0.1, # 10% for validation
    "CHECKPOINT_DIR": "./checkpoints/v2/",
    "EPOCH_SAMPLING_SIZE": None # Options: None (use min), int (e.g. 50000), "max" (use max)
}

# Define our 5 tasks
TASKS = ['jigsaw', 'goemotions', 'davidson', 'olid', 'rumour']

def get_label_counts(dataset_schema):
    """Gets the number of labels for each task from our schema."""

    # We must access the *loaded* schema from the dataset instance
    # because 'goemotions' is auto-detected.

    return {
        'jigsaw': len(JIGSAW_LABEL_COLS),
        'goemotions': len(dataset_schema['goemotions']),
        'davidson': len(DAVIDSON_LABEL_MAP),
        'olid': len(OLID_LABEL_MAP),
        'rumour': len(RUMOUR_LABEL_MAP)
    }


def compute_task_class_weights(dataset, indices, label_counts):
    """Compute class and positive weights for each task using the training subset."""

    ce_tasks = ['davidson', 'olid', 'rumour']
    bce_tasks = ['jigsaw', 'goemotions']

    ce_counts = {
        task: torch.zeros(label_counts[task], dtype=torch.float32)
        for task in ce_tasks
    }

    bce_positive_counts = {
        task: torch.zeros(label_counts[task], dtype=torch.float32)
        for task in bce_tasks
    }
    bce_total_counts = {task: 0 for task in bce_tasks}

    for idx in indices:
        task_name, item_idx = dataset.task_indices[idx]
        labels = dataset.task_data[task_name]['labels'][item_idx]

        if task_name in ce_counts:
            label_idx = int(labels)
            if 0 <= label_idx < ce_counts[task_name].numel():
                ce_counts[task_name][label_idx] += 1
        elif task_name in bce_positive_counts:
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            bce_positive_counts[task_name] += label_tensor
            bce_total_counts[task_name] += 1

    ce_weights = {}
    for task, counts in ce_counts.items():
        total = counts.sum()
        if total > 0:
            weights = torch.where(
                counts > 0,
                total / torch.clamp(counts, min=1e-6),
                torch.zeros_like(counts)
            )
        else:
            weights = torch.ones_like(counts)
        ce_weights[task] = weights

    bce_pos_weights = {}
    for task, pos_counts in bce_positive_counts.items():
        total_samples = bce_total_counts[task]
        if total_samples > 0:
            weights = torch.where(
                pos_counts > 0,
                total_samples / torch.clamp(pos_counts, min=1e-6),
                torch.zeros_like(pos_counts)
            )
        else:
            weights = torch.ones_like(pos_counts)
        bce_pos_weights[task] = weights

    return ce_weights, bce_pos_weights

def batch_to_device(batch, device):
    """Moves all tensor values in a batch dictionary to the specified device."""
    return {k: v.to(device) for k, v in batch.items()}

def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    """Performs one full training epoch."""
    model.train()
    
    # Track losses
    total_loss_sum = 0
    task_loss_sums = {task: 0 for task in TASKS}
    
    loop = tqdm(dataloader, desc=f"Training Epoch", leave=False)
    
    for batch in loop:
        # 1. Move batch to device
        batch = batch_to_device(batch, device)
        
        # 2. Get inputs and labels from batch
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        # 3. Forward pass (Model -> Logits)
        outputs = model(**inputs)
        
        # 4. Calculate losses (Logits + Labels -> 6 Losses)
        losses = loss_fn(model_outputs=outputs, batch_labels=batch)
        
        # 5. Backpropagation
        optimizer.zero_grad()
        # # --- NaN GUARD ---
        # if not torch.isnan(losses['total_loss']):
        #     losses['total_loss'].backward()
        #     optimizer.step()
        #     scheduler.step()
        # else:
        #     print("WARNING: Skipping batch due to NaN total_loss.")
        # ---------------
        losses['total_loss'].backward()
        optimizer.step()
        scheduler.step()
        # 6. Log losses
        total_loss_sum += losses['total_loss'].item()
        for task in TASKS:
            task_loss_sums[task] += losses[f'loss_{task}'].item()
            
        # Update TQDM with live loss
        loop.set_postfix(
            total_loss=losses['total_loss'].item(),
            jigsaw=losses['loss_jigsaw'].item(),
            go=losses['loss_goemotions'].item()
        )
        
    # Return average losses for the epoch
    num_batches = len(dataloader)
    avg_total_loss = total_loss_sum / num_batches
    avg_task_losses = {task: loss_sum / num_batches for task, loss_sum in task_loss_sums.items()}
    
    return avg_total_loss, avg_task_losses


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Performs one full validation run."""
    model.eval()
    
    # Store all predictions and labels for F1 calculation
    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    
    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for batch in loop:
        # 1. Move batch to device
        batch = batch_to_device(batch, device)
        
        # 2. Get inputs and labels
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        labels = batch
        
        # 3. Forward pass (Model -> Logits)
        outputs = model(**inputs)
        
        # 4. Filter active predictions and labels for each task
        for task in TASKS:
            task_logits = outputs[task]
            task_labels = labels[f'labels_{task}']
            
            # --- Find active samples ---
            if task in ['jigsaw', 'goemotions']:
                # Multi-label: active if first label is not -100
                mask = task_labels[:, 0] != -100
                if mask.sum() == 0: continue
                # Get preds: Apply sigmoid and threshold
                task_preds = (torch.sigmoid(task_logits[mask]) > 0.5).int()
            else:
                # Multi-class: active if label is not -100
                mask = task_labels != -100
                if mask.sum() == 0: continue
                # Get preds: Argmax
                task_preds = torch.argmax(task_logits[mask], dim=1)
            
            # Store the active preds and labels
            all_preds[task].append(task_preds.cpu().numpy())
            all_labels[task].append(task_labels[mask].cpu().numpy())

    # --- 5. Calculate Metrics ---
    metrics = {}
    
    for task in TASKS:
        if not all_labels[task]:
            print(f"Warning: No active samples found for task '{task}' in validation set.")
            metrics[f'{task}_f1_macro'] = 0.0 # Use macro key
            continue
            
        # Concatenate all batch results
        task_all_labels = np.concatenate(all_labels[task])
        task_all_preds = np.concatenate(all_preds[task])
        
        # --- Use 'macro' average for imbalanced data ---
        f1 = f1_score(task_all_labels, task_all_preds, average='macro', zero_division=0)
        metrics[f'{task}_f1_macro'] = f1
        
    # Calculate the main metric: Average F1
    metrics['avg_f1_macro'] = np.mean(list(metrics.values()))
    
    return metrics


def main():
    print("--- Starting V1 Multi-Task Training Pipeline ---")
    
    # --- 1. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Load and Split Dataset ---
    print(f"Loading full dataset for model: {CONFIG['MODEL_NAME']}")
    full_dataset = UnifiedDataset(tokenizer_name=CONFIG['MODEL_NAME'], max_length=CONFIG['MAX_LENGTH'])
    
    # Get label counts *after* loading dataset
    label_counts = get_label_counts(SCHEMA)
    print(f"Discovered label counts: {label_counts}")

    # =================== NEW STRATIFIED SPLIT ===================
    print("Creating task-stratified train/val split...")

    # 1. Get the list of all indices (0 to N-1)
    all_indices = list(range(len(full_dataset)))
    
    # 2. Get the corresponding task name for each index (for stratification)
    # This creates a list like ['jigsaw', 'jigsaw', ..., 'goemotions', ..., 'davidson', ...]
    labels_for_stratify = [task_name for (task_name, _) in full_dataset.task_indices]

    # 3. Use sklearn's train_test_split to get stratified indices
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=CONFIG['VALIDATION_SPLIT'],
        stratify=labels_for_stratify, # <-- Stratify by task name
        random_state=42
    )

    # 4. Create PyTorch Subset objects from these indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    # ============================================================

    # --- This is the correct patching ---
    # Give the Subset the attributes our TaskSampler needs to find
    train_dataset.task_data = full_dataset.task_data
    train_dataset.full_task_indices_list = full_dataset.task_indices
    # ------------------------------------

    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- Compute class weights for loss balancing ---
    ce_weights, bce_pos_weights = compute_task_class_weights(
        full_dataset,
        train_indices,
        label_counts
    )

    # --- 3. Create DataLoaders ---
    # Training: Use TaskSampler for balanced batches
    print(f"Initializing TaskSampler with strategy: {CONFIG['EPOCH_SAMPLING_SIZE']}")
    train_sampler = TaskSampler(
        train_dataset,
        epoch_sampling_size=CONFIG['EPOCH_SAMPLING_SIZE']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        sampler=train_sampler,
        num_workers=2  # Use 0 if you get errors, but >0 is faster
    )
    
    # Validation: Use SequentialSampler for correct metric calculation
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        sampler=val_sampler,
        num_workers=2
    )

    # --- 4. Initialize Model (MODIFIED) ---
    print("Initializing MultiTaskModel...")
    
    # --- NEW: Determine if trunk should be frozen ---
    freeze_trunk_bool = (CONFIG["BASE_LR"] is None)
    if freeze_trunk_bool:
        print("BASE_LR is None. Freezing trunk parameters.")
    else:
        print(f"BASE_LR is {CONFIG['BASE_LR']}. Trunk parameters will be fine-tuned.")
    # ------------------------------------------------
    
    model = MultiTaskModel(
        model_name=CONFIG["MODEL_NAME"],
        num_labels_jigsaw=len(data_loading.SCHEMA["jigsaw"]),
        num_labels_goemotions=len(data_loading.SCHEMA["goemotions"]),
        num_labels_davidson=len(data_loading.SCHEMA["davidson"]),
        num_labels_olid=len(data_loading.SCHEMA["olid"]),
        num_labels_rumour=len(data_loading.SCHEMA["rumour"]),
        freeze_trunk=freeze_trunk_bool  # <-- Pass the new argument
    )

    model.to(device)

    # --- 5. Initialize Loss, Optimizer, and Scheduler ---
    loss_fn = MultiTaskLoss(
        ce_weights=ce_weights,
        bce_pos_weights=bce_pos_weights
    ).to(device)
    
    # Get parameter groups for differential LR
    # This now correctly handles the frozen/unfrozen trunk
    optimizer_params = model.get_optimizer_params(
        base_lr=CONFIG['BASE_LR'],
        head_lr=CONFIG['HEAD_LR']
    )
    
    optimizer = optim.AdamW(
        optimizer_params,
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * CONFIG['EPOCHS']
    num_warmup_steps = int(num_training_steps * 0.1) # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # --- 6. Training & Evaluation Loop ---
    best_avg_f1 = 0.0
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\n--- Epoch {epoch + 1} / {CONFIG['EPOCHS']} ---")
        
        # Train
        avg_total_loss, avg_task_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_total_loss:.4f}")
        for task, loss in avg_task_losses.items():
            print(f"  - Avg Train {task} Loss: {loss:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1} Validation Metrics:")
        for task in TASKS:
            print(f"  - {task.capitalize()} F1 (Macro): {val_metrics[f'{task}_f1_macro']:.4f}")
        print(f"  - === Average F1 (Macro): {val_metrics['avg_f1_macro']:.4f} === ")
        
        # --- 7. Checkpointing ---
        if val_metrics['avg_f1_macro'] > best_avg_f1:
            best_avg_f1 = val_metrics['avg_f1_macro']
            checkpoint_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "best_model.pth")
            print(f"New best model! Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            
            # Save metrics
            metrics_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "best_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)

    print("\n--- Training Complete ---")
    print(f"Best Average F1 Score: {best_avg_f1:.4f}")
    print(f"Best model saved to {os.path.join(CONFIG['CHECKPOINT_DIR'], 'best_model.pth')}")

def initialize_training_objects(device, ce_weights=None, bce_pos_weights=None):
    """
    Factory to initialize model, loss, and optimizer exactly as used in training.
    (MODIFIED)
    """
    import data_loading
    from model import MultiTaskModel
    from loss import MultiTaskLoss

    num_labels = {k: len(v) for k, v in data_loading.SCHEMA.items()}

    # --- NEW: Add freeze logic ---
    freeze_trunk_bool = (CONFIG["BASE_LR"] is None)
    # -----------------------------

    model = MultiTaskModel(
        model_name=CONFIG["MODEL_NAME"],
        num_labels_jigsaw=num_labels["jigsaw"],
        num_labels_goemotions=num_labels["goemotions"],
        num_labels_davidson=num_labels["davidson"],
        num_labels_olid=num_labels["olid"],
        num_labels_rumour=num_labels["rumour"],
        freeze_trunk=freeze_trunk_bool # <-- Pass the new argument
    ).to(device)

    loss_fn = MultiTaskLoss(
        ce_weights=ce_weights,
        bce_pos_weights=bce_pos_weights
    ).to(device)
    
    # --- NEW: Update optimizer init ---
    # Get param groups correctly
    optimizer_params = model.get_optimizer_params(
        base_lr=CONFIG["BASE_LR"],
        head_lr=CONFIG["HEAD_LR"]
    )
    optimizer = torch.optim.AdamW(
        optimizer_params,
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    # ----------------------------------

    return model, loss_fn, optimizer


def train_step(model, loss_fn, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    outputs = model(batch["input_ids"], batch["attention_mask"])
    loss_dict = loss_fn(outputs, {k: v for k, v in batch.items() if k.startswith("labels_")})

    if isinstance(loss_dict, dict):
        for task, val in loss_dict.items():
            if torch.isnan(val):
                print(f"⚠️  NaN in loss for {task}: {val}")
        loss = loss_dict.get('total_loss', sum(loss_dict.values()))
    else:
        loss = loss_dict

    if torch.isnan(loss):
        print("⚠️  NaN total loss! Inspect per-task above.")
        for name, logits in outputs.items():
            if torch.isnan(logits).any():
                print(f"   -> NaN in logits for {name}")
        for k, v in batch.items():
            if k.startswith("labels_") and torch.isnan(v).any():
                print(f"   -> NaN in labels for {k}")
        for k, v in batch.items():
            if k.startswith("labels_"):
                print(f"{k}: min={v.min().item()}, max={v.max().item()}, dtype={v.dtype}")
    else:
        loss.backward()
        optimizer.step()
    
    return loss

if __name__ == "__main__":
    main()