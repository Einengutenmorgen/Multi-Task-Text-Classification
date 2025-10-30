# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, random_split
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
import json

# Import our custom modules
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
    "BATCH_SIZE": 16,  # Adjust based on your VRAM
    "MAX_LENGTH": 128,
    "BASE_LR": 1e-5,   # For the mxbai trunk
    "HEAD_LR": 1e-4,   # For our new classification heads
    "WEIGHT_DECAY": 0.01,
    "VALIDATION_SPLIT": 0.1, # 10% for validation
    "CHECKPOINT_DIR": "./checkpoints"
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
                # Get preds: Apply sigmoid and threshold
                task_preds = (torch.sigmoid(task_logits[mask]) > 0.5).int()
            else:
                # Multi-class: active if label is not -100
                mask = task_labels != -100
                # Get preds: Argmax
                task_preds = torch.argmax(task_logits[mask], dim=1)
            
            # Store the active preds and labels
            if mask.sum() > 0:
                all_preds[task].append(task_preds.cpu().numpy())
                all_labels[task].append(task_labels[mask].cpu().numpy())

    # --- 5. Calculate Metrics ---
    metrics = {}
    
    for task in TASKS:
        if not all_labels[task]:
            print(f"Warning: No active samples found for task '{task}' in validation set.")
            metrics[f'{task}_f1'] = 0.0
            continue
            
        # Concatenate all batch results
        task_all_labels = np.concatenate(all_labels[task])
        task_all_preds = np.concatenate(all_preds[task])
        
        # Calculate Weighted F1
        # 'weighted' accounts for label imbalance within each task
        f1 = f1_score(task_all_labels, task_all_preds, average='weighted', zero_division=0)
        metrics[f'{task}_f1'] = f1
        
    # Calculate the main metric: Average F1
    metrics['avg_f1'] = np.mean(list(metrics.values()))
    
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

    # Split the dataset
    val_size = int(len(full_dataset) * CONFIG['VALIDATION_SPLIT'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # --- This is the correct patching ---
    # Give the Subset the attributes our TaskSampler needs to find
    train_dataset.task_data = full_dataset.task_data
    train_dataset.full_task_indices_list = full_dataset.task_indices
    # ------------------------------------

    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- 3. Create DataLoaders ---
    # Training: Use TaskSampler for balanced batches
    train_sampler = TaskSampler(train_dataset)
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

    # --- 4. Initialize Model ---
    print("Initializing MultiTaskModel...")
    model = MultiTaskModel(
        model_name=CONFIG['MODEL_NAME'],
        num_labels_jigsaw=label_counts['jigsaw'],
        num_labels_goemotions=label_counts['goemotions'],
        num_labels_davidson=label_counts['davidson'],
        num_labels_olid=label_counts['olid'],
        num_labels_rumour=label_counts['rumour']
    )
    model.to(device)

    # --- 5. Initialize Loss, Optimizer, and Scheduler ---
    loss_fn = MultiTaskLoss().to(device)
    
    # Get parameter groups for differential LR
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
            print(f"  - {task.capitalize()} F1: {val_metrics[f'{task}_f1']:.4f}")
        print(f"  - === Average F1: {val_metrics['avg_f1']:.4f} === ")
        
        # --- 7. Checkpointing ---
        if val_metrics['avg_f1'] > best_avg_f1:
            best_avg_f1 = val_metrics['avg_f1']
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


if __name__ == "__main__":
    main()