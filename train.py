# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler, Subset
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
    GOEMOTIONS_LABEL_COLS, # <-- Need to import this
    DAVIDSON_LABEL_MAP, 
    OLID_LABEL_MAP, 
    RUMOUR_LABEL_MAP
)
from model import MultiTaskModel
from loss import MultiTaskLoss

# --- 1. Configuration (v5 - LoRA) ---
CONFIG = {
    "MODEL_NAME": "mixedbread-ai/mxbai-embed-large-v1",
    "EPOCHS": 3,           # <-- Let's try 5 epochs, PEFT can overfit less
    "BATCH_SIZE": 8,      # Physical batch size that fits in VRAM
    "MAX_LENGTH": 124,
    "BASE_LR": 1e-4,       # <-- LR for LoRA layers (can be higher than full fine-tune)
    "HEAD_LR": 1e-4,       # <-- Match LoRA LR
    "WEIGHT_DECAY": 0.01,
    "VALIDATION_SPLIT": 0.1,
    "CHECKPOINT_DIR": "./checkpoints/v5_lora/", # <-- NEW: v5 directory
    "EPOCH_SAMPLING_SIZE": 50000,
    
    # --- SPEED OPTIMIZATIONS ---
    "NUM_WORKERS": 8,
    "PIN_MEMORY": False,   # <-- Correct setting for MPS
    "GRADIENT_ACCUMULATION_STEPS": 8 # Effective batch size = 16 * 4 = 64
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

# --- MODIFIED: train_one_epoch with Optimizations ---
def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    """Performs one full training epoch with mixed precision and gradient accumulation."""
    model.train()
    
    # Track losses
    total_loss_sum = 0
    task_loss_sums = {task: 0 for task in TASKS}
    
    accumulation_steps = CONFIG["GRADIENT_ACCUMULATION_STEPS"]
    
    loop = tqdm(dataloader, desc=f"Training Epoch", leave=False)
    
    # --- NEW: Zero grad at the start ---
    optimizer.zero_grad()
    
    for step, batch in enumerate(loop):
        # 1. Move batch to device
        batch = batch_to_device(batch, device)
        
        # 2. Get inputs and labels from batch
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        # 3. --- NEW: Automatic Mixed Precision (AMP) ---
        # Use torch.autocast for faster float16 computations
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16):
            outputs = model(**inputs)
            losses = loss_fn(model_outputs=outputs, batch_labels=batch)
            
            # --- NEW: Scale loss for accumulation ---
            loss = losses['total_loss'] / accumulation_steps
        
        # 5. Backpropagation
        # Note: We don't need GradScaler for MPS, just backward()
        loss.backward()
        
        # 6. --- NEW: Gradient Accumulation Step ---
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            # Clip gradients to prevent exploding gradients (good practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Perform the optimizer step
            optimizer.step()
            scheduler.step()
            
            # Zero the gradients for the *next* accumulation cycle
            optimizer.zero_grad()

        # 7. Log losses (using the *unscaled* loss)
        total_loss_sum += losses['total_loss'].item()
        for task in TASKS:
            task_loss_sums[task] += losses[f'loss_{task}'].item()
            
        # Update TQDM with live *unscaled* loss
        loop.set_postfix(
            loss=losses['total_loss'].item(),
            lr=scheduler.get_last_lr()[0]
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
    
    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    
    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    
    for batch in loop:
        batch = batch_to_device(batch, device)
        
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        labels = batch
        
        # --- NEW: Use autocast for evaluation (faster) ---
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'cuda' else torch.float16):
            outputs = model(**inputs)
        
        # 4. Filter active predictions and labels for each task
        for task in TASKS:
            task_logits = outputs[task]
            task_labels = labels[f'labels_{task}']
            
            if task in ['jigsaw', 'goemotions']:
                mask = task_labels[:, 0] != -100
                if mask.sum() == 0: continue
                task_preds = (torch.sigmoid(task_logits[mask]) > 0.5).int()
            else:
                mask = task_labels != -100
                if mask.sum() == 0: continue
                task_preds = torch.argmax(task_logits[mask], dim=1)
            
            all_preds[task].append(task_preds.cpu().numpy())
            all_labels[task].append(task_labels[mask].cpu().numpy())

    # --- 5. Calculate Metrics ---
    metrics = {}
    
    for task in TASKS:
        if not all_labels[task]:
            print(f"Warning: No active samples found for task '{task}' in validation set.")
            metrics[f'{task}_f1_macro'] = 0.0 # Use macro key
            continue
            
        task_all_labels = np.concatenate(all_labels[task])
        task_all_preds = np.concatenate(all_preds[task])
        
        f1 = f1_score(task_all_labels, task_all_preds, average='macro', zero_division=0)
        metrics[f'{task}_f1_macro'] = f1
        
    metrics['avg_f1_macro'] = np.mean(list(metrics.values()))
    
    return metrics


# --- MODIFIED: main() to pre-calculate weights ---
def main():
    print("--- Starting V1 Multi-Task Training Pipeline ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading full dataset for model: {CONFIG['MODEL_NAME']}")
    full_dataset = UnifiedDataset(tokenizer_name=CONFIG['MODEL_NAME'], max_length=CONFIG['MAX_LENGTH'])
    
    # --- FIX: Pass the correct schema object ---
    label_counts = get_label_counts(SCHEMA) 
    print(f"Discovered label counts: {label_counts}")

    print("Creating task-stratified train/val split...")
    all_indices = list(range(len(full_dataset)))
    labels_for_stratify = [task_name for (task_name, _) in full_dataset.task_indices]

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=CONFIG['VALIDATION_SPLIT'],
        stratify=labels_for_stratify,
        random_state=42
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_dataset.task_data = full_dataset.task_data
    train_dataset.full_task_indices_list = full_dataset.task_indices

    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- NEW: Pre-calculate Class Weights (Phase 2) ---
    print("Calculating class weights for loss balancing...")
    ce_weights = {}
    bce_pos_weights = {}

    # Get all labels from the *training set*
    train_labels = {task: [] for task in TASKS}
    for idx in tqdm(train_dataset.indices, desc="Scanning train labels"):
        task_name, item_index = full_dataset.task_indices[idx]
        train_labels[task_name].append(full_dataset.task_data[task_name]['labels'][item_index])

    # 1. Calculate CE weights (Davidson, OLID, Rumour)
    for task in ['davidson', 'olid', 'rumour']:
        labels = np.array(train_labels[task])
        num_classes = label_counts[task]
        counts = np.bincount(labels, minlength=num_classes)
        
        # Calculate weights: N_total / (N_classes * N_class)
        total_samples = counts.sum()
        if total_samples > 0:
            weights = total_samples / (num_classes * counts + 1e-8) # Add epsilon to avoid div by zero
            ce_weights[task] = torch.tensor(weights, dtype=torch.float)
            print(f"  - {task} weights: {weights}")
        else:
            ce_weights[task] = None # No data for this task in split

    # 2. Calculate BCE pos_weights (Jigsaw, GoEmotions)
    for task in ['jigsaw', 'goemotions']:
        labels_np = np.array(train_labels[task])
        if labels_np.shape[0] > 0:
            # N_negative / N_positive
            n_pos = np.sum(labels_np, axis=0)
            n_neg = labels_np.shape[0] - n_pos
            pos_weights = n_neg / (n_pos + 1e-8)
            bce_pos_weights[task] = torch.tensor(pos_weights, dtype=torch.float)
            print(f"  - {task} pos_weights (avg): {pos_weights.mean()}")
        else:
            bce_pos_weights[task] = None
    # --- End Class Weight Calculation ---

    print(f"Initializing TaskSampler with strategy: {CONFIG['EPOCH_SAMPLING_SIZE']}")
    train_sampler = TaskSampler(
        train_dataset,
        epoch_sampling_size=CONFIG['EPOCH_SAMPLING_SIZE']
    )
    
    # --- MODIFIED: Added num_workers and pin_memory ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        sampler=train_sampler,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'] # <-- This will now correctly pass False
    )
    
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        sampler=val_sampler,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'] # <-- This will now correctly pass False
    )

    print("Initializing MultiTaskModel...")
    
    # --- MODIFIED FOR LORA ---
    # We tell the model to freeze the trunk.
    # The model.py logic will then apply LoRA.
    # The model's .get_optimizer_params will return the LoRA params.
    # The optimizer will use CONFIG["BASE_LR"] for those params.
    freeze_trunk_bool = True # <-- This now means "Use LoRA"
    
    print(f"freeze_trunk={freeze_trunk_bool}. LoRA will be enabled.")
    
    model = MultiTaskModel(
        model_name=CONFIG["MODEL_NAME"],
        num_labels_jigsaw=len(data_loading.SCHEMA["jigsaw"]),
        num_labels_goemotions=len(data_loading.SCHEMA["goemotions"]),
        num_labels_davidson=len(data_loading.SCHEMA["davidson"]),
        num_labels_olid=len(data_loading.SCHEMA["olid"]),
        num_labels_rumour=len(data_loading.SCHEMA["rumour"]),
        freeze_trunk=freeze_trunk_bool # <-- Set to True to enable PEFT/LoRA
    )

    model.to(device)

    # --- MODIFIED: Pass weights to loss function ---
    loss_fn = MultiTaskLoss(
        ce_weights=ce_weights,
        bce_pos_weights=bce_pos_weights
    ).to(device)
    
    optimizer_params = model.get_optimizer_params(
        base_lr=CONFIG['BASE_LR'], # This LR will be applied to LoRA params
        head_lr=CONFIG['HEAD_LR']
    )
    
    optimizer = optim.AdamW(
        optimizer_params,
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    # --- MODIFIED: Calculate scheduler steps based on accumulation ---
    num_training_steps = (len(train_loader) // CONFIG['GRADIENT_ACCUMULATION_STEPS']) * CONFIG['EPOCHS']
    num_warmup_steps = int(num_training_steps * 0.1) # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_avg_f1 = 0.0
    os.makedirs(CONFIG['CHECKPOINT_DIR'], exist_ok=True)
    
    for epoch in range(CONFIG['EPOCHS']):
        print(f"\n--- Epoch {epoch + 1} / {CONFIG['EPOCHS']} ---")
        
        avg_total_loss, avg_task_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )
        print(f"Epoch {epoch+1} Avg Train Loss: {avg_total_loss:.4f}")
        for task, loss in avg_task_losses.items():
            print(f"  - Avg Train {task} Loss: {loss:.4f}")

        val_metrics = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1} Validation Metrics:")
        for task in TASKS:
            print(f"  - {task.capitalize()} F1 (Macro): {val_metrics[f'{task}_f1_macro']:.4f}")
        print(f"  - === Average F1 (Macro): {val_metrics['avg_f1_macro']:.4f} === ")
        
        if val_metrics['avg_f1_macro'] > best_avg_f1:
            best_avg_f1 = val_metrics['avg_f1_macro']
            checkpoint_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "best_model.pth")
            print(f"New best model! Saving checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            
            metrics_path = os.path.join(CONFIG['CHECKPOINT_DIR'], "best_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(val_metrics, f, indent=2)

    print("\n--- Training Complete ---")
    print(f"Best Average F1 Score: {best_avg_f1:.4f}")
    print(f"Best model saved to {os.path.join(CONFIG['CHECKPOINT_DIR'], 'best_model.pth')}")

# --- MODIFIED: initialize_training_objects to pass weights ---
def initialize_training_objects(device):
    """
    Factory to initialize model, loss, and optimizer exactly as used in training.
    (MODIFIED)
    """
    import data_loading
    from model import MultiTaskModel
    from loss import MultiTaskLoss

    # --- FIX: Need to get label_counts properly ---
    # This requires loading the schema from data_loading
    num_labels = {
        'jigsaw': len(data_loading.JIGSAW_LABEL_COLS),
        'goemotions': len(data_loading.GOEMOTIONS_LABEL_COLS),
        'davidson': len(data_loading.DAVIDSON_LABEL_MAP),
        'olid': len(data_loading.OLID_LABEL_MAP),
        'rumour': len(data_loading.RUMOUR_LABEL_MAP)
    }

    # --- MODIFIED: Set freeze_trunk=True to enable LoRA ---
    freeze_trunk_bool = True 

    model = MultiTaskModel(
        model_name=CONFIG["MODEL_NAME"],
        num_labels_jigsaw=num_labels["jigsaw"],
        num_labels_goemotions=num_labels["goemotions"],
        num_labels_davidson=num_labels["davidson"],
        num_labels_olid=num_labels["olid"],
        num_labels_rumour=num_labels["rumour"],
        freeze_trunk=freeze_trunk_bool
    ).to(device)

    # --- MODIFIED: Pass dummy weights (None) ---
    # Tests don't need real weights, just the correct init signature
    loss_fn = MultiTaskLoss(ce_weights=None, bce_pos_weights=None).to(device)
    
    optimizer_params = model.get_optimizer_params(
        base_lr=CONFIG["BASE_LR"],
        head_lr=CONFIG["HEAD_LR"]
    )
    optimizer = torch.optim.AdamW(
        optimizer_params,
        weight_decay=CONFIG['WEIGHT_DECAY']
    )

    return model, loss_fn, optimizer


# --- MODIFIED: train_step for autocast and accumulation ---
def train_step(model, loss_fn, optimizer, batch):
    # This function is used by test_integration_train.py
    # We will simulate a single step with accumulation_steps=1
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Use autocast for mixed precision
    with torch.autocast(device_type=batch['input_ids'].device.type, dtype=torch.bfloat16 if batch['input_ids'].device.type == 'cuda' else torch.float16):
        outputs = model(batch["input_ids"], batch["attention_mask"])
        loss_dict = loss_fn(outputs, {k: v for k, v in batch.items() if k.startswith("labels_")})
        loss = loss_dict.get('total_loss')

    if torch.isnan(loss):
        print("⚠️  NaN total loss! Inspect per-task above.")
    else:
        # No scaler needed for MPS
        loss.backward()
        optimizer.step()
    
    return loss

if __name__ == "__main__":
    main()