#test_nan_tasks.py

import pytest
import torch
import train
import data_loading
from torch.utils.data import DataLoader # Added for the new test

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
TASKS_TO_TEST = ["davidson", "olid", "rumour"]


@pytest.mark.parametrize("task_name", TASKS_TO_TEST)
def test_diagnostic_single_task_batch_behavior(task_name):
    """
    Diagnostic test: checks that a single-task batch produces:
    1. A valid loss for the TARGET task.
    2. A NaN loss for OTHER CrossEntropy tasks.
    3. A loss of 0.0 for the robust BCE tasks.
    4. A NaN loss for total_loss.
    
    This confirms our understanding of nn.CrossEntropyLoss and our custom BCE loss.
    """
    print(f"\n===== Diagnostic Test: {task_name} =====")

    tokenizer_name = train.CONFIG.get("MODEL_NAME", "distilbert-base-uncased")
    max_length = train.CONFIG.get("MAX_LENGTH", 128)
    batch_size = train.CONFIG.get("BATCH_SIZE", 8)

    # 1. Get a batch with data from *only* one task
    dataloader = data_loading.get_dataloader_for_task(
        task_name=task_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        batch_size=batch_size,
    )

    model, loss_fn, optimizer = train.initialize_training_objects(DEVICE)
    batch = next(iter(dataloader))
    batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # 2. Replicate the loss calculation part of train_step to get the full dict
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(batch["input_ids"], batch["attention_mask"])
    
    # Get the full loss dictionary
    loss_dict = loss_fn(outputs, {k: v for k, v in batch.items() if k.startswith("labels_")})

    # --- Assertions ---
    
    # 1. The target task's loss should be a valid number
    target_loss = loss_dict[f'loss_{task_name}']
    print(f"  - Target loss ({task_name}): {target_loss.item()}")
    assert not torch.isnan(target_loss), f"Target task {task_name} unexpectedly had a NaN loss."

    # 2. The *other* CE tasks (which are all -100) should have NaN loss
    other_ce_tasks = [t for t in TASKS_TO_TEST if t != task_name]
    for other_task in other_ce_tasks:
        other_loss = loss_dict[f'loss_{other_task}']
        print(f"  - Other CE loss ({other_task}): {other_loss.item()}")
        assert torch.isnan(other_loss), f"Other CE task {other_task} did not have expected NaN loss."
    
    # 3. The BCE tasks (jigsaw, goemotions) should have a loss of 0.0
    #    This confirms our custom BCE loss is robust to all-masked batches.
    bce_loss_jigsaw = loss_dict['loss_jigsaw']
    bce_loss_goemotions = loss_dict['loss_goemotions']
    print(f"  - Jigsaw loss (BCE): {bce_loss_jigsaw.item()}")
    print(f"  - GoEmotions loss (BCE): {bce_loss_goemotions.item()}")
    assert bce_loss_jigsaw.item() == 0.0, "Jigsaw loss was not 0.0 on a masked batch."
    assert bce_loss_goemotions.item() == 0.0, "GoEmotions loss was not 0.0 on a masked batch."

    # 4. The total loss should be NaN (because (valid + 0 + 0 + nan + nan) / 5 = nan)
    total_loss = loss_dict['total_loss']
    print(f"  - Total loss: {total_loss.item()}")
    assert torch.isnan(total_loss), "Total loss was not NaN as expected for a single-task batch."


def test_training_with_tasksampler_no_nan():
    """
    Integration test: checks that a mixed-task batch from the
    TaskSampler (as used in real training) does NOT produce NaN loss.
    This is the most important test for training stability.
    """
    print("\n===== Integration Test: Mixed-Task Batch =====")
    
    tokenizer_name = train.CONFIG.get("MODEL_NAME", "distilbert-base-uncased")
    max_length = train.CONFIG.get("MAX_LENGTH", 128)
    batch_size = train.CONFIG.get("BATCH_SIZE", 8)

    # 1. Load full dataset
    print("Loading full dataset for sampler...")
    full_dataset = data_loading.UnifiedDataset(
        tokenizer_name=tokenizer_name, 
        max_length=max_length
    )
    
    # 2. Create the TaskSampler (the key component)
    print("Initializing TaskSampler...")
    sampler = data_loading.TaskSampler(full_dataset)
    
    # 3. Create DataLoader
    dataloader = DataLoader(
        full_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0 # Simpler for testing
    )

    model, loss_fn, optimizer = train.initialize_training_objects(DEVICE)
    
    # 4. Get one mixed-task batch
    print("Fetching one mixed-task batch...")
    batch = next(iter(dataloader))
    batch = {k: v.to(DEVICE) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # 5. Run one *actual* training step
    print("Running train.train_step...")
    total_loss = train.train_step(model, loss_fn, optimizer, batch)

    # --- Assertion ---
    # The total loss from a mixed batch should NEVER be NaN.
    print(f"  - Mixed-batch total_loss: {total_loss.item()}")
    assert not torch.isnan(total_loss), \
        "CRITICAL BUG: total_loss was NaN during a mixed-batch. This should not happen."