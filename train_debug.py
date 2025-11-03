import torch
from tqdm import tqdm
import logging
from loss_debug import MultiTaskLoss

logging.basicConfig(
    filename='nan_trace.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('nan_debug')


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = MultiTaskLoss()

    nan_batches = {t: 0 for t in ['jigsaw', 'goemotions', 'davidson', 'olid', 'rumour']}

    for step, batch in enumerate(tqdm(dataloader, desc='Training Epoch')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        task_names = batch['task']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, task_names)

        loss, task_losses = criterion(outputs, labels, task_names)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"[GLOBAL] NaN/Inf loss at batch {step}. Task losses: {task_losses}")
            print(f"[NAN-DEBUG] NaN loss detected at batch {step} — details logged.")
            for t, v in task_losses.items():
                if torch.isnan(v) or torch.isinf(v):
                    nan_batches[t] += 1
            continue

        loss.backward()

        # gradient audit
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"[GRADIENT] NaN gradient in {name} at batch {step}")

        optimizer.step()

    print("[NAN-DEBUG] Summary of NaN batches:")
    for t, c in nan_batches.items():
        if c > 0:
            print(f"  - {t}: {c} NaN batches (see nan_trace.log)")
            logger.warning(f"[SUMMARY] {t}: {c} NaN batches in epoch")


def validate_epoch(model, dataloader, device):
    model.eval()
    criterion = MultiTaskLoss()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc='Validating')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            task_names = batch['task']

            outputs = model(input_ids, attention_mask, task_names)
            loss, _ = criterion(outputs, labels, task_names)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"[VALIDATION] NaN/Inf loss at batch {step}")
                print(f"[NAN-DEBUG] NaN loss in validation batch {step} — logged.")

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    from data_loading_debug import UnifiedDataset, debug_collate_fn
    import torch.optim as optim
    import os

    parser = argparse.ArgumentParser(description="Run NaN-debug training loop")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--model_name", type=str, default="mixedbread-ai/mxbai-embed-large-v1")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Dummy placeholder — replace with your dataset loading logic
    # e.g., unified_datasets = load_all_datasets()
    from collections import defaultdict
    dummy_datasets = defaultdict(list)
    dummy_datasets["jigsaw"] = [{"text": "sample text", "label": 0}] * 10
    dummy_datasets["goemotions"] = [{"text": "happy", "label": 1}] * 10
    dummy_datasets["davidson"] = [{"text": "neutral", "label": 2}] * 10
    dummy_datasets["olid"] = [{"text": "offensive", "label": 1}] * 10
    dummy_datasets["rumour"] = [{"text": "claim", "label": 0}] * 10

    dataset = UnifiedDataset(dummy_datasets, args.model_name)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=debug_collate_fn)

    # Import your actual model
    from model import MultiTaskModel  # Assuming this is your model file
    num_labels_jigsaw = 6         # toxic, severe_toxic, obscene, threat, insult, identity_hate
    num_labels_goemotions = 28    # 27 emotions + neutral
    num_labels_davidson = 3       # hate, offensive, neither
    num_labels_olid = 2           # offensive / not offensive
    num_labels_rumour = 4         # true, false, unverified, non-rumour

    model = MultiTaskModel(
        args.model_name,
        num_labels_jigsaw,
        num_labels_goemotions,
        num_labels_davidson,
        num_labels_olid,
        num_labels_rumour
    )    
    model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    print(f"[NAN-DEBUG] Starting training for {args.epochs} epoch(s) on device: {args.device}")

    for epoch in range(args.epochs):
        print(f"[NAN-DEBUG] === Epoch {epoch+1}/{args.epochs} ===")
        train_epoch(model, dataloader, optimizer, args.device)
        validate_epoch(model, dataloader, args.device)

    print("[NAN-DEBUG] Training complete. Check nan_trace.log for details.")
