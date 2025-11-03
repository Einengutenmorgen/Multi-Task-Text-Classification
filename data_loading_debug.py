import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging

# === NAN-DEBUG logger setup ===
logging.basicConfig(
    filename='nan_trace.log',
    filemode='w',  # overwrite each run
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('nan_debug')

class UnifiedDataset(Dataset):
    def __init__(self, datasets, model_name, max_length=128):
        self.datasets = datasets
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.task_indices = []  # list of (task_name, local_idx)
        for task_name, ds in datasets.items():
            for i in range(len(ds)):
                self.task_indices.append((task_name, i))

    def __len__(self):
        return len(self.task_indices)

    def __getitem__(self, idx):
        task_name, local_idx = self.task_indices[idx]
        example = self.datasets[task_name][local_idx]

        text = example['text'] if 'text' in example else ''
        label = example['label'] if 'label' in example else -100

        if text is None or not isinstance(text, str):
            logger.warning(f"[{task_name}] Invalid text at idx {local_idx}: {text}")
            text = ''

        # Tokenize safely
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # === Debug check ===
        if torch.isnan(input_ids).any() or torch.isnan(attention_mask).any():
            logger.error(f"[{task_name}] NaN detected in inputs (idx={local_idx})")

        # convert label to tensor if needed
        label_tensor = torch.tensor(label)
        if torch.isnan(label_tensor).any():
            logger.error(f"[{task_name}] NaN detected in label (idx={local_idx})")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor,
            'task': task_name
        }


def debug_collate_fn(batch):
    # collect all labels per task
    task_names = [b['task'] for b in batch]
    labels = torch.stack([b['labels'] for b in batch])

    # === Check all -100 ===
    for task in set(task_names):
        mask = [i for i, t in enumerate(task_names) if t == task]
        lbls = labels[mask]
        if (lbls == -100).all():
            msg = f"[{task}] Batch has all labels -100 ({len(mask)} samples)"
            logger.warning(msg)
            print(f"[NAN-DEBUG] {msg}")

    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels': labels,
        'task': task_names
    }
