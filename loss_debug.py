import torch
import torch.nn as nn
import logging

logging.basicConfig(
    filename='nan_trace.log',
    filemode='a',  # append to same log
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('nan_debug')

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, outputs, labels, task_names):
        total_loss = 0.0
        task_losses = {}

        for task_name in outputs.keys():
            logits = outputs[task_name]
            target = labels[task_name]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                logger.error(f"[{task_name}] NaN/Inf detected in logits: range=({logits.min()}, {logits.max()})")
                print(f"[NAN-DEBUG] NaN/Inf in logits for {task_name} — logged.")

            if task_name in ['jigsaw', 'goemotions']:
                valid_mask = (target != -100)
                loss = self.bce(logits, target.float())
                valid_count = valid_mask.sum().item()
                if valid_count == 0:
                    logger.warning(f"[{task_name}] Empty valid_mask (all -100)")
                    loss_value = torch.tensor(0.0, device=logits.device)
                else:
                    loss_value = (loss * valid_mask).sum() / valid_mask.sum()
            else:
                try:
                    loss_value = self.ce(logits, target.long())
                except Exception as e:
                    logger.error(f"[{task_name}] Exception in CE loss: {str(e)}")
                    loss_value = torch.tensor(0.0, device=logits.device)

            if torch.isnan(loss_value) or torch.isinf(loss_value):
                logger.error(f"[{task_name}] Loss NaN/Inf after computation. Logits range=({logits.min()}, {logits.max()})")
                print(f"[NAN-DEBUG] NaN in {task_name} loss — logged.")

            task_losses[task_name] = loss_value

        total_loss = sum(task_losses.values()) / len(task_losses) if len(task_losses) > 0 else torch.tensor(0.0)

        if torch.isnan(total_loss):
            logger.error(f"[GLOBAL] Total loss NaN. Individual losses: {task_losses}")
            print("[NAN-DEBUG] Total loss NaN — logged.")

        return total_loss, task_losses
