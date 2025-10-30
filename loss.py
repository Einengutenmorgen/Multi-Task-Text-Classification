# loss.py

import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    """
    Computes the loss for our multi-task model.
    
    This module encapsulates the logic for:
    1. Using CrossEntropyLoss for multi-class tasks (Davidson, OLID, Rumour).
    2. Using BCEWithLogitsLoss for multi-label tasks (Jigsaw, GoEmotions).
    3. Manually masking the BCE loss to respect the -100 ignore_index.
    4. Averaging the 5 task losses to get a single total_loss.
    """
    def __init__(self):
        super().__init__()
        
        # --- Loss Functions ---
        
        # 1. For Multi-Class tasks (Davidson, OLID, Rumour)
        # The 'ignore_index' parameter automatically handles our -100 labels.
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 2. For Multi-Label tasks (Jigsaw, GoEmotions)
        # We MUST use 'reduction="none"' to get the per-element loss.
        # This allows us to manually apply a mask *before* averaging.
        self.loss_bce = nn.BCEWithLogitsLoss(reduction="none")

    def _compute_bce_loss(self, logits, labels):
        """
        Computes a masked BCE loss.
        
        Args:
            logits (torch.Tensor): Logits from the model (batch_size, num_labels)
            labels (torch.Tensor): Labels from the dataloader (batch_size, num_labels)
        
        Returns:
            torch.Tensor: A single scalar loss value for the active samples.
        """
        # 1. Get the per-element loss
        # Shape: (batch_size, num_labels)
        per_element_loss = self.loss_bce(logits, labels)
        
        # 2. Create a mask for *active* samples.
        # An inactive sample has -100 in its first label position.
        # Shape: (batch_size,)
        mask = (labels[:, 0] != -100).float()
        
        # 3. Expand mask to match loss shape
        # Shape: (batch_size, 1)
        mask_expanded = mask.unsqueeze(1)
        
        # 4. Zero out the loss for inactive samples
        # Shape: (batch_size, num_labels)
        masked_loss = per_element_loss * mask_expanded
        
        # 5. Compute the mean loss *only* for active samples
        # We sum all losses and divide by the number of active samples
        # (multiplied by the number of labels) to get a true mean.
        num_active_samples = mask.sum().clamp(min=1e-8) # Avoid division by zero
        num_labels = labels.shape[1]
        
        total_loss = masked_loss.sum() / (num_active_samples * num_labels)
        
        return total_loss

    def forward(self, model_outputs, batch_labels):
        """
        Computes all 5 task losses and the combined total_loss.
        
        Args:
            model_outputs (dict): A dict of logits from MultiTaskModel.
                                  {'jigsaw': ..., 'goemotions': ..., ...}
            batch_labels (dict): A dict of labels from the dataloader.
                                 {'labels_jigsaw': ..., 'labels_goemotions': ..., ...}
                                 
        Returns:
            dict: A dictionary of all 6 computed losses.
        """
        
        # --- 1. Multi-Class Losses (Easy) ---
        # loss_ce automatically ignores samples where the label is -100.
        loss_davidson = self.loss_ce(
            model_outputs['davidson'], 
            batch_labels['labels_davidson']
        )
        loss_olid = self.loss_ce(
            model_outputs['olid'], 
            batch_labels['labels_olid']
        )
        loss_rumour = self.loss_ce(
            model_outputs['rumour'], 
            batch_labels['labels_rumour']
        )
        
        # --- 2. Multi-Label Losses (Masked) ---
        loss_jigsaw = self._compute_bce_loss(
            model_outputs['jigsaw'],
            batch_labels['labels_jigsaw']
        )
        loss_goemotions = self._compute_bce_loss(
            model_outputs['goemotions'],
            batch_labels['labels_goemotions']
        )
        
        # --- 3. Total Loss (V1: Simple Average) ---
        # As planned, we just average the 5 task losses.
        total_loss = (
            loss_jigsaw + 
            loss_goemotions + 
            loss_davidson + 
            loss_olid + 
            loss_rumour
        ) / 5
        
        # 4. Return all losses for training and logging
        return {
            'total_loss': total_loss,
            'loss_jigsaw': loss_jigsaw,
            'loss_goemotions': loss_goemotions,
            'loss_davidson': loss_davidson,
            'loss_olid': loss_olid,
            'loss_rumour': loss_rumour
        }