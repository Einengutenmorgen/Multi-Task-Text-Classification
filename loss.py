# loss.py

import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    """
    Computes the loss for our multi-task model.
    
    This module encapsulates the logic for:
    1. Using CrossEntropyLoss for multi-class tasks (Davidson, OLID, Rumour).
    2. Using BCEWithLogitsLoss for multi-label tasks (Jigsaw, GoEmotions).
    3. Manually masking BOTH loss types to respect the -100 ignore_index.
    4. Averaging only the *active* task losses to get a single total_loss.
    """
    def __init__(self, ce_weights=None, bce_pos_weights=None):
        """
        Initializes the loss module.

        Args:
            ce_weights (dict, optional): A dictionary mapping task names 
                ('davidson', 'olid', 'rumour') to weight tensors for CE loss.
                Tensors should be on the CPU; they will be moved by .to(device).
            bce_pos_weights (dict, optional): A dictionary mapping task names
                ('jigsaw', 'goemotions') to pos_weight tensors for BCE loss.
                Tensors should be on the CPU; they will be moved by .to(device).
        """
        super().__init__()
        
        # --- Handle default weights if none provided ---
        if ce_weights is None:
            ce_weights = {'davidson': None, 'olid': None, 'rumour': None}
        if bce_pos_weights is None:
            bce_pos_weights = {'jigsaw': None, 'goemotions': None}

        # --- Loss Functions ---
        
        # 1. For Multi-Class tasks (Davidson, OLID, Rumour)
        # --- MODIFIED: Use nn.ModuleDict to register sub-modules ---
        # This ensures that .to(device) correctly moves the weight tensors.
        self.loss_ce = nn.ModuleDict({
            'davidson': nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="none",
                weight=ce_weights.get('davidson')
            ),
            'olid': nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="none",
                weight=ce_weights.get('olid')
            ),
            'rumour': nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction="none",
                weight=ce_weights.get('rumour')
            )
        })
        
        # 2. For Multi-Label tasks (Jigsaw, GoEmotions)
        # --- MODIFIED: Use nn.ModuleDict ---
        self.loss_bce = nn.ModuleDict({
            'jigsaw': nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=bce_pos_weights.get('jigsaw')
            ),
            'goemotions': nn.BCEWithLogitsLoss(
                reduction="none",
                pos_weight=bce_pos_weights.get('goemotions')
            )
        })

    def _compute_bce_loss(self, logits, labels, task_name):
        """
        Computes a masked BCE loss.
        """
        # 1. Get the per-element loss
        per_element_loss = self.loss_bce[task_name](logits, labels)
        
        # 2. Create a mask for *active* samples.
        mask = (labels[:, 0] != -100).float()
        
        # 3. Expand mask to match loss shape
        mask_expanded = mask.unsqueeze(1)
        
        # 4. Zero out the loss for inactive samples
        masked_loss = per_element_loss * mask_expanded
        
        # 5. Compute the mean loss *only* for active samples
        num_active_samples = mask.sum().clamp(min=1e-8) # Avoid division by zero
        num_labels = labels.shape[1]
        
        # --- FIX: Ensure we return 0.0 if no samples are active ---
        if num_active_samples == 0:
            return torch.tensor(0.0, device=logits.device)
            
        total_loss = masked_loss.sum() / (num_active_samples * num_labels)
        
        return total_loss

    # --- NEW: Robust helper function for Cross-Entropy Loss ---
    def _compute_ce_loss(self, logits, labels, task_name):
        """
        Computes a masked Cross-Entropy loss.
        
        Args:
            logits (torch.Tensor): Logits from the model (batch_size, num_classes)
            labels (torch.Tensor): Labels from the dataloader (batch_size,)
            task_name (str): The name of the task ('davidson', 'olid', 'rumour')
        
        Returns:
            torch.Tensor: A single scalar loss value for the active samples.
        """
        # 1. Get the per-element loss
        # Shape: (batch_size,)
        per_element_loss = self.loss_ce[task_name](logits, labels)
        
        # 2. Create a mask for *active* samples
        # Shape: (batch_size,)
        mask = (labels != -100).float()
        
        # 3. Zero out the loss for inactive samples
        # Shape: (batch_size,)
        masked_loss = per_element_loss * mask
        
        # 4. Compute the mean loss *only* for active samples
        num_active_samples = mask.sum().clamp(min=1e-8) # Avoid division by zero
        
        # --- FIX: Ensure we return 0.0 if no samples are active ---
        if num_active_samples == 0:
            return torch.tensor(0.0, device=logits.device)

        total_loss = masked_loss.sum() / num_active_samples
        
        return total_loss
    # --- END NEW ---

    def forward(self, model_outputs, batch_labels):
        """
        Computes all 5 task losses and the combined total_loss.
        """
        
        # --- 1. Multi-Class Losses (MODIFIED) ---
        # Use our new robust _compute_ce_loss helper
        loss_davidson = self._compute_ce_loss(
            model_outputs['davidson'], 
            batch_labels['labels_davidson'],
            'davidson' # <-- Pass task name
        )
        loss_olid = self._compute_ce_loss(
            model_outputs['olid'], 
            batch_labels['labels_olid'],
            'olid' # <-- Pass task name
        )
        loss_rumour = self._compute_ce_loss(
            model_outputs['rumour'], 
            batch_labels['labels_rumour'],
            'rumour' # <-- Pass task name
        )
        
        # --- 2. Multi-Label Losses (MODIFIED) ---
        loss_jigsaw = self._compute_bce_loss(
            model_outputs['jigsaw'],
            batch_labels['labels_jigsaw'],
            'jigsaw' # <-- Pass task name
        )
        loss_goemotions = self._compute_bce_loss(
            model_outputs['goemotions'],
            batch_labels['labels_goemotions'],
            'goemotions' # <-- Pass task name
        )
        
        # --- 3. Total Loss (MODIFIED: Robust Averaging) ---
        # Average the loss *only* over tasks that were active in this batch
        
        all_losses = [
            loss_jigsaw, 
            loss_goemotions, 
            loss_davidson, 
            loss_olid, 
            loss_rumour
        ]
        
        # Sum all losses (inactive tasks will be 0.0)
        loss_sum = sum(all_losses)
        
        # Count how many tasks had a loss > 0
        # We use .item() for a safe boolean check
        num_active_tasks = sum(1.0 for loss in all_losses if loss.item() > 0)
        
        # Avoid division by zero if a batch somehow has no labels for any task
        num_active_tasks = max(1.0, num_active_tasks)
        
        # Divide by the number of *active* tasks
        total_loss = loss_sum / num_active_tasks
        
        # 4. Return all losses for training and logging
        return {
            'total_loss': total_loss,
            'loss_jigsaw': loss_jigsaw,
            'loss_goemotions': loss_goemotions,
            'loss_davidson': loss_davidson,
            'loss_olid': loss_olid,
            'loss_rumour': loss_rumour
        }