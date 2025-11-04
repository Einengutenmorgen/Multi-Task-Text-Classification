# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultiTaskModel(nn.Module):
    """
    A Multi-Task model with a shared trunk and separate heads for each task.
    """
    def __init__(self, 
                 model_name: str, 
                 num_labels_jigsaw: int,
                 num_labels_goemotions: int,
                 num_labels_davidson: int,
                 num_labels_olid: int,
                 num_labels_rumour: int,
                 freeze_trunk: bool = True): # <-- NEW ARGUMENT
        
        super().__init__()
        
        # 1. Load the Shared Trunk
        # We use AutoConfig to get the hidden_size
        config = AutoConfig.from_pretrained(model_name)
        self.trunk = AutoModel.from_pretrained(model_name)
        
        # --- NEW: Conditionally freeze the trunk ---
        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False
        # If freeze_trunk is False, params will require_grad by default
        # -------------------------------------------
        
        
        # Get the output dimension from the trunk's config
        hidden_size = config.hidden_size # For mxbai-embed-large-v1, this is 1024
        
        # 2. Define the 5 Task-Specific Heads
        # We add a Dropout layer for regularization, a good practice.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Determine the intermediate size for task-specific heads
        intermediate_size = max(1, hidden_size // 2)

        # Head for Jigsaw (Multi-Label)
        self.head_jigsaw = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(intermediate_size, num_labels_jigsaw)
        )

        # Head for GoEmotions (Multi-Label)
        self.head_goemotions = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(intermediate_size, num_labels_goemotions)
        )

        # Head for Davidson (Multi-Class)
        self.head_davidson = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(intermediate_size, num_labels_davidson)
        )

        # Head for OLID (Multi-Class)
        self.head_olid = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(intermediate_size, num_labels_olid)
        )

        # Head for Rumour (Multi-Class)
        self.head_rumour = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(intermediate_size, num_labels_rumour)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Takes tokenized input and returns a dictionary of logits for each task.
        """
        # 1. Pass input through the shared trunk
        trunk_outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Get the [CLS] token embedding
        # This is the standard output for sentence-level classification
        # Shape: (batch_size, hidden_size)
        cls_embedding = trunk_outputs.last_hidden_state[:, 0]
        cls_embedding = self.dropout(cls_embedding)
        
        # 3. Pass the [CLS] embedding to each head
        logits_jigsaw = self.head_jigsaw(cls_embedding)
        logits_goemotions = self.head_goemotions(cls_embedding)
        logits_davidson = self.head_davidson(cls_embedding)
        logits_olid = self.head_olid(cls_embedding)
        logits_rumour = self.head_rumour(cls_embedding)
        
        # 4. Return all logits in a dictionary
        return {
            'jigsaw': logits_jigsaw,
            'goemotions': logits_goemotions,
            'davidson': logits_davidson,
            'olid': logits_olid,
            'rumour': logits_rumour
        }

    def get_optimizer_params(self, base_lr: float, head_lr: float):
        """
        Returns optimizer parameters with differential learning rates.
        
        If the trunk is frozen (requires_grad=False), its parameters
        are not added to the optimizer.
        """
        # Get all parameters from the 5 heads
        head_parameters = (
            list(self.head_jigsaw.parameters()) +
            list(self.head_goemotions.parameters()) +
            list(self.head_davidson.parameters()) +
            list(self.head_olid.parameters()) +
            list(self.head_rumour.parameters())
        )
        
        # Set up the parameter groups
        optimizer_params = [
            # Group 1: Head parameters (always trained)
            {'params': head_parameters, 'lr': head_lr}
        ]
        
        # --- NEW: Only add trunk if it's not frozen ---
        # We check if the parameters actually require gradients.
        if self.trunk.parameters().__next__().requires_grad:
            optimizer_params.append(
                # Group 2: Trunk parameters (conditionally trained)
                {'params': self.trunk.parameters(), 'lr': base_lr}
            )
        # ----------------------------------------------
        
        return optimizer_params