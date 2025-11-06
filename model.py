# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# --- NEW: Import PEFT (LoRA) ---
from peft import get_peft_model, LoraConfig, TaskType

class MultiTaskModel(nn.Module):
    """
    A Multi-Task model with a shared trunk and separate heads for each task.
    
    --- MODIFIED: Now integrates PEFT (LoRA) ---
    If freeze_trunk=True, it automatically wraps the trunk with LoRA layers
    for efficient fine-tuning.
    """
    def __init__(self, 
                 model_name: str, 
                 num_labels_jigsaw: int,
                 num_labels_goemotions: int,
                 num_labels_davidson: int,
                 num_labels_olid: int,
                 num_labels_rumour: int,
                 freeze_trunk: bool = True):
        
        super().__init__()
        
        # 1. Load the Shared Trunk Config
        config = AutoConfig.from_pretrained(model_name)
        self.trunk = AutoModel.from_pretrained(model_name)
        
        # --- MODIFIED: PEFT (LoRA) Integration ---
        if freeze_trunk:
            # If trunk is "frozen", we apply LoRA for efficient PEFT
            print("Trunk is frozen. Applying LoRA adapters for PEFT...")
            
            # Freeze all original trunk parameters
            for param in self.trunk.parameters():
                param.requires_grad = False

            # Define LoRA configuration
            # We target 'query', 'key', 'value' layers in the attention blocks
            peft_config = LoraConfig(
                r=16, # Rank (a key LoRA hyperparameter)
                lora_alpha=32, # Alpha
                target_modules=["query", "key", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION # Use this for encoder-only models
            )
            
            # Wrap the trunk with LoRA
            self.trunk = get_peft_model(self.trunk, peft_config)
            
            print("LoRA applied. Trainable PEFT parameters:")
            self.trunk.print_trainable_parameters()
            
        else:
            # This is for full fine-tuning (the slow v4 way)
            print("Trunk is unfrozen. Performing full fine-tuning.")
        # -------------------------------------------
        
        
        # Get the output dimension from the trunk's config
        hidden_size = config.hidden_size
        
        # 2. Define the 5 Task-Specific Heads
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.head_jigsaw = nn.Linear(hidden_size, num_labels_jigsaw)
        self.head_goemotions = nn.Linear(hidden_size, num_labels_goemotions)
        self.head_davidson = nn.Linear(hidden_size, num_labels_davidson)
        self.head_olid = nn.Linear(hidden_size, num_labels_olid)
        self.head_rumour = nn.Linear(hidden_size, num_labels_rumour)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        """
        # 1. Pass input through the shared trunk
        # This now passes through LoRA layers transparently
        trunk_outputs = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 2. Get the [CLS] token embedding
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
        Returns optimizer parameters.
        
        If the trunk is LoRA-enabled (freeze_trunk=True), this function
        will ONLY return the LoRA parameters and the head parameters.
        
        If the trunk is fully unfrozen (freeze_trunk=False), this
        returns the trunk and head parameters as separate groups.
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
        
        # --- MODIFIED: Handle LoRA vs. Full Fine-Tuning ---
        
        # We check if the *first parameter* of the trunk requires grad.
        # If we used LoRA, this will be False.
        # If we unCfrozen fully, this will be True.
        
        if not self.trunk.parameters().__next__().requires_grad:
            # CASE 1: Trunk is frozen (LoRA or fully frozen)
            # We explicitly find the LoRA parameters (if they exist)
            # PEFT model's .parameters() only returns trainable ones.
            # But we'll be explicit to be safe.
            lora_params = [
                p for n, p in self.trunk.named_parameters() if p.requires_grad
            ]
            if lora_params:
                optimizer_params.append(
                    {'params': lora_params, 'lr': base_lr} # Use base_lr for LoRA
                )
            # If no lora_params, we are in the original v3-style
            # "fully frozen" mode, and this group is just empty.
            
        else:
            # CASE 2: Trunk is fully unfrozen (v4)
            optimizer_params.append(
                {'params': self.trunk.parameters(), 'lr': base_lr}
            )
        # ----------------------------------------------
        
        return optimizer_params