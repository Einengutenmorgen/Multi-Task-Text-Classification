# predict.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# --- Import Model and Data Schema ---
# We need the model definition and the label maps to build the model
# and make the output human-readable.

from model_v3 import MultiTaskModel
from data_loading import (
    JIGSAW_LABEL_COLS,
    GOEMOTIONS_LABEL_COLS,
    DAVIDSON_LABEL_MAP,
    OLID_LABEL_MAP,
    RUMOUR_LABEL_MAP
)

# --- Configuration (Must match train.py) ---
CONFIG = {
    "MODEL_NAME": "mixedbread-ai/mxbai-embed-large-v1",
    "MAX_LENGTH": 124,
    # --- IMPORTANT: Update checkpoint path ---
    "CHECKPOINT_PATH": "./checkpoints/v3/best_model.pth",
    # --- IMPORTANT: Set based on training log ---
    # Your log said "BASE_LR is None. Freezing trunk parameters."
    # So we set freeze_trunk=True.
    "FREEZE_TRUNK": True
}

# --- Create Inverse Label Maps ---
# These maps convert the model's output index back to a string
LABEL_MAPS = {
    'davidson': {v: k for k, v in DAVIDSON_LABEL_MAP.items()},
    'olid': {v: k for k, v in OLID_LABEL_MAP.items()},
    'rumour': {v: k for k, v in RUMOUR_LABEL_MAP.items()},
    'jigsaw': JIGSAW_LABEL_COLS,
    'goemotions': GOEMOTIONS_LABEL_COLS
}


def load_model_for_inference(checkpoint_path):
    """
    Initializes the MultiTaskModel, loads the saved weights, 
    and sets it to evaluation mode.
    """
    print(f"Loading model for inference from {checkpoint_path}...")
    
    # 1. Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize the model with the exact same architecture as training
    model = MultiTaskModel(
        model_name=CONFIG["MODEL_NAME"],
        num_labels_jigsaw=len(LABEL_MAPS['jigsaw']),
        num_labels_goemotions=len(LABEL_MAPS['goemotions']),
        num_labels_davidson=len(LABEL_MAPS['davidson']),
        num_labels_olid=len(LABEL_MAPS['olid']),
        num_labels_rumour=len(LABEL_MAPS['rumour']),
        freeze_trunk=CONFIG["FREEZE_TRUNK"]
    )
    
    # 3. Load the saved weights (state_dict)
    # We use map_location to ensure it loads correctly even if
    # you trained on a GPU and are now using a CPU.
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        print("Please ensure the path is correct.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure your model.py definition matches the saved checkpoint.")
        return None, None, None
        
    # 4. Set model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # 5. Move model to the device
    model.to(device)
    
    # 6. Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_NAME"])
    
    print("Model loaded successfully.")
    return model, tokenizer, device


def predict(model, tokenizer, device, text_a, text_b=None):
    """
    Runs a single prediction for a given text (or text pair).
    
    Args:
        model (MultiTaskModel): The loaded model.
        tokenizer (AutoTokenizer): The loaded tokenizer.
        device (torch.device): The device to run on.
        text_a (str): The primary text input (e.g., the tweet, comment, or reply).
        text_b (str, optional): The secondary text for pair tasks (e.g., the source tweet for Rumour).
    
    Returns:
        dict: A dictionary containing formatted predictions for all 5 tasks.
    """
    
    # 1. Tokenize the input(s)
    if text_b:
        # Sentence pair (for Rumour)
        tokenized_input = tokenizer(
            text_a, text_b, 
            truncation='only_first',
            max_length=CONFIG["MAX_LENGTH"], 
            padding='max_length',
            return_tensors="pt"
        )
    else:
        # Single sentence
        tokenized_input = tokenizer(
            text_a, 
            #truncation=True, 
            max_length=CONFIG["MAX_LENGTH"], 
            padding='max_length',
            return_tensors="pt"
        )

    # 2. Move inputs to the correct device
    # --- FIX: Only select the keys our model's forward() method expects ---
    # This prevents the 'token_type_ids' argument from being passed.
    inputs = {
        'input_ids': tokenized_input['input_ids'].to(device),
        'attention_mask': tokenized_input['attention_mask'].to(device)
    }
    
    # 3. Get model predictions (logits)
    with torch.no_grad():
        logits = model(**inputs)

    # 4. Format the output
    predictions = {}
    
    # --- Format Multi-Class Tasks (Davidson, OLID, Rumour) ---
    for task in ['davidson', 'olid', 'rumour']:
        task_logits = logits[task]
        task_probs = F.softmax(task_logits, dim=1).squeeze()
        pred_index = torch.argmax(task_probs).item()
        pred_label = LABEL_MAPS[task][pred_index]
        
        predictions[task] = {
            "prediction": pred_label,
            "confidence": task_probs[pred_index].item(),
            "all_scores": {LABEL_MAPS[task][i]: prob.item() for i, prob in enumerate(task_probs)}
        }

    # --- Format Multi-Label Tasks (Jigsaw, GoEmotions) ---
    for task in ['jigsaw', 'goemotions']:
        task_logits = logits[task]
        task_scores = torch.sigmoid(task_logits).squeeze()
        
        # Get labels where score > 0.5
        pred_indices = (task_scores > 0.5).nonzero(as_tuple=True)[0]
        pred_labels = [LABEL_MAPS[task][i] for i in pred_indices]
        
        predictions[task] = {
            "predictions": pred_labels if pred_labels else ["None"],
            "all_scores": {LABEL_MAPS[task][i]: score.item() for i, score in enumerate(task_scores)}
        }
        
    return predictions

# --- Main execution block to demonstrate usage ---
if __name__ == "__main__":
    
    # 1. Load the model and tokenizer
    model, tokenizer, device = load_model_for_inference(CONFIG["CHECKPOINT_PATH"])
    
    if model:
        # 2. --- Example 1: Single text input (for 4/5 tasks) ---
        print("\n" + "="*70)
        print("Example 1: Single text (Jigsaw, GoEmotions, Davidson, OLID)")
        print("="*70)
        
        test_text_1 = "I can't believe you would say something so horrible. You are a terrible person."
        print(f"Input: \"{test_text_1}\"")
        
        preds_1 = predict(model, tokenizer, device, test_text_1)
        
        print("\n--- Predictions ---")
        print(f"Jigsaw:     {preds_1['jigsaw']['predictions']}")
        print(f"GoEmotions: {preds_1['goemotions']['predictions']}")
        print(f"Davidson:   {preds_1['davidson']['prediction']} (Confidence: {preds_1['davidson']['confidence']:.2%})")
        print(f"OLID:       {preds_1['olid']['prediction']} (Confidence: {preds_1['olid']['confidence']:.2%})")
        
        
        # 3. --- Example 2: Sentence pair input (for Rumour task) ---
        print("\n" + "="*70)
        print("Example 2: Sentence Pair (Rumour)")
        print("="*70)
        
        source_tweet = "BREAKING: The Eiffel Tower just collapsed!"
        reply_tweet = "No way, I'm looking at a live webcam right now and it's fine."
        
        print(f"Source Tweet: \"{source_tweet}\"")
        print(f"Reply Tweet:  \"{reply_tweet}\"")
        
        preds_2 = predict(model, tokenizer, device, reply_tweet, source_tweet)
        
        print("\n--- Predictions ---")
        # Note: The Rumour task predicts the *stance of the reply* relative to the source
        print(f"Rumour Stance: {preds_2['rumour']['prediction']} (Confidence: {preds_2['rumour']['confidence']:.2%})")
        
        # 4. --- Example 3: More nuanced text ---
        print("\n" + "="*70)
        print("Example 3: Nuanced emotional text")
        print("="*70)
        
        test_text_3 = "Wow, I finally finished my thesis after 2 years. I'm so relieved but also a little sad it's over."
        print(f"Input: \"{test_text_3}\"")
        
        preds_3 = predict(model, tokenizer, device, test_text_3)
        
        print("\n--- Predictions ---")
        print(f"Jigsaw:     {preds_3['jigsaw']['predictions']}")
        print(f"GoEmotions: {preds_3['goemotions']['predictions']}")
        print(f"Davidson:   {preds_3['davidson']['prediction']} (Confidence: {preds_3['davidson']['confidence']:.2%})")
        print(f"OLID:       {preds_3['olid']['prediction']} (Confidence: {preds_3['olid']['confidence']:.2%})")

