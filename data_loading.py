#data_loading.py

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import os
import glob
import json
import random
import zipfile

# --- Constants ---

# 1. Jigsaw
JIGSAW_PATH = '/Users/christophhau/Desktop/tweet_classifier/jigsaw/train.csv' # ASSUMES download_jigsaw.py creates this
JIGSAW_TEXT_COL = 'comment_text'
JIGSAW_LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 2. GoEmotions
GOEMOTIONS_PATH_PATTERN = '/Users/christophhau/Desktop/tweet_classifier/goEmotions/goemotions_*.csv'
GOEMOTIONS_TEXT_COL = 'text'
GOEMOTIONS_NON_LABEL_COLS = ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']
GOEMOTIONS_LABEL_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral" ]

# 3. Davidson
DAVIDSON_PATH = '/Users/christophhau/Desktop/tweet_classifier/davidson/data/labeled_data.csv'
DAVIDSON_TEXT_COL = 'tweet'
DAVIDSON_LABEL_COL = 'class'
DAVIDSON_LABEL_MAP = {0: 0, 1: 1, 2: 2} # 0:hate, 1:offensive, 2:neither

# 4. OLID (SOLID)
OLID_TWEETS_PATH = '/Users/christophhau/Desktop/tweet_classifier/SOLID/semeval_test/test_a_tweets.tsv'
OLID_LABELS_PATH = '/Users/christophhau/Desktop/tweet_classifier/SOLID/semeval_test/test_a_labels.csv'
OLID_TEXT_COL = 'tweet'
OLID_LABEL_COL = 'label' # From test_a_labels.csv
OLID_LABEL_MAP = {'NOT': 0, 'OFF': 1}

# 5. RumourEval
RUMOUR_TRAIN_PATH = '/Users/christophhau/Desktop/tweet_classifier/semEval_task7/rumoureval-2019-training-data/' # ASSUMES you unzip it
RUMOUR_LABEL_MAP = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
# Platform-specific text keys
RUMOUR_TWITTER_TEXT_KEY = 'text' # 'full_text' is often for premium APIs, 'text' is safer
RUMOUR_REDDIT_REPLY_KEY = 'body'
RUMOUR_REDDIT_SOURCE_TITLE_KEY = 'title'
RUMOUR_REDDIT_SOURCE_SELFTEXT_KEY = 'selftext'


# --- Schema Definition ---
# This MUST match the order in Canvas.md and the model heads
SCHEMA = {
    'jigsaw': JIGSAW_LABEL_COLS,
    'goemotions': GOEMOTIONS_LABEL_COLS,
    'davidson': list(DAVIDSON_LABEL_MAP.keys()),
    'olid': list(OLID_LABEL_MAP.keys()),
    'rumour': list(RUMOUR_LABEL_MAP.keys()),
}

# --- Main Dataset Class ---

class UnifiedDataset(Dataset):
    """
    Loads 5 distinct datasets for multi-task learning.
    
    `__getitem__` returns a "Master Sample" dictionary as defined in Canvas.md,
    using -100 as the ignore_index for non-applicable labels.
    """
    def __init__(self, tokenizer_name, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        self.ignore_value = -100
        self.task_data = {}
        self.task_indices = [] # Our master index

        # Load each dataset
        self._load_jigsaw()
        self._load_goemotions()
        self._load_davidson()
        self._load_olid()
        self._load_rumoureval()

        # Create the master index
        # self.task_indices will be a list of tuples: [('jigsaw', 0), ('jigsaw', 1), ..., ('goemotions', 0), ...]
        for task_name, data in self.task_data.items():
            for i in range(len(data['texts'])):
                self.task_indices.append((task_name, i))
        
        print(f"\n--- Total Samples Loaded: {len(self)} ---")
        for task, data in self.task_data.items():
            print(f"  - {task}: {len(data['texts'])} samples")

    def __len__(self):
        return len(self.task_indices)

    def _load_jigsaw(self):
        print(f"Loading Jigsaw from {JIGSAW_PATH}...")
        try:
            df = pd.read_csv(JIGSAW_PATH)
            df = df.dropna(subset=[JIGSAW_TEXT_COL] + JIGSAW_LABEL_COLS) # Drop bad rows
            self.task_data['jigsaw'] = {
                'texts': df[JIGSAW_TEXT_COL].astype(str).tolist(),
                'labels': df[JIGSAW_LABEL_COLS].values.tolist()
            }
        except FileNotFoundError:
            print(f"Error: Jigsaw file not found at {JIGSAW_PATH}")
            print("Please run the download_jigsaw.py script first.")
            self.task_data['jigsaw'] = {'texts': [], 'labels': []}

    def _load_goemotions(self):
        print(f"Loading GoEmotions from {GOEMOTIONS_PATH_PATTERN}...")
        files = glob.glob(GOEMOTIONS_PATH_PATTERN)
        if not files:
            print("Error: No GoEmotions files found.")
            self.task_data['goemotions'] = {'texts': [], 'labels': []}
            return
            
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        
        

        df = df.dropna(subset=[GOEMOTIONS_TEXT_COL] + SCHEMA['goemotions'])
        self.task_data['goemotions'] = {
            'texts': df[GOEMOTIONS_TEXT_COL].astype(str).tolist(),
            'labels': df[SCHEMA['goemotions']].values.tolist()
        }

    def _load_davidson(self):
        print(f"Loading Davidson from {DAVIDSON_PATH}...")
        try:
            df = pd.read_csv(DAVIDSON_PATH)
            df = df.dropna(subset=[DAVIDSON_TEXT_COL, DAVIDSON_LABEL_COL])
            self.task_data['davidson'] = {
                'texts': df[DAVIDSON_TEXT_COL].astype(str).tolist(),
                'labels': df[DAVIDSON_LABEL_COL].map(DAVIDSON_LABEL_MAP).tolist()
            }
        except FileNotFoundError:
            print(f"Error: Davidson file not found at {DAVIDSON_PATH}")
            self.task_data['davidson'] = {'texts': [], 'labels': []}

    def _load_olid(self):
        print(f"Loading OLID from {OLID_TWEETS_PATH} and {OLID_LABELS_PATH}...")
        try:
            df_tweets = pd.read_csv(OLID_TWEETS_PATH, sep='\t', quoting=3) # Use quoting=3 for QUOTE_NONE
            
            # --- FIX: Load with header=None and explicit names ---
            df_labels = pd.read_csv(OLID_LABELS_PATH, header=None, names=['id_norm', 'label'])
            # -----------------------------------------------------
            
            # Standardize ID columns for merging
            df_tweets.rename(columns={'id': 'id_norm'}, inplace=True)
            
            # --- REMOVED: Old renaming logic is no longer needed ---
            # df_labels.rename(columns={col: 'id_norm' for col in df_labels.columns if col.strip() == 'ID'}, inplace=True)
            # df_labels.rename(columns={col: 'label' for col in df_labels.columns if col.strip() == 'LABEL'}, inplace=True)
            # -----------------------------------------------------
            
            df = pd.merge(df_tweets, df_labels, on='id_norm')
            
            df = df.dropna(subset=[OLID_TEXT_COL, OLID_LABEL_COL])
            self.task_data['olid'] = {
                'texts': df[OLID_TEXT_COL].astype(str).tolist(),
                'labels': df[OLID_LABEL_COL].map(OLID_LABEL_MAP).tolist()
            }
        except FileNotFoundError:
            print(f"Error: OLID files not found at {OLID_TWEETS_PATH} or {OLID_LABELS_PATH}")
            self.task_data['olid'] = {'texts': [], 'labels': []}
        except Exception as e:
            print(f"Error loading OLID: {e}")
            self.task_data['olid'] = {'texts': [], 'labels': []}

    def _load_rumoureval(self):
        print(f"Loading RumourEval from {RUMOUR_TRAIN_PATH}...")
        texts_a = []
        texts_b = []
        labels = []

        if not os.path.exists(RUMOUR_TRAIN_PATH):
            print(f"Error: RumourEval directory not found at {RUMOUR_TRAIN_PATH}")
            self.task_data['rumour'] = {'texts': [], 'texts_b': [], 'labels': []}
            return

        # --- NEW FIX: Load labels from the -key.json files first ---
        label_lookup = {}
        key_files = glob.glob(os.path.join(RUMOUR_TRAIN_PATH, '*-key.json'))
        
        if not key_files:
            print(f"Error: No *-key.json files (like train-key.json) found in {RUMOUR_TRAIN_PATH}")
            self.task_data['rumour'] = {'texts': [], 'texts_b': [], 'labels': []}
            return
            
        for key_file in key_files:
            try:
                with open(key_file, 'r') as f:
                    key_data = json.load(f)
                
                # Data is often nested, e.g., {'subtaskaenglish': {id: stance, ...}}
                # We'll check common keys or just take the first dictionary we find.
                if 'subtaskaenglish' in key_data and isinstance(key_data['subtaskaenglish'], dict):
                    label_lookup.update(key_data['subtaskaenglish'])
                    print(f"  ... Loaded {len(key_data['subtaskaenglish'])} labels from {key_file} (subtaskaenglish)")
                elif 'subtaska' in key_data and isinstance(key_data['subtaska'], dict):
                    label_lookup.update(key_data['subtaska'])
                    print(f"  ... Loaded {len(key_data['subtaska'])} labels from {key_file} (subtaska)")
                # Fallback for flat {id: stance} structure
                elif all(isinstance(val, str) for val in key_data.values()) and key_data:
                     label_lookup.update(key_data)
                     print(f"  ... Loaded {len(key_data)} labels from {key_file} (root)")
                else:
                    print(f"  ... WARNING: Could not auto-detect stance data in {key_file}. Format might be unknown.")

            except Exception as e:
                print(f"Warning: Failed to parse key file {key_file}. Error: {e}")
        
        if not label_lookup:
            print("Error: Could not load any labels from *-key.json files. Stopping RumourEval load.")
            self.task_data['rumour'] = {'texts': [], 'texts_b': [], 'labels': []}
            return
            
        print(f"Loaded a total of {len(label_lookup)} stance labels into lookup dictionary.")
        # --- END NEW FIX ---

        structure_files = sorted(glob.glob(os.path.join(RUMOUR_TRAIN_PATH, '**', 'structure.json'), recursive=True))
        
        # --- You can uncomment these lines for faster debugging ---
        # print(f"Found {len(structure_files)} total threads. Loading a subset of 50...")
        # structure_files = structure_files[:50]
        # --------------------------------------------------------
        
        for structure_file in structure_files:
            try:
                thread_dir = os.path.dirname(structure_file)
                is_reddit = 'reddit' in thread_dir.lower()
                
                with open(structure_file, 'r') as f:
                    structure = json.load(f)
                
                if not structure: continue
                
                if isinstance(structure, dict):
                    source_id = list(structure.keys())[0]
                elif isinstance(structure, list):
                    source_id = structure[0]
                else:
                    continue
                
                source_text = ""
                # Use 'source-tweet' for all, as confirmed by your file tree
                source_file_pattern = 'source-tweet'
                source_file_glob = glob.glob(os.path.join(thread_dir, source_file_pattern, '*.json'))

                if source_file_glob:
                    with open(source_file_glob[0], 'r') as f:
                        source_data = json.load(f)
                    
                    # Handle Reddit's '{ "data": { ... } }' structure
                    source_post = source_data.get('data', source_data) if is_reddit else source_data

                    if is_reddit:
                        source_text = source_post.get(RUMOUR_REDDIT_SOURCE_TITLE_KEY, "") + " " + source_post.get(RUMOUR_REDDIT_SOURCE_SELFTEXT_KEY, "")
                    else:
                        source_text = source_post.get(RUMOUR_TWITTER_TEXT_KEY, "")

                if not source_text:
                    continue

                reply_files = sorted(glob.glob(os.path.join(thread_dir, 'replies', '*.json')))
                for reply_file in reply_files:
                    with open(reply_file, 'r') as f:
                        reply_data = json.load(f)
                    
                    # Handle Reddit's '{ "data": { ... } }' structure
                    reply_post = reply_data.get('data', reply_data) if is_reddit else reply_data

                    reply_text = ""
                    if is_reddit:
                        reply_text = reply_post.get(RUMOUR_REDDIT_REPLY_KEY, "")
                    else:
                        reply_text = reply_post.get(RUMOUR_TWITTER_TEXT_KEY, "")
                        
                    # --- FINAL FIX: Get stance from lookup, not from reply file ---
                    reply_id = os.path.splitext(os.path.basename(reply_file))[0]
                    stance = label_lookup.get(reply_id)
                    # -------------------------------------------------------------
                    
                    if reply_text and stance in RUMOUR_LABEL_MAP:
                        texts_a.append(reply_text.strip())
                        texts_b.append(source_text.strip())
                        labels.append(RUMOUR_LABEL_MAP[stance])
                        
            except Exception as e:
                print(f"Warning: Failed to parse thread {structure_file}. Error: {e}")

        self.task_data['rumour'] = {
            'texts': texts_a,
            'texts_b': texts_b,
            'labels': labels
        }
        print(f"Loaded {len(labels)} RumourEval stance samples.")

    def __getitem__(self, idx):
        # 1. Get the task and item index from our master list
        task_name, item_index = self.task_indices[idx]
        
        # 2. Get the text(s) and label(s) for this item
        item = self.task_data[task_name]
        text_a = item['texts'][item_index]
        text_b = item.get('texts_b', [None]*len(item['texts']))[item_index] # For RumourEval
        
        # 3. Tokenize the text
        if text_b:
            tokenized_input = self.tokenizer(
                text_a, text_b, 
                truncation='only_first', # Truncate reply first
                max_length=self.max_length, 
                padding='max_length',
                return_tensors="pt" # Return PyTorch tensors
            )
        else:
            tokenized_input = self.tokenizer(
                text_a, 
                truncation=True, 
                max_length=self.max_length, 
                padding='max_length',
                return_tensors="pt" # Return PyTorch tensors
            )

        # 4. Build the master label dictionary
        
        # --- Multi-Label (BCE) tasks ---
        # Jigsaw
        if task_name == 'jigsaw':
            labels_jigsaw = torch.tensor(item['labels'][item_index], dtype=torch.float)
        else:
            labels_jigsaw = torch.full((len(SCHEMA['jigsaw']),), self.ignore_value, dtype=torch.float)

        # GoEmotions
        if task_name == 'goemotions':
            labels_goemotions = torch.tensor(item['labels'][item_index], dtype=torch.float)
        else:
            labels_goemotions = torch.full((len(SCHEMA['goemotions']),), self.ignore_value, dtype=torch.float)

        # --- Multi-Class (CE) tasks ---
        # Davidson
        if task_name == 'davidson':
            labels_davidson = torch.tensor(item['labels'][item_index], dtype=torch.long)
        else:
            labels_davidson = torch.tensor(self.ignore_value, dtype=torch.long)
            
        # OLID
        if task_name == 'olid':
            labels_olid = torch.tensor(item['labels'][item_index], dtype=torch.long)
        else:
            labels_olid = torch.tensor(self.ignore_value, dtype=torch.long)
            
        # Rumour
        if task_name == 'rumour':
            labels_rumour = torch.tensor(item['labels'][item_index], dtype=torch.long)
        else:
            labels_rumour = torch.tensor(self.ignore_value, dtype=torch.long)

        # Squeeze tensors from tokenizer output
        return {
            'input_ids': tokenized_input['input_ids'].squeeze(0), # Remove batch dim
            'attention_mask': tokenized_input['attention_mask'].squeeze(0), # Remove batch dim
            'labels_jigsaw': labels_jigsaw,
            'labels_goemotions': labels_goemotions,
            'labels_davidson': labels_davidson,
            'labels_olid': labels_olid,
            'labels_rumour': labels_rumour
        }

# --- Task-Based Sampler  ---

class TaskSampler(Sampler):
    """
    Samples from all tasks to create a balanced "epoch".
    Oversamples small datasets and undersamples large ones by
    sampling with replacement from each task up to the size of the
    largest task.
    
    This sampler is "Subset-aware". It can be initialized with:
    1. A full UnifiedDataset (used in testing)
    2. A torch.utils.data.Subset (used in train.py)
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
        # 1. Get task names. This works for both Subset (via patching)
        #    and the Full Dataset.
        self.task_indices = {task_name: [] for task_name in dataset.task_data.keys()}
        
        # 2. Check if we're dealing with a Subset or the full dataset
        is_subset = hasattr(dataset, 'indices')
        
        if is_subset:
            # --- CASE 2: We are given a SUBSET (from train.py) ---
            # Use the attributes we patched in train.py
            full_indices_list = dataset.full_task_indices_list 
            
            for i in range(len(dataset)):
                # map local subset index 'i' to global index
                global_idx = dataset.indices[i] 
                # get task name from global index
                task_name, _ = full_indices_list[global_idx]
                
                if task_name in self.task_indices:
                    self.task_indices[task_name].append(i) # Store LOCAL index 'i'
        else:
            # --- CASE 1: We are given the FULL DATASET (from pytest) ---
            # Use the dataset's own attributes
            full_indices_list = dataset.task_indices
            
            for i in range(len(dataset)):
                # local index 'i' is the same as global index
                global_idx = i 
                task_name, _ = full_indices_list[global_idx]
                
                if task_name in self.task_indices:
                    self.task_indices[task_name].append(i) # Store index 'i'
                    
        # 3. Filter out any tasks that didn't end up in this split
        self.task_indices = {k: v for k, v in self.task_indices.items() if len(v) > 0}
        
        # 4. (FIX) Define self.task_names (was missing, needed for __len__ and __iter__)
        self.task_names = list(self.task_indices.keys())

        # 5. Find the size of the *largest* dataset
        self.max_task_size = 0
        if self.task_indices: # This line will no longer fail
            self.max_task_size = max(len(indices) for indices in self.task_indices.values())
        
        # 6. Set the epoch size
        self.epoch_size = self.max_task_size * len(self.task_names)
        
    def __iter__(self):
        all_indices = []
        if self.max_task_size == 0:
            return iter(all_indices) # Return empty iterator if no data loaded

        for task_name in self.task_names:
            # Oversample smaller tasks up to the size of the largest task
            indices = self.task_indices[task_name]
            oversampled_indices = random.choices(indices, k=self.max_task_size)
            all_indices.extend(oversampled_indices)
        
        # Shuffle all the indices together
        random.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        # The length of one "epoch" is all tasks oversampled to the max size
        return self.epoch_size
# --- Example Usage (How to test this file) ---
if __name__ == "__main__":
    
    print("--- Initializing Tokenizer and Dataset ---")
    
    # 1. Define your tokenizer
    tokenizer_name = 'distilbert-base-uncased'
    
    # 2. Create the dataset
    #    (This will print loading status and errors)
    dataset = UnifiedDataset(tokenizer_name=tokenizer_name, max_length=128)
        
    # 3. Create the Task-Balancing Sampler
    #    This is for training. For validation, use a standard SequentialSampler.
    batch_size = 8
    sampler = TaskSampler(dataset)
    
    # 4. Create the DataLoader
    from torch.utils.data import DataLoader
    # Use num_workers > 0 for faster loading
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0) # Set num_workers=0 for simple debugging
    
    # 5. Check a batch
    print("\n--- Testing DataLoader ---")
    try:
        # The DataLoader now yields a collated batch dictionary
        batch = next(iter(dataloader))
        
        print(f"Batch contains {batch['input_ids'].shape[0]} items (from DataLoader)")
        
        print("\n--- Structure of Collated Batch ---")
        print({k: v.shape for k, v in batch.items()})
        
        print("\nDecoded Text (Sample 0 from Batch):")
        print(dataset.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        
        print("\nJigsaw Labels (Sample 0 from Batch):")
        print(batch['labels_jigsaw'][0])
        
        print("\nDavidson Label (Sample 0 from Batch):")
        print(batch['labels_davidson'][0])
        
        print("\nOLID Label (Sample 0 from Batch):")
        print(batch['labels_olid'][0])

    except StopIteration:
        print("\nError: DataLoader is empty. This likely means one or more datasets failed to load.")
        print("Please check the error messages above.")
    except Exception as e:
        print(f"\nAn error occurred while testing the loader: {e}")
        import traceback
        traceback.print_exc()

