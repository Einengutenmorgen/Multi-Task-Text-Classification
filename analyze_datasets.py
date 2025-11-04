# analyze_datasets.py
#
# A script to perform a deep analysis of the 5 datasets used in the multi-task project.
# It reuses the exact file paths and constants from data_loading.py.

import pandas as pd
import numpy as np
import os
import glob
import json
from transformers import AutoTokenizer
from collections import Counter
import warnings

# --- 1. Reuse Constants from data_loading.py ---

# Paths (Update these if they are different on your system)
JIGSAW_PATH = '/Users/christophhau/Desktop/tweet_classifier/jigsaw/train.csv'
GOEMOTIONS_PATH_PATTERN = '/Users/christophhau/Desktop/tweet_classifier/goEmotions/goemotions_*.csv'
DAVIDSON_PATH = '/Users/christophhau/Desktop/tweet_classifier/davidson/data/labeled_data.csv'
OLID_TWEETS_PATH = '/Users/christophhau/Desktop/tweet_classifier/SOLID/semeval_test/test_a_tweets.tsv'
OLID_LABELS_PATH = '/Users/christophhau/Desktop/tweet_classifier/SOLID/semeval_test/test_a_labels.csv'
RUMOUR_TRAIN_PATH = '/Users/christophhau/Desktop/tweet_classifier/semEval_task7/rumoureval-2019-training-data/'

# Column and Label definitions
JIGSAW_TEXT_COL = 'comment_text'
JIGSAW_LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

GOEMOTIONS_TEXT_COL = 'text'
GOEMOTIONS_LABEL_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", 
    "confusion", "curiosity", "desire", "disappointment", "disapproval", 
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", 
    "joy", "love", "nervousness", "optimism", "pride", "realization", 
    "relief", "remorse", "sadness", "surprise", "neutral" ]

DAVIDSON_TEXT_COL = 'tweet'
DAVIDSON_LABEL_COL = 'class'
DAVIDSON_LABEL_MAP = {0: 'hate', 1: 'offensive', 2: 'neither'}

OLID_TEXT_COL = 'tweet'
OLID_LABEL_COL = 'label'
OLID_LABEL_MAP = {'NOT': 0, 'OFF': 1}

RUMOUR_LABEL_MAP = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
RUMOUR_TWITTER_TEXT_KEY = 'text'
RUMOUR_REDDIT_REPLY_KEY = 'body'
RUMOUR_REDDIT_SOURCE_TITLE_KEY = 'title'
RUMOUR_REDDIT_SOURCE_SELFTEXT_KEY = 'selftext'

# Model name from train.py
TOKENIZER_NAME = "mixedbread-ai/mxbai-embed-large-v1"


# --- 2. Helper Functions ---

def analyze_token_lengths(texts, tokenizer):
    """Tokenizes a list of texts and returns length statistics."""
    if not texts:
        return {'count': 0}
        
    # Tokenize without padding or truncation to get real lengths
    # We suppress warnings about long sequences
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        token_lengths = [len(tokenizer.encode(text)) for text in texts]
    
    return {
        'count': len(token_lengths),
        'min': np.min(token_lengths),
        'max': np.max(token_lengths),
        'mean': np.mean(token_lengths),
        'median': np.median(token_lengths),
        '25th_percentile': np.percentile(token_lengths, 25),
        '75th_percentile': np.percentile(token_lengths, 75),
        '95th_percentile': np.percentile(token_lengths, 95),
    }

def print_stats(title, stats_dict):
    """Pretty-prints the statistics."""
    print(f"\n  {title}:")
    if stats_dict.get('count', 0) == 0:
        print("    (No data to analyze)")
        return
        
    print(f"    Total Samples:    {stats_dict['count']}")
    print(f"    Min Length:       {stats_dict['min']}")
    print(f"    Max Length:       {stats_dict['max']}")
    print(f"    Mean Length:      {stats_dict['mean']:.2f}")
    print(f"    Median Length:    {stats_dict['median']:.0f}")
    print(f"    Percentiles:      (25th: {stats_dict['25th_percentile']:.0f}) (75th: {stats_dict['75th_percentile']:.0f}) (95th: {stats_dict['95th_percentile']:.0f})")


# --- 3. Analysis Functions per Dataset ---

def analyze_jigsaw(tokenizer):
    print("\n" + "="*70)
    print("Analyzing Jigsaw Dataset")
    print("="*70)
    try:
        df = pd.read_csv(JIGSAW_PATH)
        print(f"Total raw rows: {len(df)}")
        
        # Check for missing data
        missing_text = df[JIGSAW_TEXT_COL].isna().sum()
        missing_labels = df[JIGSAW_LABEL_COLS].isna().any(axis=1).sum()
        print(f"Rows with missing text: {missing_text}")
        print(f"Rows with missing labels: {missing_labels}")
        
        # Clean data (replicating data_loading.py)
        df_clean = df.dropna(subset=[JIGSAW_TEXT_COL] + JIGSAW_LABEL_COLS)
        print(f"Total clean rows: {len(df_clean)}")

        # Analyze token lengths
        texts = df_clean[JIGSAW_TEXT_COL].astype(str).tolist()
        stats = analyze_token_lengths(texts, tokenizer)
        print_stats("Token Length Statistics", stats)
        
        # Analyze label distribution
        print("\n  Label Distribution (Total Occurrences):")
        label_counts = df_clean[JIGSAW_LABEL_COLS].sum().sort_values(ascending=False)
        print(label_counts.to_string())
        
        # Check for rows with NO labels
        no_label_count = (df_clean[JIGSAW_LABEL_COLS].sum(axis=1) == 0).sum()
        print(f"\n  Rows with no toxic labels ('clean'): {no_label_count} ({no_label_count / len(df_clean):.2%})")

        return len(df_clean)

    except Exception as e:
        print(f"ERROR: Could not analyze Jigsaw. {e}")
        return 0

def analyze_goemotions(tokenizer):
    print("\n" + "="*70)
    print("Analyzing GoEmotions Dataset")
    print("="*70)
    try:
        files = glob.glob(GOEMOTIONS_PATH_PATTERN)
        if not files:
            print("ERROR: No GoEmotions files found.")
            return 0
            
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        print(f"Total raw rows: {len(df)}")

        # Check for missing data
        missing_text = df[GOEMOTIONS_TEXT_COL].isna().sum()
        missing_labels = df[GOEMOTIONS_LABEL_COLS].isna().any(axis=1).sum()
        print(f"Rows with missing text: {missing_text}")
        print(f"Rows with missing labels: {missing_labels}")
        
        # Clean data
        df_clean = df.dropna(subset=[GOEMOTIONS_TEXT_COL] + GOEMOTIONS_LABEL_COLS)
        print(f"Total clean rows: {len(df_clean)}")

        # Analyze token lengths
        texts = df_clean[GOEMOTIONS_TEXT_COL].astype(str).tolist()
        stats = analyze_token_lengths(texts, tokenizer)
        print_stats("Token Length Statistics", stats)

        # Analyze label distribution
        print("\n  Label Distribution (Total Occurrences):")
        label_counts = df_clean[GOEMOTIONS_LABEL_COLS].sum().sort_values(ascending=False)
        print(label_counts.to_string())
        
        # Check for rows with NO labels
        no_label_count = (df_clean[GOEMOTIONS_LABEL_COLS].sum(axis=1) == 0).sum()
        print(f"\n  Rows with no emotion labels: {no_label_count} ({no_label_count / len(df_clean):.2%})")
        print(f"  Avg labels per sample: {df_clean[GOEMOTIONS_LABEL_COLS].sum(axis=1).mean():.2f}")

        return len(df_clean)

    except Exception as e:
        print(f"ERROR: Could not analyze GoEmotions. {e}")
        return 0

def analyze_davidson(tokenizer):
    print("\n" + "="*70)
    print("Analyzing Davidson Dataset")
    print("="*70)
    try:
        df = pd.read_csv(DAVIDSON_PATH)
        print(f"Total raw rows: {len(df)}")
        
        # Check for missing data
        missing_text = df[DAVIDSON_TEXT_COL].isna().sum()
        missing_labels = df[DAVIDSON_LABEL_COL].isna().sum()
        print(f"Rows with missing text: {missing_text}")
        print(f"Rows with missing labels: {missing_labels}")
        
        # Clean data
        df_clean = df.dropna(subset=[DAVIDSON_TEXT_COL, DAVIDSON_LABEL_COL])
        print(f"Total clean rows: {len(df_clean)}")
        
        # Analyze token lengths
        texts = df_clean[DAVIDSON_TEXT_COL].astype(str).tolist()
        stats = analyze_token_lengths(texts, tokenizer)
        print_stats("Token Length Statistics", stats)
        
        # Analyze label distribution
        print("\n  Class Distribution:")
        df_clean['label_name'] = df_clean[DAVIDSON_LABEL_COL].map(DAVIDSON_LABEL_MAP)
        label_counts = df_clean['label_name'].value_counts(normalize=True).sort_index()
        print(label_counts.to_string())
        
        return len(df_clean)

    except Exception as e:
        print(f"ERROR: Could not analyze Davidson. {e}")
        return 0

def analyze_olid(tokenizer):
    print("\n" + "="*70)
    print("Analyzing OLID Dataset")
    print("="*70)
    try:
        df_tweets = pd.read_csv(OLID_TWEETS_PATH, sep='\t', quoting=3)
        df_labels = pd.read_csv(OLID_LABELS_PATH, header=None, names=['id_norm', 'label'])
        
        df_tweets.rename(columns={'id': 'id_norm'}, inplace=True)
        df_raw = pd.merge(df_tweets, df_labels, on='id_norm')
        print(f"Total raw merged rows: {len(df_raw)}")
        
        # Check for missing data
        missing_text = df_raw[OLID_TEXT_COL].isna().sum()
        missing_labels = df_raw[OLID_LABEL_COL].isna().sum()
        print(f"Rows with missing text: {missing_text}")
        print(f"Rows with missing labels: {missing_labels}")
        
        # Clean data
        df_clean = df_raw.dropna(subset=[OLID_TEXT_COL, OLID_LABEL_COL])
        print(f"Total clean rows: {len(df_clean)}")
        
        # Analyze token lengths
        texts = df_clean[OLID_TEXT_COL].astype(str).tolist()
        stats = analyze_token_lengths(texts, tokenizer)
        print_stats("Token Length Statistics", stats)
        
        # Analyze label distribution
        print("\n  Class Distribution:")
        label_counts = df_clean[OLID_LABEL_COL].value_counts(normalize=True).sort_index()
        print(label_counts.to_string())
        
        return len(df_clean)
        
    except Exception as e:
        print(f"ERROR: Could not analyze OLID. {e}")
        return 0

def analyze_rumour(tokenizer):
    print("\n" + "="*70)
    print("Analyzing RumourEval Dataset")
    print("="*70)
    # This logic is copied directly from data_loading.py
    try:
        texts_a = [] # Reply
        texts_b = [] # Source
        labels = []

        if not os.path.exists(RUMOUR_TRAIN_PATH):
            print(f"Error: RumourEval directory not found at {RUMOUR_TRAIN_PATH}")
            return 0

        # Load labels from -key.json files
        label_lookup = {}
        key_files = glob.glob(os.path.join(RUMOUR_TRAIN_PATH, '*-key.json'))
        for key_file in key_files:
            with open(key_file, 'r') as f:
                key_data = json.load(f)
            if 'subtaskaenglish' in key_data and isinstance(key_data['subtaskaenglish'], dict):
                label_lookup.update(key_data['subtaskaenglish'])
            elif 'subtaska' in key_data and isinstance(key_data['subtaska'], dict):
                label_lookup.update(key_data['subtaska'])
            elif all(isinstance(val, str) for val in key_data.values()) and key_data:
                 label_lookup.update(key_data)
        
        if not label_lookup:
            print("Error: Could not load any labels from *-key.json files.")
            return 0
            
        # Find all threads
        structure_files = sorted(glob.glob(os.path.join(RUMOUR_TRAIN_PATH, '**', 'structure.json'), recursive=True))
        
        for structure_file in structure_files:
            try:
                thread_dir = os.path.dirname(structure_file)
                is_reddit = 'reddit' in thread_dir.lower()
                
                with open(structure_file, 'r') as f:
                    structure = json.load(f)
                
                if not structure: continue
                
                if isinstance(structure, dict): source_id = list(structure.keys())[0]
                elif isinstance(structure, list): source_id = structure[0]
                else: continue
                
                source_text = ""
                source_file_pattern = 'source-tweet'
                source_file_glob = glob.glob(os.path.join(thread_dir, source_file_pattern, '*.json'))

                if source_file_glob:
                    with open(source_file_glob[0], 'r') as f:
                        source_data = json.load(f)
                    source_post = source_data.get('data', source_data) if is_reddit else source_data
                    if is_reddit:
                        source_text = source_post.get(RUMOUR_REDDIT_SOURCE_TITLE_KEY, "") + " " + source_post.get(RUMOUR_REDDIT_SOURCE_SELFTEXT_KEY, "")
                    else:
                        source_text = source_post.get(RUMOUR_TWITTER_TEXT_KEY, "")

                if not source_text: continue

                reply_files = sorted(glob.glob(os.path.join(thread_dir, 'replies', '*.json')))
                for reply_file in reply_files:
                    with open(reply_file, 'r') as f:
                        reply_data = json.load(f)
                    reply_post = reply_data.get('data', reply_data) if is_reddit else reply_data
                    reply_text = ""
                    if is_reddit:
                        reply_text = reply_post.get(RUMOUR_REDDIT_REPLY_KEY, "")
                    else:
                        reply_text = reply_post.get(RUMOUR_TWITTER_TEXT_KEY, "")
                        
                    reply_id = os.path.splitext(os.path.basename(reply_file))[0]
                    stance = label_lookup.get(reply_id)
                    
                    if reply_text and stance in RUMOUR_LABEL_MAP:
                        texts_a.append(reply_text.strip())
                        texts_b.append(source_text.strip())
                        labels.append(RUMOUR_LABEL_MAP[stance])
            except Exception:
                pass # Skip problematic threads
        
        print(f"Total clean rows (reply-source pairs): {len(labels)}")
        
        # Analyze token lengths
        stats_reply = analyze_token_lengths(texts_a, tokenizer)
        print_stats("Token Length Statistics (Reply Text)", stats_reply)
        
        stats_source = analyze_token_lengths(texts_b, tokenizer)
        print_stats("Token Length Statistics (Source Text)", stats_source)
        
        # Analyze combined length
        combined_texts = [a + " [SEP] " + b for a, b in zip(texts_a, texts_b)]
        stats_combined = analyze_token_lengths(combined_texts, tokenizer)
        print_stats("Token Length Statistics (Combined)", stats_combined)
        
        # Analyze label distribution
        print("\n  Class Distribution:")
        label_names = [list(RUMOUR_LABEL_MAP.keys())[list(RUMOUR_LABEL_MAP.values()).index(l)] for l in labels]
        counts = Counter(label_names)
        total = len(label_names)
        for label, count in sorted(counts.items()):
            print(f"    {label:<10}: {count} ({count/total:.2%})")

        return len(labels)
        
    except Exception as e:
        print(f"ERROR: Could not analyze RumourEval. {e}")
        return 0

# --- 4. Main execution ---

def main():
    print(f"Initializing tokenizer: {TOKENIZER_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    except Exception as e:
        print(f"FATAL: Could not load tokenizer. {e}")
        print("Please check TOKENIZER_NAME and internet connection.")
        return

    print("Starting dataset analysis...")
    
    counts = {}
    counts['jigsaw'] = analyze_jigsaw(tokenizer)
    counts['goemotions'] = analyze_goemotions(tokenizer)
    counts['davidson'] = analyze_davidson(tokenizer)
    counts['olid'] = analyze_olid(tokenizer)
    counts['rumour'] = analyze_rumour(tokenizer)
    
    print("\n" + "="*70)
    print("Final Clean Sample Counts")
    print("="*70)
    
    df_counts = pd.DataFrame.from_dict(counts, orient='index', columns=['SampleCount'])
    df_counts['Percentage'] = (df_counts['SampleCount'] / df_counts['SampleCount'].sum())
    
    print(df_counts.to_string(formatters={'Percentage': '{:.2%}'.format}))
    
    print(f"\nTotal samples: {df_counts['SampleCount'].sum()}")
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()