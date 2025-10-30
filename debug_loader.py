# debug_data_loading.py
# Quick diagnostic script to test OLID and RumourEval loading

import pandas as pd
import os
import glob
import json

print("="*70)
print("DIAGNOSING OLID DATASET")
print("="*70)

# Test OLID loading
OLID_TWEETS_PATH = '/Users/christophhau/Desktop/tweet_classifier/SOLID/semeval_test/test_a_tweets.tsv'
OLID_LABELS_PATH = '/Users/christophhau/Desktop/tweet_classifier/SOLID/semeval_test/test_a_labels.csv'

try:
    print(f"\n1. Loading tweets from: {OLID_TWEETS_PATH}")
    df_tweets = pd.read_csv(OLID_TWEETS_PATH, sep='\t', quoting=3)
    print(f"   ✓ Loaded {len(df_tweets)} tweets")
    print(f"   Columns: {df_tweets.columns.tolist()}")
    print(f"\n   First row:")
    print(df_tweets.head(1))
    
    print(f"\n2. Loading labels from: {OLID_LABELS_PATH}")
    df_labels = pd.read_csv(OLID_LABELS_PATH)
    print(f"   ✓ Loaded {len(df_labels)} labels")
    print(f"   Columns: {df_labels.columns.tolist()}")
    print(f"   Column names (with repr): {[repr(col) for col in df_labels.columns]}")
    print(f"\n   First row:")
    print(df_labels.head(1))
    
    print(f"\n3. Attempting merge...")
    # Standardize ID columns
    df_tweets.rename(columns={'id': 'id_norm'}, inplace=True)
    print(f"   Tweets columns after rename: {df_tweets.columns.tolist()}")
    
    # Find and rename the ID column in labels
    id_col = [col for col in df_labels.columns if 'ID' in col.strip().upper()]
    label_col = [col for col in df_labels.columns if 'LABEL' in col.strip().upper()]
    
    print(f"   Found ID column: {id_col}")
    print(f"   Found LABEL column: {label_col}")
    
    if id_col:
        df_labels.rename(columns={id_col[0]: 'id_norm'}, inplace=True)
    if label_col:
        df_labels.rename(columns={label_col[0]: 'label'}, inplace=True)
    
    print(f"   Labels columns after rename: {df_labels.columns.tolist()}")
    
    df = pd.merge(df_tweets, df_labels, on='id_norm')
    print(f"   ✓ Merge successful! {len(df)} samples")
    
    # Check for NaN values
    print(f"\n4. Checking for NaN values...")
    print(f"   NaN in 'tweet': {df['tweet'].isna().sum()}")
    print(f"   NaN in 'label': {df['label'].isna().sum()}")
    
    df_clean = df.dropna(subset=['tweet', 'label'])
    print(f"   After dropping NaN: {len(df_clean)} samples")
    
    # Check label mapping
    OLID_LABEL_MAP = {'NOT': 0, 'OFF': 1}
    print(f"\n5. Checking label values...")
    print(f"   Unique labels: {df_clean['label'].unique()}")
    df_clean['label_mapped'] = df_clean['label'].map(OLID_LABEL_MAP)
    print(f"   After mapping: {df_clean['label_mapped'].unique()}")
    print(f"   NaN after mapping: {df_clean['label_mapped'].isna().sum()}")
    
    if df_clean['label_mapped'].isna().sum() > 0:
        print(f"   ⚠️  WARNING: Some labels couldn't be mapped!")
        print(f"   Unmapped labels: {df_clean[df_clean['label_mapped'].isna()]['label'].unique()}")
    
    print(f"\n✓ OLID: Final count = {len(df_clean[~df_clean['label_mapped'].isna()])} samples")
    
except Exception as e:
    print(f"\n✗ ERROR loading OLID: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSING RUMOUREVAL DATASET")
print("="*70)

RUMOUR_TRAIN_PATH = '/Users/christophhau/Desktop/tweet_classifier/semEval_task7/rumoureval-2019-training-data/'

try:
    print(f"\n1. Checking directory: {RUMOUR_TRAIN_PATH}")
    if not os.path.exists(RUMOUR_TRAIN_PATH):
        print(f"   ✗ Directory does not exist!")
    else:
        print(f"   ✓ Directory exists")
        
        # List contents
        contents = os.listdir(RUMOUR_TRAIN_PATH)
        print(f"   Contents: {contents}")
        
        # Find structure.json files
        structure_files = sorted(glob.glob(
            os.path.join(RUMOUR_TRAIN_PATH, '**', 'structure.json'), 
            recursive=True
        ))
        print(f"\n2. Found {len(structure_files)} structure.json files")
        
        if len(structure_files) == 0:
            print(f"   ⚠️  No structure.json files found!")
            print(f"   Searching in subdirectories...")
            for root, dirs, files in os.walk(RUMOUR_TRAIN_PATH):
                print(f"   - {root}")
                if 'structure.json' in files:
                    print(f"     ✓ Has structure.json")
        else:
            print(f"   First 3 structure files:")
            for f in structure_files[:3]:
                print(f"   - {f}")
        
        # Try to load one thread
        if structure_files:
            print(f"\n3. Testing first thread: {structure_files[0]}")
            thread_dir = os.path.dirname(structure_files[0])
            is_reddit = 'reddit' in thread_dir.lower()
            print(f"   Platform: {'Reddit' if is_reddit else 'Twitter'}")
            
            with open(structure_files[0], 'r') as f:
                structure = json.load(f)
            
            print(f"   Structure type: {type(structure)}")
            print(f"   Structure content: {structure}")
            
            # Find source
            if isinstance(structure, dict):
                source_id = list(structure.keys())[0]
            elif isinstance(structure, list):
                source_id = structure[0]
            else:
                source_id = None
            
            print(f"   Source ID: {source_id}")
            
            # Look for source file
            source_pattern = 'source-post' if is_reddit else 'source-tweet'
            source_files = glob.glob(os.path.join(thread_dir, source_pattern, '*.json'))
            print(f"   Source files found: {len(source_files)}")
            if source_files:
                print(f"   First source: {source_files[0]}")
                with open(source_files[0], 'r') as f:
                    source = json.load(f)
                print(f"   Source keys: {source.keys()}")
            
            # Look for replies
            reply_files = glob.glob(os.path.join(thread_dir, 'replies', '*.json'))
            print(f"   Reply files found: {len(reply_files)}")
            if reply_files:
                print(f"   First reply: {reply_files[0]}")
                with open(reply_files[0], 'r') as f:
                    reply = json.load(f)
                print(f"   Reply keys: {reply.keys()}")
                if 'stance' in reply:
                    print(f"   Stance: {reply['stance']}")
        
        # Try the full loading logic
        print(f"\n4. Running full loading logic...")
        texts_a = []
        texts_b = []
        labels = []
        
        RUMOUR_LABEL_MAP = {'support': 0, 'deny': 1, 'query': 2, 'comment': 3}
        RUMOUR_TWITTER_TEXT_KEY = 'text'
        RUMOUR_REDDIT_REPLY_KEY = 'body'
        RUMOUR_REDDIT_SOURCE_TITLE_KEY = 'title'
        RUMOUR_REDDIT_SOURCE_SELFTEXT_KEY = 'selftext'
        
        for structure_file in structure_files:
            try:
                thread_dir = os.path.dirname(structure_file)
                is_reddit = 'reddit' in thread_dir.lower()
                
                with open(structure_file, 'r') as f:
                    structure = json.load(f)
                
                if not structure:
                    continue
                
                if isinstance(structure, dict):
                    source_id = list(structure.keys())[0]
                elif isinstance(structure, list):
                    source_id = structure[0]
                else:
                    continue
                
                source_text = ""
                source_file_pattern = 'source-post' if is_reddit else 'source-tweet'
                source_file_glob = glob.glob(os.path.join(thread_dir, source_file_pattern, f'{source_id}.json'))

                if source_file_glob:
                    with open(source_file_glob[0], 'r') as f:
                        source_post = json.load(f)
                    
                    if is_reddit:
                        source_text = source_post.get(RUMOUR_REDDIT_SOURCE_TITLE_KEY, "") + " " + source_post.get(RUMOUR_REDDIT_SOURCE_SELFTEXT_KEY, "")
                    else:
                        source_text = source_post.get(RUMOUR_TWITTER_TEXT_KEY, "")

                if not source_text:
                    continue

                reply_files = sorted(glob.glob(os.path.join(thread_dir, 'replies', '*.json')))
                for reply_file in reply_files:
                    with open(reply_file, 'r') as f:
                        reply_post = json.load(f)
                    
                    reply_text = ""
                    if is_reddit:
                        reply_text = reply_post.get(RUMOUR_REDDIT_REPLY_KEY, "")
                    else:
                        reply_text = reply_post.get(RUMOUR_TWITTER_TEXT_KEY, "")
                        
                    stance = reply_post.get('stance')
                    
                    if reply_text and stance in RUMOUR_LABEL_MAP:
                        texts_a.append(reply_text.strip())
                        texts_b.append(source_text.strip())
                        labels.append(RUMOUR_LABEL_MAP[stance])
                        
            except Exception as e:
                pass  # Skip problematic threads
        
        print(f"   ✓ Loaded {len(labels)} RumourEval samples")
        
        if len(labels) > 0:
            print(f"\n   Label distribution:")
            from collections import Counter
            label_counts = Counter(labels)
            for label_idx, count in sorted(label_counts.items()):
                label_name = [k for k, v in RUMOUR_LABEL_MAP.items() if v == label_idx][0]
                print(f"   - {label_name}: {count}")
        
except Exception as e:
    print(f"\n✗ ERROR loading RumourEval: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
