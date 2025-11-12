"""
PRE-PROCESSING-SKRIPT (EINMALIG AUSFÜHREN)

Zweck:
1. Lädt alle 4 Quelldatensätze (Jigsaw, GoEmotions, Davidson, OLID).
2. Mappt die Original-Labels auf die 8 "Superlabel"-Kategorien (v2-Namen).
3. Erstellt EINEN einzigen Label-Vektor ("labels"), der -100.0 für
   irrelevante/maskierte Labels verwendet (wie vom Trainer benötigt).
4. Tokenisiert alle Texte auf die maximale Länge des Modells.
5. Speichert das fertige, tokenisierte Dataset im Arrow-Format
   unter "./processed_superlabel_dataset".
"""

import torch
import pandas as pd
import numpy as np
import os
import glob
import json
from transformers import AutoTokenizer
from datasets import Dataset
import warnings

# --- 1. Konstanten (Pfade) ---
# (Passe diese Pfade an deine lokale Struktur an)
BASE_PATH = '/Users/christophhau/Desktop/tweet_classifier/'
JIGSAW_PATH = os.path.join(BASE_PATH, 'data/jigsaw/train.csv')
GOEMOTIONS_PATH_PATTERN = os.path.join(BASE_PATH, 'data/goEmotions/goemotions_*.csv')
DAVIDSON_PATH = os.path.join(BASE_PATH, 'data/davidson/data/labeled_data.csv')
OLID_TWEETS_PATH = os.path.join(BASE_PATH, 'data/SOLID/semeval_test/test_a_tweets.tsv')
OLID_LABELS_PATH = os.path.join(BASE_PATH, 'data/SOLID/semeval_test/test_a_labels.csv')

# --- 2. Modell- und Ziel-Konstanten ---
MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
# WICHTIG: Die Modellkarte bestätigt 512 als max. Sequenzlänge
MAX_LENGTH = 512
OUTPUT_DIR = "./processed_superlabel_dataset"
IGNORE_VALUE = -100.0 # Standard-Ignore-Wert für Labels

# --- 3. Superlabel-Definition (v2) ---
SUPERLABEL_SCHEMA = [
    'EXTREME_HATE_OR_THREAT', # 0
    'OFFENSIVE_TOXIC',        # 1
    'POSITIVE_HIGH_AROUSAL',  # 2
    'POSITIVE_LOW_AROUSAL',   # 3
    'NEGATIVE_SADNESS',       # 4
    'ANXIETY_FEAR',           # 5
    'COGNITIVE_REACTION',     # 6
    'NEUTRAL_FACTUAL'         # 7
]
NUM_SUPERLABELS = len(SUPERLABEL_SCHEMA)

SUPERLABEL_MAPPING = {
    'jigsaw': {
        'identity_hate': 0, 'threat': 0, 'severe_toxic': 0,
        'toxic': 1, 'obscene': 1, 'insult': 1
    },
    'davidson': {
        0: 0, # class 0 (hate_speech)
        1: 1, # class 1 (offensive)
        2: 7  # class 2 (neither)
    },
    'olid': {
        'OFF': 1,
        'NOT': 7
    },
    'goemotions': {
        'anger': 1, 'disgust': 1, 'disapproval': 1, 'annoyance': 1,
        'excitement': 2, 'joy': 2, 'amusement': 2, 'admiration': 2, 'desire': 2,
        'gratitude': 3, 'love': 3, 'caring': 3, 'relief': 3, 'optimism': 3, 'pride': 3, 'approval': 3,
        'sadness': 4, 'grief': 4, 'remorse': 4, 'disappointment': 4, 'embarrassment': 4,
        'fear': 5, 'nervousness': 5,
        'curiosity': 6, 'confusion': 6, 'realization': 6, 'surprise': 6,
        'neutral': 7
    }
}

# --- 4. Angepasste Ladefunktionen ---
# (Jede Funktion gibt eine Liste von Dictionaries zurück: [{'text': ..., 'labels': ...}, ...])

def load_jigsaw():
    print(f"Loading Jigsaw from {JIGSAW_PATH}...")
    JIGSAW_TEXT_COL = 'comment_text'
    JIGSAW_LABEL_COLS = list(SUPERLABEL_MAPPING['jigsaw'].keys())
    
    samples_list = []
    try:
        df = pd.read_csv(JIGSAW_PATH)
        df = df.dropna(subset=[JIGSAW_TEXT_COL] + JIGSAW_LABEL_COLS)
        
        # Erstelle die Superlabel-Maske für diese Aufgabe
        task_mask = [0.0] * NUM_SUPERLABELS
        for col in JIGSAW_LABEL_COLS:
            idx = SUPERLABEL_MAPPING['jigsaw'].get(col)
            task_mask[idx] = 1.0

        for _, row in df.iterrows():
            superlabel_vec = [0.0] * NUM_SUPERLABELS
            for col in JIGSAW_LABEL_COLS:
                if row[col] == 1:
                    idx = SUPERLABEL_MAPPING['jigsaw'].get(col)
                    superlabel_vec[idx] = 1.0 # OR-Logik
            
            # WICHTIG: Kombiniere Label und Maske
            final_label_vec = [
                label if task_mask[i] == 1.0 else IGNORE_VALUE
                for i, label in enumerate(superlabel_vec)
            ]
            samples_list.append({
                'text': str(row[JIGSAW_TEXT_COL]),
                'labels': final_label_vec
            })
    except FileNotFoundError:
        print(f"Error: Jigsaw file not found.")
    return samples_list

def load_goemotions():
    print(f"Loading GoEmotions from {GOEMOTIONS_PATH_PATTERN}...")
    GOEMOTIONS_TEXT_COL = 'text'
    GOEMOTIONS_LABEL_COLS = list(SUPERLABEL_MAPPING['goemotions'].keys())
    
    samples_list = []
    files = glob.glob(GOEMOTIONS_PATH_PATTERN)
    if not files:
        print("Error: No GoEmotions files found.")
        return samples_list
        
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=[GOEMOTIONS_TEXT_COL] + GOEMOTIONS_LABEL_COLS)
    
    # Erstelle die Superlabel-Maske
    task_mask = [0.0] * NUM_SUPERLABELS
    for col in GOEMOTIONS_LABEL_COLS:
        idx = SUPERLABEL_MAPPING['goemotions'].get(col)
        task_mask[idx] = 1.0
    
    for _, row in df.iterrows():
        superlabel_vec = [0.0] * NUM_SUPERLABELS
        for col in GOEMOTIONS_LABEL_COLS:
            if row[col] == 1:
                idx = SUPERLABEL_MAPPING['goemotions'].get(col)
                superlabel_vec[idx] = 1.0 # OR-Logik
        
        final_label_vec = [
            label if task_mask[i] == 1.0 else IGNORE_VALUE
            for i, label in enumerate(superlabel_vec)
        ]
        samples_list.append({
            'text': str(row[GOEMOTIONS_TEXT_COL]),
            'labels': final_label_vec
        })
    return samples_list

def load_davidson():
    print(f"Loading Davidson from {DAVIDSON_PATH}...")
    DAVIDSON_TEXT_COL = 'tweet'
    DAVIDSON_LABEL_COL = 'class'
    
    samples_list = []
    try:
        df = pd.read_csv(DAVIDSON_PATH)
        df = df.dropna(subset=[DAVIDSON_TEXT_COL, DAVIDSON_LABEL_COL])
        
        task_mask = [0.0] * NUM_SUPERLABELS
        for super_idx in SUPERLABEL_MAPPING['davidson'].values():
            task_mask[super_idx] = 1.0

        for _, row in df.iterrows():
            superlabel_vec = [0.0] * NUM_SUPERLABELS
            orig_label = row[DAVIDSON_LABEL_COL]
            super_idx = SUPERLABEL_MAPPING['davidson'].get(orig_label)
            if super_idx is not None:
                superlabel_vec[super_idx] = 1.0
            
            final_label_vec = [
                label if task_mask[i] == 1.0 else IGNORE_VALUE
                for i, label in enumerate(superlabel_vec)
            ]
            samples_list.append({
                'text': str(row[DAVIDSON_TEXT_COL]),
                'labels': final_label_vec
            })
    except FileNotFoundError:
        print(f"Error: Davidson file not found.")
    return samples_list

def load_olid():
    print(f"Loading OLID from {OLID_TWEETS_PATH}...")
    OLID_TEXT_COL = 'tweet'
    OLID_LABEL_COL = 'label'
    
    samples_list = []
    try:
        df_tweets = pd.read_csv(OLID_TWEETS_PATH, sep='\t', quoting=3)
        df_labels = pd.read_csv(OLID_LABELS_PATH, header=None, names=['id_norm', 'label'])
        df_tweets.rename(columns={'id': 'id_norm'}, inplace=True)
        df = pd.merge(df_tweets, df_labels, on='id_norm')
        df = df.dropna(subset=[OLID_TEXT_COL, OLID_LABEL_COL])
        
        task_mask = [0.0] * NUM_SUPERLABELS
        for super_idx in SUPERLABEL_MAPPING['olid'].values():
            task_mask[super_idx] = 1.0
            
        for _, row in df.iterrows():
            superlabel_vec = [0.0] * NUM_SUPERLABELS
            orig_label_str = row[OLID_LABEL_COL]
            super_idx = SUPERLABEL_MAPPING['olid'].get(orig_label_str)
            if super_idx is not None:
                superlabel_vec[super_idx] = 1.0

            final_label_vec = [
                label if task_mask[i] == 1.0 else IGNORE_VALUE
                for i, label in enumerate(superlabel_vec)
            ]
            samples_list.append({
                'text': str(row[OLID_TEXT_COL]),
                'labels': final_label_vec
            })
    except Exception as e:
        print(f"Error loading OLID: {e}")
    return samples_list


# --- 5. Main-Funktion: Laden, Tokenisieren, Speichern ---
def main():
    print("--- Start: Pre-Processing ---")
    
    # Lade alle Samples in eine einzige Liste
    all_samples = []
    all_samples.extend(load_jigsaw())
    all_samples.extend(load_goemotions())
    all_samples.extend(load_davidson())
    all_samples.extend(load_olid())
    
    print(f"\n--- Total Samples Loaded: {len(all_samples)} ---")
    
    if not all_samples:
        print("Keine Daten geladen. Überprüfe die Dateipfade. Skript wird beendet.")
        return

    # Konvertiere in Hugging Face Dataset
    print("Converting to Hugging Face Dataset...")
    hf_dataset = Dataset.from_list(all_samples)
    
    # Initialisiere Tokenizer
    print(f"Loading Tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenisierungs-Funktion
    def tokenize_function(examples):
        # Wir tokenisieren nur die 'text'-Spalte.
        # 'labels' wird einfach durchgereicht.
        return tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    # Wende Tokenisierung an (batched=True für Geschwindigkeit)
    print("Tokenizing all texts (this may take a while)...")
    with warnings.catch_warnings():
        # Unterdrücke Warnungen über lange Sequenzen, da wir sowieso 'truncation=True' verwenden
        warnings.simplefilter("ignore")
        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4, # Anzahl der CPU-Kerne für die Verarbeitung
            remove_columns=['text'] # Entferne die Roh-Text-Spalte, wir brauchen sie nicht mehr
        )
    
    # Setze das Format für PyTorch
    tokenized_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )
    
    # Speichere das fertige Dataset
    print(f"\n--- Saving processed dataset to {OUTPUT_DIR} ---")
    tokenized_dataset.save_to_disk(OUTPUT_DIR)
    
    print("\n--- Pre-Processing complete! ---")
    print(f"Das Dataset ist jetzt bereit unter: {OUTPUT_DIR}")
    print("Du kannst jetzt das 'train.py' Skript starten.")
    
    print("\nBeispiel eines verarbeiteten Samples:")
    print(tokenized_dataset[0])

if __name__ == "__main__":
    main()