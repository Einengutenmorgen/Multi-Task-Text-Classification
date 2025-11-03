# ML Design Decisions: Multi-Task Tweet Classifier

## Research on datasets
### Davidson DS 
von: https://www.kaggle.com/datasets/eldrich/hate-speech-offensive-tweets-by-davidson-et-al?resource=download
saved as: davidson/

#### file-structure 
- csv file with the following columns:
,count,hate_speech,offensive_language,neither,class,tweet
/ 26954 rows

### SOLID 

von: https://zenodo.org/records/3950379#.XxZ-aFVKipp
saved as: SOLID
#### file-structure 


### Jigsaw (Wikipedia Talkpages)

von: https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge
saved as: jigsaw
#### file-structure 

- test.csv: "id","comment_text" / 552889rows

- train.csv: "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate" / 561809 rows

- test_labels.csv: id,toxic,severe_toxic,obscene,threat,insult,identity_hate / 153160 rows

### Sem-Eval-task 7 (claim detection)

von: https://figshare.com/articles/dataset/RumourEval_2019_data/8845580?file=16188500
saved as: semEval_task7
#### file-structure 


### GoEmotions

von:    wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
        wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
        wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
saved as: goEmotions

#### file-structure 
3 csv dateien mit den folgenden columns:
text,id,author,subreddit,link_id,parent_id,created_utc,rater_id,example_very_unclear,admiration,amusement,anger,annoyance,approval,caring,confusion,curiosity,desire,disappointment,disapproval,disgust,embarrassment,excitement,fear,gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral

70k rows

# ML Design Decisions: Multi-Task Tweet Classifier

## 📊 Datasets

### 1. Davidson Dataset
- **Source**: [Kaggle - Hate Speech & Offensive Tweets](https://www.kaggle.com/datasets/eldrich/hate-speech-offensive-tweets-by-davidson-et-al?resource=download)
- **Location**: `davidson/`
- **Structure**: CSV with 26,954 rows
- **Columns**: `count`, `hate_speech`, `offensive_language`, `neither`, `class`, `tweet`

### 2. SOLID Dataset
- **Source**: [Zenodo](https://zenodo.org/records/3950379#.XxZ-aFVKipp)
- **Location**: `SOLID/`

### 3. Jigsaw (Wikipedia Talkpages)
- **Source**: [HuggingFace](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)
- **Location**: `jigsaw/`
- **Files**:
  - `test.csv`: 552,889 rows (`id`, `comment_text`)
  - `train.csv`: 561,809 rows (`id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`)
  - `test_labels.csv`: 153,160 rows (labels for test set)

### 4. SemEval Task 7 (Claim Detection)
- **Source**: [Figshare - RumourEval 2019](https://figshare.com/articles/dataset/RumourEval_2019_data/8845580?file=16188500)
- **Location**: `semEval_task7/`

### 5. GoEmotions
- **Source**: Google Research
- **Location**: `goEmotions/`
- **Structure**: 3 CSV files with ~70k rows total
- **Columns**: `text`, `id`, `author`, `subreddit`, `link_id`, `parent_id`, `created_utc`, `rater_id`, `example_very_unclear`, plus 28 emotion labels (`admiration`, `amusement`, `anger`, etc.)

---

## 🏗️ Model Architecture

### Base Configuration
- **Architecture Type**: Multi-Task Learning (MTL) with Hard Parameter Sharing
- **Implementation**: `MultiTaskModel` class
- **Base Model**: `mixedbread-ai/mxbai-embed-large-v1` (via `transformers.AutoModel`)

### Shared Trunk
- **Status**: **Frozen** (`param.requires_grad = False`)
- **Rationale**: `CONFIG["BASE_LR"] = None` triggers trunk freezing
- **Representation**: Extracts `[CLS]` token from `last_hidden_state[:, 0]`
- **Regularization**: `nn.Dropout` applied to CLS embedding

### Task-Specific Heads
- **Number of Heads**: 5 (one per task)
- **Architecture**: Single `nn.Linear` layer per head
- **Heads**: `head_jigsaw`, `head_goemotions`, `head_davidson`, `head_olid`, `head_rumour`
- **Parameter Sharing**: None (fully independent heads)

---

## 🔄 Data Processing

### Dataset Unification
- **Class**: `UnifiedDataset`
- **Strategy**: All 5 datasets loaded into memory with master index mapping global → (task, sample)

### Tokenization
- **Tokenizer**: `transformers.AutoTokenizer` (matches `MODEL_NAME`)
- **Max Length**: 256 tokens
- **Padding**: Enabled
- **Truncation**: Enabled
- **Input Types**: 
  - Single-sentence (Jigsaw, GoEmotions, Davidson, OLID)
  - Sentence-pair (Rumour)

### Label Schema
| Task | Type | Label Format |
|------|------|--------------|
| Jigsaw | Multi-label | `torch.float` |
| GoEmotions | Multi-label | `torch.float` |
| Davidson | Multi-class | `torch.long` |
| OLID | Multi-class | `torch.long` |
| Rumour | Multi-class | `torch.long` |

- **Ignore Index**: `-100` marks inactive tasks
- **Purpose**: Allows batch processing of mixed tasks

---

## 🎯 Training Strategy

### Data Splitting
- **Method**: `sklearn.model_selection.train_test_split`
- **Stratification**: By task name
- **Splits**: `train_dataset`, `val_dataset` (as `Subset` objects)

### Inter-Task Balancing
- **Sampler**: Custom `TaskSampler`
- **Configuration**: `EPOCH_SAMPLING_SIZE: None`
- **Strategy**: **Undersampling** - each task sampled with replacement up to size of smallest task
- **Effect**: Prevents large datasets from dominating training

### Validation Sampling
- **Sampler**: `SequentialSampler`
- **Purpose**: Consistent evaluation across epochs

### Loss Function (`MultiTaskLoss`)

#### Multi-Class Tasks (Davidson, OLID, Rumour)
- **Loss**: `nn.CrossEntropyLoss(reduction="none")`
- **Masking**: Helper `_compute_ce_loss` filters labels != -100
- **Aggregation**: Mean over active samples only

#### Multi-Label Tasks (Jigsaw, GoEmotions)
- **Loss**: `nn.BCEWithLogitsLoss(reduction="none")`
- **Masking**: Helper `_compute_bce_loss` filters where first label == -100
- **Aggregation**: Mean over active samples only

#### Total Loss
```
total_loss = sum(active_task_losses) / count(active_tasks)
```
- Inactive tasks contribute 0.0 to sum
- Only active tasks in batch are averaged

---

## ⚙️ Optimization

### Optimizer
- **Type**: `optim.AdamW`
- **Learning Rates**:
  - **Shared Trunk**: None (frozen)
  - **Task Heads**: `1e-4`
- **Configuration**: Via `model.get_optimizer_params`

### Learning Rate Scheduler
- **Type**: `get_linear_schedule_with_warmup`
- **Warmup**: 10% of training steps
- **Schedule**: Linear decay after warmup

### Training Hyperparameters
- **Batch Size**: 8
- **Max Sequence Length**: 256

---

## 📈 Evaluation & Model Selection

### Per-Task Metrics
- **Primary Metric**: F1-Score (`sklearn.metrics.f1_score`)
- **Averaging**: `average='macro'`
- **Filtering**: Inactive labels (-100) excluded before calculation

### Overall Metric
- **Metric**: `avg_f1_macro`
- **Calculation**: `np.mean([f1_jigsaw, f1_goemotions, f1_davidson, f1_olid, f1_rumour])`
- **Purpose**: Primary metric for model selection

### Model Selection
- **Checkpoint**: `best_model.pth`
- **Criterion**: Highest `avg_f1_macro` on validation set
- **Strategy**: Save best performing epoch

---

## ❓ Unknown/Missing Information

### Not Implemented
1. **Intra-Task Balancing**
   - No class-weighted sampling
   - No Focal Loss for imbalanced classes within tasks

2. **Dynamic Loss Weighting**
   - No Uncertainty Weighting
   - No GradNorm
   - Simple equal averaging of active task losses

3. **Ablation Studies**
   - Unknown if single-task baselines were evaluated
   - No comparison to MTL performance

### Unspecified Rationale
- **Hyperparameter Selection**: Values in `CONFIG` are hard-coded without documented justification
  - Why `HEAD_LR = 1e-4`?
  - Why `BATCH_SIZE = 8`?
  - Why `MAX_LENGTH = 256`?
  - Selection process unknown