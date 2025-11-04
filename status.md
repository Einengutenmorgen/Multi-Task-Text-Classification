# ML Design Decisions: Multi-Task Tweet Classifier



## 📊 Datasets

### 1\. Davidson Dataset

  - **Source**: [Kaggle - Hate Speech & Offensive Tweets](https://www.kaggle.com/datasets/eldrich/hate-speech-offensive-tweets-by-davidson-et-al?resource=download)
  - **Location**: `davidson/`
  - **Structure**: CSV with 26,954 rows
  - **Columns**: `count`, `hate_speech`, `offensive_language`, `neither`, `class`, `tweet`

**Analysis Results:**

```
Total clean rows: 24783

Token Length Statistics:
  Total Samples:    24783
  Min Length:       3
  Max Length:       481
  Mean Length:      30.05
  Median Length:    28
  Percentiles:      (25th: 18) (75th: 39) (95th: 57)

Class Distribution:
label_name
hate         0.057701
neither      0.167978
offensive    0.774321
```

-----

### 2\. SOLID Dataset

  - **Source**: [Zenodo](https://zenodo.org/records/3950379#.XxZ-aFVKipp)
  - **Location**: `SOLID/`

**Analysis Results (OLID Task):**

```
Total clean rows: 3887

Token Length Statistics:
  Total Samples:    3887
  Min Length:       5
  Max Length:       74
  Mean Length:      22.09
  Median Length:    21
  Percentiles:      (25th: 16) (75th: 28) (95th: 36)

Class Distribution:
label
NOT    0.722151
OFF    0.277849
```

-----

### 3\. Jigsaw (Wikipedia Talkpages)

  - **Source**: [HuggingFace](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)
  - **Location**: `jigsaw/`
  - **Files**:
      - `train.csv`: 561,809 rows (`id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`)
      - `test.csv`: 552,889 rows (`id`, `comment_text`)
      - `test_labels.csv`: 153,160 rows (labels for test set)

**Analysis Results (from `train.csv`):**

```
Total clean rows: 159571

Token Length Statistics:
  Total Samples:    159571
  Min Length:       4
  Max Length:       4950
  Mean Length:      94.89
  Median Length:    52
  Percentiles:      (25th: 26) (75th: 104) (95th: 310)

Label Distribution (Total Occurrences):
toxic            15294
obscene           8449
insult            7877
severe_toxic      1595
identity_hate     1405
threat             478

Rows with no toxic labels ('clean'): 143346 (89.83%)
```

-----

### 4\. SemEval Task 7 (RumourEval)

  - **Source**: [Figshare - RumourEval 2019](https://figshare.com/articles/dataset/RumourEval_2019_data/8845580?file=16188500)
  - **Location**: `semEval_task7/`

**Analysis Results (Rumour Task):**

```
Total clean rows (reply-source pairs): 6318

Token Length Statistics (Reply Text):
  Total Samples:    6318
  Min Length:       3
  Max Length:       1264
  Mean Length:      35.23
  Median Length:    30
  Percentiles:      (25th: 19) (75th: 39) (95th: 67)

Token Length Statistics (Source Text):
  Total Samples:    6318
  Min Length:       2
  Max Length:       60
  Mean Length:      31.44
  Median Length:    33
  Percentiles:      (25th: 24) (75th: 44) (95th: 53)

Token Length Statistics (Combined):
  Total Samples:    6318
  Min Length:       4
  Max Length:       1265
  Mean Length:      65.67
  Median Length:    64
  Percentiles:      (25th: 50) (75th: 76) (95th: 99)

Class Distribution:
  comment   : 4670 (73.92%)
  deny      : 446 (7.06%)
  query     : 487 (7.71%)
  support   : 715 (11.32%)
```

-----

### 5\. GoEmotions

  - **Source**: Google Research
  - **Location**: `goEmotions/`
  - **Structure**: 3 CSV files with \~70k rows total
  - **Columns**: `text`, `id`, `author`, `subreddit`, `link_id`, `parent_id`, `created_utc`, `rater_id`, `example_very_unclear`, plus 28 emotion labels (`admiration`, `amusement`, `anger`, etc.)

**Analysis Results:**

```
Total clean rows: 211225

Token Length Statistics:
  Total Samples:    211225
  Min Length:       3
  Max Length:       316
  Mean Length:      19.40
  Median Length:    19
  Percentiles:      (25th: 12) (75th: 26) (95th: 34)

Label Distribution (Total Occurrences):
neutral           55298
approval          17620
admiration        17131
annoyance         13618
gratitude         11625
disapproval       11424
curiosity          9692
amusement          9245
realization        8785
optimism           8715
disappointment     8469
love               8191
anger              8084
joy                7983
confusion          7359
sadness            6758
caring             5999
excitement         5629
surprise           5514
disgust            5301
desire             3817
fear               3197
remorse            2525
embarrassment      2476
nervousness        1810
pride              1302
relief             1289
grief               673

Rows with no emotion labels: 3411 (1.61%)
Avg labels per sample: 1.18
```

-----

### Overall Dataset Statistics

```
Final Clean Sample Counts
======================================================================
            SampleCount Percentage
jigsaw           159571     39.32%
goemotions       211225     52.05%
davidson          24783      6.11%
olid               3887      0.96%
rumour             6318      1.56%

Total samples: 405784
```

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