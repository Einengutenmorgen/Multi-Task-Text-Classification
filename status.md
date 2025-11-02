# ML Design Decisions: Multi-Task Tweet Classifier

# Used datasets

## Davidson DS 

von: https://www.kaggle.com/datasets/eldrich/hate-speech-offensive-tweets-by-davidson-et-al?resource=download
saved as: davidson/

### file-structure 
- csv file with the following columns:
,count,hate_speech,offensive_language,neither,class,tweet
/ 26954 rows

## SOLID 

von: https://zenodo.org/records/3950379#.XxZ-aFVKipp
saved as: SOLID
### file-structure 


## Jigsaw (Wikipedia Talkpages)

von: https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge
saved as: jigsaw
### file-structure 

- test.csv: "id","comment_text" / 552889rows

- train.csv: "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate" / 561809 rows

- test_labels.csv: id,toxic,severe_toxic,obscene,threat,insult,identity_hate / 153160 rows




## Sem-Eval-task 7 (claim detection)

von: https://figshare.com/articles/dataset/RumourEval_2019_data/8845580?file=16188500
saved as: semEval_task7
### file-structure 


## GoEmotions

von:    wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv
        wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv
        wget -P data/full_dataset/ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv
saved as: goEmotions
### file-structure 
3 csv dateien mit den folgenden columns:
text,id,author,subreddit,link_id,parent_id,created_utc,rater_id,example_very_unclear,admiration,amusement,anger,annoyance,approval,caring,confusion,curiosity,desire,disappointment,disapproval,disgust,embarrassment,excitement,fear,gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral

70k rows




## 1. Model Architecture

- Model Type: Multi-Task Learning (MTL) with a Hard Parameter Sharing architecture.

- Shared Trunk: The model uses a shared trunk based on transformers.AutoModel. The specific pre-trained model is mixedbread-ai/mxbai-embed-large-v1.

- Trunk State: The parameters of the shared trunk are frozen (param.requires_grad = False). Only the task-specific heads are trained.

- Shared Representation: The model extracts the last_hidden_state of the [CLS] token as the shared embedding for all tasks.

- Regularization: A nn.Dropout layer is applied to the [CLS] token embedding before it is fed to the heads.

- Task-Specific Heads: The model has 5 independent, non-shared classification heads, one for each task. Each head consists of a single nn.Linear layer.

## 2. Data Ingestion & Preprocessing

**Tasks**: The project loads data for 5 distinct tasks: jigsaw, goemotions, davidson, olid, and rumour.

**Dataset Class**: A single, custom UnifiedDataset class is used. It loads all 5 datasets into memory and creates a master index (self.task_indices) that maps a global index to a specific sample from a specific task.

**Tokenization**:

Uses transformers.AutoTokenizer corresponding to the MODEL_NAME.

Handles both single-sentence (e.g., Jigsaw, Davidson) and sentence-pair (RumourEval) tasks.

All inputs are padded to max_length=128 and truncated.

# Label Schema:

- Ignore Index: A value of -100 is used to mark labels for tasks that are not active for a given sample.

- Data Types: Multi-label tasks (Jigsaw, GoEmotions) use torch.float labels. Multi-class tasks (Davidson, OLID, Rumour) use torch.long labels.

## 3. Training Strategy

**Data Splitting**: The full UnifiedDataset is split into train_dataset and val_dataset (Subset objects) using sklearn.model_selection.train_test_split. This split is stratified by task name to ensure all tasks are proportionally represented in both sets.

**Inter-Task Balancing**: The train_loader uses a custom TaskSampler. This sampler performs oversampling with replacement, sampling from each task up to the size of the largest task (goemotions) to create a balanced "epoch" where each task gets equal representation.

**Validation Sampling**: The val_loader uses a standard SequentialSampler for consistent evaluation.

**Loss Function**: A custom MultiTaskLoss module is implemented.

**Multi-Class** (Davidson, OLID, Rumour): Uses nn.CrossEntropyLoss(ignore_index=-100), which automatically handles masked labels.

**Multi-Label (Jigsaw, GoEmotions)**: Uses nn.BCEWithLogitsLoss(reduction="none") combined with a manual masking operation to zero out the loss from inactive (-100) samples before averaging.

**Loss Aggregation**: The total_loss is calculated as a simple, uniform average of the 5 individual task losses (e.g., (loss_jigsaw + ... + loss_rumour) / 5).

**Optimizer**: The optimizer is optim.AdamW.

**Learning Rate**: The model uses differential learning rates.

**Shared Trunk**: BASE_LR of 1e-5.

**Task Heads**: HEAD_LR of 1e-4.

**LR Scheduler**: A get_linear_schedule_with_warmup scheduler is used, with 10% of training steps dedicated to warmup.

## 4. Evaluation & Validation

- Evaluation Metric: The primary metric for each task is F1-Score.

- Metric Aggregation (Per-Task): The F1-score for each task is calculated using average='weighted', which accounts for label imbalance within that task.

- Metric Aggregation (Overall): An avg_f1 score is computed by taking the np.mean of the 5 individual task F1 scores. This avg_f1 is the key metric for model selection.

- Model Selection: The best model checkpoint (best_model.pth) is saved based on the epoch that achieves the highest avg_f1 on the validation set.

- Evaluation Logic: The evaluate function correctly filters out inactive (-100) labels for each task before calculating the F1 score, ensuring metrics are only computed on valid samples.

## 5. Missing / Unknown Information

- Intra-Task Balancing: While inter-task balancing is implemented (via TaskSampler), no intra-task balancing (e.g., Focal Loss, class-weighted sampling) is currently implemented.

- Dynamic Loss Weighting: The loss is a static, uniform average. No dynamic weighting schemes (e.g., Uncertainty Weighting, GradNorm) from the research report are implemented.

- Hyperparameter Rationale: The CONFIG values (LRs, batch size, etc.) are hard-coded. The process for how these values were chosen is unknown.

- Ablation Studies: It is unknown if single-task baselines were run to confirm that the MTL-lift is positive, as recommended in the research report.