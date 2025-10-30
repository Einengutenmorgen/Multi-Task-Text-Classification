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

