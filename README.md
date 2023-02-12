# NEd

## Introduction

- This is a python implementation with PyTorch for our paper
- We deleted all identifiable information to ensure anonymity
- This is a early release version and would have some bugs caused by code reorganizations. 


## Python Env

```
conda create -n ned python=3.7
conda activate ned
conda install pytorch=1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install transformers==4.6.1; pip uninstall -y transformers
pip install gpustat ipython jupyter datasets==1.7 accelerate==0.3 sklearn tensorboard
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
```

## Data 

All datasets and corpora used in this paper are publicly available. 


- [Scruples](https://github.com/allenai/scruples) Project Including `Anecdotes` and `Dilemmas` datasets. 
  - Download `Anecdotes` dataset [here](https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/anecdotes.tar.gz)
  - Download `Dilemmas` dataset [here](https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/data/dilemmas.tar.gz)
- [Social Chemistry 101](https://github.com/mbforbes/social-chemistry-101) Project
    - Download `Social Chemistry 101` corpus [here](https://storage.googleapis.com/ai2-mosaic-public/projects/social-chemistry/data/social-chem-101.zip)

After unzip, we can get three folders:
- `social-chem-101`
- `anecdotes`
- `dilemmas`

Please put them in one dir named `$DATA_DIR`.


## Norm-grounding Knowledge Model

### Knowledge Model Training

Please specify `DATA_DIR` and `KNOWMODEL_DIR` in the below code block. 
 - `DATA_DIR` is a dir to downloaded `Social Chemistry 101` corpus
 - `KNOWMODEL_DIR` is a dir to store the checkpoint of Norm-grounding Knowledge Model

```
DATA_DIR="to-specify"
KNOWMODEL_DIR="to-specify"

python3 -m sc101_gen.train_sc101gen --do_train --do_eval --do_prediction \
    --loss_components [clm]-[judgment]-[char] --model_name_or_path facebook/bart-large \
    --train_file $DATA_DIR/social-chem-101/social-chem-101.v1.0.train.jsonl \
    --dev_file $DATA_DIR/social-chem-101/social-chem-101.v1.0.dev.jsonl \
    --test_file $DATA_DIR/social-chem-101/social-chem-101.v1.0.test.jsonl \
    --max_length 80 --train_batch_size 16 --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 --num_train_epochs 3 \
    --output_dir $KNOWMODEL_DIR
    --num_proc 6 --logging_steps 200 --eval_steps 4000
```

### Knowledge Model Prediction on Anecdotes

```
python3 -m sc101_gen.gen_data_anecdotes \
    --data_dir $DATA_DIR --knowmodel_dir $KNOWMODEL_DIR
```

This cmd will generate three new files (i.e., train, dev, test) ending with `action.jsonl` in `$DATA_DIR/anecdotes`

### Knowledge Model Prediction on Dilemmas

```
python3 -m sc101_gen.gen_data_dilemmas \
    --data_dir $DATA_DIR --knowmodel_dir $KNOWMODEL_DIR
```

This cmd will generate three new files (i.e., train, dev, test) ending with `action.jsonl` in `$DATA_DIR/dilemmas`

## Natural Language Infererence (NLI) Model

We directly use a RoBERTa-based NLI model available at Huggingface

### NLI Prediction on Anecdotes

```
python3 -m nli_distill.predict_nli_anecdotes \
    --data_dir $DATA_DIR
```

This cmd will generate other three new files (i.e., train, dev, test) ending with `nli.jsonl` in `$DATA_DIR/anecdotes`

### NLI Prediction on Dilemmas

```
python3 -m nli_distill.predict_nli_anecdotes \
    --data_dir $DATA_DIR
```

This cmd will generate other three new files (i.e., train, dev, test) ending with `nli.jsonl` in `$DATA_DIR/dilemmas`


## Norm-supported Judgment Model

After modeling training, model prediction and knowledge grounding we can get 
```
$DATA_DIR/anecdotes/train.scruples-anecdotes.action.nli.jsonl
$DATA_DIR/anecdotes/dev.scruples-anecdotes.action.nli.jsonl
$DATA_DIR/anecdotes/test.scruples-anecdotes.action.nli.jsonl

$DATA_DIR/dilemmas/train.scruples-dilemmas.action.nli.jsonl
$DATA_DIR/dilemmas/dev.scruples-dilemmas.action.nli.jsonl
$DATA_DIR/dilemmas/test.scruples-dilemmas.action.nli.jsonl
```

### Training and Evaluation on Anecdotes

Please specify `$OUTPUT_DIR` before running the code below. 

```
OUTPUT_DIR="to-specify"
python3 -m work_scruples.script_anecdotes --do_train --do_eval --do_prediction \
    --train_file $DATA_DIR/anecdotes/train.scruples-anecdotes.action.nli.jsonl \
    --dev_file $DATA_DIR/anecdotes/dev.scruples-anecdotes.action.nli.jsonl \
    --test_file $DATA_DIR/anecdotes/test.scruples-anecdotes.action.nli.jsonl \
    --max_length 480 --train_batch_size 32 --num_train_epochs 3 --logging_steps 100 \
    --model_name_or_path roberta-base --gradient_accumulation_steps 4 --learning_rate 5e-5 \
    --eval_steps 500 --seed 42 --output_dir $OUTPUT_DIR

```

### Training and Evaluation on Dilemmas

Please specify `$OUTPUT_DIR` before running the code below. 

```
OUTPUT_DIR="to-specify"
python3 -m work_scruples.script_dilemmas --do_train --do_eval --do_prediction \
    --train_file $DATA_DIR/train.scruples-dilemmas.action.nli.jsonl \
    --dev_file $DATA_DIR/dilemmas/test.scruples-dilemmas.action.nli.jsonl \
    --test_file $DATA_DIR/dilemmas/dev.scruples-dilemmas.action.nli.jsonl  \
    --max_length 128 --logging_steps 100 --eval_steps 500  --model_name_or_path roberta-base \
    --train_batch_size 32 --learning_rate 5e-5 --num_train_epochs 3 --seed 42 --output_dir $OUTPUT_DIR
```




