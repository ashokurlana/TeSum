We use a modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for our experiments.

## Setup

```bash
$ cd mT5/seq2seq
$ conda create python==3.7.9 pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -p ./env
$ conda activate ./env # or source activate ./env (for older versions of anaconda)
$ bash setup.sh 
```
* Use the newly created environment for running rest of the commands.

## Extracting data
```
$ python extract_data.py -i tesum_data/ -o tesum_input/
```
* This will create the source and target training and evaluation filepairs under `mt5_data/`.

## Training & Evaluation

* Minimal training example on a single GPU is given below:
```bash
$ python pipeline.py \
    --model_name_or_path "google/mt5-base" \
    --data_dir "tesum_input" \
    --output_dir "tesum_output" \
    --lr_scheduler_type="linear" \
    --learning_rate=5e-4 \
    --warmup_steps 100 \
    --weight_decay 0.01 \ 
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=16  \
    --num_train_epochs=10 \
    --save_steps 100 \
    --predict_with_generate \
    --evaluation_strategy "epoch" \
    --logging_first_step \
    --adafactor \
    --label_smoothing_factor 0.1 \
    --do_train \
    --do_eval
```  

## Testing

* To calculate rouge scores on test sets using a trained model, use the following snippet:

```bash
$ python pipeline.py \
    --model_name_or_path <path/to/trained/model/directory> \
    --data_dir "tesum_input" \
    --output_dir "tesum_output" \
    --rouge_lang "telugu" \ 
    --predict_with_generate \
    --length_penalty 0.6 \
    --no_repeat_ngram_size 2 \
    --max_source_length 512 \
    --test_max_target_length 84 \
    --do_predict
```
For a detailed example, refer to [evaluate.sh](evaluate.sh)
