python run_summarization.py \
    --model_name_or_path ai4bharat/IndicBART-XLSum \
    --do_train \
    --do_eval \
    --do_predict \
    --lang te_IN \
    --train_file train.csv \
    --validation_file dev.csv \
    --test_file test.csv \
    --max_source_length 512 \
    --max_target_length 256 \
    --val_max_target_length 256 \
    --output_dir tmp/outputs/ \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --dataloader_num_workers 4 \
    --logging_strategy "epoch" \
    --save_strategy "no" \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 15 \
    --forced_bos_token=tokenizer.lang_code_to_id["te_IN"] \
    --summary_column summary \
    --text_column text $@