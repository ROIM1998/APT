for mac_constraint in 0.3 0.8
do
    for steppoint in 0.25 0.5 0.75 1.0
    do
        output_dir="output/roberta_lora_minus_mnli_once_rescaled/step${steppoint}/batchuse${pruning_batches}/mac${mac_constraint}/"
        mkdir -p $output_dir

        python run_minus_training.py \
            --output_dir ${output_dir}\
            --task_name mnli \
            --model_name_or_path roberta-base \
            --do_train \
            --do_eval \
            --minus_scheduler \
            --save_strategy no \
            --max_seq_length 128 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --lr_scheduler_type linear\
            --warmup_ratio 0.06\
            --learning_rate 5e-4\
            --weight_decay 0.1\
            --apply_lora \
            --lora_alpha 16 \
            --lora_r 8 \
            --report_to none \
            --pruning_batches 64 \
            --mac_constraint ${mac_constraint} \
            --pruning_scheduler once \
            --pruning_start ${steppoint}
    done
done