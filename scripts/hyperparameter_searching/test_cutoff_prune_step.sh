export CUDA_VISIBLE_DEVICES=3

for pruning_frequency in 0.1 0.5 1.0 1.5
do
    output_dir="output/roberta_lora_minus_mnli_cutoff/freq${pruning_frequency}/batchuse64/mac0.6/"
    mkdir -p $output_dir

    python run_minus_training.py \
        --output_dir ${output_dir}\
        --task_name mnli \
        --model_name_or_path roberta-base \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --apply_lora \
        --lora_alpha 16 \
        --lora_r 8 \
        --report_to none\
        --pruning_batches 64 \
        --mac_constraint 0.6 \
        --pruning_frequency ${pruning_frequency}\
        --pruning_scheduler cutoff
done