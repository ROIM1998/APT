output_dir="output/efficiency_testing"
mkdir -p $output_dir

python test_pruning_efficiency.py \
    --output_dir ${output_dir}\
    --task_name mnli \
    --model_name_or_path roberta-base \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --report_to none\