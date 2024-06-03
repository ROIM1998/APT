if [ "$#" -eq 0 ]; then
    pruning_frequency=0.1
    pruning_batches=64
    mac_constraint=0.6
elif [ "$#" -eq 3 ]; then
    pruning_frequency=$1
    pruning_batches=$2
    mac_constraint=$3
fi

model_dir="output/roberta_lora_minus_mnli/freq${pruning_frequency}/batchuse${pruning_batches}/mac${mac_constraint}/"

python run_minus_training.py \
    --output_dir ${model_dir}\
    --task_name mnli \
    --model_name_or_path "./${model_dir}" \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r 8 \
    --report_to none\
    --pruning_frequency ${pruning_frequency}\
    --pruning_batches ${pruning_batches} \
    --mac_constraint ${mac_constraint}