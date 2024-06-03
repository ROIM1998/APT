model_path=$1
task_name=$2
mac_constraint=$3
num_batches=$4
if [ "$#" -eq 5 ]; then
    lora_alpha=$5
else
    lora_alpha=16
fi
output_dir="${model_path}/new_pruned/constraint_${mac_constraint}/batches_${num_batches}"

python post_training_prune.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_path} \
    --task_name ${task_name} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --pruning_batch_size 32 \
    --pruning_batches ${num_batches} \
    --mac_constraint $3 \
    --lora_alpha ${lora_alpha} \