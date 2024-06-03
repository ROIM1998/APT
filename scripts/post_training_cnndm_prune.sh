model_path=$1
mac_constraint=$2
num_batches=$3
if [ "$#" -eq 5 ]; then
    lora_alpha=$5
else
    lora_alpha=16
fi
output_dir="${model_path}/pruned/constraint_${mac_constraint}/batches_${num_batches}"

python post_training_seq2seq_prune.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_path} \
    --do_train \
    --do_eval \
    --task_name cnndm \
    --max_input_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --tf32 True \
    --pruning_batch_size 32 \
    --pruning_batches ${num_batches} \
    --mac_constraint ${mac_constraint} \
    --lora_alpha ${lora_alpha} \