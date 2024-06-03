model_path=$1
mac_constraint=$2
num_batches=$3

source_lang=en
target_lang=ro
task_name=wmt16

lora_alpha=16
output_dir="${model_path}/pruned/constraint_${mac_constraint}/batches_${num_batches}"

python post_training_seq2seq_prune.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_path} \
    --do_train \
    --do_eval \
    --task_name ${task_name} \
    --max_input_length 256 \
    --max_target_length 256 \
    --lang_pair ${target_lang}-${source_lang} \
    --source_lang ${source_lang} \
    --target_lang ${target_lang} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --tf32 True \
    --pruning_batch_size 32 \
    --pruning_batches ${num_batches} \
    --mac_constraint ${mac_constraint} \
    --lora_alpha ${lora_alpha} \