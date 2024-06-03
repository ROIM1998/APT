model_path=$1
output_dir=$2
task_name=$3
lora_r=$4
lora_alpha=$5

python merge_model_lora.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_path} \
    --task_name ${task_name} \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --apply_lora \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \