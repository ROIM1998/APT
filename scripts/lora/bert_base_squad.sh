model_name='bert-base-uncased'
task_name=squad
adapter_type=lora
learning_rate=2e-4
num_epochs=30
batch_size=32

if [ "$#" -eq 0 ]; then
    lora_r=8
    lora_alpha=16
elif [ "$#" -eq 2 ]; then
    lora_r=$1
    lora_alpha=$2
fi

teacher_param_tuning_config=q:0-11,v:0-11
output_dir="output/${model_name}_${adapter_type}_${task_name}/epoch${num_epochs}/bz${batch_size}/lora_r${lora_r}/lora_alpha${lora_alpha}"
echo $output_dir
mkdir -p $output_dir

python run_minus_squad_training.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none \