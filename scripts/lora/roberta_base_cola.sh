model_name='roberta-base'
task_name=cola
adapter_type=lora

if [ "$#" -eq 0 ]; then
    num_epochs=80
    batch_size=32
    lora_r=8
    lora_alpha=16
    learning_rate=2e-4
    seed=128
elif [ "$#" -eq 6 ]; then
    num_epochs=$1
    batch_size=$2
    lora_r=$3
    lora_alpha=$4
    learning_rate=$5
    seed=$6
fi

teacher_param_tuning_config=q:0-11,v:0-11
output_dir="output/${model_name}/${task_name}/bz${batch_size}/${adapter_type}/epoch${num_epochs}/lora_r${lora_r}/lora_alpha${lora_alpha}/param_tuning_${teacher_param_tuning_config}/lr${learning_rate}/seed${seed}"
echo $output_dir
mkdir -p $output_dir

python run_minus_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 100 \
    --log_level info \
    --log_level_replica info \
    --eval_steps 500 \
    --max_seq_length 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --seed ${seed} \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --report_to none \