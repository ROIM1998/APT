model_name='huggyllama/llama-7b'
task_name=alpaca_gpt4
adapter_type=lora

if [ "$#" -eq 0 ]; then
    num_epochs=2
    batch_size=4
    lora_r=8
    lora_alpha=16
    learning_rate=1e-4
    seed=42
elif [ "$#" -eq 6 ]; then
    num_epochs=$1
    batch_size=$2
    lora_r=$3
    lora_alpha=$4
    learning_rate=$5
    seed=$6
fi

teacher_param_tuning_config=dq:0-31,dv:0-31,di0:0-31
output_dir="llama_output/${model_name}/${task_name}/bz${batch_size}/${adapter_type}/teacher_${teacher_param_tuning_config}/epoch${num_epochs}/lora_r${lora_r}/lora_alpha${lora_alpha}/lr${learning_rate}/seed${seed}"
echo $output_dir
mkdir -p $output_dir

python run_llama_sft.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --bf16 True \
    --data_path 'data/sft/alpaca_data_cleaned.json' \
    --do_train \
    --do_eval \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 1 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_steps 500 \
    --model_max_length 512 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.03\
    --learning_rate ${learning_rate}\
    --weight_decay 0.\
    --lr_scheduler_type cosine \
    --tf32 True \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --report_to none | tee ${output_dir}/log.txt