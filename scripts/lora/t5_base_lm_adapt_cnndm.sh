model_name=google/t5-base-lm-adapt
task_name=cnndm
adapter_type=lora

if [ "$#" -eq 0 ]; then
    num_epochs=6
    batch_size=16
    lora_r=102
    lora_alpha=408
    learning_rate=5e-5
    seed=42
elif [ "$#" -eq 6 ]; then
    num_epochs=$1
    batch_size=$2
    lora_r=$3
    lora_alpha=$4
    learning_rate=$5
    seed=$6
fi

teacher_param_tuning_config=eq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11
output_dir="output/${model_name}/${task_name}/bz${batch_size}/${adapter_type}/epoch${num_epochs}/lora_r${lora_r}/lora_alpha${lora_alpha}/lr${learning_rate}/seed${seed}"
echo $output_dir
mkdir -p $output_dir

python run_minus_seq2seq_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 500 \
    --eval_steps 2000 \
    --max_input_length 512 \
    --max_target_length 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --tf32 True \
    --fp16 True \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.01\
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --report_to none | tee ${output_dir}/log.txt \