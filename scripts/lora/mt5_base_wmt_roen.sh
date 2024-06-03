#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=144:00:00             # Walltime (hh:mm:ss)

model_name=google/mt5-base
task_name=wmt16
adapter_type=lora
source_lang=ro
target_lang=en

if [ "$#" -eq 0 ]; then
    num_epochs=5
    batch_size=16
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

teacher_param_tuning_config=eq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11
output_dir="output/${model_name}/${task_name}_${source_lang}-${target_lang}/bz${batch_size}/${adapter_type}/epoch${num_epochs}/lora_r${lora_r}/lora_alpha${lora_alpha}/lr${learning_rate}/seed${seed}"
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
    --max_input_length 150 \
    --max_target_length 150 \
    --lang_pair ${target_lang}-${source_lang} \
    --source_lang ${source_lang} \
    --target_lang ${target_lang} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.01\
    --label_smoothing 0.1 \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --tf32 True \
    --fp16 True \
    --report_to none | tee ${output_dir}/log.txt \