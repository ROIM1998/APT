#!/bin/bash
#SBATCH -p gpu-rtx6k
#SBATCH -A h2lab
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Number of tasks per node (1 in this case)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=64G                 # Memory per node (total memory)
#SBATCH --gres=gpu:1               # Number of GPUs requested
#SBATCH --time=48:00:00             # Walltime (hh:mm:ss)

if [ "$#" -eq 0 ]; then
    model_name='bert-base-uncased'
    lora_r=8
    lora_alpha=16
    learning_rate=1e-4
    teacher_param_tuning_config=eq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11
elif [ "$#" -eq 5 ]; then
    model_name=$1
    lora_r=$2
    lora_alpha=$3
    learning_rate=$4
    teacher_param_tuning_config=$5
fi

adapter_type=lora
num_epochs=30
batch_size=8
suffix=''
task_name=wmt16
source_lang=en
target_lang=ro

if [ -d $model_name ]
then
    output_dir="${model_name}/loraed/epoch${num_epochs}/bz${batch_size}/lr${learning_rate}"
else
    output_dir="output/${model_name}_${adapter_type}_${task_name}${suffix}/epoch${num_epochs}/bz${batch_size}/lora_r${lora_r}/lora_alpha${lora_alpha}"
fi

echo $output_dir
mkdir -p $output_dir

python run_minus_seq2seq_training.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 1000 \
    --eval_steps 5000 \
    --task_name ${task_name} \
    --max_input_length 256 \
    --max_target_length 256 \
    --lang_pair ${target_lang}-${source_lang} \
    --source_lang ${source_lang} \
    --target_lang ${target_lang} \
    --max_input_length 512 \
    --max_target_length 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --tf32 True \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none | tee ${output_dir}/log.txt