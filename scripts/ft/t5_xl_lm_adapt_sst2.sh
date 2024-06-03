model_name=google/t5-xl-lm-adapt
task_name=sst2

if [ "$#" -eq 0 ]; then
    epoch=10
    batch_size=32
    learning_rate=1e-3
    seed=128
elif [ "$#" -eq 4 ]; then
    epoch=$1
    batch_size=$2
    learning_rate=$3
    seed=$4
fi

output_dir="output/${model_name}/${task_name}/bz${batch_size}/ft/epoch${epoch}/lr${learning_rate}/seed${seed}/"

echo $output_dir
mkdir -p $output_dir


python run_minus_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --log_level info \
    --log_level_replica info \
    --logging_steps 100 \
    --eval_steps 500 \
    --max_seq_length 128 \
    --num_train_epochs ${epoch} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --seed ${seed} \
    --report_to none 