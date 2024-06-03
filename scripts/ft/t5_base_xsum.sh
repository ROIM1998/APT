model_name="t5-base"
task_name="xsum"

if [ "$#" -eq 0 ]; then
    num_epochs=10
    batch_size=16
    learning_rate=1e-4
    seed=128
elif [ "$#" -eq 4 ]; then
    num_epochs=$1
    batch_size=$2
    learning_rate=$3
    seed=$4
fi

output_dir="output/${model_name}/${task_name}/bz${batch_size}/ft/epoch${epoch}/lr${learning_rate}/seed${seed}/"


echo $output_dir
mkdir -p $output_dir


python run_minus_seq2seq_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --max_input_length 512 \
    --max_target_length 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --report_to none \
    | tee ${output_dir}/log.txt