if [ "$#" -eq 0 ]; then
    id=default
    backbone_name='roberta-base'
    model_name='roberta-base'
    task_name=mnli
    lora_r=8
    lora_alpha=16
    batch_size=128
elif [ "$#" -eq 3 ]; then
    id=$1
    backbone_name='roberta-base'
    model_name=$2
    task_name=$3
    lora_r=8
    lora_alpha=16
    batch_size=128
elif [ "$#" -eq 7 ]; then
    id=$1
    backbone_name=$2
    model_name=$3
    task_name=$4
    lora_r=$5
    lora_alpha=$6
    batch_size=$7
fi

output_dir="output/efficiency_testing/${backbone_name}/${task_name}/${id}/bz${batch_size}/"

echo $output_dir
mkdir -p $output_dir

python efficiency_test.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 100 \
    --eval_steps 500 \
    --max_seq_length 128 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none | tee ${output_dir}/log.txt