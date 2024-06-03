if [ "$#" -eq 0 ]; then
    mac_constraint=0.4
    lora_r=8
    lora_alpha=16
    pruning_scheduler=once
    pruner_type=global
    param_allocation_strategy=free_inout
elif [ "$#" -eq 6 ]; then
    mac_constraint=$1
    lora_r=$2
    lora_alpha=$3
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
elif [ "$#" -eq 7 ]; then
    mac_constraint=$1
    lora_r=$2
    lora_alpha=$3
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
    gpu_id=$7
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

model_name=bert-base-uncased
task_name=mnli
adapter_type=lora
pruning_start=1
continuous_alloc_interval=1
pruning_batches=256
num_prunings=5
pruning_batch_size=4

learning_rate=2e-5
training_batch_size=32
num_train_epochs=30
pruning_stop=20
teacher_param_tuning_config=q:0-11,v:0-11,i:0-11


output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_${param_allocation_strategy}_nodistill_evensmallerlr/mac${mac_constraint}/epoch${num_train_epochs}/bz${training_batch_size}/numprune${num_prunings}/param${teacher_param_tuning_config}/lora_r${lora_r}/lora_alpha${lora_alpha}"
echo $output_dir
mkdir -p $output_dir

python run_minus_training.py \
    --output_dir ${output_dir}\
    --task_name ${task_name} \
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --minus_scheduler \
    --max_seq_length 128 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --lr_scheduler_type linear\
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none \
    --pruning_batches ${pruning_batches} \
    --mac_constraint ${mac_constraint} \
    --pruning_scheduler ${pruning_scheduler} \
    --param_allocation_strategy ${param_allocation_strategy} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --continuous_allocation \
    --continuous_alloc_interval ${continuous_alloc_interval} \
    --pruning_start ${pruning_start} \
    --pruning_stop ${pruning_stop} \
    --head_scorer_type gradient_l2 \
    --intermediate_scorer_type gradient_l2 \
    --pruner_type ${pruner_type} \
    --num_prunings ${num_prunings} \
    --pruning_batch_size ${pruning_batch_size} \
    --pruning_scheduler_strategy saliency