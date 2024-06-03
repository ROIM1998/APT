if [ "$#" -eq 0 ]; then
    mac_constraint=0.4
    lora_r=8
    lora_alpha=16
    pruning_scheduler=cubic_gradual
    pruner_type=running_fisher
    param_allocation_strategy=running_fisher
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
task_name=sst2
adapter_type=lora
param_resizing_strategy=tophalf_limited
pruning_start=-1
pruning_stop=3
num_prunings=10
pruning_batches=256
pruning_batch_size=4

learning_rate=2e-4
training_batch_size=32
num_train_epochs=30
warmup_param_tuning_config=q:0-11,v:0-11
teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
pre_pruning_tuning_epochs=0.5
pre_pruning_layer_warmup_epochs=1.75
suffix='_noffnstart'

output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_${param_allocation_strategy}_${param_resizing_strategy}_resizing_nodistill${suffix}/mac${mac_constraint}/epoch${num_train_epochs}/bz${training_batch_size}/numprune${num_prunings}/param${teacher_param_tuning_config}/lora_r${lora_r}/lora_alpha${lora_alpha}"
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
    --logging_steps 1000 \
    --log_level info \
    --log_level_replica info \
    --eval_steps 5000 \
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
    --warmup_param_tuning_config ${warmup_param_tuning_config} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --pruning_start ${pruning_start} \
    --pruning_stop ${pruning_stop} \
    --pre_pruning_layer_warmup_epochs ${pre_pruning_layer_warmup_epochs} \
    --head_scorer_type gradient_l2 \
    --intermediate_scorer_type gradient_l2 \
    --pruner_type ${pruner_type} \
    --num_prunings ${num_prunings} \
    --pruning_batch_size ${pruning_batch_size} \
    --pruning_scheduler_strategy saliency \
    --collect_salience \
    --salience_collecting_start 200 \
    --salience_collecting_end -1 \
    --pre_pruning_tuning_epochs ${pre_pruning_tuning_epochs} \
    --mask_lr 0.01 \
    --grafting_top_k -1 \
    --param_resizing_strategy ${param_resizing_strategy} \
    | tee ${output_dir}/log.txt