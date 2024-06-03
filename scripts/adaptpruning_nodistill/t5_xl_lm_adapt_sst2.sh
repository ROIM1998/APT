if [ "$#" -eq 0 ]; then
    mac_constraint=0.4
    lora_r=8
    pruning_start=-1
    pruning_scheduler=cubic_gradual
    pruner_type=running_fisher
    param_allocation_strategy=running_fisher
elif [ "$#" -eq 8 ]; then
    mac_constraint=$1
    lora_r=$2
    pruning_start=$3
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
elif [ "$#" -eq 9 ]; then
    mac_constraint=$1
    lora_r=$2
    pruning_start=$3
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
    gpu_id=$9
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

model_name=google/t5-xl-lm-adapt
task_name=sst2
adapter_type=lora
param_resizing_strategy=tophalf_limited
pruning_start=-1
pruning_stop=3
pruning_batches=64
num_prunings=5
pruning_batch_size=16
pre_pruning_tuning_epochs=0.5
pre_pruning_layer_warmup_epochs=1.75

learning_rate=1e-3
training_batch_size=16
num_train_epochs=20
warmup_param_tuning_config=eq:0-23,ev:0-23,dq:0-23,dv:0-23,cq:0-23,cv:0-23
teacher_param_tuning_config=eq:0-23,ev:0-23,dq:0-23,dv:0-23,cq:0-23,cv:0-23,ei0:0-23,di0:0-23

suffix='_noffnstart_nodistill'

output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_alloc_${param_allocation_strategy}_${param_resizing_strategy}${suffix}/mac${mac_constraint}/epoch${num_train_epochs}/bz${training_batch_size}/numprune${num_prunings}/param${teacher_param_tuning_config}/lora_r${lora_r}/pruning_start${pruning_start}"
echo $output_dir
mkdir -p $output_dir

python run_minus_training.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --eval_steps 1000 \
    --logging_steps 1000 \
    --minus_scheduler \
    --task_name ${task_name} \
    --max_seq_length 128 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --tf32 True \
    --lr_scheduler_type linear\
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --apply_lora \
    --lora_alpha 16 \
    --lora_r ${lora_r} \
    --report_to none \
    --pruning_batches ${pruning_batches} \
    --pruning_batch_size ${pruning_batch_size} \
    --mac_constraint ${mac_constraint} \
    --pruning_scheduler ${pruning_scheduler} \
    --param_allocation_strategy ${param_allocation_strategy} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --warmup_param_tuning_config ${warmup_param_tuning_config} \
    --pre_pruning_layer_warmup_epochs ${pre_pruning_layer_warmup_epochs} \
    --head_scorer_type gradient_l2 \
    --intermediate_scorer_type gradient_l2 \
    --pruner_type ${pruner_type} \
    --pruning_start ${pruning_start} \
    --pruning_stop ${pruning_stop} \
    --num_prunings ${num_prunings} \
    --pruning_scheduler_strategy saliency \
    --collect_salience \
    --salience_collecting_start 200 \
    --salience_collecting_end -1 \
    --pre_pruning_tuning_epochs ${pre_pruning_tuning_epochs} \
    --mask_lr 0.01 \
    --grafting_top_k -1 \
    --param_resizing_strategy ${param_resizing_strategy} \
    --tuning_expanding_ratio 4.0 \
    --max_lora_r 64 \
    --pre_tuning_scorer magnitude \
    --pre_tuning_pruner uniform \
    --pre_tuning_constraint 0.8 \
    | tee ${output_dir}/log.txt