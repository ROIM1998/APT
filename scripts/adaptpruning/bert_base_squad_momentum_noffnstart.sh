if [ "$#" -eq 0 ]; then
    mac_constraint=0.4
    lora_r=8
    pruning_start=-1
    pruning_scheduler=cubic_gradual
    pruner_type=running_fisher
    param_allocation_strategy=running_fisher
    distillation_type=self_momentum
    distill_mapping_strategy=static_teacher_dynamic_cofi_student
elif [ "$#" -eq 8 ]; then
    mac_constraint=$1
    lora_r=$2
    pruning_start=$3    
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
    distillation_type=$7
    distill_mapping_strategy=$8
elif [ "$#" -eq 9 ]; then
    mac_constraint=$1
    lora_r=$2
    pruning_start=$3
    pruning_scheduler=$4
    pruner_type=$5
    param_allocation_strategy=$6
    distillation_type=$7
    distill_mapping_strategy=$8
    gpu_id=$9
    export CUDA_VISIBLE_DEVICES=$gpu_id
fi

model_name=bert-base-uncased
task_name=squad
param_resizing_strategy=tophalf_limited
adapter_type=lora
pruning_start=-1
pruning_stop=10
distill_start=-1
distill_epoch=20
pruning_batches=64
num_prunings=10
pruning_batch_size=4
pre_pruning_tuning_epochs=3
pre_pruning_layer_warmup_epochs=6.5

learning_rate=2e-4
training_batch_size=32
num_train_epochs=40
warmup_param_tuning_config=q:0-11,v:0-11
teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
suffix=_noffnstart

output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_alloc_${param_allocation_strategy}_${distillation_type}_mapping_${distill_mapping_strategy}_distill_momentum${suffix}/mac${mac_constraint}/epoch${num_train_epochs}/bz${training_batch_size}/numprune${num_prunings}/param${teacher_param_tuning_config}/lora_r${lora_r}/pruning_start${pruning_start}/distill_epoch${distill_epoch}"
echo $output_dir
mkdir -p $output_dir

python run_minus_squad_training.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_steps 500 \
    --eval_steps 2000 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --lr_scheduler_type linear\
    --distillation_type ${distillation_type} \
    --distill_mapping_strategy ${distill_mapping_strategy} \
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
    --do_distill \
    --do_virtual_prune \
    --distill_start ${distill_start} \
    --distill_epoch ${distill_epoch} \
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
    | tee ${output_dir}/log.txt