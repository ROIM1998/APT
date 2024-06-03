# TODO: merge T5 tuning and Bert-like model tuning
if [ "$#" -eq 0 ]; then
    model_name='roberta-base'
    task_name=mnli
    mac_constraint=0.4
    lora_r=32
    lora_alpha=16
    adapter_type=lora
    distillation_type=self_interleave
    pruning_scheduler=once
    param_allocation_strategy=free_inout
    pruner_type=global
    num_train_epochs=30
    training_batch_size=128
    pruning_batches=256
    pruning_batch_size=4
    num_prunings=5
elif [ "$#" -eq 5 ]; then
    model_name=$1
    task_name=$2
    mac_constraint=$3
    lora_r=$4
    lora_alpha=$5
    adapter_type=lora
    distillation_type=self
    pruning_scheduler=linear_gradual
    param_allocation_strategy=static_inout
    pruner_type=global
    num_train_epochs=10
    training_batch_size=128
    pruning_batches=256
    pruning_batch_size=4
    num_prunings=5
elif [ "$#" -eq 14 ]; then
    model_name=$1
    task_name=$2
    mac_constraint=$3
    lora_r=$4
    lora_alpha=$5
    adapter_type=$6
    distillation_type=$7
    pruning_scheduler=$8
    param_allocation_strategy=$9
    pruner_type=${10}
    num_train_epochs=${11}
    training_batch_size=${12}
    pruning_batches=${13}
    pruning_batch_size=${14}
    num_prunings=${15}
fi

if [ "$model_name" =  "roberta-base" ]; then
    learning_rate=5e-4
elif [ "$model_name" =  "bert-base-uncased" ]; then
    learning_rate=2e-4
else
    learning_rate=5e-4
fi

teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
student_param_tuning_config=q:0-11,v:0-11,i:0-11
distill_mapping_strategy=none
output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_alloc_${param_allocation_strategy}_${distillation_type}_mapping_${distill_mapping_strategy}_distill/mac${mac_constraint}/epoch${num_train_epochs}/bz${training_batch_size}/numprune${num_prunings}/param${teacher_param_tuning_config}/lora_r${lora_r}/lora_alpha${lora_alpha}"
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
    --minus_scheduler \
    --max_seq_length 128 \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size 128 \
    --lr_scheduler_type linear\
    --distillation_type ${distillation_type} \
    --distill_mapping_strategy ${distill_mapping_strategy} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --apply_lora \
    --lora_alpha ${lora_alpha} \
    --lora_r ${lora_r} \
    --report_to none \
    --pruning_batches ${pruning_batches} \
    --pruning_batch_size ${pruning_batch_size} \
    --mac_constraint ${mac_constraint} \
    --pruning_scheduler ${pruning_scheduler} \
    --param_allocation_strategy ${param_allocation_strategy} \
    --teacher_param_tuning_config ${teacher_param_tuning_config} \
    --student_param_tuning_config ${student_param_tuning_config} \
    --head_scorer_type gradient_l2 \
    --intermediate_scorer_type gradient_l2 \
    --pruner_type ${pruner_type} \
    --do_distill \
    --distill_epoch 24 \
    --teacher_learning \
    --pruning_start 1 \
    --num_prunings ${num_prunings} \
    --pruning_scheduler_strategy saliency \
    --continuous_allocation \
    --continuous_alloc_interval 1