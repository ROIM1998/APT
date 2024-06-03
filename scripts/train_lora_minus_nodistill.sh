if [ "$#" -eq 0 ]; then
    model_name='roberta-base'
    task_name=mnli
    mac_constraint=0.4
    lora_r=8
    lora_alpha=16
elif [ "$#" -eq 1 ]; then
    model_name=$1
    task_name=mnli
    mac_constraint=0.4
    lora_r=8
    lora_alpha=16
elif [ "$#" -eq 2 ]; then
    model_name=$1
    task_name=$2
    mac_constraint=0.4
    lora_r=8
    lora_alpha=16
elif [ "$#" -eq 3 ]; then
    model_name='roberta-base'
    task_name=mnli
    mac_constraint=$1
    lora_r=$2
    lora_alpha=$3
elif [ "$#" -eq 5 ]; then
    model_name=$1
    task_name=$2
    mac_constraint=$3
    lora_r=$4
    lora_alpha=$5
fi


adapter_type=lora
pruning_scheduler=once
pruner_type=global
param_allocation_strategy=free_inout
pruning_start=1
continuous_alloc_interval=1
pruning_batches=256
num_prunings=5
pruning_batch_size=4

if [ "$model_name" =  "roberta-base" ]; then
    learning_rate=5e-4
    training_batch_size=128
    num_train_epochs=30
    pruning_stop=25
    teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
elif [ "$model_name" =  "roberta-large" ]; then
    learning_rate=3e-4
    training_batch_size=128
    num_train_epochs=10
    pruning_stop=8
    teacher_param_tuning_config=q:0-23,v:0-23,i:0-23
elif [ "$model_name" =  "bert-base-uncased" ]; then
    learning_rate=2e-4
    training_batch_size=128
    num_train_epochs=30
    pruning_stop=25
    teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
else
    learning_rate=5e-4
    training_batch_size=128
    num_train_epochs=30
    pruning_stop=25
    teacher_param_tuning_config=q:0-11,v:0-11,i:0-11
fi


output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_${param_allocation_strategy}_nodistill/mac${mac_constraint}/epoch${num_train_epochs}/bz${training_batch_size}/numprune${num_prunings}/param${teacher_param_tuning_config}/lora_r${lora_r}/lora_alpha${lora_alpha}"
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