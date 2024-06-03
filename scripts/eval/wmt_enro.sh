model_name=$1

mac_constraint=0.4
lora_r=8
pruning_start=-1
pruning_scheduler=cubic_gradual
pruner_type=none
param_allocation_strategy=running_fisher
distillation_type=self_momentum
distill_mapping_strategy=dynamic_block_teacher_dynamic_student


task_name=wmt16
adapter_type=lora
source_lang=en
target_lang=ro
param_resizing_strategy=tophalf_limited
pruning_start=-1
pruning_stop=3
distill_start=-1 # about 60%, between 3.4 and 3.8, but after 3.6 where the teacher is updated
distill_epoch=5
pruning_batches=64
num_prunings=10
pruning_batch_size=4
# pre_pruning_tuning_epochs=1
pre_pruning_tuning_steps=200
sparsity_warmup_epochs=1

learning_rate=1e-3
training_batch_size=16
num_train_epochs=10
warmup_param_tuning_config=eq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11,ei0:0-11,di0:0-11
teacher_param_tuning_config=eq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11,ei0:0-11,di0:0-11
student_param_tuning_config=eq:0-11,ev:0-11,dq:0-11,dv:0-11,cq:0-11,cv:0-11,ei0:0-11,di0:0-11


output_dir="${model_name}/eval"
echo $output_dir
mkdir -p $output_dir

python run_minus_seq2seq_training.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_eval \
    --save_strategy no \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --eval_steps 5000 \
    --logging_steps 1000 \
    --log_level info \
    --log_level_replica info \
    --minus_scheduler \
    --task_name ${task_name} \
    --max_input_length 256 \
    --max_target_length 256 \
    --lang_pair ${target_lang}-${source_lang} \
    --source_lang ${source_lang} \
    --target_lang ${target_lang} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${training_batch_size} \
    --per_device_eval_batch_size ${training_batch_size} \
    --tf32 True \
    --lr_scheduler_type linear\
    --distillation_type ${distillation_type} \
    --distill_mapping_strategy ${distill_mapping_strategy} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --seed 128 \
    --apply_lora \
    --lora_alpha 16 \
    --lora_r ${lora_r} \
    --report_to none \
    --pruner_type none \
    | tee ${output_dir}/log.txt