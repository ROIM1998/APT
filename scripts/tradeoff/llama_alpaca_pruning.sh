lora_r=8
pruning_start=-1
pruning_scheduler=cubic_gradual
pruner_type=running_fisher
param_allocation_strategy=running_fisher
distillation_type=self_momentum
distill_mapping_strategy=dynamic_block_teacher_dynamic_student
model_name='meta-llama/Llama-2-7b-hf'
param_resizing_strategy=tophalf_limited
task_name=alpaca_gpt4
adapter_type=lora
lora_alpha=16

pruning_start=-1
pruning_stop=3
distill_start=-1 # about 60%, between 3.4 and 3.8, but after 3.6 where the teacher is updated
distill_epoch=5
pruning_batches=128
num_prunings=10
pruning_batch_size=2
# pre_pruning_tuning_epochs=1
pre_pruning_tuning_steps=200
sparsity_warmup_epochs=1

learning_rate=2e-4
training_batch_size=4
num_train_epochs=10
warmup_param_tuning_config=dq:0-31,dv:0-31
teacher_param_tuning_config=dq:0-31,dv:0-31
student_param_tuning_config=dq:0-31,dv:0-31


target_constraints=(0.8 0.6 0.4)
pre_tuning_constraints=(0.9 1.0)

for mac_constraint in ${target_constraints[@]}; do
    for pre_tuning_constraint in ${pre_tuning_constraints[@]}; do
        echo "mac_constraint: $mac_constraint"
        echo "pre_tuning_constraint: $pre_tuning_constraint"
        output_dir="llama_output/${model_name}/${task_name}/bz${training_batch_size}/elastictuning_virtualprune_pre-tuning-prune-${pre_tuning_constraint}/mac${mac_constraint}/epoch${num_train_epochs}/distill_epoch${distill_epoch}/numprune${num_prunings}/sparsity_warmup${sparsity_warmup_epochs}/pruning_start${pruning_start}/pruning_stop${pruning_stop}/lora_r${lora_r}/lora_alpha${lora_alpha}/warmup_param${warmup_param_tuning_config}/teacher_param${teacher_param_tuning_config}/"
        echo $output_dir
        mkdir -p $output_dir

        python run_llama_sft.py \
            --output_dir ${output_dir}\
            --task_name ${task_name} \
            --model_name_or_path ${model_name} \
            --bf16 True \
            --data_path 'data/sft/alpaca_data_gpt4.json' \
            --do_train \
            --do_eval \
            --save_strategy no \
            --evaluation_strategy steps \
            --logging_strategy steps \
            --logging_steps 100 \
            --eval_steps 1000 \
            --log_level info \
            --log_level_replica info \
            --model_max_length 512 \
            --num_train_epochs ${num_train_epochs} \
            --per_device_train_batch_size ${training_batch_size} \
            --per_device_eval_batch_size ${training_batch_size} \
            --gradient_accumulation_steps 8 \
            --warmup_ratio 0.03\
            --learning_rate ${learning_rate}\
            --weight_decay 0.\
            --lr_scheduler_type cosine \
            --tf32 True \
            --apply_lora \
            --lora_alpha ${lora_alpha} \
            --lora_r ${lora_r} \
            --distillation_type ${distillation_type} \
            --distill_mapping_strategy ${distill_mapping_strategy} \
            --seed 128 \
            --report_to none \
            --pruning_batches ${pruning_batches} \
            --pruning_batch_size ${pruning_batch_size} \
            --mac_constraint ${mac_constraint} \
            --pruning_scheduler ${pruning_scheduler} \
            --sparsity_warmup_epochs ${sparsity_warmup_epochs} \
            --param_allocation_strategy ${param_allocation_strategy} \
            --teacher_param_tuning_config ${teacher_param_tuning_config} \
            --student_param_tuning_config ${student_param_tuning_config} \
            --head_scorer_type gradient_l1 \
            --intermediate_scorer_type gradient_l1 \
            --pruner_type ${pruner_type} \
            --do_distill \
            --distill_start ${distill_start} \
            --distill_epoch ${distill_epoch} \
            --pruning_start ${pruning_start} \
            --pruning_stop ${pruning_stop} \
            --num_prunings ${num_prunings} \
            --pruning_scheduler_strategy saliency \
            --collect_salience \
            --salience_collecting_start 200 \
            --salience_collecting_end -1 \
            --pre_pruning_tuning_steps ${pre_pruning_tuning_steps} \
            --mask_lr 0.01 \
            --grafting_top_k -1 \
            --param_resizing_strategy ${param_resizing_strategy} \
            --tuning_expanding_ratio 4.0 \
            --max_lora_r $(($lora_r * 8)) \
            --pre_tuning_scorer backward_running_hidden_states_salience \
            --pre_tuning_pruner running_fisher \
            --pre_tuning_constraint ${pre_tuning_constraint} \
            | tee ${output_dir}/log.txt
        
        bash scripts/eval/mmlu.sh "${output_dir}/best_model"
    done
done



