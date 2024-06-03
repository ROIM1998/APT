for mac_constraint in 0.05 0.4 0.5
do
    for lora_r in 8 32 128
    do
        model_name='roberta-base'
        adapter_type=lora
        pruning_scheduler=once
        pruner_type=fisher
        task_name=mnli
        lora_alpha=16
        pruning_batches=256
        pruning_batch_size=4
        steppoint=1.0

        output_dir="output/${model_name}_${adapter_type}_minus_${task_name}_${pruning_scheduler}_${pruner_type}_distill_full_exp/mac${mac_constraint}/lora_r${lora_r}/lora_alpha${lora_alpha}"
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
            --num_train_epochs 10 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --lr_scheduler_type linear\
            --warmup_ratio 0.06\
            --learning_rate 5e-4\
            --weight_decay 0.1\
            --apply_lora \
            --lora_alpha ${lora_alpha} \
            --lora_r ${lora_r} \
            --report_to none \
            --pruning_batches ${pruning_batches} \
            --pruning_batch_size ${pruning_batch_size} \
            --mac_constraint ${mac_constraint} \
            --pruning_scheduler ${pruning_scheduler} \
            --pruning_start ${steppoint} \
            --head_scorer_type gradient_l2 \
            --intermediate_scorer_type gradient_l2 \
            --pruner_type ${pruner_type} \
            --do_distill \
            --distill_epoch 8
    done
done