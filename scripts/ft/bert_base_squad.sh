model_name="bert-base-uncased"
task_name="squad"
num_epochs=10
learning_rate=1e-5
batch_size=48
output_dir="output/${model_name}_${task_name}_full/epoch${num_epochs}/bz${batch_size}"


echo $output_dir
mkdir -p $output_dir


python run_minus_squad_training.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_train \
    --do_eval \
    --save_strategy no \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --warmup_ratio 0.06\
    --learning_rate ${learning_rate}\
    --weight_decay 0.1\
    --report_to none 