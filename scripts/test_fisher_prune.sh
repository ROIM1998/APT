if [ "$#" -eq 0 ]; then
    model_name_or_path='roberta-base'
    test_mode='correlation'
elif [ "$#" -eq 1 ]; then
    model_name_or_path=$1
    test_mode='correlation'
elif [ "$#" -eq 3 ]; then
    model_name_or_path=$1
    test_mode=$2
fi

output_dir='./output/test_prune/'
mkdir -p $output_dir

python run_pruning.py \
    --output_dir ${output_dir}\
    --task_name mnli \
    --model_name_or_path ${model_name_or_path} \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --report_to none\
    --do_train\
    --do_eval\
    --test_mode ${test_mode}\
    --ratio_bound 0.1\
    --ratio_step 0.01\
    --apply_lora\
    --prune_mode fisher\