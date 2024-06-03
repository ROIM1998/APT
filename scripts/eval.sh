model_name=$1
task_name=$2
output_dir="$model_name/results"
mkdir -p $output_dir

# Evaluate
python evaluate.py \
    --output_dir ${output_dir}\
    --model_name_or_path ${model_name} \
    --do_eval \
    --task_name ${task_name}