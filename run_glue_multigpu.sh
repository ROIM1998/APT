model_name=roberta_base

gpu_available="0,1,2,3,4,5,6"
# Split the gpu_available string into an array
gpu_ids=(${gpu_available//,/ })

task_name=(sst2 stsb qqp mnli cola mrpc qnli)

# For each GPU, run the script with a different mac_constraint
for i in "${!gpu_ids[@]}"; do
    gpu_id=${gpu_ids[$i]}
    task_name=${task_name[$i]}
    echo "Running on GPU $gpu_id with mac_constraint $mac_constraint"
    bash scripts/adaptpruning_nodistill/roberta_base_${task_name}.sh 0.4 8 16 cubic_gradual global free_inout $gpu_id &
done