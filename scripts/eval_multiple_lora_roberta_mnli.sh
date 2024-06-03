for mac_constraint in 0.2 0.4 0.6
do
    for pruning_frequency in 0.1 0.5 1.5
    do
        echo "Using mac_constraint ${mac_constraint}, pruning_frequency ${pruning_frequency}"
        bash scripts/eval_lora_roberta_mnli.sh ${pruning_frequency} 64 ${mac_constraint}
    done
done