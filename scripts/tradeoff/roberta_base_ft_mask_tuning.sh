mac_constraints=(0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)

for mac_constraint in ${mac_constraints[@]}; do
    echo "mac_constraint: $mac_constraint"
    bash scripts/post_training_prune.sh 'output/roberta-base/sst2/bz32/ft/epoch60/lr2e-5/seed42/best_model' sst2 $mac_constraint 64 
done