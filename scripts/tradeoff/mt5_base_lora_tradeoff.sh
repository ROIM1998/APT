lora_rs=(102 64 32 16 8)
for lora_r in ${lora_rs[@]}; do
    echo "lora_r: $lora_r"
    bash scripts/lora/mt5_base_wmt_enro.sh 2 16 $lora_r $(($lora_r * 4)) 5e-5 42
done