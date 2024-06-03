lora_rs=(102 64 32 16 8)
for lora_r in ${lora_rs[@]}; do
    echo "lora_r: $lora_r"
    bash scripts/lora/t5_base_lm_adapt_cnndm.sh 6 16 $lora_r $(($lora_r * 4)) 5e-5 42
done