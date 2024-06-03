# # export CUDA_VISIBLE_DEVICES=0

# zero-shot
python -m eval.truthfulqa.run_eval \
    --ntrain 0 \
    --data_dir data/eval/truthfulqa \
    --save_dir results/truthfulqa/llama-7B-0shot/ \
    --model_name_or_path /mmfs1/gscratch/cse/yizhongw/llama_checkpoints/7B/ \
    --tokenizer_name_or_path /mmfs1/gscratch/cse/yizhongw/llama_checkpoints/7B/ \
    --eval_batch_size 2 \
    --load_in_8bit \
    --use_chat_format