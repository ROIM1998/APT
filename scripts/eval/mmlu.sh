# # export CUDA_VISIBLE_DEVICES=0
# zero-shot
model_name_or_path=$1

mkdir -p output/results/mmlu/llama-7B-5shot/

python run_eval_llama_mmlu.py \
    --ntrain 5 \
    --data_dir /mmfs1/home/bowen98/projects/AdaptPruning/data/eval/mmlu \
    --output_dir output/results/mmlu/llama2-7B-0shot/ \
    --model_name_or_path ${model_name_or_path} \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --eval_batch_size 2 | tee "${model_name_or_path}/mmlu-5shot.log"
    
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir /mmfs1/home/bowen98/projects/AdaptPruning/data/eval/mmlu \
#     --save_dir results/mmlu/llama2-7B-0shot/ \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
#     --eval_batch_size 2 \
#     --use_chat_format

# # zero-shot with chatgpt
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-0shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # few-shot with chatgpt
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/chatgpt-5shot/ \
#     --openai_engine "gpt-3.5-turbo-0301" \
#     --eval_batch_size 20


# # zero-shot with gpt4
# python -m eval.mmlu.run_eval \
#     --ntrain 0 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-0shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 20


# # few-shot with gpt4
# python -m eval.mmlu.run_eval \
#     --ntrain 5 \
#     --data_dir data/eval/mmlu \
#     --save_dir results/mmlu/gpt4-5shot/ \
#     --openai_engine "gpt-4-0314" \
#     --n_instances 100 \
#     --eval_batch_size 2
