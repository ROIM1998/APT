export OPENAI_API_KEY=$NEW_OPENAI_KEY
echo $OPENAI_API_KEY
model_output_path=$1

alpaca_eval --model_outputs $model_output_path