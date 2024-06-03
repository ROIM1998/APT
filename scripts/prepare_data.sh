mkdir -p data/sft
mkdir -p data/eval

# Download alpaca data for sft
wget -O data/sft/alpaca_data.json https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json

# Download MMLU eval data

# MMLU dataset
wget -O data/eval/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/eval/mmlu_data
tar -xvf data/eval/mmlu_data.tar -C data/eval/mmlu_data
mv data/eval/mmlu_data/data data/eval/mmlu && rm -r data/eval/mmlu_data data/eval/mmlu_data.tar

# TruthfulQA dataset, open-ended and multiple-choice versions
mkdir -p data/eval/truthfulqa
wget -O data/eval/truthfulqa/truthfulqa.csv https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/v0/TruthfulQA.csv
wget -O data/eval/truthfulqa/truthfulqa_mc.json https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/data/mc_task.json