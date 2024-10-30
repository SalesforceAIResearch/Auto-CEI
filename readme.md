# Automatic Curriculum Expert Iteration

## Install dependencies

The dependencies are in the `requirements.txt` file. 

```bash
pip install -r requirements.txt
```

## Generate dataset

### Blocksworld

Read the README file in the `data/blocksworld` directory.

### MATH

Run:
```bash
export PYTHONPATH="$PWD"
python3 data/MATH/data_pre.py
```

### BoardgameQA

Run:
```bash
export PYTHONPATH="$PWD"
python3 data/boardgameQA/data_pre.py
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_ORG_ID="your-openai-org-id"
python3 data/boardgameQA/data_gen.py # use GPT-4 for data generation
```

## run the code

You will need to have access to the Huggingface via
```bash
huggingface-cli login --token $HUGGINGFACE_TOKEN
```
Make sure your HuggingFace account have access to the Meta Llama3.1 model. 

Use the script in the `scripts` directory to run the code. 

```bash
bash scripts/run_blocksworld.sh
bash scripts/run_MATH.sh
bash scripts/run_boardgameQA.sh
```

