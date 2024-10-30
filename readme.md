# Automatic Curriculum Expert Iteration

Zirui Zhao, Hanze Dong, Amrita Saha, Caiming Xiong, Doyen Sahoo

## Introduction
Hallucinations (i.e., generating plausible but inaccurate content) and laziness (i.e. excessive refusals or defaulting to "I don't know") persist as major challenges in LLM reasoning. Current efforts to reduce hallucinations primarily focus on factual errors in knowledge-grounded tasks, often neglecting hallucinations related to faulty reasoning. Meanwhile, some approaches render LLMs overly conservative, limiting their problem-solving capabilities. To mitigate hallucination and laziness in reasoning tasks, we propose Automatic Curriculum Expert Iteration (Auto-CEI) to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them. In our method, Expert Iteration explores the reasoning trajectories near the LLM policy, guiding incorrect paths back on track to reduce compounding errors and improve robustness; it also promotes appropriate "I don't know" responses after sufficient reasoning attempts. The curriculum automatically adjusts rewards, incentivizing extended reasoning before acknowledging incapability, thereby pushing the limits of LLM reasoning and aligning its behaviour with these limits. We compare Auto-CEI with various SOTA baselines across logical reasoning, mathematics, and planning tasks, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness.

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


## Reference

If you found it useful, please cite
```bibtex
@article{zhao2024automatic,
  title={Automatic Curriculum Expert Iteration for Reliable LLM Reasoning},
  author={Zhao, Zirui and Dong, Hanze and Saha, Amrita and Xiong, Caiming and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2410.07627},
  year={2024}
}
```
