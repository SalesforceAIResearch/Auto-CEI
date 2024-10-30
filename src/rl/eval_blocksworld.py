# from .base import BaseEvaluator
import re
from typing import List
import datasets
import string
from collections import Counter
import copy
from src.utils.bw_utils import BWEvaluator
# output = evaluator.output_extractor(response)
# correct = evaluator.eval_output(datum, output)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class Evaluator:
    def __init__(self, idx=0):
        domain_file: str = "./data/blocksworld/LLMs-Planning/llm_planning_analysis/instances/blocksworld/generated_domain.pddl" 
        config_file: str = "./data/blocksworld/LLMs-Planning/llm_planning_analysis/configs/bw_config.yaml" 
        disable_log=False
        self.bw_evaluator = BWEvaluator(
            config_file=config_file, domain_file=domain_file, 
            init_prompt=None, 
            disable_log=disable_log, output_extractor=lambda x:x, 
            sample_prompt_type="rap",
            index=idx) # rap prompt includes cot
    

    def __call__(self, responses: List[str], instance_files: List[str], process_response=True, process_gold=True):

        correct_list = []
        for r, g in zip(responses, instance_files):
            output = self.bw_evaluator.output_extractor(r)
            correct = self.bw_evaluator.eval_output(g, output) 
            correct_list.append(float(correct))
            
        return correct_list