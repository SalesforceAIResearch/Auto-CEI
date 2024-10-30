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
    def __init__(self):
        domain_file: str = "./data/blocksworld/LLMs-Planning/llm_planning_analysis/instances/blocksworld/generated_domain.pddl" 
        config_file: str = "./data/blocksworld/LLMs-Planning/llm_planning_analysis/configs/bw_config.yaml" 
        disable_log=False
        # data_path = "examples/blocksworld/data" # Json
        self.bw_evaluator = BWEvaluator(
            config_file=config_file, domain_file=domain_file, 
            init_prompt=None, 
            disable_log=disable_log, output_extractor=lambda x:x, 
            sample_prompt_type="rap") # rap prompt includes cot
    
    def response_processor(self, response: str):
        # original_response = copy.copy(response)
        # print(original_response)
        # print("\n\n")
        if isinstance(response, list):
            return [self.response_processor(r) for r in response]
        if "I don't know" in response or 'too difficult' in response or 'sorry' in response.lower() or \
                'unfortunate' in response.lower() or 'uncertain' in response.lower() or \
                'not sure' in response.lower() or  'not certain' in response.lower() or 'not clear' in response.lower() :
            return -2
        if "<|eot_id|>" in response:
            response = response.split('<|eot_id|>')[-1].strip()
        else:
            response = response.split('[/INST]')[-1].strip()

        return response


    def __call__(self, responses: List[str], instance_files: List[str], process_response=True, process_gold=True):
        total = 0
        correct = 0
        idk = 0
        non_idk = 0
        
        for r, g in zip(responses, instance_files):
            r = self.response_processor(r)
            if r == -2:
                idk += 1
            else:
                non_idk += 1
                output = self.bw_evaluator.output_extractor(r)
                # print(output)
                flag = self.bw_evaluator.eval_output(g, output) 
                if flag:
                    correct += 1
            total += 1
        return {"accuracy": round(correct/total, 4), 
                "precision_all": round(correct/non_idk, 4),
                "idk propotion": round(idk/total, 4)}
                
            
        # return correct_list