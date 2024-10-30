# from .base import BaseEvaluator
import re
from typing import List
import datasets
import string
from collections import Counter
import copy

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
    def __init__(self, idx = 0):
        pass
    
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
        response_answer = response.split("<|eot_id|>")[0]
        if ' proved' in response_answer or "\"yes\"" in response_answer:
            return True
        elif "disproved" in response_answer or "\"no\"" in response_answer:
            return False
        elif "not provable" in response_answer or "\"unknown\"" in response_answer:
            return -1
        else:
            return None



    
    def gold_processor(self, gold: str):
        if not isinstance(gold, str):
            return [self.gold_processor(g) for g in gold]
        if gold == 'proved':
            return True
        elif gold == 'disproved':
            return False
        else: 
            return -1
        
    def __call__(self, responses: List[str], golds: List[str], process_response=True, process_gold=True):
        total = 0   
        correct_list = []
        for r, g in zip(responses, golds):
            if process_response:
                tmp_r = copy.copy(r)
                r = self.response_processor(r)
                # if r is None:
                #     print(f"==== answer cannot be detected. response: {tmp_r}====")

            if process_gold:
                g = self.gold_processor(str(g))
            if r == g:
                correct_list.append(1.0)
            elif r is None:
                correct_list.append(-1.0)
            else:
                correct_list.append(0.0)
            total += 1
        assert total == len(correct_list)
            
        return correct_list