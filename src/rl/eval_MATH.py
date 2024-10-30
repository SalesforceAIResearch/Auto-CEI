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
        
    def __call__(self, responses: List[str], golds: List[str], process_response=True, process_gold=True):
        total = 0   
        correct_list = []
        for r, g in zip(responses, golds):
            if "####" in r:
                r = r.split('\n#### ')[-1].strip()
            else:
                r = r.split('[/INST]')[-1].strip()
            if g in r:
                correct_list.append(1.0)
            else:
                correct_list.append(0.0)
            total += 1
        assert total == len(correct_list)
            
        return correct_list