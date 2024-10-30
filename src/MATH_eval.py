# from .base import BaseEvaluator
import re
from typing import List
import datasets
import string
from collections import Counter
import copy
def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        if "\\boxed" in s:
            answer = s.split("\\boxed")[-1].strip().split("$")[0]
        return answer

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
        pass
    def response_processor(self, response: str):
        original_response = copy.copy(response)
        # print("\n\n")
        if isinstance(response, list):
            return [self.response_processor(r) for r in response]
        if "I don't know" in response or 'too difficult' in response or 'sorry' in response.lower() or \
                'unfortunate' in response.lower() or 'uncertain' in response.lower() or \
                'not sure' in response.lower() or  'not certain' in response.lower() or 'not clear' in response.lower() :
            return -2
        if "####" in response:
            response = response.split('\n#### ')[-1].strip()
        elif '\\box' in response:
            resp = last_boxed_only_string(response)
            if resp is None:
                return -1
            response = remove_boxed(resp)
            if response is None:
                response = resp 
        if response is None:
            print(original_response)
            return -1
        if "no answer" in response.lower():
            return -2
        else:
            return response
    # def response_processor(self, response: str):
    #     original_response = copy.copy(response)
    #     # print("\n\n")
    #     if isinstance(response, list):
    #         return [self.response_processor(r) for r in response]
    #     reason = original_response
    #     if "I don't know" in reason or 'too difficult' in reason or 'sorry' in reason.lower() or \
    #             'unfortunate' in reason.lower() or 'uncertain' in reason.lower() or \
    #             'not sure' in reason.lower() or  'not certain' in reason.lower() or 'not clear' in reason.lower() :
    #         # print(reason)
    #         return -2
    #     if "####" in response:
    #         responses = response.split('\n#### ')
    #         reason = responses[-1]
    #         # response = response.split(".")[0]
    #     elif 'final answer' in response:
    #         response = response.split("final answer")[-1]
    #     elif "###" in response:
    #         response = response.split("###")[0]
    #     if '\\box' in response:
    #         resp = last_boxed_only_string(response)
    #         if resp is None:
    #             return -1
    #         response = remove_boxed(resp)
    #         if response is None:
    #             response = resp 
    #     if response is None:
    #         # print(original_response)
    #         return -1
    #     if "no answer" in response.lower():
    #         return -2
    #     else:
    #         return response

    
    def gold_processor(self, gold: str):
        if not isinstance(gold, str):
            return [self.gold_processor(g) for g in gold]
        gold = gold.split("####")[-1].strip().lower()
        return gold

    def __call__(self, responses: List[str], golds: List[str], 
                 process_response=True, process_gold=True, return_list=False):
        total = correct = 0
        none_num = 0
        idk = 0
        non_idk = 0
        if return_list:
            result_list = []
        for r, g in zip(responses, golds):
            if process_response:
                tmp_r = copy.copy(r)
                r = self.response_processor(r)
                if r is None:
                    print(f"==== answer cannot be detected. response: {tmp_r}====")
            if process_gold:
                g = self.gold_processor(str(g))
            if isinstance(r, str) and g in r:
                correct += 1
                if return_list:
                    result_list.append(1)
            elif r is None or r != -2:
                none_num += 1
                if return_list:
                    result_list.append(0)
            if r == -2:
                idk += 1
                if return_list:
                    result_list.append(-2)
            else:
                non_idk += 1
            total += 1
            # precision and recall when gold is not None
        # round the result to 4 decimal places
        print(f"correct: {correct}, total: {total}, idk: {idk}, non_idk: {non_idk}, totol: {total}")
        if return_list:
            return result_list, {'accuracy': round(correct / total, 4),
                'precision_all': round(correct / non_idk, 4),
                'idk propotion': round(idk / total, 4)}
        return {'accuracy': round(correct / total, 4),
                'precision_all': round(correct / non_idk, 4),
                'idk propotion': round(idk / total, 4)}