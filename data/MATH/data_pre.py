import datasets

ds = datasets.load_dataset("hendrycks/competition_math")

print(len(ds['train']))
print(len(ds['test']))

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
    
    return retval, string


def remove_boxed(s, s_original):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        if "\\boxed" in s_original:
            answer = s_original.split("\\boxed")[-1].strip().split("$")[0]
        return answer

def modify(s):
    s, s_original = last_boxed_only_string(s)
    return remove_boxed(s, s_original)

ds_train_val = ds['train'].map(lambda x: {
        'prompt': x['problem'] +"\nThink step by step and use the format of '#### \{final answer\}. <|eot_id|>' to complete your answer.", 
        'answer': modify(x['solution']),
        'completion': x['solution'].replace(". ", ".\n") + f"\n#### {modify(x['solution'])}. <|eot_id|>", 
    })



ds_test = ds['test'].map(lambda x: {
        'prompt': x['problem'] +"\nThink step by step and use the format of '#### \{final answer\}. <|eot_id|>' to complete your answer.", 
        'answer': modify(x['solution']),
        'completion': x['solution'].replace(". ", ".\n") + f"\n#### {modify(x['solution'])}. <|eot_id|>", 
    })
ds_train_val = ds_train_val.shuffle(seed=42)

ds_train_val = ds_train_val.train_test_split(train_size=0.9, seed=42)

ds_train_val['train'].save_to_disk("data/MATH/MATH_train")
ds_train_val['test'].save_to_disk("data/MATH/MATH_val")
ds_test.save_to_disk("data/MATH/MATH_test")
