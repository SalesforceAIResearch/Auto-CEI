from fastchat.model import get_conversation_template
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList, HfArgumentParser
import datasets
import json
import pandas as pd
import os
from tqdm import tqdm
import importlib.util
import sys
import random
import torch
from accelerate import Accelerator
accelerator = Accelerator()
from typing import Optional
from dataclasses import field, dataclass
from utils.utils import  generate_batched_greedy
from utils.utils import reward_processor
import numpy as np

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_path: Optional[str] = field(default='./data/models/proofwriter/llama-7b-cot-300-ind-rej/', metadata={"help": "the model path"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    model_name: Optional[str] = field(default='meta-llama/Meta-Llama-3.1-8B-Instruct', metadata={"help": "the model name"})
    num_samples: Optional[int] = field(default=10, metadata={"help": "the number of samples"})
    dataset_name: Optional[str] = field(default='train_ds_cot_ei', metadata={"help": "the dataset name"})
    task_name: Optional[str] = field(default='proofwriter', metadata={"help": "the task name"})  
    conservative_c1: Optional[float] = field(default=13, metadata={"help": "the conservative c1 parameter"})
    conservative_c2: Optional[float] = field(default=0.4, metadata={"help": "the conservative c2 parameter"})
    use_DDP: bool = field(default=True, metadata={"help": "whether to use DDP"})
    orm: bool = field(default=True, metadata={"help": "whether to use orm"})
    conservative_reward: bool = field(default=False, metadata={"help": "whether to use conservative reward"})
    resampling_temp: Optional[float] = field(default=0.6, metadata={"help": "the resampling temperature"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name)

stop_list = ["<|eot_id|>"]
stop_token_ids = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to('cuda') for x in stop_token_ids]

class StoppingCriteriaSub(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(self, args):
        if args.task_name == 'blocksworld':
            stop_list = ["<|eot_id|>", "####", "###"]
        elif args.task_name == "proofwriter": 
            stop_list = ["<|eot_id|>", "####", "###", "[END]"]  
        else:
            stop_list = ["<|eot_id|>"]
        eos_token_id = [tokenizer(x, add_special_tokens=False)['input_ids'] for x in stop_list]
        # eos_token_id = [torch.LongTensor(x).to('cuda') for x in stop_token_ids]
        # if isinstance(eos_token_id, int):
        #     eos_token_id = [eos_token_id]
        self.eos_token_id = torch.tensor(eos_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        if input_ids.device.type == "mps":
            # https://github.com/pytorch/pytorch/issues/77764#issuecomment-2067838075
            is_done = (
                input_ids[:, -1]
                .tile(self.eos_token_id.shape[0], 1)
                .eq(self.eos_token_id.unsqueeze(1).to(input_ids.device))
                .sum(dim=0)
                .bool()
                .squeeze()
            )
        else:
            is_done = torch.isin(input_ids[:, -1], self.eos_token_id.to(input_ids.device))
        return is_done
def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Can't create a module from the file path {file_path} provided.")

# prompt = module_from_file("prompt", "/mnt/hdd/kang/description_length_analysis/prompts/proofwriter.py")

model_path = script_args.model_path

# As per the peft instructions, make sure the lm_head is on gpu 0.  
# This works for Llama, not sure what to set for pythia models.
# device_map["lm_head"] = 0
init_dataset = {
    'MATH': "MATH_train",
    "boardgameQA": "train_argumented_ds", 
    "blocksworld": "blocksworld_dataset_sft"
}
init_questions_set = {
    'MATH': "MATH_train",
    "blocksworld": "blocksworld_dataset_sft", 
    "boardgameQA": "train_argumented_ds",
}
ds_cot = datasets.load_from_disk(f"./data/{script_args.task_name}/{init_dataset[script_args.task_name]}")
ds = datasets.load_from_disk(f"./data/{script_args.task_name}/{init_questions_set[script_args.task_name]}")


def build_dataset(ds):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        query_dataset (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    # ds = ds.train_test_split(train_size=0.1, seed=42)['train']
    # ds = ds.train_test_split(train_size=0.3, seed=42)['test']
    # print(len(ds))
    # ds = ds.rename_columns({"text": "review"})
    # ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)
    def tokenize(sample):
        conv = get_conversation_template(script_args.model_name)
        conv.messages = []
        conv.system = "[INST] "
        if script_args.task_name == 'proofwriter':
            conv.append_message(conv.roles[0], f"{sample['theory']}\nIs the statement\"{sample['question'][:-1]}\" True or False?")
        elif script_args.task_name == 'blocksworld': 
            conv.append_message(conv.roles[0], f"{sample['instance_id']}\n{sample['query']}")
        elif script_args.task_name == 'gsm' or script_args.task_name == 'gsm_small' or \
            'boardgameQA' in script_args.task_name or script_args.task_name == 'MATH':
            conv.append_message(conv.roles[0], f"{sample['prompt']}")
        else:
            raise NotImplementedError 
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()

        sample["input_ids"] = tokenizer.encode(query)
        sample["query"] = query
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.shuffle(seed=3)
    ds.set_format(type="torch")
    return ds


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(ds) 
# ds = load_from_disk(args.dataset_name)
# ds = ds.train_test_split(train_size=0.1, seed=42)['train']
# dataset = ds.train_test_split(train_size=0.1, seed=42)['train']


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
# response_template = get_conversation_template(args.model_name).roles[1]
tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=script_args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collator
        )
dataloader = accelerator.prepare(dataloader)

# from accelerate import infer_auto_device_map, init_empty_weights
# from transformers import AutoConfig, AutoModelForCausalLM

# config = AutoConfig.from_pretrained(script_args.model_name)
# with init_empty_weights():
#     model_ = AutoModelForCausalLM.from_config(config)

# max_memory = get_balanced_memory(model_,
#                                 low_zero=False, 
# )
# device_map = infer_auto_device_map(
#     model_,
#     max_memory=max_memory,
#     # Manually set the modules to not split based on the model.  
#     # The models' say but it's hard to figure out a this stage without doing it manually.
#     no_split_module_classes=["LlamaDecoderLayer"],
    
#     # dtype=dtype,
#     )
if script_args.use_DDP:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
else: 
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map="balanced")
model.eval()
model = accelerator.prepare(model)
index_gpu = model.device.index
evaluator_module = module_from_file("Evaluator", f"./src/rl/eval_{script_args.task_name}.py")
evaluator = evaluator_module.Evaluator(idx=index_gpu)
# model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(script_args)])
generation_kwargs = {
    "min_length": -1,
    "temperature": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "stopping_criteria": stopping_criteria,
    "num_return_sequences": script_args.num_samples,
    "max_new_tokens": 512,
    
}
if script_args.task_name == 'proofwriter':
    new_data_dict = {
        'theory': [],
        'question': [],
        'cot': [],
        'answer': [],
    }
elif script_args.task_name == 'blocksworld':
    new_data_dict = {
        'instance_id': [],
        'query': [],
        'reply': [],
        'ground_truth_plan': [], 
    }
elif script_args.task_name == 'gsm' or script_args.task_name == 'gsm_small' or \
    "boardgameQA" in script_args.task_name or script_args.task_name == 'MATH':
    new_data_dict = {
        'prompt' : [],
        'completion': [],
        'answer': [],
    }
else:
    raise NotImplementedError
# try:
idk_rate = 0
total_n = 0
idk_n = 0
pbar = tqdm(dataloader)
with torch.no_grad():
    for datum in pbar:
        query_tensors = [b.to(model.device) for b in datum["input_ids"]]
        responses = generate_batched_greedy(model, tokenizer, query_tensors, batch_size=script_args.batch_size, return_prompt=False, **generation_kwargs)
        if script_args.task_name == 'proofwriter':
            rewards = reward_processor([r.split('[END]')[0].replace('</s>', '').strip() for r in responses], 
                        [c.strip() for c in datum["cot"]  for _ in range(script_args.num_samples) ],
                        [a.strip() for a in datum["answer"]  for _ in range(script_args.num_samples) ], 
                        evaluator, script_args, pt=False)
        elif script_args.task_name == 'blocksworld':
            rewards = reward_processor([r.split('[/INST] \n')[-1].split('<|eot_id|>')[0].replace('</s>', '').strip() for r in responses],
                        None,
                        [f'./data/blocksworld/LLMs-Planning/llm_planning_analysis/instances/blocksworld/blocksworld/generated/instance-{int(a)}.pddl' for a in datum["instance_id"]  for _ in range(script_args.num_samples) ],
                        evaluator, script_args, pt=False)
        elif script_args.task_name == 'gsm' or script_args.task_name == 'gsm_small' or \
            "boardgameQA" in script_args.task_name or script_args.task_name == 'MATH':
            rewards = reward_processor([r.split('<|eot_id|>')[0].strip() if "####" in r else r.split('<|eot_id|>')[0].split('[/INST]')[-1].strip() for r in responses],
                        None,
                        [a.split('\n#### ')[-1].strip() for a in datum['answer']  for _ in range(script_args.num_samples) ],
                        evaluator, script_args, pt=False)
        else: 
            raise NotImplementedError 
        selected_rewards = [rewards[i: i + script_args.num_samples] for i in range(0, len(rewards), script_args.num_samples)]
        selected_response = [responses[i: i + script_args.num_samples] for i in range(0, len(rewards), script_args.num_samples)] 
        if script_args.resampling_temp == -1:
            sampled_index = [np.arange(len(sr)).tolist()    
                for sr in selected_rewards]
        else:
            sampled_index = [np.random.choice(np.arange(len(sr)), size=script_args.num_samples, replace=True, 
                p=np.exp(1/script_args.resampling_temp * np.array(sr))/np.sum(np.exp(1/script_args.resampling_temp * np.array(sr)))).tolist() + np.arange(len(sr)).tolist()    
                for sr in selected_rewards]
        # sampled_index = [np.random.choice(np.arange(len(sr)), size=script_args.num_samples, replace=True, 
        #                                   p=np.exp(script_args.resampling_temp * np.array(sr))/np.sum(np.exp(script_args.resampling_temp * np.array(sr)))) for sr in selected_rewards]
        # sampled_index = [list(set(idx)) for idx in sampled_index]
        sampled_responses = [[selected_response[i][j] for j in idx] for i, idx in enumerate(sampled_index)]
        sampled_rewards = [[selected_rewards[i][j] for j in idx] for i, idx in enumerate(sampled_index)]
        sampled_responses = accelerator.gather_for_metrics(sampled_responses) 
        sampled_rewards = accelerator.gather_for_metrics(sampled_rewards)
        gathered_datum = {}
        
        for key in new_data_dict:
            gathered_datum[key] = accelerator.gather_for_metrics(datum[key])

        for idx, sr, res in zip(range(len(sampled_responses)), sampled_rewards, sampled_responses):
            assert len(sr) == len(res), print(len(sr), len(res))
            for i in range(len(sr)):
                if sr[i] > -1:
                    total_n += 1
                    if sr[i] < 1:
                        idk_n += 1
                    if script_args.task_name == 'proofwriter':
                        new_data_dict['cot'].append(res[i])
                    elif script_args.task_name == 'blocksworld':
                        new_data_dict['reply'].append(res[i])
                    elif script_args.task_name == 'gsm' or script_args.task_name == 'gsm_small' \
                        or "boardgameQA" in script_args.task_name or script_args.task_name == 'MATH':
                        new_data_dict['completion'].append(res[i])
                    else:
                        raise NotImplementedError
                    for key in gathered_datum:
                        if key not in ['cot', 'reply', 'completion']:
                            new_data_dict[key].append(gathered_datum[key][idx])
        # For debug
        if total_n != 0:
            print(f"IDK rate: {idk_n/total_n}")
        else:
            print("No data")

for key in new_data_dict:
    print(len(new_data_dict[key]))
# for d in ds_cot:
#     for key in new_data_dict:
#         new_data_dict[key].append(d[key])
# for key in new_data_dict:
#     print(len(new_data_dict[key]))
# save dataset from dict
new_ds = datasets.Dataset.from_dict(new_data_dict)
new_ds.save_to_disk(script_args.dataset_name) 
                    
del model
torch.cuda.empty_cache()

