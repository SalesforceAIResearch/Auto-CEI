from fastchat.model import get_conversation_template
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList, HfArgumentParser
import datasets
from accelerate import Accelerator
accelerator = Accelerator()
import json
import pandas as pd
import os
from tqdm import tqdm
import importlib.util
import sys
import random
import torch
from typing import Optional
from dataclasses import field, dataclass
from src.utils.utils import generate_batched_greedy

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_path: Optional[str] = field(default='', metadata={"help": "the model path"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    model_name: Optional[str] = field(default='meta-llama/Meta-Llama-3.1-8B-Instruct', metadata={"help": "the model name"})
    dataset_path: Optional[str] = field(default='data/MATH/MATH_test', metadata={"help": "the dataset path"})
    use_sampling: Optional[bool] = field(default=False, metadata={"help": "use sampling"})
    sample_size: Optional[int] = field(default=8, metadata={"help": "sample size"}) 

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

    def __init__(self):
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

evaluator_module = module_from_file("Evaluator", "./src/MATH_eval.py")
evaluator = evaluator_module.Evaluator()
model_path = script_args.model_path
if len(model_path) == 0:
    model_path = f"meta-llama/Meta-Llama-3.1-8B-Instruct"
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils import get_balanced_memory
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(script_args.model_name)
with init_empty_weights():
    model_ = AutoModelForCausalLM.from_config(config)

max_memory = get_balanced_memory(model_,
                                low_zero=False,
)
device_map = infer_auto_device_map(
    model_,
    max_memory=max_memory,
    # Manually set the modules to not split based on the model.  
    # The models' say but it's hard to figure out a this stage without doing it manually.
    no_split_module_classes=["LlamaDecoderLayer"],
    
    # dtype=dtype,
    )
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)  
# model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = transformers.AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
# load the mod
model = accelerator.prepare(model)
# As per the peft instructions, make sure the lm_head is on gpu 0.  
# This works for Llama, not sure what to set for pythia models.
# device_map["lm_head"] = 0


ds = datasets.load_from_disk(script_args.dataset_path)

print(len(ds)) 

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
        conv.system = "[INST] "
        conv.messages = []
        conv.append_message(conv.roles[0], sample['prompt']) 
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        sample["input_ids"] = tokenizer.encode(query)
        return sample

    ds = ds.map(tokenize, batched=False)
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
            collate_fn = collator
        )
dataloader = accelerator.prepare(dataloader)

if script_args.model_path == '' :
    model_path = f"./data/models/MATH/llama3.1-8b-instruct-ei-baseline-0.2-128-64/iter_0/"
results = [] if not os.path.exists(os.path.join(model_path, 'results.json')) else json.load(open(os.path.join(model_path, 'results.json')))
pbar = tqdm(dataloader)

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub()])
generation_kwargs = {
    "min_length": -1,
    "do_sample": script_args.use_sampling,
    "pad_token_id": tokenizer.eos_token_id,
    "stopping_criteria": stopping_criteria,
    "num_return_sequences": script_args.sample_size if script_args.use_sampling else 1,
    "max_new_tokens": 512,
}
if script_args.use_sampling:
    generation_kwargs["temperature"] = 1.0

try:
    with torch.no_grad():
        for i, datum in enumerate(pbar):
            query_tensors = [b.to(model.device) for b in datum["input_ids"]]
            responses = generate_batched_greedy(model, tokenizer, query_tensors, batch_size=script_args.batch_size, return_prompt=False, **generation_kwargs)
            # response = tokenizer.batch_decode(model.generate(query_tensors, stopping_criteria=stopping_criteria, max_new_tokens=1024))[0]
            datum['response'] = responses
            if i == len(results):
                results.append({
                    'response': datum['response'],
                    'answer': datum['answer'],
                })
            elif i < len(results):
                results[i] = {
                    'response': datum['response'],
                    'answer': datum['answer'],
                }
            else:
                raise ValueError("Something went wrong")
            responses = accelerator.gather_for_metrics(responses) 
            results = accelerator.gather_for_metrics(results)
            # TODO: evaluator
            if script_args.use_sampling: 
                eval_result = f"{evaluator([r_.split('<|eot_id|>')[0].replace('</s>', '').strip() for r in results[:i+1] for r_ in r['response']], [r_ for r in results[:i+1] for r_ in r['answer'] for _ in range(script_args.sample_size)])}"
            else:
                eval_result = f"{evaluator([r_.split('<|eot_id|>')[0].replace('</s>', '').strip() for r in results[:i+1] for r_ in r['response']], [r_ for r in results[:i+1] for r_ in r['answer']])}"
            pbar.set_description(eval_result)
            # save eval_result in model_path/eval_result.txt



except KeyboardInterrupt:
    json.dump(results, open(os.path.join(model_path, f'results_ind.json'), 'w'))
if 'test' in script_args.dataset_path:
    with open(os.path.join(model_path, f'eval_result_test.txt'), 'w') as f:
        f.write(eval_result)
else:
    with open(os.path.join(model_path, f'eval_result_ind.txt'), 'w') as f:
        f.write(eval_result)
json.dump(results, open(os.path.join(model_path, f'results_ind.json'), 'w'))
    # print(model_path)
    # print(evaluator([r['response'].split('[/INST]')[-1].replace('</s>', '').strip() for r in results], [r['completion'].split('? = ')[-1] for r in results]))
del model
torch.cuda.empty_cache()
