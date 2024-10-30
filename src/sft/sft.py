# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import List, Optional
import os

import torch
from accelerate import Accelerator
import datasets
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from fastchat.model import get_conversation_template


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    deepspeed: Optional[str] = field(default=None, metadata={"help": "Use deepspeed"})
    dataset_name: Optional[str] = field(default='dp/data/cot_explicit_ds_5000_n=7_-100-100', metadata={"help": "the dataset name"})
    run_name: Optional[str] = field(default="dp_cot_explicit_ds_5000_n=7_-100-100", metadata={"help": "Wandb run name"})
    output_dir: Optional[str] = field(default="dp/checkpoints/cot_explicit_ds_5000_n=7_-100-100", metadata={"help": "the output directory"})
    task_name: Optional[str] = field(default="logic", metadata={"help": "the task name"})
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct", metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    report_to: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=3e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="linear", metadata={"help": "the lr scheduler type"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=4096, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    use_rej : Optional[bool] = field(default=True, metadata={"help": "Use rejection sampling"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    peft_lora_r: Optional[int] = field(default=128, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=64, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=10, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    mixed_precision: Optional[str] = field(default="bf16", metadata={"help": "Mixed precision training"})
    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "Target modules for LoRA adapters"})
    use_DDP: Optional[bool] = field(default=True, metadata={"help": "Use DDP"})
    use_flash_attention_2: Optional[bool] = field(default=False, metadata={"help": "Use Flash Attention 2"})

    # wandb_project_name: Optional[str] = field(default="description_length", metadata={"help": "Wandb project name"})
    


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
# import wandb
# wandb.login()
# os.environ["WANDB_PROJECT"] = script_args.wandb_project_name

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

if not script_args.use_DDP:
    print("Using Accelerate")
    # from accelerate import infer_auto_device_map, init_empty_weights
    # from accelerate.utils import get_balanced_memory
    # from transformers import AutoConfig, AutoModelForCausalLM

    # config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    # with init_empty_weights():
    #     model_ = AutoModelForCausalLM.from_config(config)

    # max_memory = get_balanced_memory(model_,
    #                                 low_zero=True,
    # )
    # device_map = infer_auto_device_map(
    #     model_,
    #     # max_memory=max_memor05,
    #     # Manually set the modules to not split based on the model.  
    #     # The models' say but it's hard to figure out a this stage without doing it manually.
    #     no_split_module_classes=["LlamaDecoderLayer", "lm_head"],
        
    #     # dtype=dtype,
    #     )
    # device_map["lm_head"] = 0
    device_map = "balanced_low_0"

model_kwargs = dict(
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    use_flash_attention_2=script_args.use_flash_attention_2,
)
# model = AutoModelForCausalLM.from_pretrained(
#     script_args.model_name,
#     quantization_config=quantization_config,
#     device_map=device_map,
#     trust_remote_code=script_args.trust_remote_code,
#     torch_dtype=torch_dtype,
#     use_auth_token=script_args.use_auth_token,
#     use_flash_attention_2=script_args.use_flash_attention_2,
# )

# Step 2: Load the dataset
dataset = datasets.load_from_disk(script_args.dataset_name)
# dataset = dataset.shuffle(seed=42)
# dataset = dataset.select(range(2000))
# if 'iter_' not in script_args.dataset_name:
#     dataset = dataset.train_test_split(train_size=0.3, seed=42)['train']
# epoches_num = max(12000 // len(dataset), 2)
epoches_num = script_args.num_train_epochs
# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    lr_scheduler_type=script_args.lr_scheduler_type,
    logging_steps=script_args.logging_steps,
    num_train_epochs=epoches_num,
    max_steps=script_args.max_steps,
    ddp_find_unused_parameters=False,
    deepspeed=script_args.deepspeed,
    # report_to=script_args.report_to,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    gradient_checkpointing=script_args.gradient_checkpointing,
    run_name=script_args.run_name,
    # TODO: uncomment that on the next release
    # gradient_checkpointing_kwargs=script_args.gradient_checkpointing_kwargs,
)
if script_args.batch_size != -1:
    training_args.per_device_train_batch_size = script_args.batch_size


# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=script_args.target_modules,  # default is none, taken care by peft/utils/other.py:385
    )
else:
    peft_config = None

# Step 5: Define the Trainer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func_bw(example):
    output_texts = []
    for i in range(len(example['query'])):
        conv = get_conversation_template(script_args.model_name)
        conv.system = "[INST] "
        conv.messages = []
        conv.append_message(conv.roles[0], example['query'][i])
        conv.append_message(conv.roles[1], example['reply'][i])
        output_texts.append(conv.get_prompt())
    return output_texts

def formatting_prompts_func_word(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        conv = get_conversation_template(script_args.model_name)
        conv.system = "[INST] "
        conv.messages = []
        conv.append_message(conv.roles[0], example['prompt'][i])
        conv.append_message(conv.roles[1], example['completion'][i])
        output_texts.append(conv.get_prompt())
    return output_texts

response_template_with_context = "\n### Assistant:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`

collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

if script_args.task_name == "blocksworld":
    formatting_prompts_func = formatting_prompts_func_bw
elif script_args.task_name == "gsm" or script_args.task_name == "gsm_small" or \
    'boardgameQA' in script_args.task_name or 'MATH' == script_args.task_name:
    formatting_prompts_func = formatting_prompts_func_word
else:
    raise ValueError(f"Task name {script_args.task_name} not recognized")
trainer = SFTTrainer(
    # model=model,
    model=script_args.model_name,
    model_init_kwargs=model_kwargs,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)