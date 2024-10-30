from dataclasses import dataclass, field
from typing import Optional
from src.gpt.gpt import OpenAIModel
from data.boardgameQA.prompt import prompt_unknown
from transformers import HfArgumentParser
import datasets
import tqdm

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_name: Optional[str] = field(default='gpt-4-turbo', metadata={"help": "the model name"})
    stop_words: Optional[str] = field(default='<|eot_id|>', metadata={"help": "the stop words"})
    max_new_tokens: Optional[int] = field(default=512, metadata={"help": "the max new tokens"})
    dataset_path: Optional[str] = field(default='data/boardgameQA_small/train_ds', metadata={"help": "the dataset path"})

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

openai_api = OpenAIModel(args.model_name, args.stop_words, args.max_new_tokens)

data = datasets.load_from_disk(args.dataset_path)
data_new = {
    'prompt': [],
    'completion': [],
    'answer': []
}

pbar = tqdm.tqdm(data)

for ele in pbar:
    data_new['prompt'].append(ele["prompt"])
    data_new['answer'].append(ele['answer'])
    if ele['answer'] == 'unknown':
        response = openai_api.chat_generate(prompt_unknown.format(ele["prompt"]))
        response = ".\n".join(response.split(". ")) + "<|eot_id|>"
        data_new['completion'].append(response.replace("\n\n", '\n'))
    else:
        data_new['completion'].append(ele['completion'])

data_new = datasets.Dataset.from_dict(data_new)
data_new.save_to_disk(args.dataset_path.replace("train_", "train_augmented_"))