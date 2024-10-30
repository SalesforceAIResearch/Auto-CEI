import numpy as np
import torch

from typing import Callable, List, Optional, Union

from trl.models import (
    PreTrainedModelWrapper,
)


def generate_batched_greedy(
    model: PreTrainedModelWrapper,
    tokenizer,
    query_tensors: List[torch.Tensor],
    length_sampler: Optional[Callable] = None,
    batch_size: int = 4,
    return_prompt=False,
    pad_to_multiple_of: Optional[int] = None,
    remove_padding: bool = True,
    **generation_kwargs,
):
    outputs = []

    padding_side_default = tokenizer.padding_side
    # if not ppo_trainer.is_encoder_decoder:
    tokenizer.padding_side = "left"

    # in case we have fewer examples than bs
    batch_size = min(len(query_tensors), batch_size)

    for i in range(0, len(query_tensors), batch_size):
        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()

        # prevent overflow if query tensors are not even multiple of bs
        end_index = min(len(query_tensors), i + batch_size)

        batch = query_tensors[i:end_index]
        batch_mask = [torch.ones_like(element) for element in batch]
        inputs = {"input_ids": batch, "attention_mask": batch_mask}

        padded_inputs = tokenizer.pad(
            inputs,
            padding=True,
            max_length=None,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        ).to(model.device)
        # with torch.cuda.amp.autocast():
        num_samples = generation_kwargs['num_return_sequences']
        if num_samples > 8:
            generation_kwargs['num_return_sequences'] = 8
            num_steps_sampling = num_samples // 8
            generations = []
            for _ in range(num_steps_sampling):
                generations_ = model.generate(**padded_inputs, **generation_kwargs)
                generations.append(generations_)
            generations = [g for j in range(batch_size) for i in range(num_steps_sampling) for g in generations[i][j*8:j*8+8]]
            
        else: 
            generations = model.generate(**padded_inputs, **generation_kwargs)
        for i, generation in enumerate(generations):
            output = generation[(1 - padded_inputs["attention_mask"][i//num_samples]).sum() :]
            output = output[(padded_inputs["attention_mask"][i//num_samples]).sum() :]
            if remove_padding and tokenizer.eos_token_id in output:
                pad_mask = output == tokenizer.eos_token_id
                pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                output = output[:pad_start + 1]
            outputs.append(output)
    tokenizer.padding_side = padding_side_default
    response = tokenizer.batch_decode(outputs)
    return response

def sigmoid(x, c1, c2):
    return 2 / (1 + np.exp(-c2 * (x - c1))) - 1


def reward_processor(response, cot, answer, evaluator: callable, args, pt=True):
    '''
    reward function, human annotated
    options: vanila orm, vanila prm, conservative orm, conservative prm
    evaluator: (response: str, answer: str) -> dict, take 'accuracy' as the key
    '''
    # evaluator: accuracy
    score = evaluator(response, answer)
    for i, r in enumerate(response):
        if score[i] == 1.0:
            score[i] *= 1.0
        if args.conservative_reward:
            if "i don't know" in r.lower() or 'sorry' in r.lower() or \
            'too difficult' in r.lower() or 'unfortunate' in r.lower() or 'no answer' in r.lower():
                if args.task_name == 'blocksworld':
                    depth = len(r.lower().split("sorry")[0].split('since'))
                elif args.task_name == "MATH":
                    depth = len(r.split('.\n'))
                else:
                    depth = len(r.split('\n'))
                depth_score = sigmoid(depth, args.conservative_c1, args.conservative_c2)
                score[i] = depth_score
            
            elif score[i] != 1.0:
                score[i] = -1.0
        elif score[i] != 1.0:
            if score[i] == 0:
                score[i] = -1.0
            else:
                score[i] = -2
    if pt: 
        return [torch.tensor(s) for s in score]
    else:
        return score


        