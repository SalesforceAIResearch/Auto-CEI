import os
import json
import importlib.util
from dataclasses import field, dataclass
from typing import Optional
from transformers import HfArgumentParser
import math

init_dataset = {
    "MATH": "MATH_train",
    "blocksworld": "blocksworld_dataset_q",
    "boardgameQA": "train_argumented_ds", 
}
validation_dataset = {
    "MATH": "MATH_val",
    "blocksworld": "blocksworld_dataset_val",
    "boardgameQA": "valid_ds",
}


def from_string_to_dict(string):
    string = string.strip()
    # remove {} from the string
    string = string[1:-1]
    string = string.replace("'", "")
    
    return dict(map(lambda x: x.split(': '), string.split(', ')))

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Can't create a module from the file path {file_path} provided.")


def get_objective_function(precision, idk_rate, Lambda):
    return (1 - Lambda) * precision + Lambda * (1 - idk_rate)

def if_converge(new_f, curr_f, threshold=0.003):
    return abs(new_f - curr_f) < threshold

def test_c1(EI_MODEL_PATH, args, model_name, LAMBDA):
    total_steps = []
    with open(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA,  args.lora_r, args.lora_alpha, 0, 'results_ind.json')) as f:
        # turns json file into dictionary
        results = json.load(f)
    num_steps = []
    for result in results:
        responses = result['response']
        for i, response in enumerate(responses):
            if script_args.task_name == 'blocksworld':
                if "sorry" in response.lower() or "i don't know" in response.lower():
                    response = response.lower().split("sorry")[0]
                    num_steps.append(len(response.split("since")))
                else:
                    response = response.lower().split("goal conditions are satisfied")[0]
                total_steps.append(len(response.split("since")))
            else:
                if "sorry" in response.lower() or "i don't know" in response.lower():
                    num_steps.append(len(response.split(".\n")))
                total_steps.append(len(response.split(".\n")))
                
    mean = sum(num_steps) / len(num_steps)
    total_mean = sum(total_steps) / len(total_steps)
    total_std = (sum([(x - total_mean) ** 2 for x in total_steps]) / len(total_steps)) ** 0.5
    step = min(total_std * 0.2, 0.5)
    c2 = - math.log(0.1) / total_std /4
    return mean + step, step, c2

def initialisation(EI_INITIAL_MODEL_PATH, args, model_name, LAMBDA, num_gpus, gpu_memory, log_file):
    iter_idx = 0
    if not os.path.exists(EI_INITIAL_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, 0, 'eval_result_ind.txt')):
        os.system(f"""
        export PYTHONPATH="$PWD"
        export CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}
        torchrun --nproc_per_node {num_gpus} src/sft/sft.py --dataset_name "./data/{args.task_name}/{init_dataset[args.task_name]}/" \
            --run_name llama-7b-cot-300-ind \
            --output_dir "./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_0"\
            --batch_size {int(gpu_memory/40)} \
            --use_rej True \
            --deepspeed config/deepspeed.json \
            --task_name {args.task_name} \
            --gradient_checkpointing {args.gradient_checkpointing} \
            --gradient_accumulation_steps {int(256 / num_gpus)} \
            --num_train_epochs {10 if 'small' in args.task_name or args.task_name == 'blocksworld' else 6}
        python3 src/sft/run_{args.task_name}.py  --model_path "./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_0" \
            --dataset_path "./data/{args.task_name}/{validation_dataset[args.task_name]}" \
            --batch_size {int(gpu_memory * num_gpus/40)}
        """) 
    
    with open(EI_INITIAL_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, "eval_result_ind.txt")) as f:
        lines = f.readlines()[0]
        # turns lines into dictionary
        eval_result = from_string_to_dict(lines)
    # current_c1, step_size, c2 = test_c1()
    # log_file.write(f"current_c1: {current_c1}\n")
    log_file.flush()
    current_precision = float(eval_result['precision_all'])
    current_idk_rate = float(eval_result['idk propotion'])
    current_f = get_objective_function(current_precision, current_idk_rate, LAMBDA)     
    log_file.write(f"precision: {current_precision}, idk_rate: {current_idk_rate}\ncurrent f: {current_f}\n") 
    log_file.flush()
    iter_idx += 1
    temp = current_precision
    ei_iter_nums = 0
    while True:
        log_file.write(f"iter_idx: {iter_idx}\n")
        log_file.flush()
        if not os.path.exists(EI_INITIAL_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'eval_result_ind.txt')):
            try:
                if not os.path.exists(f"./data/{args.task_name}/train_ds_ei_baseline_{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}/state.json"):
                    os.system(f"""
                    export PYTHONPATH="$PWD"
                    export CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}
                    torchrun --nproc_per_node {num_gpus}  src/rl/sft_patching_generation.py  \
                        --model_path "./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx-1}" \
                        --dataset_name "./data/{args.task_name}/train_ds_ei_baseline_{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                        --num_samples {args.n_samples} \
                        --task_name {args.task_name} \
                        --resampling_temp {temp} \
                        --batch_size {int(gpu_memory/40)} 
                        """)
                os.system(f"""
                export PYTHONPATH="$PWD"
                export CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}
                torchrun --nproc_per_node {num_gpus} src/sft/sft.py --dataset_name "./data/{args.task_name}/train_ds_ei_baseline_{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                    --deepspeed config/deepspeed.json \
                    --gradient_checkpointing {args.gradient_checkpointing} \
                    --run_name llama-7b-cot-300-ind-rej \
                    --output_dir "./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                    --task_name {args.task_name} \
                    --batch_size {int(gpu_memory/40)} \
                    --use_rej True \
                    --gradient_accumulation_steps {int(256 / num_gpus)} \
                    --num_train_epochs 1
                
                python3 src/sft/run_{args.task_name}.py  --model_path "./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                    --dataset_path "./data/{args.task_name}/{validation_dataset[args.task_name]}" \
                    --batch_size {int(gpu_memory * num_gpus/40)}
                python3 src/sft/run_{args.task_name}.py  --model_path "./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}"  \
                    --batch_size {int(gpu_memory * num_gpus/40)} 
                """) 
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                exit(0)
        
        with open(EI_INITIAL_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'eval_result_ind.txt')) as f:
            lines = f.readlines()[0]
            # turns lines into dictionary
            eval_result = from_string_to_dict(lines)
        new_precision = float(eval_result['precision_all'])
        new_idk_rate = float(eval_result['idk propotion'])
        new_f = get_objective_function(new_precision, new_idk_rate, LAMBDA) 
        log_file.write(f"precision: {new_precision}, idk_rate: {new_idk_rate}, new_f: {new_f}\n")
        ei_iter_nums += 1
        if new_idk_rate > args.idk_threshold:
            log_file.write(f"new_idk_rate > args.idk_threshold\n")
            log_file.flush()
            break 

        elif ei_iter_nums >= 1:
            log_file.write(f"ei_iter_nums >= 1\n")
            log_file.flush()
            break
        iter_idx += 1
    
    return iter_idx, temp


def ei_step(iter_idx, EI_MODEL_PATH, args, model_name, LAMBDA, current_c1,
            c2, temp, num_gpus, gpu_memory, log_file, current_precision, current_idk_rate):
    ei_iter_nums = 0
    best_iter = 0
    best_f = 0
    while True:
        log_file.write(f"iter_idx: {iter_idx}\n")
        log_file.write(f"ei_iter_nums: {ei_iter_nums}\n")
        log_file.flush()
        if not os.path.exists(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'eval_result_ind.txt')):
            try:
                if not os.path.exists(f"./data/{args.task_name}/train_ds_auto_cei_{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}/state.json"):
                    os.system(f"""
                    export PYTHONPATH="$PWD"
                    export CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}
                    torchrun --nproc_per_node {num_gpus}  src/rl/ei_generation.py  \
                        --model_path "./data/models/{args.task_name}/{model_name}-auto-cei-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx-1}" \
                        --dataset_name "./data/{args.task_name}/train_ds_auto_cei_{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                        --num_samples {args.n_samples} \
                        --task_name {args.task_name} \
                        --conservative_reward True \
                        --conservative_c1 {current_c1} \
                        --conservative_c2 {c2} \
                        --resampling_temp {temp} \
                        --batch_size {int(gpu_memory/40)} 
                    """)
                if not os.path.exists(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'adapter_config.json')):
                    os.system(f"""
                    torchrun --nproc_per_node {num_gpus} src/sft/sft.py --dataset_name "./data/{args.task_name}/train_ds_auto_cei_{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                        --deepspeed config/deepspeed.json \
                        --gradient_checkpointing {args.gradient_checkpointing} \
                        --run_name llama-7b-cot-300-ind-rej \
                        --output_dir "./data/models/{args.task_name}/{model_name}-auto-cei-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}"\
                        --task_name {args.task_name} \
                        --batch_size {int(gpu_memory/40)} \
                        --use_rej True \
                        --peft_lora_r {args.lora_r} \
                        --peft_lora_alpha {args.lora_alpha} \
                        --gradient_accumulation_steps {int(256 / num_gpus)} \
                        --num_train_epochs 1
                        """)
                os.system(f""" 
                python3 src/sft/run_{args.task_name}.py  --model_path "./data/models/{args.task_name}/{model_name}-auto-cei-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}" \
                    --batch_size {int(gpu_memory * num_gpus/40)} \
                    --dataset_path "./data/{args.task_name}/{validation_dataset[args.task_name]}" 
                """) 
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                exit(0)
        
        with open(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'eval_result_ind.txt')) as f:
            lines = f.readlines()[0]
            # turns lines into dictionary
            eval_result = from_string_to_dict(lines)
        new_precision = float(eval_result['precision_all'])
        new_idk_rate = float(eval_result['idk propotion'])
        new_f = get_objective_function(new_precision, new_idk_rate, LAMBDA) 
        log_file.write(f"precision: {new_precision}, idk_rate: {new_idk_rate}, new_f: {new_f}\n")
        ei_iter_nums += 1
        if new_f > best_f:
            best_iter = iter_idx
            best_f = new_f
        if if_converge(new_precision, current_precision) and if_converge(new_idk_rate, current_idk_rate):
            new_ei_f_score = best_f
            log_file.write(f"converge\n")
            log_file.flush()
            for i in range(best_iter + 1, iter_idx + 1):
                if os.path.exists(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, i, 'eval_result_ind.txt')):
                    # remove the file
                    os.remove(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, i, 'eval_result_ind.txt'))
            iter_idx = best_iter + 1
            break
        elif ei_iter_nums >= 3:
            new_ei_f_score = best_f
            log_file.write(f"ei_iter_nums >= 3\n")
            log_file.flush()
            iter_idx = best_iter + 1
            break

        iter_idx += 1
    return new_ei_f_score, iter_idx


def auto_cei_hill_climbing(args):

    LAMBDA = args.Lambda
    EI_MODEL_PATH = "./data/models/{}/{}-auto-cei-{}-{}-{}/iter_{}/{}"
    EI_INITIAL_MODEL_PATH = "./data/models/{}/{}-ei-baseline-{}-{}-{}/iter_{}/{}"
    if not os.path.exists(f"logs/ei/{args.task_name}"):
        os.makedirs(f"logs/ei/{args.task_name}")
    log_file = open(f"logs/ei/{args.task_name}/log-{LAMBDA}.txt", "w")
    model_name = "llama3.1-8b-instruct"
    iter_idx = 0
    num_gpus = args.num_gpus
    gpu_memory = args.gpu_memory

    iter_idx, temp = initialisation(EI_INITIAL_MODEL_PATH, args, model_name, LAMBDA, num_gpus, gpu_memory, log_file)

    with open(EI_INITIAL_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, "eval_result_ind.txt")) as f:
        lines = f.readlines()[0]
        # turns lines into dictionary
        eval_result = from_string_to_dict(lines)
    os.system(f"""
    mkdir -p ./data/models/{args.task_name}/{model_name}-auto-cei-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_0/
    cp -r ./data/models/{args.task_name}/{model_name}-ei-baseline-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx}/* ./data/models/{args.task_name}/{model_name}-auto-cei-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_0
    """)
    iter_idx = 0
    current_c1, step_size, c2 = test_c1(EI_MODEL_PATH, args, model_name, LAMBDA)
    log_file.write(f"current_c1: {current_c1}\n")
    log_file.flush()
    current_precision = float(eval_result['precision_all'])
    current_idk_rate = float(eval_result['idk propotion'])
    current_f = get_objective_function(current_precision, current_idk_rate, LAMBDA)     
    log_file.write(f"precision: {current_precision}, idk_rate: {current_idk_rate}\ncurrent f: {current_f}\n") 
    log_file.flush()
    curr_ei_f_score = 0
    best_iter_idx = 0
    iter_idx += 1
    c1_to_best_f_score = {}
    best_c1 = current_c1
    move_direction = 1
    while True:
        new_ei_f_score, iter_idx = ei_step(iter_idx, EI_MODEL_PATH, args, model_name, LAMBDA, current_c1, c2, temp, num_gpus, gpu_memory, log_file, current_precision, current_idk_rate)
        
        os.system(f"""python3 src/sft/run_{args.task_name}.py  --model_path "./data/models/{args.task_name}/{model_name}-auto-cei-{LAMBDA}-{args.lora_r}-{args.lora_alpha}/iter_{iter_idx - 1}" \
                        --batch_size {int(gpu_memory * num_gpus/40)} 
                    """)
        move_direction = int(new_ei_f_score > curr_ei_f_score)* 2 - 1
        new_c1 = current_c1 + move_direction * step_size
        if new_ei_f_score > curr_ei_f_score:
            current_c1 = new_c1
            curr_ei_f_score = new_ei_f_score
            c1_to_best_f_score[current_c1] = curr_ei_f_score
            best_iter_idx = iter_idx
            best_c1 = current_c1
            log_file.write(f"current_f: {curr_ei_f_score}\n")
            log_file.write(f"new_c1: {new_c1}\n")
            log_file.write(f"new_ei_f_score: {new_ei_f_score}\n")
            log_file.flush()
            if os.path.exists(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'eval_result_ind.txt')):
                # remove the file
                os.remove(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, iter_idx, 'eval_result_ind.txt'))
        else:
            best_iter_idx = iter_idx - 1
            c1_to_best_f_score[current_c1] = new_ei_f_score
            log_file.write(f"current_f: {curr_ei_f_score}\n")
            log_file.write(f"new_c1: {new_c1}\n")
            log_file.write(f"new_ei_f_score: {new_ei_f_score}\n")
            log_file.flush()
        # if the current best score's neighbour are all lower than the current best score, finish the ei
        if new_c1 in c1_to_best_f_score:
            if c1_to_best_f_score[new_c1] < curr_ei_f_score:
                log_file.write(f"new_c1 in c1_to_best_f_score\n")
                log_file.flush()
                break
        
    with open(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA, args.lora_r, args.lora_alpha, best_iter_idx, 'eval_result_ind.txt')) as f:
        lines = f.readlines()[0]
        # turns lines into dictionary
        eval_result = from_string_to_dict(lines)
    # save the information of the best model
    with open(EI_MODEL_PATH.format(args.task_name, model_name, LAMBDA,  args.lora_r, args.lora_alpha, best_iter_idx, 'best_model_info.txt'), 'w') as f:
        f.write(str(eval_result)) 
    
     
if __name__ == "__main__":
    @dataclass
    class ScriptArguments:
        """
        The name of the Casual LM model we wish to fine with SFTTrainer
        """
        num_gpus: Optional[int] = field(default=8, metadata={"help": "Number of GPUs to use"})
        gpu_memory: Optional[int] = field(default=40, metadata={"help": "GPU memory to use"})
        task_name: Optional[str] = field(default="proofwriter", metadata={"help": "The name of the task"})
        cuda_visible_devices: Optional[str] = field(default="0,1,2,3,4,5,6,7", metadata={"help": "gpu index"})
        Lambda: Optional[float] = field(default=0.2, metadata={"help": "Lambda"})
        lora_r: Optional[float] = field(default=128, metadata={"help": "lora_r"})
        lora_alpha: Optional[float] = field(default=64, metadata={"help": "lora_alpha"})
        n_samples: Optional[int] = field(default=16, metadata={"help": "n_samples"})
        idk_threshold: Optional[float] = field(default=0.25, metadata={"help": "idk_threshold"})
        gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "gradient_checkpointing"})
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    try:
        auto_cei_hill_climbing(script_args)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        exit(0)