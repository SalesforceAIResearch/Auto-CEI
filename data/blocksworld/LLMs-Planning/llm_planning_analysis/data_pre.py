import datasets
import json
file = './data/blocksworld/LLMs-Planning/llm_planning_analysis/prompts/blocksworld/task_1_plan_generation_state_tracking.json'

with open(file) as f:
    dict_data = json.load(f)

new_list = dict_data['instances']
data_dict = {}
# length_dict = {}
for ele in new_list:
    length_reply = len(ele['ground_truth_plan'].split('\n')) - 1
    # if length_reply not in length_dict:
    #     length_dict[length_reply] = 0
    # length_dict[length_reply] += 1
    if length_reply <= 10:
        for key in ele:
            if key not in data_dict:
                data_dict[key] = []
            if key == 'reply':
                data_dict[key].append(ele[key].replace(
                    "The goal conditions are satisfied in the final state. Hence, the above plan is valid.\n<|eot_id|>\n",
                    ". The goal conditions are satisfied in the final state. Hence, the above plan is valid.\n<|eot_id|>\n",
                    ))
                pass
            elif key == 'instance_id':
                data_dict[key].append(str(ele[key]))
            else:
                data_dict[key].append(ele[key])
# print(length_dict)
print(data_dict['instance_id'][0])
print(isinstance(data_dict['instance_id'][0], str))
data = datasets.Dataset.from_dict(data_dict)
len_data = len(data)
# print(len_data)
data.shuffle(seed=42)
data_train_validation = data.select(range(int(len(data) * 0.5)))
data_test_ = data.select(range(int(len(data) * 0.5), len(data), 1))
data_sft = data_train_validation.select(range(1500))
# total_len = []
data_q = data_train_validation.select(range(1500, len(data_train_validation), 1))
index = []
for i, ele in enumerate(data_q):
    if len(ele['ground_truth_plan'].split('\n')) - 1 <= 7:
        index.append(i)
data_q = data_q.select(index)
data_q = datasets.concatenate_datasets([data_q, data_sft])
new_q = {
    
}
for ele in data_q:
    for key in ele:
        if key not in new_q:
            new_q[key] = []
        if key == 'reply':
            reply = ele[key].replace("[END]", "<|eot_id|>")
            reply = reply.replace(". Since", ".\nSince")
            new_q[key].append(reply)
        else:
            new_q[key].append(ele[key])
new_data_q = datasets.Dataset.from_dict(new_q)
data_q = new_data_q

print(len(data_q))
index = {}
for i, ele in enumerate(data_test_):
    length = len(ele['ground_truth_plan'].split('\n')) - 1
    if length not in index:
        index[length] = []
    index[length].append(i)

index_val = []
index_test = []
for ele in index:
    if ele < 11:
        if len(index[ele]) < 200:
            index_val = index_val + index[ele][:int(len(index[ele]) * 0.5)]
            index_test = index_test + index[ele][int(len(index[ele]) * 0.5):]
        else:
            index_val = index_val + index[ele][:100]
            index_test = index_test + index[ele][100:200]


data_val = data_test_.select(index_val)
data_test = data_test_.select(index_test)
data_sft.save_to_disk('./data/blocksworld/blocksworld_dataset_sft')
data_q.save_to_disk('./data/blocksworld/blocksworld_dataset_q')
data_val.save_to_disk('./data/blocksworld/blocksworld_dataset_val')
data_test.save_to_disk('./data/blocksworld/blocksworld_dataset_test')