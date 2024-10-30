import datasets
import random
random.seed(42)
total_len = []
data_sft= datasets.load_from_disk('./data/blocksworld/blocksworld_dataset_sft')
print(len(data_sft))
len_dict = {}
argumented_data = {}
lengths = []
for d in data_sft:
    length= len(d['reply'].split('\n'))
    lengths.append(length)
    if length >= 9:
        if random.choice([True, True, False]):
            num = random.choice([1, 2])
            for _ in range(num):
                index = random.choice(range(4, length-1, 1))
                if random.choice([True, False]):
                    reply = "\n".join(d['reply'].split('\n')[:index]) + "\nSorry, the problem is too difficult. I cannot solve it. [END]"
                else:
                    reply = "\n".join(d['reply'].split('\n')[:index]) + "\nSorry, I am not sure if the plan is correct. [END]"
                for key in d:
                    if key not in argumented_data:
                        argumented_data[key] = []
                    if key != 'reply':
                        argumented_data[key].append(d[key])
                    else:
                        argumented_data[key].append(reply)
    elif length >= 7:
        if random.choice([True, True, False]):
            num = random.choice([1, 2])
            for _ in range(num):
                index = random.choice(range(4, length-1, 1))
                if random.choice([True, False]):
                    reply = "\n".join(d['reply'].split('\n')[:index]) + "\nSorry, the problem is too difficult. I cannot solve it. [END]"
                else:
                    reply = "\n".join(d['reply'].split('\n')[:index]) + "\nSorry, I am not sure if the plan is correct. [END]"
                for key in d:
                    if key not in argumented_data:
                        argumented_data[key] = []
                    if key != 'reply':
                        argumented_data[key].append(d[key])
                    else:
                        argumented_data[key].append(reply)
    elif length >= 5:
        if random.choice([True, False, False]):
            index = random.choice(range(3, length-1, 1))
            if random.choice([True, False]):
                reply = "\n".join(d['reply'].split('\n')[:index]) + "\nSorry, the problem is too difficult. I cannot solve it. [END]"
            else:
                reply = "\n".join(d['reply'].split('\n')[:index]) + "\nSorry, I am not sure if the plan is correct. [END]"
            for key in d:
                if key not in argumented_data:
                    argumented_data[key] = []
                if key != 'reply':
                    argumented_data[key].append(d[key])
                else:
                    argumented_data[key].append(reply)

mean_length = sum(lengths) / len(lengths)
std_length = (sum([(x - mean_length) ** 2 for x in lengths]) / len(lengths)) ** 0.5

print(mean_length, std_length)

data_sft_arg = datasets.Dataset.from_dict(argumented_data)
# print(len(data_sft_arg))
# print(len(data_sft))
# merge the two datasets
data_sft_new = datasets.concatenate_datasets([data_sft, data_sft_arg])
print(len(data_sft_new))
data_sft_new.save_to_disk('./data/blocksworld/blocksworld_dataset_sft_arg')

