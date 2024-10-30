import datasets
import json
import os
os.system("""
          cd data/boardgameQA
          wget https://storage.googleapis.com/gresearch/BoardgameQA/BoardgameQA.zip
          unzip BoardgameQA.zip
          """)


# save 

path = "data/boardgameQA/BoardgameQA/BoardgameQA-Main-depth{}/{}.json"

for n in ['train', 'valid', 'test']:
    dataset_dict = {
        'prompt': [],
        'completion': [],
        'answer': []
    }
    # read json file
    for i in range(1, 4, 1):
        with open(path.format(i, n)) as f:
            dict_data = json.load(f)

        for ele in dict_data:
            dataset_dict['prompt'].append(ele['example'])
            completion = ".\n".join(ele['proof'].split(". ")) + "<|eot_id|>"
            completion = completion.replace(", so", ".\nSo").replace(", but", ".\nBut")
            completion = completion.replace(", and", ".\nAnd")
            dataset_dict['completion'].append(completion)
            dataset_dict['answer'].append(ele['label'])

    ds = datasets.Dataset.from_dict(dataset_dict)
    if n == 'valid':
        ds = ds.shuffle(seed=42)
        ds = ds.select(range(400))
    ds.save_to_disk(f'data/boardgameQA/{n}_ds')


