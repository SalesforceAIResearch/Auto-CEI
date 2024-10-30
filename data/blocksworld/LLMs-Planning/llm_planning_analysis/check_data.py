import datasets

ds = datasets.load_from_disk('data/blocksworld/train_ds_ei_baseline_0.2-128-64/iter_1')
for ele in ds:
    if 'sorry' in ele['reply'].lower():
        print(ele['reply'])
        break