import datasets

ds = datasets.load_from_disk('data/blocksworld/train_ds_ei_baseline_0.2-128-64/iter_1')
ds_dict = {}
for ele in ds:
    if 'sorry' in ele['reply'].lower():
        reply = ele['reply'].split("The goal conditions are satisfied.")[0] \
            + "Sorry, I am not sure if I answer the question correctly. There might be mistakes in my answer. <|eot_id|> "
    else:
        reply = ele['reply']

    for key in ele:
        if key not in ds_dict:
            ds_dict[key] = []
        if key == 'reply':
            ds_dict[key].append(reply)
        else:
            ds_dict[key].append(ele[key])

new_ds = datasets.Dataset.from_dict(ds_dict)
new_ds.save_to_disk('data/blocksworld/train_ds_ei_baseline_0.2-128-64/iter_1')