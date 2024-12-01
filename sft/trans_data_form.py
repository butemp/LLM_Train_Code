import json
import os

ori_data = '/mnt/new_disk/qjw/my_llm_code/sft/train_class.json'

out_data = '/mnt/new_disk/qjw/my_llm_code/sft/train_data.json'

res = []

with open(ori_data, 'r') as fp:
    data = json.load(fp)

    for item in data:
        tmp = {}
        tmp['messages'] = [
            {
                "content": item['instruction'],
                "role": "user"
            },
            {
                "content": item['output'],
                "role": "assistant"
            },
        ]
        tmp['images'] = item['image']
        res.append(tmp)

with open(out_data,'w') as fp:
    json.dump(res,fp,indent=4,ensure_ascii=False) 