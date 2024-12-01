import json
import torch
import os
import sys
import pandas as pd

from PIL import Image
from typing import Dict, List, Tuple
from torch.utils.data import Dataset,DataLoader
from dataclasses import dataclass
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


@dataclass
class text_image:
    q_input_ids : torch.Tensor
    pixel_values : torch.Tensor
    a_input_ids : torch.Tensor

class LlavaDataset(Dataset):
    def __init__(self, data_file, image_dir):
        super().__init__()

        self.data_file = data_file
        self.image_dir = image_dir
        self.chat_data = pd.read_json(data_file).to_dict(orient="records")

    def __len__(self):
        return len(self.chat_data)
    
    def __getitem__(self, index):
        data = self.chat_data[index]
        messages = data['messages']
        human_input = messages[0]['content']
        gpt_output = messages[1]['content']
        images = data['images'] # List
        images = [os.path.join(self.image_dir,image) for image in images]
        return human_input, gpt_output, images
    
def process_qa_and_image(qa_and_image : List[Tuple[str,str,List[str]]], processor : AutoProcessor):
    # 需要直接传一个组数据，并组成官方可以支持的数据格式

    messages = []

    gpt_output_list = []
    for one_piece_data in qa_and_image:
        human_input, gpt_output, images = one_piece_data
        gpt_output_list.append(gpt_output)

        ins = human_input.split('<image>')
        content = []
        for idx in range(len(ins)):
            s = ins[idx]

            if s != '':
                content.append({
                    "type":"text",
                    "text":s
                })

            if idx != len(ins) - 1:
                content.append({
                    "type":"image",
                    "image":images[idx]
                })
        message = {
            "role":"user",
            "content":content,
        }
        # 注意每一条数据都是一个列表，不是字典
        messages.append([message])
    

    # ====================官方模板处理====================     
    # 再加模板之前，需要转为标准格式，详细见modelscope官网，以便于组batch
    # 填充是用 '<|endoftext|>' 是left_padding
    prompt = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, video_inputs = process_vision_info(messages)


    q_inputs_list = processor(
        text=prompt,
        images=image_inputs,
        videos=video_inputs,
        padding=False,
    )
    # ====================官方模板处理====================
    # inputs = dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'])
    a_input_ids = processor.tokenizer(
        gpt_output_list,
        truncation=True,
    )['input_ids']


    return q_inputs_list, a_input_ids
    



class LlavaDataCollator:
    def __init__(self,processor: AutoProcessor):
        self.processor = processor
        self.ignore_index = -100

    def processqa_for_train(self, q_input_ids, a_input_ids):
        # 两者形状都是[1,X],直接concat
        # 因为是sft 最后要加上eos token

        if type(q_input_ids) == list:
            q_input_ids = torch.tensor(q_input_ids).reshape(1,-1)
            a_input_ids = torch.tensor(a_input_ids).reshape(1,-1)

        eos_token_id = torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1,-1)

        inputs = torch.concat(
            [q_input_ids,a_input_ids,eos_token_id],
            dim=1
        )
        # sft格式的instruction对应的labels中的值设置为-100
        # 最后的eos token也要计算loss
        labels = torch.concat(
            [torch.full_like(q_input_ids, fill_value=self.ignore_index),a_input_ids,eos_token_id],
            dim=1
        )

        # 设置训练数据的最大长度为2048
        # cutoff_len = 1024
        # inputs = inputs[:cutoff_len]
        # labels = labels[:cutoff_len]
        return inputs, labels


    
    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        label_ids_list = []
        pixel_values_list = []
        max_len_list = []
        image_grid_thw_list = []

        q_inputs_list, a_input_ids = process_qa_and_image(features,self.processor)

        # for feature in features:
        #     input_ids ,labels = self.processqa_for_train(feature.q_input_ids, feature.a_input_ids)
        #     input_ids_list.append(input_ids)
        #     label_ids_list.append(labels)
        #     pixel_values_list.append(feature.pixel_values)
        #     image_grid_thw_list.append(feature.image_grid_thw)
        #     max_len_list.append(input_ids.shape[1])
        
        # 要将这些数据组成一个batch，这里主要是做了padding
        # 实际sft时还要做截断


        # 将q和a组成训练所需要的格式
        max_len = 0
        for q,a in zip(q_inputs_list['input_ids'], a_input_ids):
            input_ids ,labels = self.processqa_for_train(q, a)
            input_ids_list.append(input_ids)
            label_ids_list.append(labels)
            max_len = max(max_len, input_ids.shape[1])

        # pixel_values_list = torch.tensor(q_inputs_list['pixel_values'])
        # image_grid_thw_list = torch.tensor(q_inputs_list['image_grid_thw'])
            
            
        pad_token_id = self.processor.tokenizer.pad_token_id

        # 这里实现的是right padding 但是训练是无所谓的

        final_input_ids = torch.concat(
            [torch.concat([input_ids,torch.full((1,max_len - input_ids.shape[1]), pad_token_id)],dim=1) for input_ids in input_ids_list],
            dim=0
        )
        final_labels_ids = torch.concat(
            [torch.concat([labels_ids,torch.full((1,max_len - labels_ids.shape[1]), pad_token_id)],dim=1) for labels_ids in label_ids_list],
            dim=0
        )

        # qwen_vl还需要个image_grid_thw参数
        # final_image_grid_thw = torch.concat(
        #     [image_grid_thw for image_grid_thw in image_grid_thw_list],
        #     dim=0
        # )

        # # pixel_values不需要padding
        # final_pixel_values = torch.concat(pixel_values_list,dim=0)
        final_pixel_values = torch.tensor(q_inputs_list['pixel_values'])
        final_image_grid_thw = torch.tensor(q_inputs_list['image_grid_thw'])

        # attenion_mask:pad token的值应该设置为0
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "input_ids":final_input_ids,
            "labels":final_labels_ids,
            "pixel_values":final_pixel_values,
            'image_grid_thw':final_image_grid_thw,
            "attention_mask":attention_mask
        }



if __name__ == "__main__":
    data_file = '/mnt/new_disk/qjw/my_llm_code/sft/mllm_demo.json'
    image_dir = '/mnt/new_disk/qjw/LLaMA-Factory/data/'
    dataset = LlavaDataset(data_file=data_file,image_dir=image_dir)

    model_dir = '/mnt/new_disk/qjw/ckpt/qwen/Qwen2-VL-7B-Instruct'
    processor = AutoProcessor.from_pretrained(model_dir)

    data_loader = DataLoader(dataset,2,collate_fn=LlavaDataCollator(processor=processor))

    for batch in data_loader:
        print(batch)