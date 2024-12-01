import logging
import os
import torch

from transformers import Qwen2ForCausalLM, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import Trainer,TrainingArguments
import transformers

from dataset import LlavaDataCollator, LlavaDataset
from dataclasses import dataclass, field

## 定义一些训练所需要的必要的参数

@dataclass
class ModelArguments:
    model_name_or_path : str = field(default='/mnt/new_disk/qjw/ckpt/Qwen/Qwen2-VL-2B-Instruct')
    train_type : str = field(default='full_sft')

@dataclass
class DataArguments:
    data_path : str = field(default='/mnt/new_disk/qjw/my_llm_code/sft/train_data.json')
    image_dir : str = field(default='/mnt/new_disk/qjw/LLaMA-Factory/data')


@dataclass
class TrainArguments:
    deepspeed : str = field(default='')

def load_model_and_processor(model_args : ModelArguments):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16
    ).cuda()

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    return model, processor

def load_dataset_and_collator(data_args : DataArguments, processor : AutoProcessor):
    dataset = LlavaDataset(data_args.data_path, data_args.image_dir)

    collator = LlavaDataCollator(processor)

    return dataset, collator


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_and_processor(model_args)

    dataset, collator = load_dataset_and_collator(data_args, processor)

    train_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5.0e-6,
        num_train_epochs=3,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        save_steps=0.2,
        bf16=True,
        logging_steps=1,
        logging_strategy="steps",
        output_dir='./output',
        deepspeed=training_args.deepspeed,
        overwrite_output_dir = True,
        dataloader_num_workers=8,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        args=train_args
    )

    trainer.train()

if __name__ == "__main__":
    train()