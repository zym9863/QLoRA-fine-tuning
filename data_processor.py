# -*- coding: utf-8 -*-
"""
数据处理模块
从BelleGroup/train_0.5M_CN数据集中选取500条数据并格式化为指令微调格式
"""

import json
import random
from datasets import load_dataset
from typing import List, Dict, Any


class DataProcessor:
    """数据处理器，负责加载和预处理训练数据"""

    def __init__(self, dataset_name: str = "BelleGroup/train_0.5M_CN", sample_size: int = 500):
        """
        初始化数据处理器

        Args:
            dataset_name: 数据集名称
            sample_size: 采样数量
        """
        self.dataset_name = dataset_name
        self.sample_size = sample_size

    def load_and_sample_data(self) -> List[Dict[str, Any]]:
        """
        加载数据集并随机采样指定数量的数据

        Returns:
            采样后的数据列表
        """
        print(f"正在加载数据集: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split="train")

        # 随机采样
        total_size = len(dataset)
        if total_size < self.sample_size:
            print(f"警告: 数据集大小({total_size})小于采样大小({self.sample_size})")
            sample_indices = list(range(total_size))
        else:
            sample_indices = random.sample(range(total_size), self.sample_size)

        sampled_data = [dataset[i] for i in sample_indices]
        print(f"成功采样 {len(sampled_data)} 条数据")

        return sampled_data

    def format_for_instruction_tuning(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        将数据格式化为指令微调格式

        Args:
            data: 原始数据列表

        Returns:
            格式化后的数据列表
        """
        formatted_data = []

        for item in data:
            # Belle数据集通常包含instruction和output字段
            if "instruction" in item and "output" in item:
                formatted_item = {
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"]
                }
                formatted_data.append(formatted_item)
            elif "conversations" in item:
                # 处理对话格式数据
                conversations = item["conversations"]
                if len(conversations) >= 2:
                    instruction = conversations[0].get("value", "")
                    output = conversations[1].get("value", "")
                    formatted_item = {
                        "instruction": instruction,
                        "input": "",
                        "output": output
                    }
                    formatted_data.append(formatted_item)

        print(f"成功格式化 {len(formatted_data)} 条数据")
        return formatted_data

    def create_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        创建训练用的提示文本

        Args:
            instruction: 指令文本
            input_text: 输入文本

        Returns:
            格式化的提示文本
        """
        if input_text:
            prompt = f"### 指令:\n{instruction}\n\n### 输入:\n{input_text}\n\n### 回答:\n"
        else:
            prompt = f"### 指令:\n{instruction}\n\n### 回答:\n"

        return prompt

    def prepare_training_data(self) -> List[Dict[str, str]]:
        """
        准备训练数据

        Returns:
            准备好的训练数据列表
        """
        # 加载和采样数据
        raw_data = self.load_and_sample_data()

        # 格式化数据
        formatted_data = self.format_for_instruction_tuning(raw_data)

        # 创建训练样本
        training_data = []
        for item in formatted_data:
            prompt = self.create_prompt(item["instruction"], item["input"])
            training_sample = {
                "text": prompt + item["output"]
            }
            training_data.append(training_sample)

        return training_data


def save_data_to_json(data: List[Dict[str, str]], output_path: str):
    """
    将数据保存为JSON文件

    Args:
        data: 要保存的数据
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到: {output_path}")


if __name__ == "__main__":
    # 测试数据处理器
    processor = DataProcessor()
    training_data = processor.prepare_training_data()

    # 保存处理后的数据
    save_data_to_json(training_data, "processed_data.json")

    # 显示样本
    if training_data:
        print("\n样本数据:")
        print(training_data[0]["text"][:200] + "...")