# -*- coding: utf-8 -*-
"""
训练核心模块
实现QLoRA训练逻辑，包括模型初始化、训练循环、损失计算和优化器配置
"""

import os
import torch
from datasets import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from typing import List, Dict, Any
from data_processor import DataProcessor
from model_config import setup_model_for_training


class QLoRATrainer:
    """QLoRA训练器，负责执行完整的训练流程"""

    def __init__(self, data_path: str, model_path: str, output_path: str,
                 epochs: int = 3, batch_size: int = 1):
        """
        初始化QLoRA训练器

        Args:
            data_path: 数据路径（如果为None则使用默认数据集）
            model_path: 模型路径
            output_path: 输出路径
            epochs: 训练轮数
            batch_size: 批次大小
        """
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.epochs = epochs
        self.batch_size = batch_size

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

    def prepare_dataset(self) -> Dataset:
        """
        准备训练数据集

        Returns:
            处理好的数据集
        """
        print("正在准备数据集...")

        if self.data_path and os.path.exists(self.data_path):
            # 从文件加载数据
            import json
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 使用数据处理器生成数据
            processor = DataProcessor()
            data = processor.prepare_training_data()

        # 转换为Dataset格式
        dataset = Dataset.from_list(data)
        print(f"数据集准备完成，共 {len(dataset)} 条数据")

        return dataset

    def tokenize_function(self, examples: Dict[str, List[str]], tokenizer) -> Dict[str, Any]:
        """
        分词函数

        Args:
            examples: 输入样本
            tokenizer: 分词器

        Returns:
            分词后的结果
        """
        # 分词
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512,
            return_overflowing_tokens=False,
        )

        # 设置labels为input_ids的副本（用于语言建模）
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    def train(self):
        """
        执行训练流程
        """
        print("开始QLoRA训练...")

        # 1. 设置模型
        print("正在设置模型...")
        model, tokenizer, training_args = setup_model_for_training(
            model_path=self.model_path,
            output_dir=self.output_path,
            epochs=self.epochs,
            batch_size=self.batch_size
        )

        # 2. 准备数据集
        dataset = self.prepare_dataset()

        # 3. 分词
        print("正在进行分词...")
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
        )

        # 4. 创建数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 不使用掩码语言建模
        )

        # 5. 创建训练器
        print("正在创建训练器...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # 6. 开始训练
        print("开始训练...")
        trainer.train()

        # 7. 保存模型
        print("正在保存模型...")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_path)

        print(f"训练完成！模型已保存到: {self.output_path}")

    def evaluate_model(self, test_prompts: List[str] = None):
        """
        评估训练后的模型

        Args:
            test_prompts: 测试提示列表
        """
        if test_prompts is None:
            test_prompts = [
                "### 指令:\n请介绍一下人工智能的发展历史。\n\n### 回答:\n",
                "### 指令:\n解释什么是机器学习。\n\n### 回答:\n",
                "### 指令:\n写一首关于春天的诗。\n\n### 回答:\n"
            ]

        print("正在评估模型...")

        # 加载训练后的模型
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self.output_path)
        model = AutoModelForCausalLM.from_pretrained(self.output_path)

        # 设置为评估模式
        model.eval()

        # 生成回答
        for i, prompt in enumerate(test_prompts):
            print(f"\n=== 测试样本 {i+1} ===")
            print(f"输入: {prompt}")

            # 分词
            inputs = tokenizer(prompt, return_tensors="pt")

            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 解码
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"输出: {response}")


def run_training(data_path: str, model_path: str, output_path: str,
                epochs: int, batch_size: int):
    """
    运行训练的主函数

    Args:
        data_path: 数据路径
        model_path: 模型路径
        output_path: 输出路径
        epochs: 训练轮数
        batch_size: 批次大小
    """
    # 创建训练器
    trainer = QLoRATrainer(
        data_path=data_path,
        model_path=model_path,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size
    )

    # 执行训练
    trainer.train()

    # 评估模型
    trainer.evaluate_model()


if __name__ == "__main__":
    # 测试训练流程
    run_training(
        data_path=None,  # 使用默认数据集
        model_path="google/gemma-3-4b-it",
        output_path="./output",
        epochs=1,
        batch_size=1
    )