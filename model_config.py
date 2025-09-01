# -*- coding: utf-8 -*-
"""
模型配置模块
配置google/gemma-3-4b-it模型，设置QLoRA参数
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Tuple, Any


class ModelConfig:
    """模型配置器，负责设置和初始化QLoRA模型"""

    def __init__(self, model_name: str = "google/gemma-3-4b-it"):
        """
        初始化模型配置器

        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

    def create_bnb_config(self) -> BitsAndBytesConfig:
        """
        创建BitsAndBytes量化配置

        Returns:
            量化配置对象
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                      # 启用4bit量化
            bnb_4bit_use_double_quant=True,         # 使用双重量化
            bnb_4bit_quant_type="nf4",              # 量化类型
            bnb_4bit_compute_dtype=torch.bfloat16,  # 计算数据类型
        )
        return bnb_config

    def create_lora_config(self) -> LoraConfig:
        """
        创建LoRA配置

        Returns:
            LoRA配置对象
        """
        lora_config = LoraConfig(
            r=16,                                   # LoRA秩
            lora_alpha=32,                          # LoRA缩放参数
            target_modules=[                        # 目标模块
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.1,                       # LoRA dropout
            bias="none",                            # 偏置设置
            task_type="CAUSAL_LM",                  # 任务类型
        )
        return lora_config

    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """
        加载模型和分词器

        Returns:
            模型和分词器的元组
        """
        print(f"正在加载模型: {self.model_name}")

        # 创建量化配置
        bnb_config = self.create_bnb_config()

        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right",
            add_eos_token=True,
        )

        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        print("模型加载完成")
        return model, tokenizer

    def setup_peft_model(self, model: Any) -> Any:
        """
        设置PEFT模型（QLoRA）

        Args:
            model: 基础模型

        Returns:
            配置好的PEFT模型
        """
        print("正在设置QLoRA模型...")

        # 准备模型进行k-bit训练
        model = prepare_model_for_kbit_training(model)

        # 创建LoRA配置
        lora_config = self.create_lora_config()

        # 应用LoRA配置
        model = get_peft_model(model, lora_config)

        # 打印可训练参数
        model.print_trainable_parameters()

        print("QLoRA模型设置完成")
        return model

    def create_training_arguments(self, output_dir: str, epochs: int, batch_size: int) -> TrainingArguments:
        """
        创建训练参数

        Args:
            output_dir: 输出目录
            epochs: 训练轮数
            batch_size: 批次大小

        Returns:
            训练参数对象
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=500,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",  # 禁用TensorBoard（避免依赖问题）
            save_total_limit=3,
            # 禁用评估策略（适用于简单的指令微调）
            eval_strategy="no",
        )
        return training_args


def setup_model_for_training(model_path: str, output_dir: str, epochs: int, batch_size: int):
    """
    设置模型进行训练的完整流程

    Args:
        model_path: 模型路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小

    Returns:
        配置好的模型、分词器和训练参数
    """
    # 创建模型配置器
    config = ModelConfig(model_path)

    # 加载模型和分词器
    model, tokenizer = config.load_model_and_tokenizer()

    # 设置QLoRA
    model = config.setup_peft_model(model)

    # 创建训练参数
    training_args = config.create_training_arguments(output_dir, epochs, batch_size)

    return model, tokenizer, training_args


if __name__ == "__main__":
    # 测试模型配置
    model_name = "google/gemma-3-4b-it"
    model, tokenizer, training_args = setup_model_for_training(
        model_path=model_name,
        output_dir="./output",
        epochs=3,
        batch_size=1
    )
    print("模型配置测试完成")