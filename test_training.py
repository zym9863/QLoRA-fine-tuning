#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA训练流程测试脚本
验证各个模块的功能是否正常
"""

import os
import sys
import json
import torch
from typing import List, Dict


def test_data_processor():
    """
    测试数据处理模块
    """
    print("=" * 50)
    print("测试数据处理模块")
    print("=" * 50)

    try:
        from data_processor import DataProcessor

        # 创建数据处理器（使用较小的样本数量进行测试）
        processor = DataProcessor(sample_size=10)

        # 测试数据加载和处理
        print("正在测试数据加载和处理...")
        training_data = processor.prepare_training_data()

        # 验证结果
        assert len(training_data) > 0, "训练数据为空"
        assert "text" in training_data[0], "数据格式不正确"

        print(f"✓ 数据处理测试通过，生成了 {len(training_data)} 条训练数据")
        print(f"样本数据预览: {training_data[0]['text'][:100]}...")

        return True

    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        return False


def test_model_config():
    """
    测试模型配置模块
    """
    print("\n" + "=" * 50)
    print("测试模型配置模块")
    print("=" * 50)

    try:
        from model_config import ModelConfig

        # 创建模型配置器
        config = ModelConfig("google/gemma-3-4b-it")

        # 测试配置创建
        print("正在测试配置创建...")
        bnb_config = config.create_bnb_config()
        lora_config = config.create_lora_config()
        training_args = config.create_training_arguments("./test_output", 1, 1)

        # 验证配置
        assert bnb_config is not None, "BnB配置创建失败"
        assert lora_config is not None, "LoRA配置创建失败"
        assert training_args is not None, "训练参数创建失败"

        print("✓ 模型配置测试通过")
        print(f"  - LoRA rank: {lora_config.r}")
        print(f"  - LoRA alpha: {lora_config.lora_alpha}")
        print(f"  - 量化类型: {bnb_config.bnb_4bit_quant_type}")

        return True

    except Exception as e:
        print(f"✗ 模型配置测试失败: {e}")
        return False


def test_command_line_interface():
    """
    测试命令行接口
    """
    print("\n" + "=" * 50)
    print("测试命令行接口")
    print("=" * 50)

    try:
        # 模拟命令行参数
        test_args = [
            "--model_path", "google/gemma-3-4b-it",
            "--epochs", "1",
            "--batch_size", "1",
            "--output_path", "./test_output"
        ]

        # 保存原始sys.argv
        original_argv = sys.argv.copy()

        try:
            # 设置测试参数
            sys.argv = ["train.py"] + test_args

            # 导入并测试参数解析
            from train import parse_arguments, validate_arguments

            args = parse_arguments()
            is_valid = validate_arguments(args)

            # 验证结果
            assert args.model_path == "google/gemma-3-4b-it", "模型路径解析错误"
            assert args.epochs == 1, "epochs解析错误"
            assert args.batch_size == 1, "batch_size解析错误"
            assert is_valid, "参数验证失败"

            print("✓ 命令行接口测试通过")
            print(f"  - 模型路径: {args.model_path}")
            print(f"  - 训练轮数: {args.epochs}")
            print(f"  - 批次大小: {args.batch_size}")

            return True

        finally:
            # 恢复原始sys.argv
            sys.argv = original_argv

    except Exception as e:
        print(f"✗ 命令行接口测试失败: {e}")
        return False


def test_environment():
    """
    测试环境依赖
    """
    print("\n" + "=" * 50)
    print("测试环境依赖")
    print("=" * 50)

    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "bitsandbytes",
        "accelerate"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (未安装)")
            missing_packages.append(package)

    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA可用 (设备数量: {torch.cuda.device_count()})")
    else:
        print("⚠ CUDA不可用，将使用CPU训练（速度较慢）")

    if missing_packages:
        print(f"\n缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    return True


def create_sample_data():
    """
    创建示例数据文件用于测试
    """
    sample_data = [
        {
            "text": "### 指令:\n请介绍一下人工智能。\n\n### 回答:\n人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        },
        {
            "text": "### 指令:\n什么是机器学习？\n\n### 回答:\n机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。"
        }
    ]

    with open("sample_data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)

    print("✓ 创建示例数据文件: sample_data.json")


def main():
    """
    运行所有测试
    """
    print("QLoRA训练流程测试")
    print("=" * 60)

    # 创建示例数据
    create_sample_data()

    # 运行测试
    tests = [
        ("环境依赖", test_environment),
        ("数据处理模块", test_data_processor),
        ("模型配置模块", test_model_config),
        ("命令行接口", test_command_line_interface),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name}测试出现异常: {e}")

    # 输出测试结果
    print("\n" + "=" * 60)
    print(f"测试完成: {passed}/{total} 通过")

    if passed == total:
        print("✓ 所有测试通过！可以开始训练。")
        print("\n使用示例:")
        print("python train.py --model_path google/gemma-3-4b-it --epochs 3 --batch_size 1 --output_path ./output")
    else:
        print("✗ 部分测试失败，请检查环境配置。")

    # 清理测试文件
    if os.path.exists("sample_data.json"):
        os.remove("sample_data.json")


if __name__ == "__main__":
    main()