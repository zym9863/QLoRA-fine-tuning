#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA指令微调主程序
支持命令行参数：data_path, model_path, epochs, batch_size, output_path
"""

import argparse
import sys
import os
from trainer import run_training


def parse_arguments():
    """
    解析命令行参数

    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="QLoRA指令微调训练程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train.py --data_path ./data.json --model_path google/gemma-2-9b-it --epochs 3 --batch_size 1 --output_path ./output

  python train.py --model_path google/gemma-2-9b-it --epochs 5 --batch_size 2 --output_path ./my_model
        """
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="训练数据路径（JSON格式）。如果不指定，将自动从BelleGroup/train_0.5M_CN下衳2000条数据"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="预训练模型路径或HuggingFace模型名称（如：google/gemma-2-9b-it）"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数（默认：3）"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="批次大小（默认：1）"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="模型输出路径"
    )

    return parser.parse_args()


def validate_arguments(args):
    """
    验证命令行参数

    Args:
        args: 解析后的参数对象

    Returns:
        验证是否通过
    """
    # 验证数据路径
    if args.data_path and not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return False

    # 验证训练参数
    if args.epochs <= 0:
        print(f"错误: epochs必须大于0，当前值: {args.epochs}")
        return False

    if args.batch_size <= 0:
        print(f"错误: batch_size必须大于0，当前值: {args.batch_size}")
        return False

    # 验证输出路径
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"创建输出目录: {output_dir}")
        except Exception as e:
            print(f"错误: 无法创建输出目录 {output_dir}: {e}")
            return False

    return True


def print_training_info(args):
    """
    打印训练配置信息

    Args:
        args: 解析后的参数对象
    """
    print("=" * 60)
    print("QLoRA指令微调训练配置")
    print("=" * 60)
    print(f"数据路径:     {args.data_path if args.data_path else '自动下载BelleGroup/train_0.5M_CN(2000条)'}")
    print(f"模型路径:     {args.model_path}")
    print(f"训练轮数:     {args.epochs}")
    print(f"批次大小:     {args.batch_size}")
    print(f"输出路径:     {args.output_path}")
    print("=" * 60)


def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 验证参数
        if not validate_arguments(args):
            sys.exit(1)

        # 打印训练信息
        print_training_info(args)

        # 确认开始训练
        response = input("\n是否开始训练？(y/N): ").strip().lower()
        if response not in ['y', 'yes', '是']:
            print("训练已取消")
            sys.exit(0)

        # 开始训练
        print("\n开始QLoRA指令微调训练...")
        run_training(
            data_path=args.data_path,
            model_path=args.model_path,
            output_path=args.output_path,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        print("\n训练完成！")

    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()