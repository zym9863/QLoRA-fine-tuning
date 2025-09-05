[English Version](README-EN.md)

# QLoRA指令微调系统

基于QLoRA技术的语言模型指令微调实现，支持google/gemma-3-4b-it模型和BelleGroup/train_0.5M_CN数据集。

## 项目特性

- ✅ 支持QLoRA（4-bit量化 + LoRA）高效微调
- ✅ 使用google/gemma-3-4b-it预训练模型
- ✅ 自动从BelleGroup/train_0.5M_CN数据集采朷2000条数据
- ✅ 完整的命令行接口
- ✅ 模块化设计，易于扩展
- ✅ 包含测试和验证脚本

## 项目结构

```
QLoRA fine-tuning/
├── requirements.txt          # 依赖包列表
├── data_processor.py         # 数据处理模块
├── model_config.py          # 模型配置模块
├── trainer.py               # 训练核心模块
├── train.py                 # 命令行主程序
├── test_training.py         # 测试验证脚本
└── README.md               # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖

- torch>=2.0.0
- transformers>=4.36.0
- datasets>=2.14.0
- peft>=0.7.0
- bitsandbytes>=0.41.0
- accelerate>=0.24.0

## 快速开始

### 1. 测试环境

首先运行测试脚本确保环境配置正确：

```bash
python test_training.py
```

### 2. 开始训练

使用指定的命令行格式进行训练：

```bash
python train.py \
  --model_path google/gemma-3-4b-it \
  --epochs 3 \
  --batch_size 1 \
  --output_path ./output
```

### 3. 使用自定义数据

如果有自己的训练数据，可以指定data_path参数：

```bash
python train.py \
  --data_path ./my_data.json \
  --model_path google/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 2 \
  --output_path ./my_model
```

## 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_path` | str | 否 | None | 训练数据路径（JSON格式）。如不指定，自动使用BelleGroup/train_0.5M_CN |
| `--model_path` | str | 是 | - | 预训练模型路径或HuggingFace模型名称 |
| `--epochs` | int | 否 | 3 | 训练轮数 |
| `--batch_size` | int | 否 | 1 | 批次大小 |
| `--output_path` | str | 是 | - | 模型输出路径 |

## 数据格式

### 自动数据格式

程序会自动从BelleGroup/train_0.5M_CN数据集中采朷2000条数据，并格式化为指令微调格式。

### 自定义数据格式

如果使用自定义数据，JSON文件应包含以下格式的数据：

```json
[
  {
    "text": "### 指令:\n请介绍一下人工智能。\n\n### 回答:\n人工智能（AI）是计算机科学的一个分支..."
  },
  {
    "text": "### 指令:\n什么是机器学习？\n\n### 回答:\n机器学习是人工智能的一个子领域..."
  }
]
```

## 模型配置

### QLoRA参数

- **LoRA rank (r)**: 16
- **LoRA alpha**: 32
- **LoRA dropout**: 0.1
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 量化配置

- **量化位数**: 4-bit
- **量化类型**: nf4
- **双重量化**: 启用
- **计算数据类型**: bfloat16

## 训练配置

- **优化器**: paged_adamw_32bit
- **学习率**: 2e-4
- **权重衰减**: 0.001
- **梯度累积步数**: 4
- **最大梯度范数**: 0.3
- **学习率调度器**: constant
- **预热比例**: 0.03

## 使用示例

### 基本训练示例

```bash
# 使用默认参数训练
python train.py --model_path google/gemma-3-4b-it --output_path ./output

# 自定义训练参数
python train.py \
  --model_path google/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 2 \
  --output_path ./my_fine_tuned_model
```

### 使用自定义数据训练

```bash
# 准备数据文件 my_data.json
python train.py \
  --data_path ./my_data.json \
  --model_path google/gemma-3-4b-it \
  --epochs 3 \
  --batch_size 1 \
  --output_path ./custom_model
```

## 模块说明

### data_processor.py
- **DataProcessor类**: 负责数据加载、采样和格式化
- **主要功能**: 从HuggingFace数据集加载数据，随机采样指定数量，格式化为指令微调格式

### model_config.py
- **ModelConfig类**: 负责模型和QLoRA配置
- **主要功能**: 创建量化配置、LoRA配置、训练参数，加载模型和分词器

### trainer.py
- **QLoRATrainer类**: 负责执行完整的训练流程
- **主要功能**: 数据预处理、模型训练、模型保存、模型评估

### train.py
- **主程序**: 提供命令行接口
- **主要功能**: 参数解析、参数验证、训练流程控制

### test_training.py
- **测试脚本**: 验证各模块功能
- **主要功能**: 环境检查、模块测试、功能验证

## 硬件要求

### 推荐配置
- **GPU**: NVIDIA RTX 4090 或更高
- **显存**: 24GB 或更多
- **内存**: 32GB 或更多
- **存储**: 50GB 可用空间

### 最低配置
- **GPU**: NVIDIA RTX 3080 或同等性能
- **显存**: 12GB 或更多
- **内存**: 16GB 或更多
- **存储**: 30GB 可用空间

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案: 减小batch_size或使用gradient_accumulation_steps
   ```

2. **依赖包版本冲突**
   ```bash
   pip install --upgrade transformers datasets peft bitsandbytes
   ```

3. **模型下载失败**
   ```bash
   # 设置HuggingFace镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

4. **权限错误**
   ```bash
   # 确保输出目录有写权限
   chmod 755 ./output
   ```

## 性能优化

### 训练速度优化
- 使用更大的batch_size（如果显存允许）
- 启用gradient_checkpointing
- 使用混合精度训练（bf16）

### 内存优化
- 减小max_length参数
- 使用更小的LoRA rank
- 启用gradient_accumulation_steps

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 致谢

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Google Gemma](https://ai.google.dev/gemma)
- [BelleGroup](https://github.com/LianjiaTech/BELLE)