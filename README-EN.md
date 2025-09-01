# QLoRA Instruction Fine-tuning System

QLoRA-based language model instruction fine-tuning implementation, supporting google/gemma-3-4b-it model and BelleGroup/train_0.5M_CN dataset.

[中文版本](README.md)

## Project Features

- ✅ Support for QLoRA (4-bit quantization + LoRA) efficient fine-tuning
- ✅ Use google/gemma-3-4b-it pre-trained model
- ✅ Automatically sample 500 data entries from BelleGroup/train_0.5M_CN dataset
- ✅ Complete command-line interface
- ✅ Modular design, easy to extend
- ✅ Includes testing and validation scripts

## Project Structure

```
QLoRA fine-tuning/
├── requirements.txt          # Dependency package list
├── data_processor.py         # Data processing module
├── model_config.py          # Model configuration module
├── trainer.py               # Training core module
├── train.py                 # Command-line main program
├── test_training.py         # Testing validation script
└── README.md               # Project documentation
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

- torch>=2.0.0
- transformers>=4.36.0
- datasets>=2.14.0
- peft>=0.7.0
- bitsandbytes>=0.41.0
- accelerate>=0.24.0

## Quick Start

### 1. Test Environment

First, run the test script to ensure the environment is configured correctly:

```bash
python test_training.py
```

### 2. Start Training

Use the specified command-line format for training:

```bash
python train.py \
  --model_path google/gemma-3-4b-it \
  --epochs 3 \
  --batch_size 1 \
  --output_path ./output
```

### 3. Use Custom Data

If you have your own training data, you can specify the data_path parameter:

```bash
python train.py \
  --data_path ./my_data.json \
  --model_path google/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 2 \
  --output_path ./my_model
```

## Command-line Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--data_path` | str | No | None | Training data path (JSON format). If not specified, automatically use BelleGroup/train_0.5M_CN |
| `--model_path` | str | Yes | - | Pre-trained model path or HuggingFace model name |
| `--epochs` | int | No | 3 | Number of training epochs |
| `--batch_size` | int | No | 1 | Batch size |
| `--output_path` | str | Yes | - | Model output path |

## Data Format

### Automatic Data Format

The program will automatically sample 500 data entries from the BelleGroup/train_0.5M_CN dataset and format them for instruction fine-tuning.

### Custom Data Format

If using custom data, the JSON file should contain data in the following format:

```json
[
  {
    "text": "### Instruction:\nPlease introduce artificial intelligence.\n\n### Response:\nArtificial intelligence (AI) is a branch of computer science..."
  },
  {
    "text": "### Instruction:\nWhat is machine learning?\n\n### Response:\nMachine learning is a subfield of artificial intelligence..."
  }
]
```

## Model Configuration

### QLoRA Parameters

- **LoRA rank (r)**: 16
- **LoRA alpha**: 32
- **LoRA dropout**: 0.1
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Quantization Configuration

- **Quantization bits**: 4-bit
- **Quantization type**: nf4
- **Double quantization**: Enabled
- **Compute data type**: bfloat16

## Training Configuration

- **Optimizer**: paged_adamw_32bit
- **Learning rate**: 2e-4
- **Weight decay**: 0.001
- **Gradient accumulation steps**: 4
- **Maximum gradient norm**: 0.3
- **Learning rate scheduler**: constant
- **Warmup ratio**: 0.03

## Usage Examples

### Basic Training Example

```bash
# Train with default parameters
python train.py --model_path google/gemma-3-4b-it --output_path ./output

# Custom training parameters
python train.py \
  --model_path google/gemma-3-4b-it \
  --epochs 5 \
  --batch_size 2 \
  --output_path ./my_fine_tuned_model
```

### Training with Custom Data

```bash
# Prepare data file my_data.json
python train.py \
  --data_path ./my_data.json \
  --model_path google/gemma-3-4b-it \
  --epochs 3 \
  --batch_size 1 \
  --output_path ./custom_model
```

## Module Description

### data_processor.py
- **DataProcessor class**: Responsible for data loading, sampling, and formatting
- **Main functions**: Load data from HuggingFace dataset, randomly sample specified quantity, format for instruction fine-tuning

### model_config.py
- **ModelConfig class**: Responsible for model and QLoRA configuration
- **Main functions**: Create quantization configuration, LoRA configuration, training parameters, load model and tokenizer

### trainer.py
- **QLoRATrainer class**: Responsible for executing the complete training process
- **Main functions**: Data preprocessing, model training, model saving, model evaluation

### train.py
- **Main program**: Provides command-line interface
- **Main functions**: Parameter parsing, parameter validation, training process control

### test_training.py
- **Test script**: Validate module functions
- **Main functions**: Environment check, module testing, function validation

## Hardware Requirements

### Recommended Configuration
- **GPU**: NVIDIA RTX 4090 or higher
- **VRAM**: 24GB or more
- **RAM**: 32GB or more
- **Storage**: 50GB available space

### Minimum Configuration
- **GPU**: NVIDIA RTX 3080 or equivalent
- **VRAM**: 12GB or more
- **RAM**: 16GB or more
- **Storage**: 30GB available space

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```
   Solution: Reduce batch_size or use gradient_accumulation_steps
   ```

2. **Dependency package version conflicts**
   ```bash
   pip install --upgrade transformers datasets peft bitsandbytes
   ```

3. **Model download failure**
   ```bash
   # Set HuggingFace mirror
   export HF_ENDPOINT=https://hf-mirror.com
   ```

4. **Permission error**
   ```bash
   # Ensure output directory has write permissions
   chmod 755 ./output
   ```

## Performance Optimization

### Training Speed Optimization
- Use larger batch_size (if VRAM allows)
- Enable gradient_checkpointing
- Use mixed precision training (bf16)

### Memory Optimization
- Reduce max_length parameter
- Use smaller LoRA rank
- Enable gradient_accumulation_steps

## License

This project is open-sourced under the MIT license.

## Contributions

Welcome to submit Issues and Pull Requests to improve this project.

## Acknowledgments

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Google Gemma](https://ai.google.dev/gemma)
- [BelleGroup](https://github.com/LianjiaTech/BELLE)
