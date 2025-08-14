# C# Programming Training with HRM

This guide provides complete instructions for training the Hierarchical Reasoning Model (HRM) on C# programming tasks.

## Quick Start 🚀

### Option 1: Automated Pipeline (Recommended)
```bash
# Run complete training pipeline with defaults
python train_csharp_hrm.py

# Custom configuration
python train_csharp_hrm.py \
    --dataset-dir data/my_csharp_data \
    --subsample-size 2000 \
    --epochs 10000 \
    --batch-size 256 \
    --num-gpus 2
```

### Option 2: Manual Step-by-Step
```bash
# 1. Setup environment
chmod +x setup_csharp_training.sh
./setup_csharp_training.sh

# 2. Build dataset
python dataset/build_csharp_dataset.py \
    --input-dir data/csharp_raw \
    --output-dir data/csharp-programming-1k \
    --subsample-size 1000

# 3. Train model
python pretrain.py --config-name cfg_csharp data_path=data/csharp-programming-1k

# 4. Evaluate model
python evaluate_csharp.py checkpoint=checkpoints/path/to/checkpoint
```

## Dataset Format 📊

Your C# dataset should be structured as JSON files with input/target pairs:

### Directory Structure
```
data/csharp_raw/
├── train.json
└── test.json
```

### JSON Format
```json
[
    {
        "input": "// Complete this method\\npublic static int Add(int a, int b)\\n{\\n    // TODO\\n}",
        "target": "public static int Add(int a, int b)\\n{\\n    return a + b;\\n}"
    },
    {
        "input": "// Fix the bug\\npublic static bool IsEven(int n)\\n{\\n    return n % 2 = 0;\\n}",
        "target": "public static bool IsEven(int n)\\n{\\n    return n % 2 == 0;\\n}"
    }
]
```

## Task Types Supported 🎯

1. **Code Completion**: Complete partial C# methods/classes
2. **Bug Fixing**: Correct syntax and logic errors
3. **Code Refactoring**: Improve code quality and structure
4. **Function Generation**: Generate complete functions from descriptions

## Configuration ⚙️

### Key Parameters in `config/cfg_csharp.yaml`

```yaml
# Model Architecture
arch:
  H_cycles: 6        # High-level planning steps
  L_cycles: 12       # Low-level implementation steps
  hidden_size: 512   # Model dimension
  
# Training
epochs: 5000         # Total training epochs
global_batch_size: 128
lr: 5e-5            # Learning rate

# Data
max_seq_length: 1024 # Maximum code sequence length
```

### Hyperparameter Tuning Guidelines

| Parameter | Small Dataset (1K) | Large Dataset (10K+) | Description |
|-----------|-------------------|---------------------|-------------|
| `lr` | 5e-5 to 1e-4 | 1e-4 to 3e-4 | Learning rate |
| `H_cycles` | 4-6 | 6-8 | Planning depth |
| `L_cycles` | 8-12 | 12-16 | Implementation detail |
| `epochs` | 5000-10000 | 1000-3000 | Training iterations |

## Evaluation Metrics 📈

The evaluation framework measures:

- **Exact Match**: Identical code generation
- **BLEU Score**: Token-level similarity
- **Compilation Success**: Syntactically valid C# code
- **Syntax Correctness**: Basic syntax validation
- **Functional Correctness**: Code compiles and runs

## Performance Expectations 📊

### Small Sample (1K examples):
- Training Time: ~6-8 hours on RTX 4070
- Expected Metrics:
  - Exact Match: 15-25%
  - BLEU Score: 60-75%
  - Compilation Success: 70-85%

### Large Sample (10K+ examples):
- Training Time: ~24-48 hours on 8x A100
- Expected Metrics:
  - Exact Match: 35-50%
  - BLEU Score: 75-85%
  - Compilation Success: 85-95%

## Troubleshooting 🔧

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python train_csharp_hrm.py --batch-size 64

# Or use gradient accumulation by modifying config
```

**2. Slow Training**
```bash
# Use multiple GPUs
python train_csharp_hrm.py --num-gpus 4

# Reduce sequence length
# Modify max_seq_length in config/cfg_csharp.yaml
```

**3. Poor Generation Quality**
- Increase `H_cycles` for better planning
- Add more training examples
- Improve data quality and diversity

**4. Compilation Errors in Generated Code**
- Check tokenizer vocabulary coverage
- Increase `L_cycles` for more detailed generation
- Validate training data quality

### Debugging Tips

1. **Monitor W&B Dashboard**: Track loss curves and metrics
2. **Check Intermediate Outputs**: Use `eval_save_outputs` in config
3. **Validate Dataset**: Manually inspect processed data files
4. **Test Tokenizer**: Run tokenizer tests on your specific code patterns

## Advanced Usage 🔬

### Custom Tokenizer
Modify `dataset/csharp_tokenizer.py` to handle domain-specific patterns:
```python
# Add custom tokens for your codebase
custom_tokens = ['MyCustomClass', 'SpecialMethod']
vocab_list.extend(custom_tokens)
```

### Multi-Task Training
Structure your dataset with task-specific prefixes:
```json
{
    "input": "[COMPLETION] // Complete this method\\n...",
    "target": "public static void Method() { ... }"
}
```

### Transfer Learning
Start from a pre-trained checkpoint:
```bash
python pretrain.py --config-name cfg_csharp \
    checkpoint=path/to/pretrained/model \
    lr=1e-5  # Lower learning rate for fine-tuning
```

## File Reference 📁

| File | Purpose |
|------|---------|
| `train_csharp_hrm.py` | Complete training pipeline |
| `dataset/csharp_tokenizer.py` | C# code tokenization |
| `dataset/build_csharp_dataset.py` | Dataset preprocessing |
| `config/cfg_csharp.yaml` | Training configuration |
| `evaluate_csharp.py` | Model evaluation |
| `setup_csharp_training.sh` | Environment setup |

## Best Practices 💡

1. **Data Quality**: Ensure clean, well-formatted C# code
2. **Balanced Dataset**: Include diverse coding patterns and difficulty levels
3. **Validation Split**: Keep 10-20% of data for testing
4. **Regular Checkpointing**: Monitor training progress frequently
5. **Hyperparameter Search**: Experiment with different H_cycles/L_cycles ratios
6. **Early Stopping**: Stop training when validation metrics plateau

## Contributing 🤝

To extend this C# training framework:

1. **Add New Task Types**: Modify tokenizer and dataset builder
2. **Custom Evaluation Metrics**: Extend `CSharpCodeEvaluator` class
3. **Architecture Improvements**: Modify HRM model configuration
4. **Dataset Augmentation**: Add new augmentation strategies

## Citation 📜

If you use this C# training framework, please cite the original HRM paper:

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```