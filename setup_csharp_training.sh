#!/bin/bash

# Setup script for C# Programming Training with HRM
# Run this script to prepare the environment and start training

set -e  # Exit on any error

echo "🚀 Setting up C# Programming Training with HRM"

# Check if running in the correct directory
if [ ! -f "pretrain.py" ]; then
    echo "❌ Error: Please run this script from the HRM repository root directory"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. Make sure CUDA is properly installed."
fi

# Check Python environment
echo "📋 Checking Python environment..."
python3 --version
pip3 --version

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip3 install -r requirements.txt
else
    echo "⚠️  requirements.txt not found. Installing core dependencies..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip3 install einops tqdm pydantic argdantic wandb omegaconf hydra-core huggingface_hub
fi

# Check if dataset directory exists
DATASET_DIR="data/csharp_raw"
if [ ! -d "$DATASET_DIR" ]; then
    echo "📁 Creating dataset directory: $DATASET_DIR"
    mkdir -p "$DATASET_DIR"
    
    echo "📝 Creating sample dataset structure..."
    # Create sample training data structure
    cat > "$DATASET_DIR/train.json" << 'EOF'
[
    {
        "input": "// Complete this method to calculate factorial\npublic static int Factorial(int n)\n{\n    // TODO: implement\n}",
        "target": "public static int Factorial(int n)\n{\n    if (n <= 1) return 1;\n    return n * Factorial(n - 1);\n}"
    },
    {
        "input": "// Fix the bug in this method\npublic static bool IsPrime(int n)\n{\n    if (n < 2) return false;\n    for (int i = 2; i < n; i++)\n        if (n % i == 0) return false;\n    return true;\n}",
        "target": "public static bool IsPrime(int n)\n{\n    if (n < 2) return false;\n    for (int i = 2; i <= Math.Sqrt(n); i++)\n        if (n % i == 0) return false;\n    return true;\n}"
    }
]
EOF

    cat > "$DATASET_DIR/test.json" << 'EOF'
[
    {
        "input": "// Complete this method to reverse a string\npublic static string Reverse(string input)\n{\n    // TODO: implement\n}",
        "target": "public static string Reverse(string input)\n{\n    char[] chars = input.ToCharArray();\n    Array.Reverse(chars);\n    return new string(chars);\n}"
    }
]
EOF

    echo "✅ Sample dataset created. Replace with your actual C# dataset."
fi

# Build the dataset
echo "🔨 Building C# programming dataset..."
python3 dataset/build_csharp_dataset.py \
    --input-dir "$DATASET_DIR" \
    --output-dir "data/csharp-programming-1k" \
    --subsample-size 1000 \
    --num-aug 100

# Check if W&B is configured
if ! wandb status &> /dev/null; then
    echo "📊 Weights & Biases not configured. Please run 'wandb login' to enable experiment tracking."
    echo "   You can also disable W&B by setting WANDB_MODE=disabled"
fi

# Set environment variables for optimal performance
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "✅ Setup complete!"
echo ""
echo "🎯 To start training, run:"
echo "   # Single GPU:"
echo "   python3 pretrain.py --config-name cfg_csharp data_path=data/csharp-programming-1k"
echo ""
echo "   # Multi-GPU (8 GPUs):"
echo "   OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py --config-name cfg_csharp data_path=data/csharp-programming-1k"
echo ""
echo "📈 Monitor training progress:"
echo "   - Check W&B dashboard for metrics and losses"
echo "   - Checkpoints will be saved in checkpoints/ directory"
echo "   - Evaluation runs every 500 epochs by default"
echo ""
echo "🔧 Configuration files:"
echo "   - Main config: config/cfg_csharp.yaml"
echo "   - Tokenizer: dataset/csharp_tokenizer.py"
echo "   - Dataset builder: dataset/build_csharp_dataset.py"