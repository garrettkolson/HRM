"""
Complete Training Pipeline for C# Programming with HRM
Automated script to train HRM model on C# code generation tasks
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

import wandb
import torch


class CSharpHRMTrainer:
    """Complete training pipeline for C# programming with HRM"""
    
    def __init__(self, 
                 dataset_dir: str = "data/csharp_raw",
                 output_dir: str = "data/csharp-programming-1k", 
                 subsample_size: int = 1000,
                 num_augmentations: int = 100,
                 use_wandb: bool = True):
        
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.subsample_size = subsample_size
        self.num_augmentations = num_augmentations
        self.use_wandb = use_wandb
        
        self.project_name = "CSharp-Programming-HRM"
        self.checkpoint_dir = None
        
    def validate_environment(self):
        """Validate that the environment is set up correctly"""
        print("🔍 Validating environment...")
        
        # Check Python packages
        required_packages = [
            'torch', 'einops', 'tqdm', 'pydantic', 'argdantic', 
            'omegaconf', 'hydra-core', 'huggingface_hub'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("   Run: pip install " + " ".join(missing_packages))
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available. Training will be slower on CPU.")
        
        # Check required files
        required_files = [
            "pretrain.py",
            "dataset/csharp_tokenizer.py", 
            "dataset/build_csharp_dataset.py",
            "config/cfg_csharp.yaml",
            "evaluate_csharp.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing required files: {', '.join(missing_files)}")
            return False
            
        print("✅ Environment validation passed")
        return True
    
    def prepare_dataset(self):
        """Prepare the C# programming dataset"""
        print("📊 Preparing C# programming dataset...")
        
        # Check if dataset directory exists
        if not os.path.exists(self.dataset_dir):
            print(f"📁 Creating dataset directory: {self.dataset_dir}")
            os.makedirs(self.dataset_dir, exist_ok=True)
            
            # Create sample data if none exists
            self._create_sample_dataset()
        
        # Build the processed dataset
        print("🔨 Building processed dataset...")
        cmd = [
            sys.executable, "dataset/build_csharp_dataset.py",
            "--input-dir", self.dataset_dir,
            "--output-dir", self.output_dir,
            "--subsample-size", str(self.subsample_size),
            "--num-aug", str(self.num_augmentations)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Dataset building failed:")
            print(result.stderr)
            return False
        
        print("✅ Dataset prepared successfully")
        return True
    
    def _create_sample_dataset(self):
        """Create a sample dataset for demonstration"""
        print("📝 Creating sample C# programming dataset...")
        
        sample_train_data = [
            {
                "input": "// Complete this method to calculate factorial\npublic static int Factorial(int n)\n{\n    // TODO: implement\n}",
                "target": "public static int Factorial(int n)\n{\n    if (n <= 1) return 1;\n    return n * Factorial(n - 1);\n}"
            },
            {
                "input": "// Fix the bug in this method\npublic static bool IsPrime(int n)\n{\n    if (n < 2) return false;\n    for (int i = 2; i < n; i++)\n        if (n % i == 0) return false;\n    return true;\n}",
                "target": "public static bool IsPrime(int n)\n{\n    if (n < 2) return false;\n    for (int i = 2; i <= Math.Sqrt(n); i++)\n        if (n % i == 0) return false;\n    return true;\n}"
            },
            {
                "input": "// Implement binary search\npublic static int BinarySearch(int[] arr, int target)\n{\n    // TODO: implement\n}",
                "target": "public static int BinarySearch(int[] arr, int target)\n{\n    int left = 0, right = arr.Length - 1;\n    while (left <= right)\n    {\n        int mid = left + (right - left) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}"
            }
        ]
        
        sample_test_data = [
            {
                "input": "// Complete this method to reverse a string\npublic static string Reverse(string input)\n{\n    // TODO: implement\n}",
                "target": "public static string Reverse(string input)\n{\n    char[] chars = input.ToCharArray();\n    Array.Reverse(chars);\n    return new string(chars);\n}"
            }
        ]
        
        # Duplicate sample data to reach subsample_size
        train_data = []
        for i in range(max(self.subsample_size, len(sample_train_data))):
            train_data.append(sample_train_data[i % len(sample_train_data)])
        
        # Save sample data
        with open(os.path.join(self.dataset_dir, "train.json"), 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(os.path.join(self.dataset_dir, "test.json"), 'w') as f:
            json.dump(sample_test_data, f, indent=2)
        
        print(f"✅ Created sample dataset with {len(train_data)} training examples")
    
    def setup_wandb(self):
        """Set up Weights & Biases logging"""
        if not self.use_wandb:
            os.environ["WANDB_MODE"] = "disabled"
            return True
        
        try:
            # Check if user is logged in
            result = subprocess.run(['wandb', 'status'], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️  W&B not logged in. Run 'wandb login' to enable experiment tracking")
                print("   Continuing without W&B logging...")
                os.environ["WANDB_MODE"] = "disabled"
            else:
                print("✅ W&B configured successfully")
            return True
        except FileNotFoundError:
            print("⚠️  W&B CLI not found. Install with: pip install wandb")
            print("   Continuing without W&B logging...")
            os.environ["WANDB_MODE"] = "disabled"
            return True
    
    def train_model(self, 
                   epochs: int = 5000,
                   eval_interval: int = 500,
                   global_batch_size: int = 128,
                   learning_rate: float = 5e-5,
                   num_gpus: int = 1):
        """Train the HRM model"""
        print("🚀 Starting HRM training...")
        
        # Set up environment variables
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Construct training command
        if num_gpus > 1:
            cmd = [
                "torchrun", 
                f"--nproc-per-node={num_gpus}",
                "pretrain.py",
                "--config-name=cfg_csharp"
            ]
        else:
            cmd = [sys.executable, "pretrain.py", "--config-name=cfg_csharp"]
        
        # Add parameters
        cmd.extend([
            f"data_path={self.output_dir}",
            f"epochs={epochs}",
            f"eval_interval={eval_interval}",
            f"global_batch_size={global_batch_size}",
            f"lr={learning_rate}",
            f"puzzle_emb_lr={learning_rate}",
            f"project_name={self.project_name}"
        ])
        
        print(f"🔧 Training command: {' '.join(cmd)}")
        
        # Run training
        try:
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                print("✅ Training completed successfully")
                return True
            else:
                print(f"❌ Training failed with return code {result.returncode}")
                return False
                
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            return False
        except Exception as e:
            print(f"❌ Training failed with exception: {e}")
            return False
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint from training"""
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            return None
        
        # Look for project directory
        project_dirs = [d for d in os.listdir(checkpoints_dir) 
                       if os.path.isdir(os.path.join(checkpoints_dir, d))]
        
        if not project_dirs:
            return None
        
        # Find latest checkpoint
        latest_checkpoint = None
        for project_dir in project_dirs:
            project_path = os.path.join(checkpoints_dir, project_dir)
            run_dirs = [d for d in os.listdir(project_path) 
                       if os.path.isdir(os.path.join(project_path, d))]
            
            for run_dir in run_dirs:
                run_path = os.path.join(project_path, run_dir)
                checkpoints = [f for f in os.listdir(run_path) 
                             if f.startswith("step_") and not f.endswith(".json")]
                
                if checkpoints:
                    # Get latest step
                    latest_step = max(int(f.split("_")[1]) for f in checkpoints)
                    checkpoint_path = os.path.join(run_path, f"step_{latest_step}")
                    if os.path.exists(checkpoint_path):
                        latest_checkpoint = checkpoint_path
        
        return latest_checkpoint
    
    def evaluate_model(self, checkpoint_path: Optional[str] = None):
        """Evaluate the trained model"""
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None:
            print("❌ No checkpoint found for evaluation")
            return False
        
        print(f"📊 Evaluating model: {checkpoint_path}")
        
        cmd = [
            sys.executable, "evaluate_csharp.py",
            "--config-name=cfg_csharp",
            f"checkpoint={checkpoint_path}"
        ]
        
        result = subprocess.run(cmd, text=True)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully")
            return True
        else:
            print(f"❌ Evaluation failed")
            return False
    
    def run_complete_pipeline(self,
                            epochs: int = 5000,
                            eval_interval: int = 500,
                            global_batch_size: int = 128,
                            learning_rate: float = 5e-5,
                            num_gpus: int = 1,
                            evaluate_after_training: bool = True):
        """Run the complete training pipeline"""
        print("🎯 Starting Complete C# HRM Training Pipeline")
        print("-" * 60)
        
        # Step 1: Validate environment
        if not self.validate_environment():
            print("❌ Environment validation failed")
            return False
        
        # Step 2: Prepare dataset
        if not self.prepare_dataset():
            print("❌ Dataset preparation failed")
            return False
        
        # Step 3: Setup W&B
        self.setup_wandb()
        
        # Step 4: Train model
        if not self.train_model(
            epochs=epochs,
            eval_interval=eval_interval,
            global_batch_size=global_batch_size,
            learning_rate=learning_rate,
            num_gpus=num_gpus
        ):
            print("❌ Model training failed")
            return False
        
        # Step 5: Evaluate model
        if evaluate_after_training:
            self.evaluate_model()
        
        print("🎉 Complete pipeline finished successfully!")
        print("-" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(description="Train HRM model on C# programming tasks")
    
    parser.add_argument("--dataset-dir", default="data/csharp_raw", 
                       help="Directory containing raw C# dataset")
    parser.add_argument("--output-dir", default="data/csharp-programming-1k",
                       help="Directory for processed dataset")
    parser.add_argument("--subsample-size", type=int, default=1000,
                       help="Number of training examples to use")
    parser.add_argument("--num-augmentations", type=int, default=100,
                       help="Number of augmentations per example")
    
    parser.add_argument("--epochs", type=int, default=5000,
                       help="Number of training epochs")
    parser.add_argument("--eval-interval", type=int, default=500,
                       help="Evaluation interval")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Global batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs to use")
    
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--no-evaluation", action="store_true",
                       help="Skip evaluation after training")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CSharpHRMTrainer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        subsample_size=args.subsample_size,
        num_augmentations=args.num_augmentations,
        use_wandb=not args.no_wandb
    )
    
    # Run pipeline
    success = trainer.run_complete_pipeline(
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        global_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_gpus=args.num_gpus,
        evaluate_after_training=not args.no_evaluation
    )
    
    if success:
        print("🎯 Training pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Training pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()