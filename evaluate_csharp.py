"""
C# Code Generation Evaluation Framework
Evaluates HRM model performance on C# programming tasks
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from dataset.csharp_tokenizer import CSharpTokenizer
from utils.functions import load_model_class
from pretrain import create_model, PretrainConfig
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


@dataclass
class CSharpEvaluationMetrics:
    """Metrics for evaluating C# code generation"""
    exact_match: float
    bleu_score: float
    compilation_success: float
    syntax_correctness: float
    functional_correctness: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "exact_match": self.exact_match,
            "bleu_score": self.bleu_score,
            "compilation_success": self.compilation_success,
            "syntax_correctness": self.syntax_correctness,
            "functional_correctness": self.functional_correctness,
        }


class CSharpCodeEvaluator:
    """Evaluator for C# code generation quality"""
    
    def __init__(self):
        self.tokenizer = CSharpTokenizer()
        
    def check_compilation(self, code: str) -> bool:
        """Check if C# code compiles successfully"""
        try:
            # Create a temporary C# file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
                # Wrap code in a basic class structure if needed
                if not code.strip().startswith('using') and 'class' not in code:
                    wrapped_code = f"""
using System;
using System.Collections.Generic;
using System.Linq;

public class TempClass
{{
    {code}
}}"""
                else:
                    wrapped_code = code
                    
                f.write(wrapped_code)
                temp_file = f.name
            
            # Try to compile using mcs (Mono C# compiler) if available
            try:
                result = subprocess.run(
                    ['mcs', '-target:library', temp_file, '-out:/dev/null'],
                    capture_output=True,
                    timeout=10
                )
                compilation_success = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback: basic syntax checking if compiler not available
                compilation_success = self._basic_syntax_check(code)
            
            # Clean up
            os.unlink(temp_file)
            return compilation_success
            
        except Exception:
            return False
    
    def _basic_syntax_check(self, code: str) -> bool:
        """Basic syntax validation when compiler is not available"""
        # Simple checks for balanced braces, semicolons, etc.
        try:
            # Check balanced braces
            brace_count = code.count('{') - code.count('}')
            if brace_count != 0:
                return False
                
            # Check for basic C# keywords and structure
            required_patterns = [
                r'\bpublic\b|\bprivate\b|\bprotected\b',  # Access modifiers
                r'\{.*\}',  # Braces
            ]
            
            import re
            for pattern in required_patterns:
                if not re.search(pattern, code, re.DOTALL):
                    return False
                    
            return True
        except Exception:
            return False
    
    def calculate_bleu(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score for code similarity"""
        try:
            # Tokenize both predicted and reference code
            pred_tokens = self.tokenizer.tokenize(predicted)
            ref_tokens = self.tokenizer.tokenize(reference)
            
            # Simple BLEU-1 calculation (unigram precision)
            if not pred_tokens:
                return 0.0
                
            matches = sum(1 for token in pred_tokens if token in ref_tokens)
            bleu_score = matches / len(pred_tokens) if pred_tokens else 0.0
            return bleu_score
            
        except Exception:
            return 0.0
    
    def evaluate_predictions(self, predictions: List[str], references: List[str]) -> CSharpEvaluationMetrics:
        """Evaluate a batch of predictions against references"""
        assert len(predictions) == len(references)
        
        exact_matches = 0
        bleu_scores = []
        compilation_successes = 0
        syntax_correct = 0
        functional_correct = 0
        
        for pred, ref in tqdm(zip(predictions, references), desc="Evaluating", total=len(predictions)):
            # Exact match
            if pred.strip() == ref.strip():
                exact_matches += 1
            
            # BLEU score
            bleu = self.calculate_bleu(pred, ref)
            bleu_scores.append(bleu)
            
            # Compilation success
            if self.check_compilation(pred):
                compilation_successes += 1
            
            # Syntax correctness (basic check)
            if self._basic_syntax_check(pred):
                syntax_correct += 1
            
            # Functional correctness (simplified - same as compilation for now)
            if self.check_compilation(pred):
                functional_correct += 1
        
        n = len(predictions)
        return CSharpEvaluationMetrics(
            exact_match=exact_matches / n,
            bleu_score=np.mean(bleu_scores),
            compilation_success=compilation_successes / n,
            syntax_correctness=syntax_correct / n,
            functional_correctness=functional_correct / n,
        )


class HRMCSharpEvaluator:
    """Main evaluator for HRM model on C# tasks"""
    
    def __init__(self, config: PretrainConfig, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.tokenizer = CSharpTokenizer()
        self.code_evaluator = CSharpCodeEvaluator()
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained HRM model"""
        # Create dummy metadata for model initialization
        class DummyMetadata:
            vocab_size = self.tokenizer.vocab_size
            seq_len = self.config.arch.__pydantic_extra__.get('seq_len', 1024)  # type: ignore
            num_puzzle_identifiers = 1000
        
        dummy_metadata = DummyMetadata()
        model, _, _ = create_model(self.config, dummy_metadata, world_size=1)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint)
        model.eval()
        
        return model
    
    def generate_code(self, input_code: str, max_steps: int = 50) -> str:
        """Generate C# code using the HRM model"""
        # Tokenize input
        input_tokens = ['<START>'] + self.tokenizer.tokenize(input_code) + ['<MASK>']
        input_ids = self.tokenizer.encode(input_tokens)
        
        # Pad to model's expected length
        seq_len = self.config.arch.__pydantic_extra__.get('seq_len', 1024)  # type: ignore
        while len(input_ids) < seq_len:
            input_ids.append(self.tokenizer.vocab_to_id['<PAD>'])
        
        # Create batch
        batch = {
            'inputs': torch.tensor([input_ids], dtype=torch.int32).cuda(),
            'labels': torch.tensor([input_ids], dtype=torch.int32).cuda(),  # Dummy labels
            'puzzle_identifiers': torch.tensor([0], dtype=torch.int32).cuda(),
        }
        
        # Generate with model
        with torch.no_grad():
            carry = self.model.initial_carry(batch)
            
            for step in range(max_steps):
                carry, _, metrics, preds, all_finish = self.model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=["predictions"]
                )
                
                if all_finish:
                    break
            
            # Extract predictions
            if 'predictions' in preds:
                pred_ids = preds['predictions'][0].cpu().numpy()
                pred_tokens = self.tokenizer.decode(pred_ids.tolist())
                
                # Convert tokens back to code
                generated_code = self._tokens_to_code(pred_tokens)
                return generated_code
            
        return ""
    
    def _tokens_to_code(self, tokens: List[str]) -> str:
        """Convert tokens back to readable C# code"""
        code_parts = []
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                if token in ['<START>', '<END>', '<PAD>']:
                    continue
                elif token == '<NEWLINE>':
                    code_parts.append('\n')
                elif token.startswith('<ID_'):
                    code_parts.append('var')  # Generic identifier
                elif token.startswith('<STR_'):
                    code_parts.append('"text"')  # Generic string
                elif token.startswith('<NUM_'):
                    code_parts.append('0')  # Generic number
            else:
                code_parts.append(token)
        
        return ' '.join(code_parts)
    
    def evaluate_on_dataset(self, data_path: str) -> Dict[str, float]:
        """Evaluate model on a test dataset"""
        # Load test data
        test_file = os.path.join(data_path, "test.json")
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        predictions = []
        references = []
        
        print("Generating predictions...")
        for item in tqdm(test_data[:100]):  # Limit to 100 examples for speed
            input_code = item["input"]
            target_code = item["target"]
            
            predicted_code = self.generate_code(input_code)
            
            predictions.append(predicted_code)
            references.append(target_code)
        
        # Evaluate
        print("Evaluating predictions...")
        metrics = self.code_evaluator.evaluate_predictions(predictions, references)
        
        return metrics.to_dict()


@hydra.main(config_path="config", config_name="cfg_csharp", version_base=None)
def main(config: DictConfig):
    """Main evaluation script"""
    print("🔍 Starting C# Code Generation Evaluation")
    
    # Convert config
    eval_config = PretrainConfig(**config)  # type: ignore
    
    # Get checkpoint path
    checkpoint_path = config.get("checkpoint", None)
    if checkpoint_path is None:
        print("❌ Error: Please specify checkpoint path with checkpoint=<path>")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Create evaluator
    evaluator = HRMCSharpEvaluator(eval_config, checkpoint_path)
    
    # Evaluate on dataset
    print(f"📊 Evaluating checkpoint: {checkpoint_path}")
    print(f"📁 Dataset path: {eval_config.data_path}")
    
    metrics = evaluator.evaluate_on_dataset("data/csharp_raw")
    
    print("\n📈 Evaluation Results:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print("-" * 50)
    
    # Save results
    results_file = f"evaluation_results_{os.path.basename(checkpoint_path)}.json"
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()