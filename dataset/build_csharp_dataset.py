"""
Build C# Programming Dataset for HRM Training
Converts C# code completion/generation tasks into HRM-compatible format
"""

from typing import Optional, Dict, List, Tuple
import os
import json
import numpy as np
from pathlib import Path

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata
from csharp_tokenizer import CSharpTokenizer


cli = ArgParser()


class CSharpDatasetConfig(BaseModel):
    input_dir: str = "data/csharp_raw"  # Directory with C# training data
    output_dir: str = "data/csharp-programming-1k"
    
    subsample_size: Optional[int] = 1000
    num_aug: int = 100  # Number of augmentations per example
    max_seq_length: int = 1024
    
    task_types: List[str] = ["completion", "bug_fix", "refactor"]


class CSharpDatasetBuilder:
    def __init__(self, config: CSharpDatasetConfig):
        self.config = config
        self.tokenizer = CSharpTokenizer()
        
    def load_raw_data(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load raw C# code pairs from input directory"""
        data = {"train": [], "test": []}
        
        # Expected format: JSON files with input/output pairs
        for split in ["train", "test"]:
            split_file = os.path.join(self.config.input_dir, f"{split}.json")
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    raw_data = json.load(f)
                    
                for item in raw_data:
                    input_code = item.get("input", "")
                    target_code = item.get("target", "")
                    if input_code and target_code:
                        data[split].append((input_code, target_code))
                        
        return data
    
    def augment_code_pair(self, input_code: str, target_code: str) -> List[Tuple[str, str]]:
        """Generate augmented versions of a code pair"""
        augmented_pairs = [(input_code, target_code)]  # Original
        
        # Simple augmentation strategies
        for i in range(self.config.num_aug):
            # Variable name substitutions
            aug_input, aug_target = self._substitute_variables(input_code, target_code, seed=i)
            augmented_pairs.append((aug_input, aug_target))
            
        return augmented_pairs
    
    def _substitute_variables(self, input_code: str, target_code: str, seed: int) -> Tuple[str, str]:
        """Simple variable name substitution for augmentation"""
        np.random.seed(seed)
        
        # Common variable names to substitute
        var_mappings = {
            'i': np.random.choice(['index', 'counter', 'idx']),
            'j': np.random.choice(['jIndex', 'j2', 'innerIdx']),
            'temp': np.random.choice(['tmp', 'temporary', 'tempVar']),
            'result': np.random.choice(['res', 'output', 'answer']),
            'value': np.random.choice(['val', 'item', 'element']),
        }
        
        aug_input = input_code
        aug_target = target_code
        
        for old_var, new_var in var_mappings.items():
            # Simple word boundary replacement
            import re
            pattern = r'\b' + re.escape(old_var) + r'\b'
            aug_input = re.sub(pattern, new_var, aug_input)
            aug_target = re.sub(pattern, new_var, aug_target)
            
        return aug_input, aug_target
    
    def process_split(self, split: str, raw_pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        """Process a data split into HRM format"""
        
        # Subsample if specified
        if split == "train" and self.config.subsample_size:
            if len(raw_pairs) > self.config.subsample_size:
                indices = np.random.choice(len(raw_pairs), self.config.subsample_size, replace=False)
                raw_pairs = [raw_pairs[i] for i in indices]
        
        # Generate augmented data
        all_pairs = []
        for input_code, target_code in tqdm(raw_pairs, desc=f"Processing {split}"):
            if split == "train":
                aug_pairs = self.augment_code_pair(input_code, target_code)
                all_pairs.extend(aug_pairs)
            else:
                all_pairs.append((input_code, target_code))
        
        # Convert to token sequences
        inputs = []
        labels = []
        puzzle_identifiers = []
        puzzle_indices = [0]  # Start index
        group_indices = [0]  # Start index
        
        current_puzzle_id = 0
        current_example_id = 0
        
        for pair_idx, (input_code, target_code) in enumerate(tqdm(all_pairs, desc="Tokenizing")):
            # Encode the code pair
            input_ids, target_ids = self.tokenizer.encode_code_pair(
                input_code, target_code, self.config.max_seq_length
            )
            
            inputs.append(input_ids)
            labels.append(target_ids)
            puzzle_identifiers.append(current_puzzle_id)
            current_example_id += 1
            
            # Each code pair is treated as a separate puzzle
            current_puzzle_id += 1
            puzzle_indices.append(current_example_id)
            
        # All examples are in one group for simplicity
        group_indices.append(current_puzzle_id)
        
        return {
            "inputs": np.array(inputs, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
            "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
            "group_indices": np.array(group_indices, dtype=np.int32),
        }
    
    def build_dataset(self):
        """Build the complete dataset"""
        print("Loading raw data...")
        raw_data = self.load_raw_data()
        
        if not raw_data["train"]:
            raise ValueError(f"No training data found in {self.config.input_dir}")
        
        print(f"Found {len(raw_data['train'])} training examples, {len(raw_data['test'])} test examples")
        
        # Process each split
        for split in ["train", "test"]:
            if not raw_data[split]:
                continue
                
            print(f"\nProcessing {split} split...")
            split_data = self.process_split(split, raw_data[split])
            
            # Create output directory
            split_dir = os.path.join(self.config.output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Save data arrays
            for field_name, data_array in split_data.items():
                output_path = os.path.join(split_dir, f"default__{field_name}.npy")
                np.save(output_path, data_array)
                print(f"Saved {field_name}: {data_array.shape}")
        
        # Create metadata
        metadata = PuzzleDatasetMetadata(
            pad_id=self.tokenizer.vocab_to_id['<PAD>'],
            ignore_label_id=self.tokenizer.vocab_to_id['<PAD>'],  # Ignore padding in loss
            blank_identifier_id=self.tokenizer.vocab_to_id['<UNK>'],
            
            vocab_size=self.tokenizer.vocab_size,
            seq_len=self.config.max_seq_length,
            num_puzzle_identifiers=len(raw_data["train"]) * (1 + self.config.num_aug),
            
            total_groups=1,  # Single group containing all examples
            mean_puzzle_examples=1.0,  # Each puzzle has exactly one example
            
            sets=["default"]  # Single set name
        )
        
        # Save metadata for each split
        for split in ["train", "test"]:
            if raw_data[split]:
                metadata_path = os.path.join(self.config.output_dir, split, "dataset.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.model_dump(), f, indent=2)
        
        print(f"\nDataset created successfully in {self.config.output_dir}")
        print(f"Vocabulary size: {metadata.vocab_size}")
        print(f"Sequence length: {metadata.seq_len}")
        print(f"Total puzzles: {metadata.num_puzzle_identifiers}")


@cli.main(CSharpDatasetConfig)
def main(config: CSharpDatasetConfig):
    """Main entry point for dataset building"""
    builder = CSharpDatasetBuilder(config)
    builder.build_dataset()


if __name__ == "__main__":
    main()