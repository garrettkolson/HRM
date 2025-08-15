import json
import os
import sys

def split_json_file(input_file, max_size_mb=100):
    """
    Split a large JSON file into smaller chunks while preserving JSON structure.
    
    Args:
        input_file (str): Path to the input JSON file
        max_size_mb (int): Maximum size of each output file in MB
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    print(f"Reading {input_file}...")
    
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain an array at the root level")
    
    print(f"Total items in dataset: {len(data)}")
    
    # Calculate approximate items per chunk
    total_size = os.path.getsize(input_file)
    items_per_chunk = int((len(data) * max_size_bytes) / total_size)
    
    # Ensure we have at least some items per chunk
    if items_per_chunk < 1:
        items_per_chunk = 1
    
    print(f"Estimated items per chunk: {items_per_chunk}")
    
    chunk_number = 1
    start_idx = 0
    
    while start_idx < len(data):
        # Determine end index for this chunk
        end_idx = min(start_idx + items_per_chunk, len(data))
        
        # Create chunk
        chunk = data[start_idx:end_idx]
        
        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_part_{chunk_number:03d}.json"
        
        # Write chunk to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=1, ensure_ascii=False)
        
        # Check file size and adjust if necessary
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"Created {output_file}: {len(chunk)} items, {file_size_mb:.2f} MB")
        
        # If file is too large, reduce chunk size and try again
        if file_size > max_size_bytes and len(chunk) > 1:
            os.remove(output_file)
            # Reduce chunk size by 80%
            reduced_items = max(1, int(len(chunk) * 0.8))
            end_idx = start_idx + reduced_items
            chunk = data[start_idx:end_idx]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=1, ensure_ascii=False)
            
            file_size = os.path.getsize(output_file)
            file_size_mb = file_size / (1024 * 1024)
            print(f"Adjusted {output_file}: {len(chunk)} items, {file_size_mb:.2f} MB")
        
        start_idx = end_idx
        chunk_number += 1
    
    print(f"\nSplit complete! Created {chunk_number - 1} files.")

if __name__ == "__main__":
    input_file = "c-codes-in-train-dataset.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)
    
    try:
        split_json_file(input_file, max_size_mb=100)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)