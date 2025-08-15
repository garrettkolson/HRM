import json
import os

def fix_large_file():
    """Fix the part_004 file that exceeds 100MB"""
    
    # Load the large file
    with open('c-codes-in-train-dataset_part_004.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Part 004 has {len(data)} items")
    
    # Split it into two parts
    mid_point = len(data) // 2
    part_4a = data[:mid_point]
    part_4b = data[mid_point:]
    
    # Write the first half back to part_004
    with open('c-codes-in-train-dataset_part_004.json', 'w', encoding='utf-8') as f:
        json.dump(part_4a, f, indent=1, ensure_ascii=False)
    
    # Write the second half to part_005
    with open('c-codes-in-train-dataset_part_005.json', 'w', encoding='utf-8') as f:
        json.dump(part_4b, f, indent=1, ensure_ascii=False)
    
    # Check file sizes
    size_004 = os.path.getsize('c-codes-in-train-dataset_part_004.json') / (1024 * 1024)
    size_005 = os.path.getsize('c-codes-in-train-dataset_part_005.json') / (1024 * 1024)
    
    print(f"Part 004: {len(part_4a)} items, {size_004:.2f} MB")
    print(f"Part 005: {len(part_4b)} items, {size_005:.2f} MB")

if __name__ == "__main__":
    fix_large_file()