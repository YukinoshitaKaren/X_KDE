import os
import json
import re
import random
import argparse

def natural_sort_key(file_name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', file_name)]

def merge_json_files(input_dir, output_file):
    merged_data = []
    cnt = 0
    file_list = sorted(os.listdir(input_dir), key=natural_sort_key)
    for file_name in file_list:
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):  # 确保文件内容是列表
                        cnt += 1
                        merged_data.extend(data)
                    else:
                        print(f"Skipping {file_path}: Content is not a list.")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    random.shuffle(merged_data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    print(cnt)
    print(f"Merged JSON data has been saved to {output_file}")

parser = argparse.ArgumentParser(description="Translate JSON file using OpenAI API.")
parser.add_argument("--input_dir", default="your_path")
parser.add_argument("--output_file", default="your_path")
args = parser.parse_args()

input_dir = args.input_dir 
output_file = args.output_file
merge_json_files(input_dir, output_file)