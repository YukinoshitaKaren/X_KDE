from typing import Dict
import re
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from functools import partial
import json
from tqdm import tqdm
import argparse

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_single_example(example, tokenizer):
    texts = []
    ids = []

    conversations = []
    for conv in example['conversations']:
        role = "user" if conv['from'] == "human" else "assistant"
        conversations.append({"role": role, "content": conv['value']})
    
    text = tokenizer.apply_chat_template(conversations, tokenize=False)
    
    texts.extend([text])
    ids.extend([example['id']])
    
    return {"text": texts, "id": ids}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate JSON file using OpenAI API.")
    parser.add_argument("--input_file", default="your_input_file")
    parser.add_argument("--tokenizer_path", default="your_tokenizer_path")
    args = parser.parse_args()

    input_file = args.input_file
    input_name = input_file.split('/')[-1].rstrip('.json') + '_tok'
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    processed_data = []
    for example in tqdm(data, desc="process"):
        result = process_single_example(example, tokenizer)
        for text, id in zip(result['text'], result['id']):
            processed_data.append({"text": text, "id": id})
    
    dataset = Dataset.from_list(processed_data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(f"sft/data/{input_name}")
    