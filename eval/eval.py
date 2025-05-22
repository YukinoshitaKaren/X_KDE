import typing
from itertools import chain
from typing import List, Optional
import numpy as np
import random
import argparse
import math
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from tqdm import tqdm
def knowledge_edit_template(prompt, target_new, question, edit_language):
    if edit_language == 'en':
        return "Please acknowledge the updated information provided below and respond to the subsequent query in English.\n\n[Updated Information]:\n" \
            + prompt + " " + target_new + "\n\n[Query]:\n" + question
    else:
        return "Please acknowledge the updated information provided below and respond to the subsequent query in Chinese.\n\n[Updated Information]:\n" \
            + prompt + " " + target_new + "\n\n[Query]:\n" + question

def icl_lm_eval(
        model,
        model_name,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:0')
    model_name = model_name.split('/')[-1]
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'phi' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(f'<|endoftext|><|user|>\n{x}<|end|>\n<|assistant|>\n{target}<|endoftext|>', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-(target_ids.size(1)+2):-2].squeeze()
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
            ans.squeeze().detach().cpu().numpy().tolist(), \
            target_ids.squeeze().detach().cpu().numpy().tolist()
    elif 'llama3.1' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{x}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target}<|eot_id|>', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-(target_ids.size(1)+2):-2].squeeze()
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
            ans.squeeze().detach().cpu().numpy().tolist(), \
            target_ids.squeeze().detach().cpu().numpy().tolist()
    elif 'llama2' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(f' [INST] {x} [/INST] ', return_tensors='pt')   
        # encodings = tokenizer(f'{x} {target}', return_tensors='pt')     
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]   
        
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
            ans.squeeze().detach().cpu().numpy().tolist(), \
            target_ids.squeeze().detach().cpu().numpy().tolist()
    elif 'qwen' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-(target_ids.size(1)+2):-2].squeeze()
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
            ans.squeeze().detach().cpu().numpy().tolist(), \
            target_ids.squeeze().detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist(), \
                ans.squeeze().detach().cpu().numpy().tolist(), \
                target_ids.squeeze().detach().cpu().numpy().tolist()

def main(model_name, model_true_name, data_path, save_path, source, target):
    device_map = 'auto'
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.pad_token_id = tok.eos_token_id
    test_data = json.load(open(data_path, 'r', encoding='utf-8'))
    all_metrics = []
    for record in tqdm(test_data):
        new_fact = knowledge_edit_template(record[source]["src"],record[source]["alt"],record[target]["src"],target)
        edit_acc, ans_ids, target_ids = icl_lm_eval(model, model_true_name, tok, [""],
                                                    record[target]["alt"], new_fact)
        ret = {
            "prompt" : new_fact,
            "rewrite_acc": edit_acc,
            "target": tok.decode(target_ids),
            "answer": tok.decode(ans_ids)
        }
        # print(edit_acc)
        ret['locality'] = {}
        ret['portability'] = {}
        ret['neighborhood'] = {}
        
        rephrase_acc, ans_ids, target_ids = icl_lm_eval(model, model_true_name, tok, [''],
                                record[target]["alt"], knowledge_edit_template(record[source]["src"],record[source]["alt"],record[target]["rephrase"],target))
        ret['rephrase_acc'] = rephrase_acc
            

        pre_neighbor = icl_lm_eval(model, model_true_name, tok, [''], record[target]['loc_ans'],
                                record[target]['loc'] , neighborhood=True)
        post_neighbor = icl_lm_eval(model, model_true_name, tok, [''], record[target]['loc_ans'],
                                    knowledge_edit_template(record[source]["src"],record[source]["alt"], record[target]['loc'],target), neighborhood=True)
        if type(pre_neighbor) is not list:
            pre_neighbor = [pre_neighbor, ]
        if type(post_neighbor) is not list:
            post_neighbor = [post_neighbor, ]
        assert len(pre_neighbor) == len(post_neighbor)

        ret['locality']['loc_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
        ret['locality']['loc_prompt'] = record[target]['loc']
        ret['locality']['loc_ground_truth'] = record[target]['loc_ans']
        ret['locality']['loc_pre_tokens'] = tok.decode(pre_neighbor)
        ret['locality']['loc_post_tokens'] = tok.decode(post_neighbor)
                
        portability_acc, ans_ids, target_ids = icl_lm_eval(model, model_true_name, tok, [''], record[target]['portability']['New Answer'],
                                            knowledge_edit_template(record[source]["src"], record[source]["alt"], record[target]['portability']['New Question'],target))
        ret['portability'][f'acc'] = portability_acc
        ret['portability'][f'target'] = tok.decode(target_ids)
        ret['portability'][f'answer'] = tok.decode(ans_ids)
        all_metrics.append(ret)

    print(source + '->' + target)
    print("Edit Success: ", round(sum([i["rewrite_acc"] for i in all_metrics])/len(all_metrics) * 100, 2))
    print("Rephrase Success: ", round(sum([i["rephrase_acc"] for i in all_metrics])/len(all_metrics) * 100, 2))
    print("Locality: ", round(sum([i["locality"]["loc_acc"] for i in all_metrics])/len(all_metrics) * 100, 2)) 
    print("Portability Reasoning: ", round(sum([i["portability"]['acc'] for i in all_metrics])/len(all_metrics) * 100, 2))
   
    if 'sft' in model_name:
        directory = f'./result/result/sft/'
    else:
        directory = f'./result/orpo/'
    os.makedirs(directory, exist_ok=True)
    file_path = f'{directory}{source}_{target}_{save_path}.json'
    with open(file_path, 'w', encoding='utf-8') as wf:
        json.dump(all_metrics, wf, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Translate JSON file using OpenAI API.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--source", default="zh")
    parser.add_argument("--target", default="en") 
    parser.add_argument("--model", default="your_path")
    parser.add_argument("--data_path", default="your_path", help="Path to the input JSON file.")
    args = parser.parse_args()
    model_name = args.model
    data_path = args.data_path 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    source = args.source
    target = args.target

    if 'llama2' in model_name:
        model_true_name = 'llama2'
    elif 'Qwen' in model_name:
        model_true_name = 'qwen'
    else:
        model_true_name = 'qwen'

    save_path = model_name.split("/")[-1]

    main(model_name, model_true_name, data_path, save_path, source, target)