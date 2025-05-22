from vllm import LLM, SamplingParams
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
from tqdm import tqdm

def main(model_name, data_path, save_path, n_iter, n_temp):
    # tok.pad_token_id = tok.eos_token_id
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    data = data
    # templet = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
    input_prompt = ["[INST]" + i['conversations'][0]['value']+" [/INST] " for i in data]
    # Define a function to split list into chunks
    def chunk_list(input_list, chunk_size):
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]
    sampling_params = SamplingParams(n=n_iter,temperature=n_temp,max_tokens=2048)
    llm = LLM(model=model_name,dtype=torch.bfloat16,tensor_parallel_size=4, gpu_memory_utilization=0.6,swap_space=6)
    generated_ = []
    chunk_size = 10000
    # Split the input prompts into chunks and process each chunk
    for prompt_chunk in chunk_list(input_prompt, chunk_size):
        generations = llm.generate(prompt_chunk, sampling_params)
        for output in generations:
            generate_text = [o.text for o in output.outputs]
            generated_.append(generate_text)

    # for output in generations:
    result = []
    assert len(data) == len(generated_)
    for i,g in zip(data,generated_):
        i["answer"] = []
        g = list(set(g))

        for temp in g:
            # if temp == 
            ret = {
                "prompt": "[INST]" + i['conversations'][0]['value']+" [/INST] ",
                "generated": temp
            }
            i["answer"].append(ret)
        result.append(i)

    with open(f'./ORPO/result/vllm/{save_path}.json', 'w', encoding="utf-8") as wf:
        json.dump(result, wf,indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate JSON file using OpenAI API.")
    parser.add_argument("--model", default="your_mode_path")
    parser.add_argument("--data_path", default="your_data_path", help="Path to the input JSON file.")
    args = parser.parse_args()
    model_name = args.model
    data_path = args.data_path 
    save_path = model_name.split("/")[-3]
    n_iter = 5
    n_temp = 0.8
    main(model_name, data_path, save_path, n_iter, n_temp)