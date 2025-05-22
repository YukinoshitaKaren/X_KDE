from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch
from datasets import load_dataset
import json
from tqdm import tqdm
from random import sample
import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
loss_fn = nn.CrossEntropyLoss(reduction='none')
parser = argparse.ArgumentParser(description="sampling argument")
parser.add_argument('--begin_index', type=int,default=0)
parser.add_argument('--data_length', type=int,default=51232)
parser.add_argument('--data_file', type=str)
args = parser.parse_args()

data_path = args.data_file
save_path = data_path.split("/")[-1]
data = []
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

#Set here
#config the NLLB model here
rm_model_base = AutoModelForSeq2SeqLM.from_pretrained("huggingface/nllb-200-distilled-600M",device_map='auto')
rm_model_base = rm_model_base.eval()
rm_tokenizer = AutoTokenizer.from_pretrained("huggingface/nllb-200-distilled-600M")
rm_model = (rm_model_base,rm_tokenizer)

def MultiLigual_Alighment_reward_fuction(tokenizer,rm_model,prediction,labels=None):
    model = rm_model[0]
    tokenizer = rm_model[1]
    target_lang = 'eng_Latn'   
    source_lang = 'zho_Hans'
    tokenizer.src_lang = source_lang
    x = tokenizer(prediction, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
    tokenizer.src_lang = target_lang
    y = tokenizer(labels, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
    results = []
    with torch.no_grad():
        output = model(**x, labels=y.input_ids)
        loss = output.loss

        for i in range(output.logits.size(0)):
            pre = output.logits[i]
            lab = y.input_ids[i]
            result = loss_fn(pre.view(-1, output.logits.size(-1)), lab.view(-1)).mean().cpu().detach().numpy().tolist()
            results.append(1/result)
    
    torch.cuda.empty_cache()
    return results


begin_index = args.begin_index
data_length = args.data_length
end_index = min(len(data), begin_index + data_length)
# end_index = min(len(data), begin_index + data_length)
print("begin_index: {}, end_index: {}".format(begin_index, end_index))

result = []
for i in tqdm(range(begin_index, end_index)):
    item = data[i]
    lang = "Chinese"
    item["answers"] = []
    # unique_list = list(dict.fromkeys(item['answer']))
    for id, it in tqdm(enumerate(item['answer'])): 
        ret = {
            # 'prompt' : 
            'generated' : it
        }
        
        reward_list = []
        # match = re.search(r'\?(.*?)\n\[Query\]:', item['conversations'][0]['value'])
        # output = match.group(1).strip()
        input_answer = it['generated'].lstrip()
        # output = item['alt']
        output = item['conversations'][1]['value']
        if input_answer == item['conversations'][1]['value']:
            continue
        reward_list = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model,input_answer,output)
        torch.cuda.empty_cache()
        it['nllb-200-distilled-600M-reward-mean'] =  sum(reward_list)/len(reward_list)
        it['nllb-200-distilled-600M-reward-max'] =  max(reward_list)
        it['nllb-200-distilled-600M-reawrdlist'] =  reward_list
        item["answers"].append(it)
    item.pop('answer')
    if len(item['answers']) != 0 :
        result.append(item)
    
with open(f'./ORPO/result/vllm/pre/{save_path}_{begin_index}.json', 'w', encoding='utf-8') as fw:
    json.dump(result, fw, indent=2, ensure_ascii=False)