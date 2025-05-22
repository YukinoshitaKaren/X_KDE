import json
import re
import argparse
parser = argparse.ArgumentParser(description="Translate JSON file using OpenAI API.")
parser.add_argument("--input_dir", default="your_path")
args = parser.parse_args()
data_path = args.input_dir 

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0
    
f = open(data_path)
data = json.load(f)
target = data_path.split("/")[-1]

def process(input_str):
    input_str = re.sub(u"\\<<.*?\\>>", "", input_str)
    return input_str

english_instruction2data = {}

for i in data:
    seen_generated = set()
    unique_data = []

    for item in i['answers']:
        generated_value = item["generated"]
        
        if generated_value not in seen_generated:
            unique_data.append(item)
            seen_generated.add(generated_value)
    i['answers'] = unique_data
    
    sorted_output = [g for g in sorted(i['answers'],key=lambda x:x["nllb-200-distilled-600M-reward-mean"],reverse=True)]
    temp = english_instruction2data.get(i['conversations'][0]['value'],[])

    for j in range(len(sorted_output)-1):
        predict_answer = sorted_output[j]['generated']
        for l in range(j+1,len(sorted_output)):
            sample = {}
            sample['prompt'] = sorted_output[0]['prompt']
            sample['choice'] = sorted_output[j]['generated']
            sample['reject'] = sorted_output[l]['generated']
            sample['score-diff'] = sorted_output[j]['nllb-200-distilled-600M-reward-mean']-sorted_output[l]['nllb-200-distilled-600M-reward-mean']
            if sorted_output[j]['nllb-200-distilled-600M-reward-mean'] != sorted_output[l]['nllb-200-distilled-600M-reward-mean'] and process(sorted_output[j]['generated']) != process(sorted_output[l]['generated']):
                temp.append(sample)
    if len(sorted_output):
        sample = {}
        sample['prompt'] = sorted_output[0]['prompt']
        sample['choice'] = i['conversations'][1]['value']
        sample['reject'] = sorted_output[0]['generated']
        sample['score-diff'] = sorted_output[0]['nllb-200-distilled-600M-reward-mean']
        temp.append(sample)
    english_instruction2data[i['conversations'][0]['value']] = temp


ratio = 10
index = 0
train_data = []
dev_data = []
for i in english_instruction2data:
    if index % ratio == 0:
        dev_data.extend(english_instruction2data[i])
    else:
        train_data.extend(english_instruction2data[i])
    index += 1
    
print(len(train_data))
print(len(dev_data))


with open("./ORPO/result/vllm/after/{}-dev.json".format(target), 'w', encoding='utf-8') as fw:
    json.dump(dev_data, fw, indent=2, ensure_ascii=False)

with open("./ORPO/result/vllm/after/{}-train.json".format(target), 'w', encoding='utf-8') as fw:
    json.dump(train_data, fw, indent=2, ensure_ascii=False)