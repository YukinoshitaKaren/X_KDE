# Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs (ACL 2025 Findings)
We present the Cross-Lingual Knowledge Democracy Edit (**X-KDE**) framework, which facilitates knowledge editing across languages in large language models (LLMs). 
By integrating Cross-lingual Edition Instruction Tuning (XE-IT) and Target-language Preference Optimization (TL-PO), X-KDE efficiently transfers knowledge from a source language to a target language while maintaining strong performance in monolingual settings. 


## Requirements
- Python 3.8+
- CUDA 11.8+
- See `requirements.txt` for detailed dependencies

## Project Structure

```
.
├── scripts/          # Training and evaluation scripts
├── data/            # Dataset directory
├── LLaMA-Factory/   # Base training framework
├── ORPO/           # ORPO optimization code
├── eval/           # Evaluation code
└── requirements.txt # Project dependencies
```

## Training
Please download the training data from [HuggingFace](https://huggingface.co/datasets/YukinoKaren/X_KDE_train_data). Moreover, we have also uploaded some ready-to-use demo models that you can download here: [HuggingFace](https://huggingface.co/datasets/YukinoKaren/X_KDE_train_data(https://huggingface.co/YukinoKaren/X_KDE_Demo_EN_ZH), [HuggingFace](https://huggingface.co/YukinoKaren/X_KDE_Demo).


### 1. Instruction Tuning (XE-IT)
- Using LLaMA2 model:
  ```bash
  bash full_train_llama2.sh
  ```
- Using Qwen2.5 model:
  ```bash
  bash full_train_qwen25.sh
  ```

### 2. Preference Optimization (TL-PO)
- Data sampling:
  ```bash
  bash sampling.sh
  ```
- ORPO optimization:
  ```bash
  bash orpo.sh
  ```

## Evaluation

Run evaluation script:
```bash
bash eval.sh
```

## Notes

- Ensure all dependencies are installed
- Check GPU memory before training
- DeepSpeed is recommended for distributed training
