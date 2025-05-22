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