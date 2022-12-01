## Text Classification with HuggingFace Transformers
Train text classification models using HuggingFace Transformers

### **Train**
1. First install requirements:
```bash
pip install -r requirements.txt
```

2. Configure properties in `config.yaml`
```yaml
name: sentiment_marketing
model_name: bert-fa-zwnj-base
device: 'cuda'
lm_checkpoint: 'weights/${model_name}'
data_path: 'data/${name}'
batch_size: 16
max_length: 256
invalid_chars: ['rt:', '@', '#']
lr: 2e-5
num_epochs: 10
save_ckpt_freq: 1
ckpt_dir: 'checkpoints/${name}/${model_name}'
weight_path: 'checkpoints/${name}/${model_name}/5'
log_dir: 'logs'

```

3. Run training script:
```bash
python train.py --config configs/sentiment.yaml
```

4. Checkpoints will be saved at `checkpoints` and logs will be saved at `runs`.
- Inspect logs by running:
```bash
tensorboard --logdir runs/
# Browse to http://localhost:6006/
```

## API
Run the following to start up FastAPI server:
```bash
uvicorn api.app:app --port 5000
```

