## Text Classification with HuggingFace Transformers
Train text classification models using HuggingFace Transformers

### **Train**
1. First install requirements:
```bash
pip install -r requirements.txt
```

2. Configure properties in `config.yaml`
```yaml
name: ${model.name}-${data.name}
data:
  name: sentiment-marketing
  invalid_chars: ['rt:', '@', '#']
  max_length: 256
model:
  name: bert-fa-zwnj-base
  lm_path: 'weights/${model.name}'
train:
  device: 'cuda'
  batch_size: 16
  lr: 2e-5
  num_epochs: 10
  save_ckpt_freq: 1
  log_dir: 'logs'
inference:
  device: 'cpu'
  weights_path: 'weights/${model.name}-${data.name}'
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

