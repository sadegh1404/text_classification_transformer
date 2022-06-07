## Shahab with BERT

In this project, a DistilBERT model is trained for text classification. There are two datasets:

- Opposition classification
- Security classification

Datasets are located at `data`.

### **Train**
1. First install requirements:
```bash
pip install -r requirements.txt
```

2. Configure properties in `config.yaml`
```yaml
device: 'cpu'
lm_checkpoint: 'HooshvareLab/distilbert-fa-zwnj-base'
data_path: 'data/opposition_classification_telegram_channel_post_1581158577_half_vector_w10_d100_top_text_based_tlg'
batch_size: 4
max_length: 150
invalid_chars: ['rt:', '@', '#']
dropout: 0.1
hidden_size: 768
lr: 2e-5
num_classes: 2
num_epochs: 100
save_ckpt_freq: 1
ckpt_dir: 'checkpoints'
resume_checkpoint: null
log_dir: null

```

3. Run training script:
```bash
python train.py --config security.yaml
```

4. Checkpoints will be saved at `checkpoints` and logs will be saved at `runs`.
- Resume Training: pass in a path to a checkpoint file (.pt) to `resume_checkpoint` in `config.yaml`
- Inspect logs by running:
```bash
tensorboard --logdir runs/
# Browse to http://localhost:6006/
```

## Prediction
Run `predict.py`:
```bash
python predict.py --config security.yaml
# Enter text to get predictions as [0, 1]
```

### TODO
- [ ] Add docstring to codes
- [ ] Generalize implementation for other tasks
- [ ] Integrate mlflow
