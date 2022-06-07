import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BertTextClassification(nn.Module):
    """
    A Bert-based transformer model for text classification. Not necessarily the exact BERT model but any BERT derivative
    and depends on `config.model_checkpoint`.

    Args:
        config: a DictConfig object containing model properties
    """
    def __init__(self, config):
        super(BertTextClassification, self).__init__()
        self.config = config
        model_config = AutoConfig.from_pretrained(config.model_checkpoint)
        self.bert = AutoModel.from_config(model_config)
        self.fc_hidden = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

    def load_lm_weights(self):
        self.bert = AutoModel.from_pretrained(self.config.model_checkpoint)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        outputs = outputs[:, 0]  # take out the [CLS] token output for classification (distilbert has no pooler layer)
        outputs = self.relu(self.fc_hidden(outputs))
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits
