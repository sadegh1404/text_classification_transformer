from typing import *
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding
from .utils import clean_text, time_it


class TransformerTextClassification(nn.Module):
    """
    Args:
        config: a DictConfig object containing model properties
    """

    def __init__(self, config, mode: Literal['training', 'inference']):
        super().__init__()
        self.config = config
        self.model = self.build_model(mode=mode)
        self.tokenizer = self.build_tokenizer()

    def build_model(self, mode):
        if mode == 'training':
            model = AutoModelForSequenceClassification.from_pretrained(self.config.lm_checkpoint,
                                                                       id2label=dict(self.config.id2label))
        elif mode == 'inference':
            model = AutoModelForSequenceClassification.from_pretrained(self.config.weight_path)
        else:
            raise ValueError(f'Invalid `mode`: {mode}')

        logging.info(f'Loaded model `{model.name_or_path}`')

        model.to(self.config.device)

        return model

    def build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.lm_checkpoint)
        logging.info(f'Loaded model `{tokenizer.name_or_path}`')
        return tokenizer

    def freeze_lm_weights(self):
        self.model.base_model.eval()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

    @time_it
    def predict(self, inputs: Union[str, List]):
        # preprocess
        raw_inputs = inputs[:]
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = [clean_text(x, self.config.invalid_chars) for x in inputs]
        inputs = self.tokenizer(inputs, padding=True,
                                truncation=True,
                                max_length=self.config.max_length,
                                return_tensors='pt')
        inputs.to(self.config.device)
        # model forward
        with torch.inference_mode():
            output = self.model(**inputs)
        # post-process
        logits = output['logits'].detach().cpu()
        prediction_ids = logits.softmax(1).argmax(1)
        prediction_probs = logits.softmax(1).max(1)
        prediction_labels = [self.model.config.id2label[x.item()] for x in prediction_ids]

        outputs = []
        for x, y, prob in zip(raw_inputs, prediction_labels, prediction_probs):
            outputs.append({
                "input": x,
                "label": y,
                "prob": prob.item()
            })

        return outputs

    def save_weights(self, path):
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
