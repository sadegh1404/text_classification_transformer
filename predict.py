import re

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.model import BertTextClassification
from src.utils import time_it


class Predictor:
    def __init__(self, config):
        self.config = config
        self.valid_chars = list(config.valid_chars)
        print(self.valid_chars)
        self.weights_file = config.weights_file  # doesn't exist in security.yaml file. must be assigned internally (see api/app.py)
        checkpoint = torch.load(self.weights_file)
        self.idx2label = checkpoint['idx2label']
        self.num_classes = len(list(self.idx2label.keys()))
        self.config.num_classes = self.num_classes
        self.model = BertTextClassification(config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(config.lm_checkpoint)

    def clean_text(self, text):
        text = text.split(' ')
        for word in text:
            if word.isascii() and not word.isnumeric():
                text.remove(word)
                print(f'removed {word}')
        text = " ".join(text)
        for char in text:
            if char not in self.valid_chars:
                text = text.replace(char, '')
        text = text.replace('_', ' ').replace('-', ' ')
        return text

    @time_it
    def __call__(self, input_text: str):
        print(input_text)
        input_text = self.clean_text(input_text)
        print(input_text)
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors='pt')
            print(inputs['input_ids'])
            outputs = self.model(**inputs).softmax(1).squeeze(0)
        return outputs.numpy()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to yaml config file', default='security.yaml')
    parser.add_argument('--weights_file', help='path to the weights file')

    args = parser.parse_args()

    config_path = args.config
    config = OmegaConf.load(config_path)
    config.weights_file = args.weights_file

    predictor = Predictor(config)

    while True:
        text = input('Enter Text:\n')
        if text == 'quit':
            break
        else:
            results = predictor(text)
            print(results)
