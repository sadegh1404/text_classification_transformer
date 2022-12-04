import os
import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_and_save_model(model_hub_path, save_dir='weights'):
    save_path = f'{save_dir}/{model_hub_path.split("/")[-1]}'
    model = AutoModelForSequenceClassification.from_pretrained(model_hub_path, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_hub_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f'saved {model_hub_path} to {save_path}!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hub_paths',
                        help='pass in model hub paths and separate them with `#` in between them.',
                        default='Dataak/bert-fa-zwnj-base-sentiment-marketing#'
                                'Dataak/distilbert-fa-zwnj-base-sentiment-marketing')
    parser.add_argument('--save_dir',
                        help='Directory path to save models',
                        default='./weights')

    args = parser.parse_args()
    hub_paths = args.hub_paths.split('#')
    save_dir = args.save_dir

    for path in hub_paths:
        load_and_save_model(path, save_dir)
