import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.data import DataCollatorWithPadding


class CSVTextDataset(Dataset):

    def __init__(self, config, split):
        super(CSVTextDataset, self).__init__()
        self.config = config
        self.invalid_chars: list = config.invalid_chars
        self.split = split
        self.df = self._load_raw()
        self._extract_labels(self.df)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.lm_checkpoint, max_length=config.max_length)
        self.collate_fn = DataCollatorWithPadding(self.tokenizer)

    def _load_raw(self):
        csv_path = f'{self.config.data_path}_{self.split}.csv'
        df = pd.read_csv(csv_path)
        return df

    def _extract_labels(self, df):
        labels_list = df['label'].unique().tolist()
        self.id2label = {str(k): str(v) for k, v in dict(list(enumerate(labels_list))).items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(labels_list)

    def __len__(self):
        return len(self.df)

    def _clean_text(self, text):
        for char in self.invalid_chars:
            text = text.replace(char, '')
        return text

    def __getitem__(self, index):
        text, label = self.df.loc[index]
        if len(self.invalid_chars):
            text = self._clean_text(text)
        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                truncation=True,
                                padding=True,
                                return_attention_mask=False,
                                return_token_type_ids=False)
        label_idx = int(self.label2id[str(label)])
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['labels'] = label_idx

        return inputs


if __name__ == '__main__':
    import omegaconf

    cfg = omegaconf.OmegaConf.load('../configs/security.yaml')
    dataset = CSVTextDataset(cfg, 'train')
    data_collator = DataCollatorWithPadding(dataset.tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
    itr = iter(loader)
    x = next(itr)
    print(x)
