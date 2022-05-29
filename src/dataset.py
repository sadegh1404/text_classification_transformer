from select import select
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
        self.collate_fn = DataCollatorWithPadding(self.tokenizer)

    def _load_raw(self):
        csv_path = f'{self.config.data_path}_{self.split}.csv'
        df = pd.read_csv(csv_path)
        return df

    def _extract_labels(self, df):
        labels_list = df['label'].unique().tolist()
        self.idx2label = dict(list(enumerate(labels_list)))
        self.label2idx = {v: k for k, v in self.idx2label.items()}
        self.num_classes = len(labels_list)

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
        inputs = self.tokenizer(text, truncation=True, return_tensors='pt', return_attention_mask=False)
        label_idx = self.label2idx[label]
        label_idx = torch.tensor(label_idx, dtype=torch.long)
        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['labels'] = label_idx

        return inputs


if __name__ == '__main__':
    import omegaconf
    from transformers.data import DataCollatorWithPadding

    cfg = omegaconf.OmegaConf.load('../configs/security.yaml')
    dataset = CSVTextDataset(cfg, 'train')
    data_collator = DataCollatorWithPadding(dataset.tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
    itr = iter(loader)
    x = next(itr)
    print(x)
