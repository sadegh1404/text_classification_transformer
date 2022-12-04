import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from omegaconf import OmegaConf
from torchmetrics import F1Score

from .model import TransformerTextClassification
from .dataset import CSVTextDataset
from .utils import *


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.train.device
        self.batch_size = self.config.train.batch_size
        self.num_epochs = self.config.train.num_epochs
        self.save_ckpt_freq = self.config.train.save_ckpt_freq
        self.ckpt_dir = self.config.train.ckpt_dir
        self.bar_format = '{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}'

        self.train_data = CSVTextDataset(config, 'train')
        self.test_data = CSVTextDataset(config, 'test')

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                       shuffle=True, collate_fn=self.train_data.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.test_data.collate_fn)
        self.config.data['id2label'] = self.train_data.id2label

        self.model = TransformerTextClassification(self.config, mode='training')

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.train.lr)

        self.tensorboard = SummaryWriter()

        self.metrics = {'loss': self.criterion, 'f1': F1Score(num_classes=len(self.config.data.id2label))}
        self.metric_trackers = {metric_name: AverageMeter(metric_name) for metric_name in self.metrics.keys()}

    def maybe_save_checkpoint(self, epoch_num):
        if epoch_num % self.save_ckpt_freq == 0:
            path = f'{self.ckpt_dir}/{epoch_num}'
            self.model.save_weights(path)
            print(f'Saved checkpoint to `{path}`')

    def train_step(self, batch):
        batch = batch.to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']
        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results = compute_metrics(self.metrics, logits.detach().cpu(), labels.detach().cpu())

        return results

    def test_step(self, batch):
        batch = batch.to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)['logits']

        results = compute_metrics(self.metrics, logits.detach().cpu(), labels.detach().cpu())

        return results

    def train_one_epoch(self, epoch):
        self.model.train()
        self.model.freeze_lm_weights()
        reset_trackers(self.metric_trackers)
        with tqdm(self.train_loader,
                  unit="batch",
                  desc=f'Epoch: {epoch}/{self.num_epochs} ',
                  bar_format=self.bar_format,
                  ascii=" #") as iterator:

            for batch in iterator:
                results = self.train_step(batch)
                update_trackers(self.metric_trackers, results)
                results_avg = get_trackers_avg(self.metric_trackers)
                iterator.set_postfix(**results_avg)

        return results_avg

    def evaluate(self):
        self.model.eval()
        reset_trackers(self.metric_trackers)
        with tqdm(self.test_loader,
                  unit="batch",
                  desc=f'Evaluating... ',
                  bar_format=self.bar_format,
                  ascii=" #") as iterator:

            with torch.no_grad():
                for batch in iterator:
                    results = self.test_step(batch)
                    update_trackers(self.metric_trackers, results)
                    results_avg = get_trackers_avg(self.metric_trackers)
                    iterator.set_postfix(**results_avg)

        return results_avg

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            print()
            train_results = self.train_one_epoch(epoch)
            val_results = self.evaluate()

            # tensorboard
            write_to_tensorboard(self.tensorboard, train_results, 'train', epoch)
            write_to_tensorboard(self.tensorboard, val_results, 'val', epoch)

            self.maybe_save_checkpoint(epoch)
