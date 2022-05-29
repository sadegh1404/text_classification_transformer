from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from omegaconf import OmegaConf
from torchmetrics import F1Score

from src.model import BertTextClassification
from src.dataset import CSVTextDataset
from src.utils import *


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.batch_size = self.config.batch_size
        self.num_epochs = self.config.num_epochs
        self.save_ckpt_freq = self.config.save_ckpt_freq
        self.ckpt_dir = self.config.ckpt_dir

        self.train_data = CSVTextDataset(config, 'train')
        self.test_data = CSVTextDataset(config, 'test')

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                       shuffle=True, collate_fn=self.train_data.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=self.test_data.collate_fn)

        self.config['num_classes'] = self.train_data.num_classes
        self.model = BertTextClassification(self.config)
        self.maybe_load_from_checkpoint()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        self.tensorboard = SummaryWriter()

        self.metrics = {'loss': self.criterion, 'f1': F1Score(num_classes=self.config.num_classes)}
        self.metric_trackers = {metric_name: AverageMeter(metric_name) for metric_name in self.metrics.keys()}

    def maybe_save_checkpoint(self, epoch_num):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        path = os.path.join(self.ckpt_dir, f'{epoch_num}.pt')
        if epoch_num % self.save_ckpt_freq == 0:
            checkpoint = {'model': self.model.state_dict(),
                          'epoch': epoch_num,
                          'idx2label': self.train_data.idx2label}
            torch.save(checkpoint, path)
            print(f'Saved checkpoint to `{path}`')

    def maybe_load_from_checkpoint(self):
        ckpt_path = self.config.resume_checkpoint
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['model'])
            self.start_epoch = ckpt['epoch'] + 1
            print(f'\nResuming training from checkpoint: `{ckpt_path}`')
        else:
            self.start_epoch = 1
            print(f'\nTraining from scratch since no checkpoint was given in the config file...')

    def train_step(self, batch):
        self.model.bert.requires_grad_(False)
        batch = batch.to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        results = compute_metrics(self.metrics, logits.detach(), labels.detach())

        return results

    def test_step(self, batch):
        batch = batch.to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        results = compute_metrics(self.metrics, logits, labels)

        return results

    def train_one_epoch(self, epoch):
        self.model.train()
        reset_trackers(self.metric_trackers)
        with tqdm(self.train_loader, unit="batch", desc=f'Epoch: {epoch}/{self.num_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for batch in iterator:
                results = self.train_step(batch)
                update_trackers(self.metric_trackers, results)
                results_avg = get_trackers_avg(self.metric_trackers)
                iterator.set_postfix(**results_avg)

        return results_avg

    def evaluate(self):
        self.model.eval()
        reset_trackers(self.metric_trackers)
        with tqdm(self.test_loader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.no_grad():
                for batch in iterator:
                    results = self.test_step(batch)
                    update_trackers(self.metric_trackers, results)
                    results_avg = get_trackers_avg(self.metric_trackers)
                    iterator.set_postfix(**results_avg)

        return results_avg

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print()
            train_results = self.train_one_epoch(epoch)
            val_results = self.evaluate()

            # tensorboard
            write_to_tensorboard(self.tensorboard, train_results, 'train', epoch)
            write_to_tensorboard(self.tensorboard, val_results, 'val', epoch)

            self.maybe_save_checkpoint(epoch)
