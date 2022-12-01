import os
import functools
from time import perf_counter

import torch
from torch.utils.tensorboard import SummaryWriter
import transformers


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_metrics(metrics: dict, preds: torch.Tensor, labels: torch.Tensor):
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = metric(preds, labels).item()

    return results


def reset_trackers(trackers: dict):
    for metric in trackers.values():
        metric.reset()


def update_trackers(trackers: dict, metrics_results: dict):
    for metric_name, metric in trackers.items():
        metric.update(metrics_results[metric_name])


def get_trackers_avg(trackers: dict):
    avg_results = {}
    for metric_name, metric in trackers.items():
        avg_results[metric_name] = metric.avg

    return avg_results


def write_to_tensorboard(writer: SummaryWriter, logs: dict, mode: str, step: int):
    for metric_name, value in logs.items():
        writer.add_scalar(f'{mode}/{metric_name}', value, step)


def maybe_save_checkpoint(model, path, epoch_num, save_freq):
    """
    Save a checkpoint
    Args:
        model: a nn.Module instance
        path: path to save checkpoint to
        epoch_num: current epoch number
        save_freq: save frequency based on epoch number

    """
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f'{epoch_num}.pt')
    if epoch_num % save_freq == 0:
        checkpoint = {'model': model.state_dict(),
                      'epoch': epoch_num}
        torch.save(checkpoint, path)
        print(f'Saved checkpoint to `{path}`')


def get_lm_module_name(model):
    """
    Return the name of the LM part e.g, distilbert, bert, roberta since we don't know which language model is used
    when loading model using AutoModel
    """
    modules = list(model.named_modules())[1:]  # ignore the main module
    for name, m in modules:
        if issubclass(m.__class__, transformers.PreTrainedModel):
            return name


def time_it(method):
    @functools.wraps(method)
    def inner(self, *args, **kwargs):
        start = perf_counter()
        out = method(self, *args, **kwargs)
        if type(out) != tuple:
            out = out,
        end = perf_counter()
        return tuple([*out, end - start])

    return inner


def clean_text(text, invalid_chars):
    text_ = text.split()
    text_ = [word for word in text_ if not word.isascii()]
    cleaned = ''
    valid = True
    for i, word in enumerate(text_):
        for x in invalid_chars:
            if x in word:
                valid = False
                break
        if valid:
            cleaned += f'{word} '
    return cleaned.strip()
