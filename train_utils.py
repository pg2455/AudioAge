import random
import torch
import torch.nn as nn
import logger as logger
import os
from sklearn.model_selection import train_test_split

class AgeDataHandler(object):
    CLASS_DICT =  {'teens':0, 'twenties':1, 'thirties':2, 'fourties':3, 'fifties':4, 'sixties':5, 'seventies':6, 'eighties':7}
    def __init__(self, datadir, batch_size=64):
        self.datadir = datadir
        self.batch_size = 64
        self.classdir = [x for x in os.listdir(self.datadir) if os.path.isdir(os.path.join(datadir, x))]
        self.soundfiles = []
        for x in self.classdir:
            for f in os.listdir(os.path.join(self.datadir, x)):
                self.soundfiles.append((os.path.join(self.datadir, x, f), self.CLASS_DICT[x.split("_")[-1]]))

        random.shuffle(self.soundfiles)
        self.soundfiles = self.soundfiles[:40]
        self.obs = len(self.soundfiles)
        self.train, self.val = train_test_split(self.soundfiles, test_size=0.1)
        self.idx = 0

    def train_val_split(self):
        return FileIterator(self.train, self.batch_size), FileIterator(self.val, self.batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self):
            self.idx = 0
            random.shuffle(self.soundfiles)
            raise StopIteration
        else:
            out = self.soundfiles[self.idx : self.idx + self.batch_size]
            self.idx += self.batch_size
            return out

    def __len__(self):
        return len(self.soundfiles)

class FileIterator(object):
    def __init__(self, files, batch):
        self.files = files
        self.idx = 0
        self.batch_size = batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self):
            self.idx = 0
            random.shuffle(self.files)
            raise StopIteration
        else:
            out = self.files[self.idx : self.idx + self.batch_size]
            self.idx += self.batch_size
            return out

    def __len__(self):
        return len(self.files)

    def tolist(self):
        return self.files

def init_visdom(env_name, config):
    assert type(config) == dict

    visdom_opts = {"server":'http://localhost', "port":8787}
    stats = logger.Experiment(env_name, log_git_hash =False, use_visdom=True, visdom_opts = visdom_opts, time_indexing = False)

    val_metrics = stats.ParentWrapper(tag="validation", name="parent",
                                        children=(
                                            stats.SimpleMetric(name='loss'),
                                            stats.SimpleMetric(name="accuracy")
                                        ))

    train_metrics = stats.ParentWrapper(tag="training", name="parent",
                                        children=(
                                            stats.AvgMetric(name='loss'),
                                            stats.AvgMetric(name="accuracy")
                                        ))

    stats.log_config(config)

    def update_metrics(loss, acc, key = 'train'):
        if key == 'train':
            train_metrics.update(loss = loss, accuracy = acc)
        elif key == "val":
            val_metrics.update(loss = loss, accuracy = acc)

    def log_metrics():
        stats.log_metric(train_metrics)
        stats.log_metric(val_metrics)
        train_metrics.reset()
        val_metrics.reset()

    #
    norm = stats.ParentWrapper(tag="norm", name="parent2",
        children = (
                stats.SimpleMetric(name="norm"),
            ))

    def plot_norm(val):
        norm.update(norm = val)
        stats.log_metric(norm)
        norm.reset()

    return update_metrics, log_metrics, plot_norm


class AnomModel(nn.Module):
    def __init__(self, feature_model):
        super(AnomModel, self).__init__()
        self.model = feature_model
        DIM = self.model.fc.out_features
        self.out = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            nn.Linear(DIM,2)
        )

    def forward(self, input):
        return self.out(self.model(input))
