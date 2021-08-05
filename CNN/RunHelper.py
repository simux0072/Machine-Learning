from torch.utils.tensorboard import SummaryWriter
import RunVariables

import pandas as pd
import json
import time

from collections import OrderedDict
from collections import namedtuple
from itertools import product

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())
        runs = []

        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class RunManager():
    def __init__(self):
        self.epoch = RunVariables.Epoch(0, 0, 0, None)
        self.run = RunVariables.Run(None, 0, [], 0, 0, None)

        self.network = None
        self.tb = None

    def begin_run(self, run, network):
        self.run.start_time = time.time()

        self.run.params = run
        self.run.id += 1

        self.run.loss = 0
        self.run.acc = 0

        self.network = network
        self.tb = SummaryWriter(comment = f'-{run}')

    def end_run(self):
        # Getting data for TensorBoard output
        self.tb.add_scalar('Test Set Loss', self.run.loss, self.run.id)
        self.tb.add_scalar('Test Set Accuracy', self.run.acc, self.run.id)
        self.tb.close()
        self.epoch.id = 0

    def begin_epoch(self):
        self.epoch.start_time = time.time()

        self.epoch.id += 1
        self.epoch.loss = 0
        self.epoch.acc = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time

        self.tb.add_scalar('Training Set Loss', self.epoch.loss, self.epoch.id)
        self.tb.add_scalar('Training Set Accuracy', self.epoch.acc, self.epoch.id)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch.id)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.id)
        # Saving data needed for post training analisis
        results = OrderedDict()
        results['run'] = self.run.id
        results['epoch'] = self.epoch.id
        results['train loss'] = self.epoch.loss
        results['train accuracy'] = self.epoch.acc
        results['test loss'] = ''
        results['test accuracy'] = ''
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run.params._asdict().items(): results[k] = v
        self.run.data.append(results)

    def train_track_loss(self, loss):
        self.epoch.loss = loss
    
    def train_track_acc(self, acc):
        self.epoch.acc = acc
    
    def test_track_loss(self, loss):
        self.run.loss = loss
    
    def test_track_acc(self, acc):
        self.run.acc = acc
    # Saving data for post testing analysis
    def write_test(self):
        results = OrderedDict()
        results['run'] = ''
        results['epoch'] = ''
        results['train loss'] = ''
        results['train accuracy'] = ''
        results['test loss'] = self.run.loss
        results['test accuracy'] = self.run.acc
        results['epoch duration'] = ''
        results['run duration'] = ''
        for k, v in self.run.params._asdict().items(): results[k] = v
        self.run.data.append(results)
    # Saving data to a csv (for excel analysis) and a json files
    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run.data,
            orient = 'columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding = 'utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii = False, indent = 4)


