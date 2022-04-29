
import os
import numpy as np
from skorch.callbacks import Callback
from sklearn.metrics import f1_score

class ModelStorer(Callback):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, net, **kwargs):
        epoch = net.history[-1]['epoch']
        net.save_params(f_params=os.path.join(self.path, str(epoch) + '.pkl'))


class ModelEvaluator(Callback):

    def __init__(self, dataset, test_set, model_path):
        super().__init__()
        self.test_set = test_set
        self.dataset = dataset

        self.model_path = model_path
        self.optimum = -1
        self.optimum_epoch = None

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):

        gt = np.argmax((self.dataset.y[self.test_set.indices]).numpy(), axis=1)
        pred = np.argmax(net.predict(self.test_set), axis=1)

        score = f1_score(gt, pred, average='weighted')
        if score > self.optimum:
            self.optimum = score
            self.optimum_epoch = net.history[-1]['epoch']

    def on_train_end(self, net, X=None, y=None, **kwargs):
        net.load_params(os.path.join(self.model_path, str(self.optimum_epoch) + ".pkl"))




