
import numpy as np
import torch
import skorch

from torch import nn
from loss import CapsuleReconstructionLoss
from models import CapsNet, Decoder
from skorch import NeuralNet
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

from callbacks import ModelStorer, ModelEvaluator
from dataset import Dataset


mnist = MNIST(root="./data", download=True, transform=ToTensor())
dataset = Dataset(mnist, class_balanced_subset=0.2, random_state=1000)

train_size = int(len(dataset)*0.9)
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1000))

model = CapsNet()
epochs = 15
callbacks = [('storer', ModelStorer("./parameter_recon")),
             ('evaluator', ModelEvaluator(dataset, test_set, "./parameter_recon"))]

net = NeuralNet(module=model,
                criterion=CapsuleReconstructionLoss,
                criterion__reconstruction_weight=0.0005,
                optimizer=Adam,
                optimizer__weight_decay=0,
                optimizer__lr=0.0005,
                batch_size=128,
                max_epochs=epochs,
                callbacks=callbacks,
                train_split=None,
                verbose=1)

net.fit(train_set)


