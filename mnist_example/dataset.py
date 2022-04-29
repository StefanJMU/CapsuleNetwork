
import torch
from torch import nn

from sklearn.utils import check_random_state
from typing import Union
from numpy.random import RandomState

class Dataset:

    """
        MNIST dataset wrapper

        Parameters
        ----------
        dataset : torch.Dataset
            MNIST dataset
        class_balanced_subset : float, default=None
            If set with a float in [0,1] only a subset with balanced class distribution of this size is considered
    """

    def __init__(self,
                 dataset,
                 class_balanced_subset: float = None,
                 random_state: Union[int, RandomState] = 123456):

        x = torch.unsqueeze(dataset.data, dim=1)
        y = dataset.targets

        if class_balanced_subset is not None:

            if class_balanced_subset < 0 or class_balanced_subset > 1:
                raise ValueError(f'class_balanced_subset is required to be in [0,1]. Got {class_balanced_subset}')

            random_state = check_random_state(random_state)

            sorting = torch.argsort(y)
            y = y[sorting]
            x = x[sorting]

            split_points = torch.reshape(torch.nonzero(torch.diff(y, n=1) > 0), shape=(-1,)) + 1
            split_sizes = torch.diff(split_points, n=1)
            split_sizes = [split_points[0]] + split_sizes.numpy().tolist() + [x.shape[0] - split_points[-1]]

            x_groups = list(torch.split(x, split_sizes, dim=0))
            y_groups = list(torch.split(y, split_sizes, dim=0))

            for i in range(len(x_groups)):
                n = int(x_groups[i].shape[0] * class_balanced_subset)
                perm = random_state.permutation(x_groups[i].shape[0])
                x_groups[i] = x_groups[i][perm][:n]
                y_groups[i] = y_groups[i][:n]

            x = torch.cat(x_groups, dim=0)
            y = torch.cat(y_groups)

        self.x = x.type(torch.float32) / 255
        self.y = nn.functional.one_hot(y, num_classes=10).to(torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
