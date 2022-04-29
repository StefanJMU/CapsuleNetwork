
import torch
from torch import nn
from capsule import CapsuleLayer


class PrimaryCaps(nn.Module):

    def __init__(self,
                 conv1_features: int = 16,
                 conv2_features: int = 32,
                 cap_size: int = 8):
        super().__init__()
        self.cap_size = cap_size

        if conv2_features % cap_size != 0:
            raise ValueError('conv2_features has to be multiple of cap_size')

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_features,
            kernel_size=9,
            padding='valid'
        )

        self.conv_2 = nn.Conv2d(
            in_channels=conv1_features,
            out_channels=conv2_features,
            kernel_size=9,
            stride=2,
            padding='valid'
        )

    def forward(self, x):

        x = self.conv_1(x)
        x = self.conv_2(x)  # (Batch, conv2_features, Height, Width)

        #reshape into primary capsules
        x = torch.reshape(x, (x.shape[0], self.cap_size, -1))  #(Batch, cap_size, n_caps)
        x = torch.transpose(x, dim0=1, dim1=2)  # (Batch, n_caps, caps_size)
        return x


class Decoder(nn.Module):

    def __init__(self, dims=[16, 32, 64, 784]):
        super().__init__()

        self.dims = dims

        # Simple linear-layer-based decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_features=dims[0], out_features=dims[1]),
            nn.ReLU(),
            nn.Linear(in_features=dims[1], out_features=dims[2]),
            nn.ReLU(),
            nn.Linear(in_features=dims[2], out_features=dims[3]),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        image_dim = int(self.dims[3]**.5)
        return torch.reshape(x, shape=(x.shape[0], 1, image_dim, image_dim))


class CapsNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.primal_capsules = PrimaryCaps()
        self.capsule_layer = CapsuleLayer(n_in_caps=144,
                                          in_caps_size=8,
                                          n_out_caps=10,
                                          out_caps_size=16)
        self.decoder = Decoder()

    def forward(self, input):
        x = self.primal_capsules(input)
        x = self.capsule_layer(x)

        #calculate l2-norm in order to retrieve class probabilities
        pred = torch.sqrt(torch.sum(torch.square(x), axis=-1))

        return pred, x, input, self.decoder

