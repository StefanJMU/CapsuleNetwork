
import torch

from torch import nn
from typing import Tuple


class CapsuleReconstructionLoss(nn.Module):

    """
        Capsule Network loss, consisting of a Binary Cross Entropy loss and l
        a Mean Squared Error reconstruction loss

        Parameters
        ----------
        reconstruction_weight: int
            scaling factor of the reconstruction component of the loss
    """

    def __init__(self, reconstruction_weight):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight

        self.classification_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()

    def forward(self,
                outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, nn.Module],
                targets: torch.Tensor):

        """

            Parameters
            ----------
            outputs: [torch.tensor, torch.tensor, torch.tensor, nn.Module]
                Output of the forwarding of the CapsNet: Prediction, output capsules, input images, decoder module
            targets: [torch.tensor]
                ground truth tensor (Batch, #Classes)

        """

        prediction, capsules, gt_images, decoder = outputs
        batch_size = targets.shape[0]

        # select the capsules of the ground truth classes
        gt_capsules = capsules[(torch.arange(end=batch_size), torch.argmax(targets, 1))]
        reconstructed = decoder(gt_capsules)

        recon_loss = self.reconstruction_loss(reconstructed, gt_images)
        pred_loss = self.classification_loss(prediction, targets)

        return pred_loss + self.reconstruction_weight * recon_loss






