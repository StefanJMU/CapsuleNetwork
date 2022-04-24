import torch
from torch import nn
from torch.nn import functional as func


class Capsule(nn.Module):

    def __init__(self,
                 n_in_caps: int,
                 in_caps_size: int,
                 n_out_caps: int,
                 out_caps_size: int,
                 agreement_iterations: int = 5):

        """
            Parameters:
            -----------

            n_in_caps: int
                Number of input capsules. Referred to in comments as 'N'
            in_caps_size: int
                Number of neurons per input capsule. Referred to in comments as 'Cin'
            n_out_caps: int
                Number of output capsules. Referred to in comments as 'M'
            out_caps_size: int
                Number of neurons per output capsule. Referred to in comments as 'Cout'
            agreement_iterations: int
                number of iterations of the dynamic agreement routing algorithm
        """
        super().__init__()

        self.transforms = nn.Parameter(torch.rand((n_out_caps,
                                                   n_in_caps,
                                                   in_caps_size,
                                                   out_caps_size)))  # (M, N, Cin, Cout)

        self.n_in_caps = n_in_caps
        self.in_caps_size = in_caps_size
        self.n_out_caps = n_out_caps
        self.out_caps_size = out_caps_size
        self.agreement_iterations = agreement_iterations

    def _forward(self, expanded_x, weights):

        weights = torch.unsqueeze(weights, dim=-1)  # (B, M, N, 1)
        weighted_caps = expanded_x * weights  # (B, M, N, Cin)

        expanded_weighted_caps = torch.unsqueeze(weighted_caps, dim=3)  # (B, M, N, 1, Cin)
        transformed_x = torch.squeeze(expanded_weighted_caps @ self.transforms, dim=3)  # (B, M, N, Cout)

        caps_consensus = torch.sum(transformed_x, dim=2)  # (B, M, Cout)
        return transformed_x, caps_consensus

    def _dynamic_agreement_routing(self, x):

        # Constants
        expanded_x = torch.unsqueeze(x, dim=1)  # (B, 1, N, Cin)

        # initialize assignment-logits
        weights = (1 / self.n_in_caps) * torch.ones((x.shape[0], self.n_out_caps, self.n_in_caps))  # (B, M, N)

        for i in range(self.agreement_iterations):
            transformed_x, caps_consensus = self._forward(expanded_x, weights)  # (B, M, N, Cout), (B, M, Cout)
            caps_consensus = torch.unsqueeze(caps_consensus, dim=-1)  # (B, M, Cout, 1)

            caps_assignment_logits = torch.squeeze(transformed_x @ caps_consensus, dim=-1)  #(B, M, N)
            weights = func.softmax(caps_assignment_logits, dim=1)  # (B, M, N)

        return weights  # (B, M, N)

    def _squash(self, x):
        """
            Calculates the squash activation function for Capsule layers

            Parameters
            ----------
            x: torch.tensor
                tensor of shape (B, M, Cout), where B is the batch size
        """
        lengths = torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
        unit_vectors = x / lengths

        scale = lengths / (lengths + 1)
        return scale * unit_vectors

    def forward(self, x):
        """
            Calculates the forward pass of Capsule Networks

            Parameters
            ---------
            x: torch.tensor
                Torch tensor of shape (B, N, Cin), where B is the batch
        """
        weights = self._dynamic_agreement_routing(x)  # (B, M, N)
        expanded_x = torch.unsqueeze(x, dim=1)  # (B, 1, N, Cin)

        _, res = self._forward(expanded_x, weights)  # _, (B, M, Cout)

        return self._squash(res)



















