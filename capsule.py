import torch
from torch import nn
from torch.nn import functional as func


"""
TODO:
-> modellation decision: implicit or explicit modellation of the capsules
-> batched dynamic agreement routing --> the routing can be arbitrary for elements of the batch

--> these are some rather unpleasant concepts

Planning:

Let B the batch, N the input capsules, Cin the in capsule size, Cout the out capsule size, M the number of output capsules

Input: (B,N,Cin)

For the calculation of the output capsule activations
Assignment weights: (B,M,N) --> expanded to (B, M, N, 1) (broadcastable)
Expansion of the input: (B,1,N,Cin)
Elementwise multiplication yields: (B, M, N, Cin)
Contraction along the penultimate axis: (B, M, Cin) --> yields the consensus for each output capsule

#Dot product calculation

Tranpose input: (B,Cin,N)
-> Calculate the matrix product with (B, M, Cin) ---> yields (B, M, N): the raw new weights
-> Calculate the softmax normalization along PENULTIMATE axis --> (B, M, N)

---> iterate: fix number of dimensions or until the change in each batch dimension is below a threshold?

--> Cout does not play any role here (obviously)

"""

# a raw implementation:

def dynamic_agreement_routing(input : torch.tensor,
                              n_output_capsules,
                              max_iterations: int = 5,
                              convergence_threshold: float = 0.1):

    # Constants
    transposed_input = torch.transpose(input, dim0=2, dim1=1)  # B x Cin x N
    expanded_input = torch.unsqueeze(input, dim=1)  #B x 1 x N x Cin

    #initialize assignment-logis
    weights = (1/input.shape[1]) * torch.ones((input.shape[0], n_output_capsules, input.shape[1]))
    prev_weights = torch.zeros((input.shape[0], n_output_capsules, input.shape[1]))

    for i in range(max_iterations):

        weights = torch.unsqueeze(weights, dim=-1)  # yields B x M x N x 1

        weighted_capsules = expanded_input * weights
        capsule_consensus_vector = torch.sum(weighted_capsules, dim=2)  # yields B x M x Cin

        capsule_assignment_logits = capsule_consensus_vector @ transposed_input  # yields B x M x N

        weights = func.softmax(capsule_assignment_logits, dim=1)

        # early termination criteria
        diff = torch.sum(torch.abs(prev_weights - weights) / (weights + 1e-10)) / (input.shape[0] * input.shape[1])
        if diff < 1:  # that needs to be fixed
            break

    # exiting the loop, weights will have the correct shape (B, M, N)
    return weights

a = dynamic_agreement_routing(torch.ones(10, 9, 8), 5, max_iterations=100)


"""
 Regarding the forward pass:
 
 expanded weights: B x M x N x 1
 
 expanded input : B x 1 x N x Cin
 
 --> elementwise multiplication yiels: B x M x N x Cin --> the weighted input capsules for every output capsule
 
----> Forward propagation:

--> reshape: B x M x N x Cin --->  B x M x (NxCin) x 1
--> weight matrix is M x Cout x (NxCin)

--> matrix multiplication M x Cout x (NxCin) @ B x M x (NxCin) x 1 ----> yields B x M x Cout x 1
 
"""

def forward(input, n_output_capsules, output_capsule_size):

    weights = dynamic_agreement_routing(input, n_output_capsules, max_iterations=10)

    weights = torch.unsqueeze(weights, dim=-1)  #alright we expand again after all
    expanded_input = expanded_input = torch.unsqueeze(input, dim=1)  #B x 1 x N x Cin

    weighted_input = weights * expanded_input

    weighted_input = torch.reshape(weighted_input, shape=(input.shape[0], n_output_capsules, -1, 1))

    # some default value:
    weight_tensor = torch.ones((n_output_capsules, output_capsule_size, weighted_input.shape[2]))

    res = torch.squeeze(weight_tensor @ weighted_input, dim=-1)

    print(res.shape)  # B x n_output_capsules x output_capsule_size

forward(torch.ones(10, 9, 8), 5, 4)


# 1.25h of work ---> acceptable...




















