from functools import partial

import torch


def tempered_log(x, t):
    if t == 1.0:
        return torch.log(x)
    return 1/(1-t) * (torch.pow(x, 1-t) - 1)


def tempered_exp(x, t):
    if t == 1.0:
        exp_res = torch.exp(x)
    else:
        base  = 1 + (1-t)*x
        exp_res = torch.pow(base, 1/(1-t))
    return torch.clamp(exp_res, min=0.0) # torch way of managing maximum(arg, 0)


def reduce2d(func, tensor, dim=1):
    """ apply a given function along a given 2d tensor axis, return updated reduced tensor 
        Args:
            func (callable): function to be applied to an axis
            tensor (Tensor): target tensor
            dim (int): dimension to reduce along
        Returns:
            Tensor with the function applied
    """
    updates = [func(vec) for vec in tensor.unbind(dim = 1 - dim)] # unbind drops the dimension we want to lose, this keeps it consistent with rest of torch
    updated_tensor = torch.stack(updates, dim = 1 - dim) # reconstruct the unbound dim
    return updated_tensor


@torch.no_grad()
def iterative_normalization_calc(x, t2):
    """ iteratively solve for the normalization matrix for your activations per the algo in the appendix
        Args:
            x (Tensor): the data Tensor to normalize
            t2 (float): the temperature used to control your heavy tail
        Returns:
            the normalization Tensor for calculating your softmax    
    """
    mu = x.max()
    a_tilde = x - mu
    update_delta = 1.0
    updates = 0

    while update_delta != 0.0:
        z_a_tilde = torch.sum(tempered_exp(a_tilde, t2))
        updated_a_tilde = torch.pow(z_a_tilde, 1-t2) * (x - mu)
        update_delta = torch.abs(updated_a_tilde - a_tilde).sum()
        a_tilde = updated_a_tilde

    z_a_tilde = torch.sum(tempered_exp(a_tilde, t2))
    norm_tensor = -tempered_log(1/z_a_tilde, t2) + mu
    return norm_tensor


def tempered_softmax(data, t2):
    """ return the tempered softmax for a given Tensor of activations
        Args:
            data (Tensor): the Tensor of data activations
            t2 (float): the heavy tailed temperature hyperparam, controls how far out that fat tail goes
        Returns:
            the tempered softmax output for your Tensor
    """
    it_norm_c = partial(iterative_normalization_calc, **{'t2':t2})
    normalization_tensor = reduce2d(it_norm_c, data).reshape(-1, 1)
    softmax_output = tempered_exp(data - normalization_tensor, t2)
    softmax_output.sum().allclose(torch.ones(1))
    return softmax_output


def tempered_logistic_loss(class_probabilities, labels, t1):
    """ calculate the tempered log loss for your softmax activations
        Args:
            class_probabilities (Tensor): the output of your tempered_softmax
            labels (Tensor): correct class label Tensor to compare against
            t1 (float): your boundedness temperature, controls the max magnitude of error a misclassifcation can generate so outliers don't blow your knees out
        Returns:
            your summed log loss
    """
    loss = torch.sum(labels * (tempered_log(labels, t1) * tempered_log(class_probabilities, t1))\
                         - (1/(2-t1))*(torch.pow(labels, 2-t1) - torch.pow(class_probabilities, t1))
            )
    return loss