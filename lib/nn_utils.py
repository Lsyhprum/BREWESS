import contextlib

import torch
import numpy as np
from collections import OrderedDict
import torch.nn as nn


class OneCycleSchedule:

    def __init__(self, optimizer, **kwargs):
        self.learning_rate_opts = kwargs
        self.opt = optimizer
        self.step_count = 0

    def step(self, **kwargs):
        self.current_lr = self.update_learning_rate(t=self.step_count, **self.learning_rate_opts)
        res = self.opt.step(**kwargs)
        self.step_count += 1
        return res

    def state_dict(self, **kwargs):
        return OrderedDict([
            ('optimizer_state_dict', self.opt.state_dict(**kwargs)),
            ('learning_rate_opts', self.learning_rate_opts),
            ('step_count', self.step_count)
        ])

    def load_state_dict(self, state_dict, load_step=True, load_opts=True, **kwargs):
        self.learning_rate_opts = state_dict['learning_rate_opts'] if load_opts else self.learning_rate_opts
        self.step_count = state_dict['step_count'] if load_step else self.step_count
        return self.opt.load_state_dict(state_dict['optimizer_state_dict'], **kwargs)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.opt, attr)

    def update_learning_rate(self, t, learning_rate_base=1e-3, warmup_steps=10000,
                             decay_rate=0.2, learning_rate_min=1e-5):
        lr = learning_rate_base * np.minimum(
            (t + 1.0) / warmup_steps,
            np.exp(decay_rate * ((warmup_steps - t - 1.0) / warmup_steps)),
        )
        lr = np.maximum(lr, learning_rate_min)
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        return lr


def to_one_hot(y, depth):
    y_flat = y.to(torch.int64).view(-1, 1)
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot

def gumbel_noise(*sizes, eps=1e-20, **kwargs):
    U = torch.rand(*sizes, **kwargs)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(dist, noise=1.0, hard=True, **kwargs):
    if noise != 0:
        z = gumbel_noise(*dist.shape, device=dist.device, dtype=dist.dtype)
        dist = dist + noise * z

    probs_gumbel = torch.softmax(dist, dim=-1)

    if hard:
        _, argmax_indices = torch.max(probs_gumbel, dim=-1)
        hard_argmax_onehot = to_one_hot(argmax_indices, depth=dist.shape[-1])
        probs_gumbel = (hard_argmax_onehot - probs_gumbel).detach() + probs_gumbel

    return probs_gumbel

class GumbelSoftmax(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.opts = kwargs

    def forward(self, dist, **kwargs):
        opts = dict(self.opts)
        if not self.training:
            opts['noise'] = 0.0
            opts['hard'] = True
        return gumbel_softmax(dist, **opts, **kwargs)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            return param_group['lr']
    raise ValueError("Could not infer learning rate from optimizer {}".format(optimizer))

class BatchNorm(nn.BatchNorm1d):
    """ Batch Normalization that always normalizes over last dim """
    def forward(self, x, **kwargs):
        return super().forward(x.view(-1, x.shape[-1]), **kwargs).view(*x.shape)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class CallMethod(nn.Module):
    """ on forward, calls a method from another module """
    def __init__(self, module, method):
        super().__init__()
        self.module, self.method = module, method

    def forward(self, *args, **kwargs):
        return getattr(self.module, self.method)(*args, **kwargs)

@contextlib.contextmanager
def training_mode(*modules, is_train:bool):
    was_training = [module.training for module in modules]
    try:
        yield [module.train(is_train) for module in modules]
    finally:
        for module, was_training_i in zip(modules, was_training):
            module.train(was_training_i)

class Feedforward(nn.Sequential):
    def __init__(self, input_dim, hidden_dim=None, num_layers=2, output_dim=None,
                 BatchNorm=BatchNorm, Activation=nn.ReLU, bias=True, **kwargs):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or hidden_dim
        for i in range(num_layers):
            self.add_module('layer%i' % i, nn.Linear(
                input_dim if i == 0 else hidden_dim,
                output_dim if i == (num_layers - 1) else hidden_dim,
                bias=bias))
            self.add_module('layer%i_bn' % i, BatchNorm(hidden_dim))
            self.add_module('layer%i_activation' % i, Activation())


DISTANCES = {
    'euclidian': lambda a, b: torch.norm(a - b, dim=-1),
    'euclidian_squared': lambda a, b: ((a - b) ** 2).sum(-1),
}
