import torch
import math
import torch.nn as nn
from .utils import check_numpy
from .nn_utils import Lambda, GumbelSoftmax
from .parametrizations import orthogonal, transpose


class PQ(nn.Module):
    def get_codes(self, x):
        raise NotImplementedError()

    def get_distances(self, x):
        raise NotImplementedError()


class DPQ(PQ):
    def __init__(self, dsub, M, K, encoder, decoder, **kwargs):
        super().__init__()
        self.M, self.K = M, K

        self.encoder = encoder
        self.codebook = nn.Parameter(torch.randn(M, K, dsub))
        self.log_temperatures = nn.Parameter(data=torch.zeros(M) * float('nan'), requires_grad=True)
        self.gumbel_softmax = GumbelSoftmax(**kwargs)
        self.decoder = decoder

    def compute_score(self, x, add_temperatures=True):
        x = self.encoder(x)
        norm_1 = torch.sum(x ** 2, dim=-1, keepdim=True)  # (bs, D, 1)
        norm_2 = torch.unsqueeze(torch.sum(self.codebook ** 2, dim=-1), 0)  # (1, D, K)
        dot = torch.matmul(x.permute(1, 0, 2), self.codebook.permute(0, 2, 1))  # (D, bs, K)
        score = - norm_1 + 2 * dot.permute(1, 0, 2) - norm_2  # (bs, D, K)
        
        if add_temperatures:
            if not self.is_initialized(): self.initialize(x)
            score *= torch.exp(-self.log_temperatures[:, None])
        return score

    def forward(self, x, return_intermediate_values=False):
        if not self.is_initialized(): self.initialize(x)
        score_raw = self.compute_score(x, add_temperatures=False)
        score = score_raw * torch.exp(-self.log_temperatures[:, None])
        codes = self.gumbel_softmax(score)  # [..., num_codebooks, codebook_size]
        x_reco = self.decoder(codes)

        if return_intermediate_values:
            dist = - score_raw
            return x_reco, dict(x=x, score=score, codes=codes, x_reco=x_reco, dist=dist)
        else:
            return x_reco

    def get_codes(self, x):
        return self.compute_score(x).argmax(dim=-1)
    
    def get_distances(self, x):
        return - self.compute_score(x, add_temperatures=False)
    
    def is_initialized(self):
        # note: can't set this as property because https://github.com/pytorch/pytorch/issues/13981
        return check_numpy(torch.isfinite(self.log_temperatures.data)).all()

    def initialize(self, x):
        """ Initialize codes and log_temperatures given data """
        with torch.no_grad():
            chosen_ix = torch.randint(0, x.shape[0], size=[self.M * self.K], device=x.device)
            chunk_ix = torch.arange(self.M * self.K, device=x.device) // self.K
            initial_keys = self.encoder(x)[chosen_ix, chunk_ix].view(*self.codebook.shape).contiguous()
            self.codebook.data[:] = initial_keys

            base_logits = self.compute_score(
                x, add_temperatures=False).view(-1, self.M, self.K)
            # ^-- [batch_size, num_codebooks, codebook_size]

            log_temperatures = torch.tensor([
                0 for codebook_logits in check_numpy(base_logits).transpose(1, 0, 2)
            ], device=x.device, dtype=x.dtype)
            self.log_temperatures.data[:] = log_temperatures


class RPQ(DPQ):
    def __init__(self, d, *, M=8, K=256, **kwargs):
        
        linear = nn.Linear(d, d, bias=False)
        linear2 = nn.Linear(d, d, bias=False)
        orth_linear = orthogonal(linear) # householder, cayley, exp (, orthogonal_map="householder")
        trans_linear = transpose(linear, linear2)

        encoder = nn.Sequential(
            orth_linear,
            Lambda(lambda x: x.view(*x.shape[:-1], M, d // M))
        )
        
        decoder = nn.Sequential()
        decoder.add_module('embed', Lambda(lambda x: torch.einsum('bnc,ncd->bnd', x, self.codebook.to(device=x.device)))) # (N, M, K) x (M, K, dsub) = (N, M, dsub)
        decoder.add_module('reshape', Lambda(lambda x: x.contiguous().view(*x.shape[:-2], -1))) # N, M * dsub
        decoder.add_module('rt', trans_linear)
        
        super().__init__(d // M, M, K, encoder, decoder, **kwargs)


