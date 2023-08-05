import os
import warnings

import numpy as np
import torch
import random
from .utils import ivecs_read, mmap_fvecs, mmap_bvecs
import os.path as osp


class Dataset:

    def __init__(self, dataset, data_path='./data', normalize=False, random_state=50, **kwargs):
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        random.seed(random_state)

        if dataset in DATASETS:
            data_dict = DATASETS[dataset](osp.join(data_path, dataset), **kwargs)
        else:
            assert all(key in kwargs for key in ('train_vectors', 'test_vectors', 'query_vectors'))
            data_dict = kwargs

        self.train_vectors = torch.tensor(data_dict['train_vectors']).type(torch.float32)
        self.test_vectors = torch.tensor(data_dict['test_vectors']).type(torch.float32)
        self.query_vectors = torch.tensor(data_dict['query_vectors']).type(torch.float32)
        self.ground_vectors = data_dict['ground_vectors']
        assert self.train_vectors.shape[1] == self.test_vectors.shape[1] == self.query_vectors.shape[1]
        self.vector_dim = self.train_vectors.shape[1]

        print(self.train_vectors.shape, self.test_vectors.shape, self.query_vectors.shape)

        mean_norm = self.train_vectors.norm(p=2, dim=-1).mean().item()
        if normalize:
            self.train_vectors = self.train_vectors / mean_norm
            self.test_vectors = self.test_vectors / mean_norm
            self.query_vectors = self.query_vectors / mean_norm
        else:
            if mean_norm < 0.1 or mean_norm > 10.0:
                warnings.warn("Mean train_vectors norm is {}, consider normalizing")


def fetch_DEEP1M(path, train_size=1 * 10 ** 5, test_size=10 ** 6, ):
    base_path = osp.join(path, 'deep_base.fvecs')
    learn_path = osp.join(path, 'deep_learn.fvecs')
    query_path = osp.join(path, 'deep_query.fvecs')
    ground_path = osp.join(path, 'deep_groundtruth.ivecs')
    xb = mmap_fvecs(base_path)
    xq = mmap_fvecs(query_path)
    xt = mmap_fvecs(learn_path)
    gt = ivecs_read(ground_path)

    return dict(
        train_vectors=xt[:train_size],
        test_vectors=xb[:test_size],
        query_vectors=xq,
        ground_vectors=gt
    )

def fetch_BIGANN1M(path, train_size=5*10**5, test_size=10**6):
    dbsize = int(test_size / 10 ** 6)
    base_path = osp.join(path, 'bigann_base.bvecs')
    learn_path = osp.join(path, 'bigann_learn.bvecs')
    query_path = osp.join(path, 'bigann_query.bvecs')
    ground_path = osp.join(path, 'gnd/idx_%dM.ivecs'% dbsize)
    xb = mmap_bvecs(base_path)
    xq = mmap_bvecs(query_path)
    xt = mmap_bvecs(learn_path)
    gt = ivecs_read(ground_path)

    return dict(
        train_vectors=xt[:train_size],
        test_vectors=xb[:test_size],
        query_vectors=xq,
        ground_vectors=gt
    )

def fetch_SIFT1M(path, train_size=None, test_size=None):
    base_path = osp.join(path, 'sift_base.fvecs')
    learn_path = osp.join(path, 'sift_learn.fvecs')
    query_path = osp.join(path, 'sift_query.fvecs')
    ground_path = osp.join(path, 'sift_groundtruth.ivecs')
    return dict(
        train_vectors=mmap_fvecs(learn_path)[:train_size],
        test_vectors=mmap_fvecs(base_path)[:test_size],
        query_vectors=mmap_fvecs(query_path),
        ground_vectors=ivecs_read(ground_path)
    )

def fetch_GIST1M(path, train_size=None, test_size=None):
    base_path = osp.join(path, 'gist_base.fvecs')
    learn_path = osp.join(path, 'gist_learn.fvecs')
    query_path = osp.join(path, 'gist_query.fvecs')
    ground_path = osp.join(path, 'gist_groundtruth.ivecs')
    return dict(
        train_vectors=mmap_fvecs(learn_path)[:train_size],
        test_vectors=mmap_fvecs(base_path)[:test_size],
        query_vectors=mmap_fvecs(query_path),
        ground_vectors=ivecs_read(ground_path)
    )

def fetch_UKBENCH1M(path, train_size=None, test_size=None):
    base_path = osp.join(path, 'ukbench1M_base.fvecs')
    learn_path = osp.join(path, 'ukbench1M_learn.fvecs')
    query_path = osp.join(path, 'ukbench1M_query.fvecs')
    ground_path = osp.join(path, 'ukbench1M_groundtruth.ivecs')
    return dict(
        train_vectors=mmap_fvecs(learn_path)[:train_size],
        test_vectors=mmap_fvecs(base_path)[:test_size],
        query_vectors=mmap_fvecs(query_path),
        ground_vectors=ivecs_read(ground_path)
    )


DATASETS = {
    'DEEP1M': fetch_DEEP1M,
    'BIGANN1M': fetch_BIGANN1M,
    'sift': fetch_SIFT1M,
    'gist': fetch_GIST1M, 
    'ukbench': fetch_UKBENCH1M,
}
