from warnings import warn

import numpy as np
import torch
import torch.nn as nn

from lib.model import PQ, DPQ
from .wrapper import PG
from ..knn import NNS
from ..nn_utils import DISTANCES, CallMethod
from ..utils import process_in_chunks, check_numpy


class PGSearcher(PG , NNS):
    def __init__(self, base, *, quantizer: PQ, batch_size=None,
                 device_ids=None, **kwargs):
        """
        A search algorithm that quickly finds nearest neighbors using pre-computed distances
        :param base: dataset to search over, [base_size, vector_dim]
        :param quantizer: quantized search method, typically neural network-based
        :param batch_size: processes base in chunks of this size
        :param quantization_context:
        """
        self.batch_size = batch_size
        self.quantizer = quantizer
        self.base = base
        self.device_ids, self.opts = device_ids, kwargs
        if quantizer.training:
            warn("quantizer was asked to process base in training mode (with dropout/bn)")

        if self.device_ids is None:
            base_to_codes = quantizer.get_codes
        else:
            base_to_codes = nn.DataParallel(CallMethod(quantizer, 'get_codes'),
                                            device_ids=device_ids, **kwargs)
        
        with torch.no_grad():
            base_codes = process_in_chunks(base_to_codes, base,
                                           batch_size=batch_size or len(base))
            # ^-- [base_size, num_codebooks] of uint8
        super().__init__(check_numpy(base_codes).astype(np.uint8))

    def search(self, query, exact_nn, k=1, batch_size=None):
        assert len(query.shape) == 2
        if self.quantizer.training:
            warn("quantizer was asked to search in training mode (with dropout/bn")

        if self.device_ids is None:
            query_to_distances = self.quantizer.get_distances
        else:
            query_to_distances = nn.DataParallel(CallMethod(self.quantizer, 'get_distances'),
                                             device_ids=self.device_ids, **self.opts)

        with torch.no_grad():
            distances_shape = [len(query), self.quantizer.M, self.quantizer.K]
            distances = process_in_chunks(query_to_distances, query,
                                          out=torch.zeros(*distances_shape, dtype=torch.float32, device='cpu'),
                                          batch_size=batch_size or self.batch_size or len(query))
            # ^-- [num_queries, num_codebooks, codebook_size]

        r = 40
        out = np.zeros((distances.shape[0], k), dtype=np.int32)
        out2 = np.zeros((distances.shape[0], (k+r+1) * 50 + 1), dtype=np.int32)
        out3 = np.zeros((distances.shape[0], (k+r+1) * 50 + 1), dtype=np.float32) # true distance
        exact_nn = exact_nn.type(torch.int32)
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/_graph/sift1m_nsg.graph"
        graph_path = "/home/zjlab/ANNS/yq/BREWESS/_graph/sift10k_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/_graph/gist_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/_graph/deep1m_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/_graph/bigann1m_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/brewess_v1/_graph/glove1m_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/brewess_v1/_graph/audio30k_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/brewess_v1/_graph/crawl1m_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/brewess_v1/_graph/msong800k_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/brewess_v1/_graph/ukbench_nsg.graph"
        # graph_path = "/home/zjlab/ANNS/yq/BREWESS/brewess_v1/_graph/nuswide_nsg.graph"
        super().search(graph_path, check_numpy(distances), check_numpy(self.base), check_numpy(query), check_numpy(out), check_numpy(exact_nn), check_numpy(out2), check_numpy(out3))
        
        return out, out2, out3

        
class PGSearch(NNS):
    def __init__(self, base, *, model:DPQ,
                 batch_size, device_ids=None, **kwargs
                 ):
        with torch.no_grad():
            self.knn = PGSearcher(
                base, quantizer=model, batch_size=batch_size,
                device_ids=device_ids, **kwargs)

    def search(self, query, exact_nn, k=1, **kwargs):
        return self.knn.search(query, exact_nn, k=k, **kwargs)