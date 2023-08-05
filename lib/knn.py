import faiss
from .utils import check_numpy, free_memory


class NNS:
    def __init__(self, base, **kwargs):
        raise NotImplementedError()

    def search(self, query, k=1):
        raise NotImplementedError()


class ExactNNS(NNS):
    def __init__(self, base, device_id=0):
        assert len(base.shape) == 2
        dim = base.shape[1]
        self.index_flat = faiss.IndexFlatL2(dim)
        self.res = faiss.StandardGpuResources()
        self.res.noTempMemory()
        self.index_flat = faiss.index_cpu_to_gpu(self.res, device_id, self.index_flat)
        self.index_flat.add(check_numpy(base))

    def search(self, query, k=1):
        free_memory()
        assert len(query.shape) == 2
        _, neighbors = self.index_flat.search(check_numpy(query), k)
        free_memory()
        return neighbors
