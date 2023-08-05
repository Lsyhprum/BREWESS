import os
import inspect
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import RPQ
from .utils import check_numpy, get_latest_file
from .knn import ExactNNS
from .nn_utils import OneCycleSchedule, DISTANCES, training_mode, gumbel_softmax
# from tensorboardX import SummaryWriter


class Trainer(nn.Module):
    def __init__(self, *, model, experiment_name=None,
                 Loss=None, loss_opts=None, optimizer=None,
                 LearnedSimilaritySearch, SimilaritySearch=ExactNNS,
                 NegativeSimilaritySearch=None, 
                 verbose=False, drop_large_grads=False,
                 device_ids=None, output_device=None, batch_dim=0, **kwargs):
        """
        A class that handles model training, checkpointing and evaluation
        :type model: NeuralQuantizationModel
        :param experiment_name: a path where all logs and checkpoints are saved
        :type Loss: module that computes loss function, see LossBase
        :param loss_opts: a dictionary of parameters for self.compute_loss
        :param Optimizer: function(parameters) -> optimizer
        :param SimilaritySearch: knn engine to be used for recall evaluation
        :param device_ids: if given, performs data-parallel training on these device ids
        :param output_device: gathers loss on this device id
        :param batch_dim: if device_ids is specified, batch tensors will be split between devices along this axis
        """
        super().__init__()
        self.model = model
        self.loss = (Loss)(model, **loss_opts)
        if device_ids is not None:
            self.loss = nn.DataParallel(self.loss, device_ids, output_device=output_device, dim=batch_dim)
        self.opt = optimizer or OneCycleSchedule(torch.optim.Adam(model.parameters(), amsgrad=True), **kwargs)
        self.NegativeSimilaritySearch = NegativeSimilaritySearch or LearnedSimilaritySearch
        self.LearnedSimilaritySearch = LearnedSimilaritySearch
        self.SimilaritySearch = SimilaritySearch
        self.verbose = verbose
        self.drop_large_grads = drop_large_grads
        self.drops = 0
        self.step = 0

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)
        self.experiment_path = os.path.join('logs/', experiment_name)
        assert not os.path.exists(self.experiment_path), 'experiment {} already exists'.format(experiment_name)
        # self.writer = SummaryWriter(self.experiment_path, comment=experiment_name)

        # frame = inspect.currentframe()
        # args, _, _, values = inspect.getargvalues(frame)
        # for arg in args:
        #     self.writer.add_text(str(arg), str(values[arg]), global_step=self.step)

    def train_on_batch(self, *batch, prefix='train/', **kwargs):
        self.opt.zero_grad()
        with training_mode(self.model, self.loss, is_train=True):
            metrics = self.loss(*batch, **kwargs)
        metrics['loss'].mean().backward()
                
        self.opt.step()
        self.step += 1
        # for metric in metrics:
        #     self.writer.add_scalar(prefix + metric, metrics[metric].mean().item(), self.step)
        return metrics

    def evaluate_recall(self, base, query, k=1, prefix='dev/', **kwargs):
        """ Computes average recall @ k """
        with torch.no_grad(), training_mode(self.model, is_train=False):
            reference_indices = self.SimilaritySearch(base, **kwargs).search(query, k=k)
            reference_indices2 = torch.as_tensor(reference_indices, device=base.device)
            predicted_indices = self.LearnedSimilaritySearch(base, **kwargs).search(query, exact_nn=reference_indices2, k=k)[0]
            predicted_indices, reference_indices = map(check_numpy, (predicted_indices, reference_indices))
            mean_acc = 0
            for bs in range(reference_indices.shape[0]):
                acc = 0
                for r in reference_indices[bs]:
                    for p in predicted_indices[bs]:
                        if r == p:
                            acc += 1
                            break
                mean_acc += (acc / k)
            recall = mean_acc / reference_indices.shape[0]

        # self.writer.add_scalar('{}recall@{}'.format(prefix, k), recall, self.step)
        return recall

    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            tag = self.step
        if path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(OrderedDict([
            ('model', self.state_dict(**kwargs)),
            ('opt', self.opt.state_dict()),
            ('step', self.step)
        ]), path)
        if self.verbose:
            print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert tag is None or path is None, "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            path = get_latest_file(os.path.join(self.experiment_path, 'checkpoint_*.pth'))
        elif tag is not None and path is None:
            path = os.path.join(self.experiment_path, "checkpoint_{}.pth".format(tag))
        checkpoint = torch.load(path)

        self.load_state_dict(checkpoint['model'], **kwargs)

        self.opt.load_state_dict(checkpoint['opt'])
        self.step = int(checkpoint['step'])
        if self.verbose:
            print('Loaded ' + path)
        return self

    def get_true_nearest_ids(self, base, query, *, k, exclude_self=True):
        """ returns indices of k nearest neighbors for each vector in original space """
        if self.verbose:
            print(end="Computing ground truth neighbors... ", flush=True)
        k = k or self.positive_neighbors
        with torch.no_grad():
            train_neighbors_index = self.SimilaritySearch(base).search(
                query, k=k + int(exclude_self))[:, int(exclude_self):]
            original_neighbors_index = torch.as_tensor(train_neighbors_index, device=base.device)
        if self.verbose:
            print(end="Done\n", flush=True)
        return original_neighbors_index
    
    def get_retrieval_ids(self, base, query, positive_ids, exact_nn, *, k, skip_k=0): # get_negative_ids
        """
        returns indices of top-k nearest neighbors in learned space excluding positive_ids
        :param base: float matrix [num_vectors, vector_dim]
        :param positive_ids: int matrix [num_vectors, num_positive_neighbors]
        :param k: number of negative samples for each vector
        :param skip_k: excludes this many nearest indices from nearest ids (used for recall@10/100)
        """
        if self.verbose:
            print(end="Computing negative candidates... ", flush=True)
        assert query.shape[0] == positive_ids.shape[0]
        num_vectors, k_positives = positive_ids.shape
        # k_total = k + skip_k + k_positives
        k_total = k

        with torch.no_grad():
            with training_mode(self.model, is_train=False):
                learned_nearest_ids, learned_cand_his_ids, dist = self.NegativeSimilaritySearch(base).search(query, exact_nn, k=k_total)
                learned_nearest_ids = torch.as_tensor(learned_nearest_ids, device=base.device)
            # ^-- [base_size, k_total]

            idendity_ids = torch.arange(len(positive_ids), device=positive_ids.device)[:, None]  # [batch_size, 1]
            forbidden_ids = torch.cat([idendity_ids, positive_ids], dim=1)
            # ^-- [base_size, 1 + k_positives]

            negative_mask = (learned_nearest_ids[..., None] != forbidden_ids[..., None, :]).all(-1)
            # ^-- [base_size, k_total]
            negative_ii, negative_jj = negative_mask.nonzero().t()
            negative_values = learned_nearest_ids[negative_ii, negative_jj]
            # shape(negative_ii, negative_jj, negative_values) = [sum(negative_mask)] (1D)

            # beginning of each row in negative_ii
            slices = torch.cat([torch.zeros_like(negative_ii[:1]),
                                1 + (negative_ii[1:] != negative_ii[:-1]).nonzero()[:, 0]])
            # ^--[base_size]

            # column indices of negative samples
            squashed_negative_jj = torch.arange(len(negative_jj), device=negative_jj.device) - slices[negative_ii]

            # a matrix with nearest elements in learned_nearest_index
            # that are NOT PRESENT in positive_ids for that element
            new_negative_ix = torch.stack((negative_ii, squashed_negative_jj), dim=0)
            negative_ids = torch.sparse_coo_tensor(new_negative_ix, negative_values,
                                                   size=learned_nearest_ids.shape).to_dense()[:, skip_k: k + skip_k]
            # ^--[base_size, k + skip_k]

        if self.verbose:
            print(end="Done\n", flush=True)
        return negative_ids, learned_cand_his_ids, dist


class LossBase(nn.Module):
    """ A module that implements loss function. compatible with nn.DataParallel """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.opts = kwargs

    def forward(self, *batch, **kwargs):
        metrics = self.compute_loss(self.model, *batch, **dict(self.opts, **kwargs))
        for key in metrics:
            if not torch.is_tensor(metrics[key]):
                continue
            if len(metrics[key].shape) == 0:
                metrics[key] = torch.unsqueeze(metrics[key], 0)
        return metrics

    @staticmethod
    def compute_loss(model, *batch, **kwargs):
        """
        Something that takes data batch and outputs a dictionary of tensors
        All tensors should be of fixed shape except for (optionally) batch dim
        """
        raise NotImplementedError()


class RoutingLoss(LossBase):
    @staticmethod
    def compute_loss(model: RPQ,
                    test_base, x_batch, x_positives=None, x_negatives=None, record=None, vertex_id_to_distance=None, *, hard_codes=True,
                     reconstruction_coeff=0.0, reconstruction_distance=DISTANCES['euclidian_squared'],
                     triplet_coeff=0.0, triplet_delta=0.0, eps=1e-6, lambda_cut=10, **kwargs):
        assert (x_positives is not None and x_negatives is not None) or triplet_coeff == 0

        # compute logits with manually applied temperatures for performance reasons
        x_reconstructed, activations = model.forward(x_batch, return_intermediate_values=True)
        # ^-- all: [batch_size, num_codebooks, codebook_size]

        metrics = dict(loss=torch.zeros([], device=x_batch.device))
        if reconstruction_coeff != 0:
            reconstruction_distances = reconstruction_distance(x_batch, x_reconstructed)
            reconstruction_loss = reconstruction_distances.mean()
            metrics['reconstruction_loss'] = reconstruction_loss
            metrics['loss'] += reconstruction_loss * reconstruction_coeff

        if triplet_coeff != 0:
            distances_to_codes = activations['dist'] # (bs, M, K)
            # print(distances_to_codes.shape)

            pos_codes = gumbel_softmax(
                model.compute_score(x_positives),
                noise=0.0, hard=hard_codes, dim=-1
            )
            pos_distances = (pos_codes * distances_to_codes).sum(dim=[-1, -2])

            neg_codes = gumbel_softmax(
                model.compute_score(x_negatives),
                noise=0.0, hard=hard_codes, dim=-1
            )
            neg_distances = (neg_codes * distances_to_codes).sum(dim=[-1, -2])

            triplet_loss = F.relu(triplet_delta + pos_distances - neg_distances).mean()
            metrics['triplet_loss'] = triplet_loss
            metrics['loss'] += triplet_coeff * triplet_loss

        return_k = 50
        objectives = []
        for i in range(len(record)):
            objective = []
            for j in range(record[i][0]):
                vertex_num = record[i][1 + j * (return_k + 40 + 1)]
                if vertex_num > 0:
                    candidate_vertices = record[i][1 + j * (return_k + 40 + 1) + 1 : 1 + j * (return_k + 40 + 1) + 1 + vertex_num]
                    candidate_distances = [vertex_id_to_distance[i][v] for v in candidate_vertices]
                    optimal_distance = min(candidate_distances)
                    objective.append(dict(
                        positive_ids=[v for v, d in zip(candidate_vertices, candidate_distances)
                                    if d <= optimal_distance],
                        negative_ids=[v for v, d in zip(candidate_vertices, candidate_distances)
                                    if d > optimal_distance],
                        weight=1. / vertex_num
                    ))
            objectives.append(objective)

        # compute all distances
        is_numerator, row_ix, col_ix, row_weights = [], [], [], []
        row_index = 0
        query_ix, vertex_ix = [], []
        for i, rec in enumerate(objectives):
            for row in rec:
                num_vertices = (len(row['positive_ids']) + len(row['negative_ids']))
                query_ix.extend([i] * num_vertices)
                vertex_ix.extend([vertex_id for vertex_id in row['positive_ids'] + row['negative_ids']])
                is_numerator.extend([1] * len(row['positive_ids']) + [0] * len(row['negative_ids'])) # pos. / neg. 标志
                row_ix.extend([row_index] * num_vertices)  # row id
                col_ix.extend(range(num_vertices))         # hop num
                row_weights.append(row.get('weight', 1.0))
                row_index += 1

        # route distance
        # print(len(vertex_ix))
        vertex_vectors = test_base[vertex_ix]
        # print(vertex_vectors.shape)
        codes = gumbel_softmax(
            model.compute_score(vertex_vectors),
            noise=0.0, hard=hard_codes, dim=-1
        )
        # print(len(query_ix))
        distances_to_codes = activations['dist']
        distances_to_code = distances_to_codes[query_ix]
        # print(distances_to_code.shape)
        distances = (codes * distances_to_code).sum(dim=[-1,-2])
        # print(distances.shape)
        logits = -distances

        is_numerator = torch.tensor(is_numerator, dtype=torch.uint8, device=x_batch.device)
        row_ix = torch.tensor(row_ix, dtype=torch.int64, device=x_batch.device)
        col_ix = torch.tensor(col_ix, dtype=torch.int64, device=x_batch.device)
        row_weights = torch.tensor(row_weights, dtype=torch.float32, device=x_batch.device)

        # construct two matrices, both of shape [num_training_instances, num_vertices_per_instance]
        # first matrix contains all logits, padded by -inf
        all_logits_matrix = torch.full([row_index, col_ix.max() + 1], -1e9, device=x_batch.device)   # 包含全部距离
        all_logits_matrix[row_ix, col_ix] = logits

        # second matrix only contains reference logits only
        ref_logits_matrix = torch.full([row_index, col_ix.max() + 1], -1e9, device=x_batch.device)   # 仅包含 positive 距离
        ref_logits_matrix[row_ix, col_ix] = torch.where(is_numerator, logits, torch.full_like(logits, -1e9))

        logp_any_ref = torch.logsumexp(ref_logits_matrix, dim=1) \
                       - torch.logsumexp(all_logits_matrix, dim=1)

        # xent = -torch.sum(logp_any_ref * row_weights) / torch.sum(row_weights)
        xent = -torch.sum(logp_any_ref * row_weights) / torch.sum(row_weights)
        # acc = torch.sum(torch.ge(torch.exp(logp_any_ref), 0.5).to(dtype=torch.float32)
        #                 * row_weights) / torch.sum(row_weights)
        # print(xent)
        metrics['xent'] = xent
        metrics['loss'] += xent
        return metrics