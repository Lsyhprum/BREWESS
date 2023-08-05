import time
import torch
import os, sys
import argparse
import numpy as np
from functools import partial
from torch_optimizer import QHAdam

import lib

sys.path.insert(0, '..')
os.environ['OMP_NUM_THREADS'] = "40"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


# python sift1m.py --dataset_name sift --data_path /home/zjlab/ANNS/yq/dataset/origin/ --M 32
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=float, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--positive_k", type=float, default=10)
    parser.add_argument("--negative_k", type=float, default=50)
    parser.add_argument("--reconstruction_coeff", type=float, default=1)
    parser.add_argument("--triplet_coeff", type=float, default=0.1)
    parser.add_argument("--triplet_delta", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=float, default=1000)
    parser.add_argument("--decay_rate", type=float, default=0.2)
    parser.add_argument("--negative_obtain_epochs", type=int, default=10)
    args = parser.parse_args()
    print(args)

    device_ids = list(range(torch.cuda.device_count()))

    experiment_name = '{}_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}:{:0>2d}'.format(args.dataset_name, *time.localtime()[:6])
    print("experiment:", experiment_name)


    dataset = lib.Dataset(args.dataset_name, data_path=args.data_path, normalize=True)
    model = lib.RPQ(d=dataset.vector_dim, M=args.M, K=args.K).cuda()

    with torch.no_grad():
        model(dataset.train_vectors[:1000].cuda())

    trainer = lib.Trainer(
        model=model, 
        Loss=lib.RoutingLoss, 
        loss_opts=dict(
            reconstruction_distance=lib.DISTANCES['euclidian_squared'],
            reconstruction_coeff=args.reconstruction_coeff, 
            triplet_coeff=args.triplet_coeff, 
            triplet_delta=args.triplet_delta
        ),
        optimizer=lib.OneCycleSchedule(
            QHAdam(model.parameters(), nus=(0.8, 0.7), betas=(0.95, 0.998)), 
            learning_rate_base=args.lr, warmup_steps=args.warmup_steps, decay_rate=args.decay_rate),
        
        LearnedSimilaritySearch=partial(lib.PGSearch, model=model, batch_size=1000, device_ids=device_ids),
        NegativeSimilaritySearch=partial(lib.PGSearch, model=model, batch_size=1000, device_ids=device_ids),
        SimilaritySearch=lib.ExactNNS,  # reference nearest vectors will be mined this way
        device_ids=device_ids,
        experiment_name=experiment_name, 
        verbose=True,
    )

    best_recall, best_tr_recall = 0.0, 0.0
    loss_history = []

    test_base = dataset.test_vectors.cuda()
    train_base = dataset.train_vectors.cuda()
    train_gt = trainer.get_true_nearest_ids(test_base, train_base, k=args.positive_k, exclude_self=False)
    train_gt2 = trainer.get_true_nearest_ids(test_base, train_base, k=args.negative_k, exclude_self=True) # ver2

    # def fetch_retrieval():
    #     negatives = trainer.get_retrieval_ids(test_base.cpu(), train_base.cpu(), train_gt.cpu(), k=args.negative_k)
    #     return negatives.cuda()
    def fetch_retrieval():
        return_k = args.negative_k
        negatives, records, dist = trainer.get_retrieval_ids(test_base.cpu(), train_base.cpu(), train_gt.cpu(), train_gt2.cpu(), k=return_k) # dist:pq距离 dist2:真实距离

        max_visited_ids = 0
        #records[bs][times + times * (r_num + K *id + r_num * id)]
        for i in range(len(records)): # bs
            visited_ids = set()
            for j in range(records[i][0]): # hops
                for k in range(records[i][1 + j * (return_k + 40 + 1)]): # r_num + k
                    # print(j * (return_k + 40 + 1))
                    # print(records[i][1 + j * (return_k + 40 + 1)])
                    visited_ids.add(records[i][1 + j * (return_k + 40 + 1) + 1 + k])
            visited_ids = list(visited_ids)     
            max_visited_ids = max(max_visited_ids, len(visited_ids))
        
        visited_ids_matrix = np.full([train_base.shape[0], max_visited_ids], -1, dtype=np.int32)
        distances_matrix = np.full([train_base.shape[0], max_visited_ids], -1, dtype=np.float32)
        for i in range(len(records)): # bs
            visited_ids = set()
            pos = 0
            for j in range(records[i][0]): # hop
                for k in range(records[i][1 + j * (return_k + 40 + 1)]): # neighbor
                    id = records[i][1 + j * (return_k + 40 + 1) + 1 + k]
                    if id not in visited_ids:
                        visited_ids.add(id)
                        visited_ids_matrix[i][pos] = id
                        distances_matrix[i][pos] = dist[i][1 + j * (return_k + 40 + 1) + 1 + k] # 真实距离
                        pos += 1

        vertex_id_to_distance = np.array([
            dict(zip(visited_ids_i, distances_i))
            for visited_ids_i, distances_i in zip(visited_ids_matrix, distances_matrix)
        ])

        return negatives.long().cuda(), records, vertex_id_to_distance

    def sample_uniform(base, ids):
        return base[ids[torch.arange(ids.shape[0]), torch.randint(0, ids.shape[1], size=[ids.shape[0]])]]

    for epoch_i in range(args.epochs):
        lib.free_memory()

        if epoch_i % args.negative_obtain_epochs == 0: # every ~250 steps  modify!!!!!!!
            retrievals, records, vertex_id_to_distance = fetch_retrieval()
        
        for x_batch, nearest_batch, retrieval_batch, record_batch, id2dist_batch in lib.iterate_minibatches(
            train_base, train_gt, retrievals, records, vertex_id_to_distance, batch_size=args.batch_size):
            metrics_t = trainer.train_on_batch(test_base,
                                               x_batch=x_batch,
                                                x_positives=sample_uniform(test_base, nearest_batch),
                                                x_negatives=sample_uniform(test_base, retrieval_batch),
                                                record=record_batch,
                                                vertex_id_to_distance=id2dist_batch)
            loss_history.append(metrics_t['loss'].mean().item())
        
        if epoch_i % 10 == 0: # every ~500 steps
            metrics_t = {key: lib.check_numpy(value) for key, value in metrics_t.items()}
            lib.free_memory()
            
            recall_tr = trainer.evaluate_recall(dataset.test_vectors.cuda(), dataset.train_vectors.cuda(), k=10)
            if recall_tr > best_tr_recall:
                best_tr_recall = recall_tr
                trainer.save_checkpoint('best')

            recall_t = trainer.evaluate_recall(dataset.test_vectors.cuda(), dataset.query_vectors.cuda(), k=10)
            if recall_t > best_recall:
                best_recall = recall_t

            print("epoch = %i \t step = %i \t mean loss = %.5f \t lr = %.5f \t best dev recall = %.5f \t best tr recall = %.5f" % (
                epoch_i, trainer.step, np.mean(loss_history[-100:]),
                lib.get_learning_rate(trainer.opt), best_recall, best_tr_recall))
            for k, v in metrics_t.items():
                print('{} = {}'.format(k, np.mean(lib.check_numpy(v))))


if __name__ == "__main__":
    main()