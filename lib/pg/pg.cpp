#include "pg.h"
#include <vector>
#include <stdint.h>
#include <iostream>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <index_nsg.h>
#include "omp.h"

PG::PG(uint8_t *input_codes, int base_size, int num_chunks) : base_size_(base_size), num_chunks_(num_chunks)
{   
    std::copy(input_codes, input_codes + base_size * num_chunks, std::back_inserter(codes));
}

void PG::search(const char* graph_path, 
                float *distance_array, int batchsize, int num_chunks, int dict_size, 
                float *base, int num, int dim,
                float *query, int num2, int dim2,
                int *res_ranking, int batchsize2, int k,
                int *nn, int batchsize3, int k2,
                int *cand_his, int batchsize4, int cand_size,
                float *cand_d, int batchsize5, int cand_size2
                )
{
    assert(batchsize == batchsize2);
    assert(batchsize == batchsize3);
    assert(batchsize == batchsize4);
    assert(batchsize == batchsize5);
    assert(num_chunks_ == num_chunks);

    std::string graph(graph_path);
    const unsigned L = 100; // 设置太小可能导致负采样失败

    efanna2e::IndexNSG index(num_chunks, base_size_, efanna2e::L2, nullptr);

    index.Load(&graph[0]);
    index.SetDataset(base);

    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", L);
    paras.Set<unsigned>("P_search", L);

    unsigned int *res = new unsigned int[k * batchsize];
    unsigned int *cand = new unsigned int[cand_size * batchsize];
    float *cand2 = new float[cand_size * batchsize];
    index.BatchSearchWithCodes(distance_array, batchsize, num_chunks, dict_size, 
                                codes.data(), k, paras, 
                                query, dim2,
                                nn, 
                                res,
                                cand, cand_size,
                                cand2
                                );
    for(int i = 0; i < k * batchsize; i ++) {
        res_ranking[i] = res[i];
    }
    for(int i = 0; i < cand_size * batchsize; i ++) {
        cand_his[i] = cand[i];
    }
    for(int i = 0; i < cand_size2 * batchsize2; i ++) {
        cand_d[i] = cand2[i];
    }
}
