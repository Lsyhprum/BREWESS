#ifndef PG_H
#define PG_H

#include <vector>
#include <stdint.h>

class PG
{
    std::vector<uint8_t> codes;
    int base_size_;
    int num_chunks_;
public:
    PG(uint8_t *input_codes, int base_size, int num_chunks);
    void search(const char* graph_path, 
                float *distance_array, int batchsize, int num_chunks, int dict_size, 
                float *base, int num, int dim,
                float *query, int num2, int dim2,
                int *res_ranking, int batchsize2, int k,
                int *nn, int batchsize3, int k2,
                int *cand_his, int batchsize4, int cand_size,
                float *cand_d, int batchsize5, int cand_size2
                );
};

#endif
