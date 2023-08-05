%module pg

// release GIL by default for all functions
%exception {
    Py_BEGIN_ALLOW_THREADS
    $action
    Py_END_ALLOW_THREADS
}

 
%{
#include <omp.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "pg.h"
#include <numpy/arrayobject.h>
#include <iostream>
%}

%include "numpy.i"

%init%{
    import_array();
%}

%apply (const char* INPUT, size_t LENGTH) { (const char* str, size_t len) }
%apply (uint8_t* IN_ARRAY2, int DIM1, int DIM2) {(uint8_t *input_codes, int base_size, int num_chunks)}

%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *distance_array, int batchsize, int num_chunks, int dict_size)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *base, int num, int dim)}
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *query, int num2, int dim2)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int *res_ranking, int batchsize2, int k)}
%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int *nn, int batchsize3, int k2)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int *cand_his, int batchsize4, int cand_size)}
%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) {(float *cand_d, int batchsize5, int cand_size2)}

%include "pg.h"
