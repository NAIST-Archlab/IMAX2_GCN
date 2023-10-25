// EMAX6/7 GCN Test Program            //
// sparse.c                            //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/sparse.h"
#include <stdlib.h>
#if !(defined(EMAX6) || defined(EMAX7) || defined(USE_CUDA))
#include <stdio.h>
#ifdef USE_MP
#include <omp.h>
#endif

void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix) {
    printf("<<CPU>>\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int k = 0; k < matrix->col_size; k++) {
        for (int i = 0; i < sp_matrix->row_size; i++) {
            int col_index_of_index = sp_matrix->row_p[i];
            float sum = 0;
            for (int j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
                int col_index = sp_matrix->col_p[j];
                sum += sp_matrix->val[j] * matrix->val[col_index * matrix->col_size + k];
            }
            result->val[i * matrix->col_size + k] = sum;
        }
    }
}

void mm(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b) {
    printf("<<CPU>>\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < a->row_size; i++) {
        for (int j = 0; j < a->col_size; j++) {
            float sum = 0;
            for (int k = 0; k < b->col_size; k++) {
                sum += a->val[i * a->col_size + k] * b->val[k * b->col_size + j];
            }
            result->val[i * b->col_size + j] = sum;
        }
    }
}

void relu(DenseMatrix *result, DenseMatrix *a) {
    printf("<<CPU>>\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < (a->col_size * a->row_size); i++) {
        result->val[i] = (a->val[i] > 0) ? a->val[i] : 0;
    }
}
#endif

#if !defined(USE_CUDA)
void allocDenseMatrix(DenseMatrix *matrix) {
    matrix->val = (float*) malloc(sizeof(float)*matrix->row_size*matrix->col_size);
}

void freeDenseMatrix(DenseMatrix *matrix) {
    free(matrix->val);
}

void allocSparseMatrix(SparseMatrix *sp_matrix) {
    sp_matrix->row_p = (int*) malloc(sizeof(int)*(sp_matrix->row_size+1));
    sp_matrix->col_p = (int*) malloc(sizeof(int)*(sp_matrix->nnz));
    sp_matrix->val = (float*) malloc(sizeof(float)*(sp_matrix->nnz));
}

void freeSparseMatrix(SparseMatrix *sp_matrix) {
    free(sp_matrix->row_p);
    free(sp_matrix->col_p);
    free(sp_matrix->val);
}
#endif