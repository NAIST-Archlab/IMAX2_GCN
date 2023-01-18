#include "../include/sparse.h"
#include <stdio.h>
#ifdef USE_MP
#include <omp.h>
#endif

void spmm(Uint *result, SparseMatrix *sp_matrix, SparseMatrixParams *sp_params, Uint *matrix, int mm_col) {
    int i, j, k;

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (k = 0; k < mm_col; k++) {
        for (i = 0; i < sp_matrix->row_size; i++) {
            int col_index_of_index = sp_matrix->row_p[i];
            float sum = 0;
            for (j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
                int col_index = sp_matrix->col_p[j];
                sum += *(float *)&sp_matrix->val[j] * *(float *)&matrix[col_index * mm_col + k];
            }
            result[i * mm_col + k] = (Uint *)&sum;
        }
    }
}

void mm(Uint *result, Uint *a, Uint *b, int row_a, int col_a, int col_b) {
    int i, j, k;

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (i = 0; i < row_a; i++) {
        for (j = 0; j < col_a; j++) {
            float sum = 0; 
            for (k = 0; k < col_b; k++) {
                sum += *(float*)&a[i*col_a+k] * *(float*)&b[k*col_b+j];
            }
            result[i*col_b+j] = (Uint*)&sum;
        }
    }
}

void relu(Uint *result, Uint *a, int size) {
    int i;
    float zero = 0;

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (i = 0; i < size; i++) {
        float val = *(float*)&a[i];
        result = (val > 0) ? *(Uint*)&val : *(Uint*)&zero;
    }
}