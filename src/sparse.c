#include "../include/sparse.h"
#ifdef USE_MP
#include <omp.h>
#endif

void spmm(Uint *result, SparseMatrix *sp_matrix, SparseMatrixParams *sp_params, Uint *matrix, int row, int col) {
    int i, j, k;

    #ifdef USE_MP
    #pragma omp parallel for
    #endif

    for (i = 0; i < sp_matrix->row_size; i++) {
        int col_index_at_row = sp_matrix->row_p[i];
        int col_size = sp_matrix->row_p[i+1] - sp_matrix->row_p[i];

        for (j = col_index_at_row; j < sp_matrix->row_p[i+1]; j++) {
            int col_index = sp_matrix->col_p[j];
            float sum = 0; 
            for (k = 0; k < col_size; k++) {
                sum += *(float*)&sp_matrix->val[i*sp_matrix->row_size+k] * *(float*)&matrix[k*sp_matrix->col_size+j];
            }
            result[i*sp_matrix->row_size+j] = (Uint*)&sum;
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
                sum += *(float*)&a[i*row_a+k] * *(float*)&b[k*col_a+j];
            }
            result[i*row_a+j] = (Uint*)&sum;
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