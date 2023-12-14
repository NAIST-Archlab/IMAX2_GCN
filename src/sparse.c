// EMAX6/7 GCN Test Program            //
// sparse.c                            //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/sparse.h"
#if !(defined(EMAX6) || defined(EMAX7) || defined(USE_CUDA))
#ifdef USE_MP
#include <omp.h>
#endif

void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix) {
    printf("<<CPU>>\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int k = 0; k < matrix->col_size; k++) {
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < sp_matrix->row_size; i++) {
            int col_index_of_index = sp_matrix->row_p[i];
            float sum = 0;
            #ifdef USE_MP
            #pragma omp parallel for reduction(+:sum)
            #endif
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
        #ifdef USE_MP
        #pragma omp parallel for
        #endif
        for (int j = 0; j < b->col_size; j++) {
            float sum = 0;
            #ifdef USE_MP
            #pragma omp parallel for reduction(+:sum)
            #endif
            for (int k = 0; k < a->col_size; k++) {
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

void d_relu(DenseMatrix *result, DenseMatrix *a) {
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

void gcn_preprocessing(SparseMatrix *matrix) {
    for (int i = 0; i < matrix->row_size; i++) {
        for (int j = matrix->row_p[i]; j < matrix->row_p[i+1]; j++) {
            int col = matrix->col_p[j];
            float d_row = 1 / sqrt(matrix->row_p[i + 1] - matrix->row_p[i] + 1);
            float d_col = 1 / sqrt(matrix->row_p[col + 1] - matrix->row_p[col] + 1);
            matrix->val[j] = d_row * d_col;
        }
    }
    #ifdef USE_CUDA
        sendSparseMatrixToGPU(matrix);
    #endif
}

void spia(SparseMatrix *result, SparseMatrix *sp_matrix) {
    int nnz = sp_matrix->nnz;
    int k = 0;
    char is_added = 0;

    for (int i = 0; i < sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        is_added = 0;
        for (int j = col_index_of_index; j < sp_matrix->row_p[i+1]; j++) {
            int col_index = sp_matrix->col_p[j];
            if (col_index == i) {
                is_added = 1;
                break;
            }
        }

        if (!is_added) nnz++;
    }

    result->nnz = nnz;
    result->col_size = sp_matrix->col_size;
    result->row_size = sp_matrix->row_size;
    allocSparseMatrix(result);

    if (nnz != 0)
        result->row_p[0] = 0;

    for (int i = 0; i < sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        int sub = sp_matrix->row_p[i+1] - sp_matrix->row_p[i];
        is_added = 0;
        for (int j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
            int col_index = sp_matrix->col_p[j];
            result->col_p[k++] = col_index;
            if (col_index == i) {
                is_added = 1;
                break;
            }
        }

        if (!is_added) {result->row_p[i+1] = result->row_p[i] + sub + 1;result->col_p[k++]=i;}
        else result->row_p[i+1] = result->row_p[i];
        is_added = 0;
    }
}

void softmax(DenseMatrix *result) {
    for (int i = 0; i < result->row_size; i++) {
        float max = max_in_array(&(result->val[i * result->col_size]), result->col_size);
        float log_max = log(max);
        float sum = 0;

        if (max <= 1) log_max = 0;
        for (int j = 0; j < result->col_size; j++) {
            sum += exp(result->val[i * result->col_size + j] + log_max);
        }
        for (int j = 0; j < result->col_size; j++) {
            result->val[i * result->col_size + j] = exp(result->val[i * result->col_size + j] + log_max) / sum;
        }
    }
}

void d_softmax(DenseMatrix *result) {
    for (int i = 0; i < result->row_size; i++) {
        for (int j = 0; j < result->col_size; j++) {
            if (i == j)
                result->val[i * result->col_size + j] = result->val[i * result->col_size + j] * (1 - result->val[i * result->col_size + j]);
            else
                result->val[i * result->col_size + j] = result->val[i * result->col_size + j] * (-result->val[i * result->col_size + j]);
        }
    }
}

float max_in_array(float *array, int size) {
    int i;
    float max = -INFINITY;

    for (i = 0; i < size; i++) {
        if (max < array[i])
            max = array[i];
    }

    return max;
}

void msub(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b) {
    for (int i = 0; i < a->row_size; i++) {
        for (int j = 0; j < a->col_size; j++) {
            result->val[i * a->col_size + j] = a->val[i * a->col_size + j] - b->val[i * a->col_size + j];
        }
    }
}

float mmeans(DenseMatrix *a) {
    float sum = 0;
    for (int i = 0; i < a->row_size; i++) {
        for (int j = 0; j < a->col_size; j++) {
            sum += a->val[i * a->col_size + j];
        }
    }
    return sum / (a->row_size * a->col_size);
}

void transpose(DenseMatrix *result, DenseMatrix *a) {
    for (int i = 0; i < a->row_size; i++) {
        for (int j = 0; j < a->col_size; j++) {
            result->val[j * a->row_size + i] = a->val[i * a->col_size + j];
        }
    }
}

void expand_labels(DenseMatrix *labels, Uchar *vlabels)  {
    for (int i = 0; i < labels->row_size; i++) {
        for (int j = 0; j < labels->col_size; j++) {
            if (j == (int)vlabels[i])
                labels->val[i*labels->col_size + j] = 1.0;
            else
                labels->val[i*labels->col_size + j] = 0.0;
        }
    }
}

#endif
