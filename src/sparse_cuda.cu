// EMAX6/7 GCN Test Program            //
// sparse_cuda.cu                      //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifdef USE_CUDA
#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/sparse.h"
#include "../include/utils.h"

extern "C"
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

extern "C"
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
        else result->row_p[i+1] = result->row_p[i] + sub;
        is_added = 0;
    }
}

extern "C"
void allocSparseMatrix(SparseMatrix *sp_matrix) {
    sp_matrix->row_p = (int*) malloc(sizeof(int)*(sp_matrix->row_size+1));
    sp_matrix->col_p = (int*) malloc(sizeof(int)*(sp_matrix->nnz));
    sp_matrix->val = (float*) malloc(sizeof(float)*(sp_matrix->nnz));
    CHECK(cudaMalloc((void**) &sp_matrix->cuda_row_p, sizeof(int)*(sp_matrix->row_size+1)));
    CHECK(cudaMalloc((void**) &sp_matrix->cuda_col_p, sizeof(int)*(sp_matrix->nnz)));
    CHECK(cudaMalloc((void**) &sp_matrix->cuda_val,   sizeof(float)*(sp_matrix->nnz)));
}

extern "C"
void sendSparseMatrixToGPU(SparseMatrix *sp_matrix) {
    CHECK(cudaMemcpy(sp_matrix->cuda_row_p, sp_matrix->row_p, sizeof(int)*(sp_matrix->row_size+1), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_matrix->cuda_col_p, sp_matrix->col_p, sizeof(int)*(sp_matrix->nnz),        cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sp_matrix->cuda_val,   sp_matrix->val,   sizeof(float)*(sp_matrix->nnz),        cudaMemcpyHostToDevice));
}

extern "C"
void sendSparseMatrixToCPU(SparseMatrix *sp_matrix) {
    CHECK(cudaMemcpy(sp_matrix->row_p, sp_matrix->cuda_row_p, sizeof(int)*(sp_matrix->row_size+1), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sp_matrix->col_p, sp_matrix->cuda_col_p, sizeof(int)*(sp_matrix->nnz),        cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(sp_matrix->val,   sp_matrix->cuda_val,   sizeof(float)*(sp_matrix->nnz),      cudaMemcpyDeviceToHost));
}

extern "C"
void freeGPUSparseMatrix(SparseMatrix *sp_matrix) {
    CHECK(cudaFree(sp_matrix->cuda_row_p));
    CHECK(cudaFree(sp_matrix->cuda_col_p));
    CHECK(cudaFree(sp_matrix->cuda_val));
}

extern "C"
void freeSparseMatrix(SparseMatrix *sp_matrix) {
    free(sp_matrix->row_p);
    free(sp_matrix->col_p);
    free(sp_matrix->val);
    freeGPUSparseMatrix(sp_matrix);
}

cusparseHandle_t cusparse_handle;

extern "C"
void createCusparse() {
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
}

extern "C"
void destroyCusparse() {
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
}

extern "C"
void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix) {
    struct timespec t1, t2;
    cusparseSpMatDescr_t Adescr;
    cusparseDnMatDescr_t Bdescr;
    cusparseDnMatDescr_t Cdescr;

    printf("<<CUDA>>\n");

    CHECK_CUSPARSE(
        cusparseCreateCsr(&Adescr, 
            sp_matrix->row_size, sp_matrix->col_size, sp_matrix->nnz, 
            sp_matrix->cuda_row_p, sp_matrix->cuda_col_p, sp_matrix->cuda_val, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
        )
    );

    CHECK_CUSPARSE(
        cusparseCreateDnMat(&Bdescr, 
            sp_matrix->col_size, matrix->col_size, matrix->col_size, matrix->cuda_val, 
            CUDA_R_32F, CUSPARSE_ORDER_ROW
        )
    );

    CHECK_CUSPARSE(
        cusparseCreateDnMat(&Cdescr, 
            sp_matrix->row_size, matrix->col_size, matrix->col_size, result->cuda_val, 
            CUDA_R_32F, CUSPARSE_ORDER_ROW
        )
    );

    float alpha = 1;
    float beta = 0;
    size_t buffer_size;
    void *buffer;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUSPARSE(
        cusparseSpMM_bufferSize(cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        (void*)&alpha, Adescr, Bdescr, (void*)&beta, Cdescr, 
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size)
    );

    CHECK(cudaMalloc(&buffer, buffer_size));
    CHECK_CUSPARSE(
        cusparseSpMM_preprocess(cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        (void*)&alpha, Adescr, Bdescr, (void*)&beta, Cdescr, 
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer)
    );

    CHECK_CUSPARSE(
        cusparseSpMM(cusparse_handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        (void*)&alpha, Adescr, Bdescr, (void*)&beta, Cdescr, 
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer)
    );
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);
    printf("cuSPARSE SpMM: %lf usec.\n", cal_time(&t2, &t1));
    all_nanosec[SPMM][0] += (unsigned long long) cal_time(&t2, &t1)*1000;

    CHECK(cudaFree(buffer));
    
    CHECK_CUSPARSE(cusparseDestroySpMat(Adescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(Bdescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(Cdescr));
}
#endif
