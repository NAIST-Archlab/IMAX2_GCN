// EMAX6/7 GCN Test Program            //
// sparse_cuda.c                       //
//         Copyright (C) 2023 by NAIST //
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

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

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
void allocDenseMatrix(DenseMatrix *matrix) {
    matrix->val = (float*) malloc(sizeof(float)*(matrix->row_size*matrix->col_size));
    CHECK(cudaMalloc((void**) &matrix->cuda_val, sizeof(float)*(matrix->row_size*matrix->col_size)));
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
void sendDenseMatrixToGPU(DenseMatrix *matrix) {
    CHECK(cudaMemcpy(matrix->cuda_val, matrix->val, sizeof(float)*(matrix->row_size*matrix->col_size), cudaMemcpyHostToDevice));
}

extern "C"
void sendDenseMatrixToCPU(DenseMatrix *matrix) {
    CHECK(cudaMemcpy(matrix->val, matrix->cuda_val, sizeof(float)*(matrix->row_size*matrix->col_size), cudaMemcpyDeviceToHost));
}

extern "C"
void freeGPUDenseMatrix(DenseMatrix *matrix) {
    CHECK(cudaFree(matrix->cuda_val));
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

extern "C"
void freeDenseMatrix(DenseMatrix *matrix) {
    free(matrix->val);
    freeGPUDenseMatrix(matrix);
}

cusparseHandle_t cusparse_handle;
cublasHandle_t cublas_handle;

extern "C"
void createCusparse() {
    CHECK_CUSPARSE(cusparseCreate(&cusparse_handle));
}

extern "C"
void createCublas() {
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
}

extern "C"
void destroyCusparse() {
    CHECK_CUSPARSE(cusparseDestroy(cusparse_handle));
}

extern "C"
void destroyCublas() {
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
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

extern "C"
void mm(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b) {
    struct timespec t1, t2;

    printf("<<CUDA>>\n");

    float alpha = 1;
    float beta = 0;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUBLAS(
        cublasSgemm(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            b->col_size, a->row_size, a->col_size,
            &alpha, b->cuda_val, b->col_size, a->cuda_val, a->col_size, 
            &beta, result->cuda_val, b->col_size
        )
    );
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);
    all_nanosec[MM][0] += (unsigned long long) cal_time(&t2, &t1)*1000;

    printf("cuBLAS MM: %lf usec.\n", cal_time(&t2, &t1));
}

__global__
void threshold(float *gpuResult, float *input, float threshold, int size) {
    unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (xIdx < size) {
        float input_val = input[xIdx];
        gpuResult[xIdx] = input_val;
        if (input_val < 0) {
            gpuResult[xIdx] = 0;
        }
    }
}

#define BLOCK_SIZE_CUDA 128

extern "C"
void relu(DenseMatrix *result, DenseMatrix *a) {
    struct timespec t1, t2;
    dim3 dimBlock(BLOCK_SIZE_CUDA);
    dim3 dimGrid((a->row_size * a->col_size) / BLOCK_SIZE_CUDA);

    printf("<<CUDA>>\n");

    timespec_get(&t1, TIME_UTC);
    threshold<<<dimGrid, dimBlock>>>(result->cuda_val, a->cuda_val, 0, a->row_size * a->col_size);
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);

    printf("CUDA ReLU: %lf usec.\n", cal_time(&t2, &t1));
}

#endif
