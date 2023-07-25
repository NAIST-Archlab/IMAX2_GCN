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
void spmm(float *result, SparseMatrix *sp_matrix, float *matrix, int mm_col) {
    int *a_row, *a_col;
    float *a_val;
    float *b, *c;
    struct timespec t1, t2;
    cusparseHandle_t handle;
    cusparseSpMatDescr_t Adescr;
    cusparseDnMatDescr_t Bdescr;
    cusparseDnMatDescr_t Cdescr;

    printf("<<CUDA>>\n");
    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK(cudaMalloc((void**)&a_row, sizeof(int)*(sp_matrix->row_size+1)));
    CHECK(cudaMalloc((void**)&a_col, sizeof(int)*(sp_matrix->nnz)));
    CHECK(cudaMalloc((void**)&a_val, sizeof(float)*(sp_matrix->nnz)));
    CHECK(cudaMalloc((void**)&b,     sizeof(float)*(sp_matrix->col_size*mm_col)));
    CHECK(cudaMalloc((void**)&c,     sizeof(float)*(sp_matrix->row_size*mm_col)));

    CHECK(cudaMemcpy(a_row, sp_matrix->row_p, sizeof(int)*(sp_matrix->row_size+1), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a_col, sp_matrix->col_p, sizeof(int)*(sp_matrix->nnz), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a_val, sp_matrix->val,   sizeof(float)*(sp_matrix->nnz), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b,     matrix,           sizeof(float)*(sp_matrix->col_size*mm_col), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(c,     0,                sizeof(float)*(sp_matrix->row_size*mm_col)));

    CHECK_CUSPARSE(
        cusparseCreateCsr(&Adescr, 
            sp_matrix->row_size, sp_matrix->col_size, sp_matrix->nnz, 
            a_row, a_col, a_val, 
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
        )
    );

    CHECK_CUSPARSE(
        cusparseCreateDnMat(&Bdescr, 
            sp_matrix->col_size, mm_col, mm_col, b, 
            CUDA_R_32F, CUSPARSE_ORDER_ROW
        )
    );

    CHECK_CUSPARSE(
        cusparseCreateDnMat(&Cdescr, 
            sp_matrix->row_size, mm_col, mm_col, c, 
            CUDA_R_32F, CUSPARSE_ORDER_ROW
        )
    );

    float alpha = 1;
    float beta = 0;
    size_t buffer_size;
    void *buffer;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUSPARSE(
        cusparseSpMM_bufferSize(handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        (void*)&alpha, Adescr, Bdescr, (void*)&beta, Cdescr, 
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size)
    );

    CHECK(cudaMalloc(&buffer, buffer_size));
    CHECK_CUSPARSE(
        cusparseSpMM_preprocess(handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        (void*)&alpha, Adescr, Bdescr, (void*)&beta, Cdescr, 
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer)
    );

    CHECK_CUSPARSE(
        cusparseSpMM(handle, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        (void*)&alpha, Adescr, Bdescr, (void*)&beta, Cdescr, 
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, buffer)
    );
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);
    printf("cuSPARSE SpMM: %lf usec.\n", cal_time(&t2, &t1));

    CHECK(cudaMemcpy(result, c, sizeof(float)*(sp_matrix->row_size*mm_col), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(a_row));
    CHECK(cudaFree(a_col));
    CHECK(cudaFree(a_val));
    CHECK(cudaFree(b));
    CHECK(cudaFree(c));
    CHECK(cudaFree(buffer));
    
    CHECK_CUSPARSE(cusparseDestroySpMat(Adescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(Bdescr));
    CHECK_CUSPARSE(cusparseDestroyDnMat(Cdescr));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

extern "C"
void mm(float *result, float *a, float *b, int row_a, int col_a, int col_b) {
    float *dA, *dB, *dC;
    struct timespec t1, t2;
    cublasHandle_t handle;

    printf("<<CUDA>>\n");
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK(cudaMalloc((void**)&dA,  sizeof(float)*(row_a*col_a)));
    CHECK(cudaMalloc((void**)&dB,  sizeof(float)*(col_a*col_b)));
    CHECK(cudaMalloc((void**)&dC,  sizeof(float)*(row_a*col_b)));

    CHECK(cudaMemcpy(dA, a, sizeof(float)*(row_a*col_a), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, b, sizeof(float)*(col_a*col_b), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, sizeof(float)*(row_a*col_b)));

    float alpha = 1;
    float beta = 0;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUBLAS(
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            col_b, row_a, col_a,
            &alpha, dB, col_b, dA, col_a, 
            &beta, dC, col_b
        )
    );
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);

    printf("cuBLAS MM: %lf usec.\n", cal_time(&t2, &t1));

    CHECK(cudaMemcpy(result, dC, sizeof(float)*(row_a*col_b), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));
}

__global__
void threshold(float *result, float *input, float threshold, int size) {
    unsigned int xIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if (xIdx < size) {
        float input_val = input[xIdx];
        result[xIdx] = input_val;
        if (input_val < 0) {
            result[xIdx] = 0;
        }
    }
}

#define BLOCK_SIZE_CUDA 128

extern "C"
void relu(float *result, float *a, int size) {
    float *dA, *dR;
    struct timespec t1, t2;
    dim3 dimBlock(BLOCK_SIZE_CUDA);
    dim3 dimGrid(size / BLOCK_SIZE_CUDA);

    printf("<<CUDA>>\n");

    CHECK(cudaMalloc((void**)&dA,  sizeof(float)*(size)));
    CHECK(cudaMalloc((void**)&dR,  sizeof(float)*(size)));

    CHECK(cudaMemcpy(dA, a, sizeof(float)*(size), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dR, 0, sizeof(float)*(size)));

    timespec_get(&t1, TIME_UTC);
    threshold<<<dimGrid, dimBlock>>>(dR, dA, 0, size);
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);

    printf("CUDA ReLU: %lf usec.\n", cal_time(&t2, &t1));

    CHECK(cudaMemcpy(result, dR, sizeof(float)*(size), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dR));
}

#endif
