// EMAX6/7 GCN Test Program            //
// linalg_cuda.c                       //
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
#include "../include/linalg.h"
#include "../include/utils.h"

extern "C"
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

extern "C"
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

extern "C"
float max_in_array(float *array, int size) {
    int i;
    float max = -INFINITY;

    for (i = 0; i < size; i++) {
        if (max < array[i])
            max = array[i];
    }

    return max;
}

extern "C"
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


extern "C"
void allocDenseMatrix(DenseMatrix *matrix) {
    matrix->val = (float*) malloc(sizeof(float)*(matrix->row_size*matrix->col_size));
    CHECK(cudaMalloc((void**) &matrix->cuda_val, sizeof(float)*(matrix->row_size*matrix->col_size)));
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
void freeDenseMatrix(DenseMatrix *matrix) {
    free(matrix->val);
    freeGPUDenseMatrix(matrix);
}
cublasHandle_t cublas_handle;

extern "C"
void createCublas() {
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
}

extern "C"
void destroyCublas() {
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
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

extern "C"
void msub (DenseMatrix *result, DenseMatrix *a, DenseMatrix *b) {
    struct timespec t1, t2;

    float alpha = 1;
    float beta = 0;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUBLAS(
        cublasSgeam(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            a->row_size, a->col_size,
            &alpha, a->cuda_val, a->row_size, &beta, b->cuda_val, b->row_size, result->cuda_val, a->row_size
        )
    );
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);

    printf("cuBLAS MSUB: %lf usec.\n", cal_time(&t2, &t1));
}

extern "C"
float mmeans(DenseMatrix *a) {
    struct timespec t1, t2;
    float sum = 0;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUBLAS(
        cublasSasum(cublas_handle,
            a->row_size * a->col_size,
            a->cuda_val, 1, &sum
        )
    );
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);

    printf("cuBLAS MMEANS: %lf usec.\n", cal_time(&t2, &t1));

    return sum / (a->row_size * a->col_size);
}

extern "C"
void transpose(DenseMatrix *result, DenseMatrix *a) {
    struct timespec t1, t2;

    float alpha = 1;
    float beta = 0;

    timespec_get(&t1, TIME_UTC);
    CHECK_CUBLAS(
        cublasSgeam(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            a->col_size, a->row_size,
            &alpha, a->cuda_val, a->row_size, &beta, a->cuda_val, a->row_size, result->cuda_val, a->col_size
        )
    );
    timespec_get(&t2, TIME_UTC);

    CHECK(cudaDeviceSynchronize());
    printf("cuBLAS Transpose: %lf usec.\n", cal_time(&t2, &t1));
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

__global__
void d_threshold(float *gpuResult, float *input, float threshold, int size) {
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

extern "C"
void d_relu(DenseMatrix *result, DenseMatrix *a) {
    struct timespec t1, t2;
    dim3 dimBlock(BLOCK_SIZE_CUDA);
    dim3 dimGrid((a->row_size * a->col_size) / BLOCK_SIZE_CUDA);

    printf("<<CUDA>>\n");

    timespec_get(&t1, TIME_UTC);
    d_threshold<<<dimGrid, dimBlock>>>(result->cuda_val, a->cuda_val, 0, a->row_size * a->col_size);
    CHECK(cudaDeviceSynchronize());
    timespec_get(&t2, TIME_UTC);

    printf("CUDA ReLU: %lf usec.\n", cal_time(&t2, &t1));
}

#endif
