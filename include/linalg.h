// EMAX6/7 GCN Test Program            //
// linalg.h                            //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __LINALG_H__
#define __LINALG_H__
#include "imax.h"

enum { SPMM, MM, RELU, SOFTMAX, ATTENTION, ELU, LOG_SOFTMAX, NUM_CLASS };
#if defined(EMAX6) || defined(EMAX7)
unsigned long long all_nanosec[NUM_CLASS][8];
#elif defined(USE_CUDA)
unsigned long long all_nanosec[NUM_CLASS][3];
#else
unsigned long long all_nanosec[NUM_CLASS];
#endif

#ifdef LMM128
#define LMM_SIZE 0x8000
#define MAX_COL_SIZE 0x800 // LMMのサイズの最大値にすると構造上問題が発生するため、1/2にしている
#else
#define LMM_SIZE 0x4000
#define MAX_COL_SIZE 0x400 // LMMのサイズの最大値にすると構造上問題が発生するため、1/2にしている
#endif
#ifdef UNIT32
#ifndef HARD_UNIT32
#define MM_H 32
#else
#define MM_H 16
#endif
#else
#define MM_H 32
#endif

#define MM_MIN 8

typedef struct dense_matrix {
    int   row_size;  // row size
    int   col_size;  // column size
    int   depth_size;
    float      *val; // values
    float *cuda_val; // values for CUDA
} DenseMatrix;

#if defined(EMAX6) || defined(EMAX7)
typedef struct dense_matrix_imax2 {
    int        row_size; // real row size
    int        col_size; // real column size
    int row_padded_size; // row size padded
    int col_padded_size; // column size padded
    int    blk_row_size; // Size of row each block
    int    blk_col_size; // Size of column each block
    Uint           *val; // values
} IMAXDenseMatrix;

void mm(IMAXDenseMatrix *result, IMAXDenseMatrix *imax_a, IMAXDenseMatrix *imax_b);
void relu(DenseMatrix *result, DenseMatrix *a);
void d_relu(DenseMatrix *result, DenseMatrix *a);
#endif
#if __cplusplus
extern "C" {
#endif
#if !(defined(EMAX6) || defined(EMAX7))
void mm(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b);
void relu(DenseMatrix *result, DenseMatrix *a);
void d_relu(DenseMatrix *result, DenseMatrix *a);
#endif
void softmax(DenseMatrix *end_vectors);
void d_softmax(DenseMatrix *result);
float max_in_array(float *array, int size);
void msub (DenseMatrix *result, DenseMatrix *a, DenseMatrix *b);
float mmeans(DenseMatrix *a);
void transpose(DenseMatrix *result, DenseMatrix *a);
void expand_labels(DenseMatrix *labels, Uchar *vlabels);
void allocDenseMatrix(DenseMatrix *matrix);
void freeDenseMatrix(DenseMatrix *matrix);
#ifdef USE_CUDA
void sendDenseMatrixToGPU(DenseMatrix *matrix);
void sendDenseMatrixToCPU(DenseMatrix *matrix);
void freeGPUDenseMatrix(DenseMatrix *matrix);
void createCublas();
void destroyCublas();
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

#endif
#if __cplusplus
}
#endif

#endif