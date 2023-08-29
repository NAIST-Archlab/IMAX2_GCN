#ifndef __SPARSE_H__
#define __SPARSE_H__
#ifdef EMAX6
#include "emax6.h"
#else
#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char Uchar;
typedef unsigned short Ushort;
typedef unsigned int Uint;
typedef unsigned long long Ull;
typedef long long int Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct { Ull u[2]; } Dll;
#endif
#endif
#endif

#define KERNEL_MODE_1 1
#define KERNEL_MODE_2 2
#define LMM_SIZE 0x4000

typedef struct sparse_matrix {
    int nnz;      // Number of Non-zero values
    int col_size; // column size
    int row_size; // row size
    int *row_p;   // row pointer
    int *col_p;   // col pointer
    float *val;   // value of each index/Floating
    int *cuda_row_p; // for CUDA
    int *cuda_col_p; // for CUDA
    float *cuda_val; // for CUDA
} SparseMatrix;

#ifdef EMAX6
typedef struct sparse_matrix_sub_imax2 {
    int nnz;
    int *row_nnz;
    Uint *row_num;
    Uint *col_num;
    Uint *val;
} IMAXSparseMatrixSub;

typedef struct sparse_matrix_imax2 {
    int nnz;
    int row_size;
    int col_size;
    int row_padded_size;
    int col_padded_size;
    int row_blk_size;
    int col_blk_size;
    int col_blk_min;
    int nnz_col_blk_size;
    IMAXSparseMatrixSub **sub;
} IMAXSparseMatrix;

typedef struct dense_matrix_imax2 {
    int row_size;
    int col_size;
    int row_padded_size;
    int col_padded_size;
    int row_blk_size;
    int col_blk_size;
    Uint *val;
} IMAXDenseMatrix;

void spmm(IMAXDenseMatrix *result, IMAXSparseMatrix *imax_sp_matrix, IMAXDenseMatrix *matrix);
void mm(IMAXDenseMatrix *result, IMAXDenseMatrix *imax_a, IMAXDenseMatrix *imax_b, int is_relu);
void sysinit(Uchar **membase, Uint memsize, Uint alignment);
void mem_release(Uchar **membase, Uint memsize);
#else
#if __cplusplus
extern "C" {
#endif
void spmm(float **gpuResult, SparseMatrix *sp_matrix, float *matrix, int mm_col);
void mm(float **gpuResult, float *a, float *b, int col_a, int row_a, int row_b);
void relu(float **gpuResult, float *a, int size);
void sendSparseMatrixToGPU(SparseMatrix *sp_matrix);
void sendSparseMatrixToCPU(SparseMatrix *sp_matrix);
void sendDenseMatrixToGPU(float **gpuMatrix, float *matrix, int row, int col);
void sendDenseMatrixToCPU(float *matrix, float *gpuMatrix, int row, int col);
void freeGPUDenseMatrix(float *gpuMatrix);
#if __cplusplus
}
#endif
#endif

#ifndef USE_CUDA
void spmm(float *result, SparseMatrix *sp_matrix, float *matrix, int mm_col);
void mm(float *result, float *a, float *b, int col_a, int row_a, int row_b);
void relu(float *result, float *a, int size);
#endif

#endif