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

typedef struct dense_matrix {
    int row_size;
    int col_size;
    float *val;
    float *cuda_val; // for CUDA
} DenseMatrix;

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
void mm(IMAXDenseMatrix *result, IMAXDenseMatrix *imax_a, IMAXDenseMatrix *imax_b);
void relu(DenseMatrix *result, DenseMatrix *a);
void sysinit(Uchar **membase, Uint memsize, Uint alignment);
void imax_add_alloc(Uchar **membase, Uint memsize, Uint alignment);
void mem_release(Uchar **membase, Uint memsize);
#endif
#if defined(USE_CUDA)
#if __cplusplus
extern "C" {
#endif
void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix);
void mm(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b);
void relu(DenseMatrix *result, DenseMatrix *a);
void allocSparseMatrix(SparseMatrix *sp_matrix);
void allocDenseMatrix(DenseMatrix *matrix);
void sendSparseMatrixToGPU(SparseMatrix *sp_matrix);
void sendSparseMatrixToCPU(SparseMatrix *sp_matrix);
void sendDenseMatrixToGPU(DenseMatrix *matrix);
void sendDenseMatrixToCPU(DenseMatrix *matrix);
void freeGPUDenseMatrix(DenseMatrix *matrix);
void freeGPUSparseMatrix(SparseMatrix *sp_matrix);
void freeDenseMatrix(DenseMatrix *matrix);
void freeSparseMatrix(SparseMatrix *sp_matrix);
#if __cplusplus
}
#endif
#else
void allocSparseMatrix(SparseMatrix *sp_matrix);
void freeSparseMatrix(SparseMatrix *sp_matrix);
void allocDenseMatrix(DenseMatrix *matrix);
void freeDenseMatrix(DenseMatrix *matrix);
#endif

#if !defined(EMAX6) && !defined(USE_CUDA)
    void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix);
    void mm(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b);
    void relu(DenseMatrix *result, DenseMatrix *a);
#endif

#endif