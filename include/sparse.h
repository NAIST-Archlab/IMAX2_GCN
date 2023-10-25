// EMAX6/7 GCN Test Program            //
// sparse.c                            //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __SPARSE_H__
#define __SPARSE_H__
#if defined(EMAX7) || defined(EMAX6)
#if defined(EMAX7)
#include "../conv-c2d/emax7.h"
#elif defined(EMAX6)
#include "../conv-c2c/emax6.h"
#endif
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

enum { SPMM, MM, RELU, SOFTMAX, NUM_CLASS };
#if defined(EMAX6) || defined(EMAX7)
unsigned long long all_nanosec[NUM_CLASS][8];
#else
unsigned long long all_nanosec[NUM_CLASS];
#endif

#define KERNEL_MODE_1 1
#define KERNEL_MODE_2 2
#define LMM_SIZE 0x4000
#define MAX_COL_SIZE 0x800
#ifdef UNIT32
#define MM_H 16
#define SPMM_H 20
#else
#define MM_H 32
#define SPMM_H 46
#endif

typedef struct sparse_matrix {
    int         nnz; // Number of Non-zero values
    int    col_size; // column size
    int    row_size; // row size
    int      *row_p; // row pointer
    int      *col_p; // col pointer
    float      *val; // value of each index/Floating
    int *cuda_row_p; // for CUDA
    int *cuda_col_p; // for CUDA
    float *cuda_val; // for CUDA
} SparseMatrix;

typedef struct dense_matrix {
    int   row_size;  // row size
    int   col_size;  // column size
    float      *val; // values
    float *cuda_val; // values for CUDA
} DenseMatrix;

#if defined(EMAX6) || defined(EMAX7)
typedef struct sparse_matrix_sub_imax2 {
    int              nnz; // Number of Non-zero values in a sub matrix
    int nnz_row_blk_size; // Size of row in Non-zero values block
    Uint        *row_blk; // Start index of each row block in Non-zero values block
    Uint        *row_nnz; // Number of Non-zero values in each row
    Uint        *row_num; // Number of row in each non-zero values block
    Uint        *col_num; // Number of column in each non-zero values
    Uint            *val; // value of each column
} IMAXSparseMatrixSub;

typedef struct sparse_matrix_imax2 {
    int                  nnz; // Number of Non-zero values
    int             mem_size; // Size of memory
    int             row_size; // real row size
    int             col_size; // real column size
    int      row_padded_size; // row size padded
    int      col_padded_size; // column size padded
    int         row_blk_size; // Size of row each block
    int         col_blk_size; // Size of column each block
    int          col_blk_min; // Minimum size of column each block
    int     nnz_row_blk_size; // Size of row in Non-zero values block
    int     nnz_col_blk_size; // Size of column in Non-zero values block
    IMAXSparseMatrixSub *sub; // Sub matrix
} IMAXSparseMatrix;

typedef struct dense_matrix_imax2 {
    int        row_size; // real row size
    int        col_size; // real column size
    int row_padded_size; // row size padded
    int col_padded_size; // column size padded
    int    row_blk_size; // Size of row each block
    int    col_blk_size; // Size of column each block
    Uint           *val; // values
} IMAXDenseMatrix;

void spmm(IMAXDenseMatrix *result, IMAXSparseMatrix *imax_sp_matrix, IMAXDenseMatrix *matrix);
void mm(IMAXDenseMatrix *result, IMAXDenseMatrix *imax_a, IMAXDenseMatrix *imax_b);
void relu(DenseMatrix *result, DenseMatrix *a);
Uchar* sysinit(Uint memsize, Uint alignment);
void imemcpy(Uint *dst, Uint *src, int words);
void xmax_bzero(Uint *dst, int words);
#endif
#ifdef USE_CUDA
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
void createCusparse();
void createCublas();
void destroyCusparse();
void destroyCublas();
#if __cplusplus
}
#endif
#else
void allocSparseMatrix(SparseMatrix *sp_matrix);
void freeSparseMatrix(SparseMatrix *sp_matrix);
void allocDenseMatrix(DenseMatrix *matrix);
void freeDenseMatrix(DenseMatrix *matrix);
#endif

#if !(defined(EMAX6) || defined(EMAX7) || defined(USE_CUDA))
void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix);
void mm(DenseMatrix *result, DenseMatrix *a, DenseMatrix *b);
void relu(DenseMatrix *result, DenseMatrix *a);
#endif

#endif