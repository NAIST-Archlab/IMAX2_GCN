#ifndef __SPARSE_H__
#define __SPARSE_H__
#ifdef USE_IMAX2
#include "emax6.h"
#else
#ifndef UTYPEDEF
#define UTYPEDEF
typedef unsigned char      Uchar;
typedef unsigned short     Ushort;
typedef unsigned int       Uint;
typedef unsigned long long Ull;
typedef long long int      Sll;
#if __AARCH64EL__ == 1
typedef long double Dll;
#else
typedef struct {Ull u[2];} Dll;
#endif
#endif
#endif

#define KERNEL_MODE_1 1
#define KERNEL_MODE_2 2
#define LMM_SIZE 0x4000

typedef struct sparse_matrix {
    int nnz; // Number of Non-zero values
    int col_size; //column size
    int row_size; //row size
    int *row_p; //row pointer
    int *col_p; //col pointer
    float *val; //value of each index/Floating
} SparseMatrix;

typedef struct sparse_matrix_params {
    int mode; //KERNEL_MODE_x
    int padding; //pading option
} SparseMatrixParams;

#ifdef USE_IMAX2
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
#endif

#ifdef USE_IMAX2
void spmm(IMAXDenseMatrix* result, IMAXSparseMatrix *imax_sp_matrix, IMAXDenseMatrix* matrix);
void sysinit(Uchar **membase, Uint memsize, Uint alignment);
void mem_release(Uchar **membase, Uint memsize);
#else
void spmm(float* result, SparseMatrix *sp_matrix, float* matrix, int mm_col);
#endif
void mm(float *result, float *a, float *b, int col_a, int row_a, int row_b);
void relu(float *result, float *a, int size);

#endif