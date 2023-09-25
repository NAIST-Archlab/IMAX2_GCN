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

#if defined(EMAX6) || defined(EMAX7)
typedef struct sparse_matrix_sub_imax2 {
    int nnz;
    int nnz_row_blk_size;
    Uint *row_nnz;
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
    int nnz_row_blk_size;
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
void createCublase();
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