// EMAX6/7 GCN Test Program            //
// sparse.h                            //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __SPARSE_H__
#define __SPARSE_H__
#include "linalg.h"

#ifdef UNIT32
#define SPMM_H 20
#else
#define SPMM_H 46
#endif

typedef struct sparse_matrix {
    int         nnz; // Number of Non-zero values
    int    col_size; // column size
    int    row_size; // row size
    int  depth_size;
    int      *row_p; // row pointer
    int      *col_p; // col pointer
    float      *val; // value of each index/Floating
    int *cuda_row_p; // for CUDA
    int *cuda_col_p; // for CUDA
    float *cuda_val; // for CUDA
} SparseMatrix;

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
    int         blk_row_size; // Size of row each block
    int         blk_col_size; // Size of column each block
    int          blk_min_col; // Minimum size of column each block
    int     nnz_blk_col_size; // Size of column in Non-zero values block
    IMAXSparseMatrixSub *sub; // Sub matrix
} IMAXSparseMatrix;

void spmm(IMAXDenseMatrix *result, IMAXSparseMatrix *imax_sp_matrix, IMAXDenseMatrix *matrix);
#endif
#if __cplusplus
extern "C" {
#endif
#if !(defined(EMAX6) || defined(EMAX7))
void spmm(DenseMatrix *result, SparseMatrix *sp_matrix, DenseMatrix *matrix);
#endif
void gcn_preprocessing(SparseMatrix *matrix);
void spia(SparseMatrix *result, SparseMatrix *sp_matrix);
void allocSparseMatrix(SparseMatrix *sp_matrix);
void freeSparseMatrix(SparseMatrix *sp_matrix);
#ifdef USE_CUDA
void sendSparseMatrixToGPU(SparseMatrix *sp_matrix);
void sendSparseMatrixToCPU(SparseMatrix *sp_matrix);
void freeGPUSparseMatrix(SparseMatrix *sp_matrix);
void createCusparse();
void destroyCusparse();
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
#endif
#if __cplusplus
}
#endif

#endif