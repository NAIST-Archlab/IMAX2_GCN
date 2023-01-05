#ifndef __SPARSE_H__
#define __SPARSE_H__
#include "emax6.h"
#include "emax6lib.c"

#define KERNEL_MODE_0 0
#define KERNEL_MODE_1 1
#define KERNEL_MODE_2 2

typedef struct sparse_matrix {
    int nnz; // Number of Non-zero values
    int col_size; //column size
    int row_size; //row size
    int *row_p; //row pointer
    int *col_p; //col pointer
    Uint *val; //value of each index
} SparseMatrix;

typedef struct sparse_matrix_params {
    int mode; //KERNEL_MODE_x
    int padding; //pading option
} SparseMatrixParams;

void spmm(Uint* result, SparseMatrix *sp_matrix, SparseMatrixParams *sp_params, Uint* matrix, int row, int col);

#endif