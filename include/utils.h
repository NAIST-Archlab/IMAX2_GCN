// EMAX6/7 GCN Test Program            //
// utils.h                             //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __UTILS_H__
#define __UTILS_H__
#include "sparse.h"
#include <time.h>

#if __cplusplus
extern "C" {
#endif
double cal_time(struct timespec *end, struct timespec *start);
#if __cplusplus
}
#endif

#if defined(EMAX6) || defined(EMAX7)
Uchar* sysinit(Uint memsize, Uint alignment);
Uchar* imax_alloc(Uint memsize, Uint alignment);
void imax_dealloc(Uint memsize, Uint alignment);
void imemcpy(Uint *dst, Uint *src, int words);
void xmax_bzero(Uint *dst, int words);
void imax_dense_format_init(IMAXDenseMatrix *imax_m, int row, int col, int row_padded, int col_padded, int row_blk, int col_blk);
void imax_sparse_format_init(IMAXSparseMatrix *imax_sp, int row, int col, int sp_col_blk, int m_col_blk_min);
void imax_dense_format_init_from_sparse(IMAXDenseMatrix *imax_m, IMAXSparseMatrix *imax_sp, int m_col, int m_col_blk_min);
void imax_matrix_init_spmm(IMAXDenseMatrix *c, IMAXSparseMatrix *a, IMAXDenseMatrix *b, int matrix_flag);
void imax_matrix_init_mm(IMAXDenseMatrix *c, IMAXDenseMatrix *a, IMAXDenseMatrix *b, int matrix_flag);
void imax_sparse_allocation(IMAXSparseMatrix *imax_sp);
void imax_sparse_deallocation(IMAXSparseMatrix *imax_sp);
void imax_dense_allocation(IMAXDenseMatrix *imax_m);
void imax_dense_deallocation(IMAXDenseMatrix *imax_m);

void convert_imax_dense_format(IMAXDenseMatrix *imax_m, DenseMatrix *m);
void convert_dense_format(DenseMatrix *m, IMAXDenseMatrix *imax_m);
void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp);

#define FIT_TO_SPARSE 0
#define FIT_TO_DENSE  1
#endif

#endif