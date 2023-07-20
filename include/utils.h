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

#ifdef EMAX6
void imax_dense_format_init(IMAXDenseMatrix *imax_m, int row, int col, int row_padded, int col_padded, int row_blk, int col_blk);
void imax_sparse_format_init(IMAXSparseMatrix *imax_sp, int row, int col, int sp_col_blk, int m_col_blk_min);
void imax_dense_format_init_from_sparse(IMAXDenseMatrix *imax_m, IMAXSparseMatrix *imax_sp, int m_col, int m_col_blk_min);
void imax_allocation(Uchar *membase, IMAXSparseMatrix *imax_sp, IMAXDenseMatrix *imax_m, IMAXDenseMatrix *imax_r);
void imax_deallocation(Uchar *membase, IMAXSparseMatrix *imax_sp, IMAXDenseMatrix *imax_m, IMAXDenseMatrix *imax_r);

void convert_imax_dense_format(IMAXDenseMatrix *imax_m, Uint *m);
void convert_dense_format(Uint *m, IMAXDenseMatrix *imax_m);
void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp);
#endif

#endif