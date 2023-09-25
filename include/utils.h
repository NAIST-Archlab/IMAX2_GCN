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
void imax_dense_format_init(IMAXDenseMatrix *imax_m, int row, int col, int row_padded, int col_padded, int row_blk, int col_blk);
void imax_sparse_format_init(IMAXSparseMatrix *imax_sp, int row, int col, int sp_col_blk, int m_col_blk_min);
void imax_dense_format_init_from_sparse(IMAXDenseMatrix *imax_m, IMAXSparseMatrix *imax_sp, int m_col, int m_col_blk_min);
void imax_gcn_allocation(IMAXSparseMatrix *imax_sp, IMAXDenseMatrix *imax_h, IMAXDenseMatrix *imax_spmm, IMAXDenseMatrix *imax_w, IMAXDenseMatrix *imax_mm);

void convert_imax_dense_format(IMAXDenseMatrix *imax_m, DenseMatrix *m);
void convert_dense_format(DenseMatrix *m, IMAXDenseMatrix *imax_m);
void convert_imax_sparse_format(IMAXSparseMatrix *imax_sp, SparseMatrix *sp);
#endif

#endif