#include "../include/layer.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    SparseMatrix sp;
    DenseMatrix m;
    HiddenLayer result;
    struct timespec t0, t1, t2;
    int i, j;

    if (argc < 4) {
        printf("Usage: %s row_size row_nnz out_size\n", argv[0]);
        return 1;
    }

    int out_size = atoi(argv[3]);

    int row_nnz = atoi(argv[2]);
    int row = atoi(argv[1]);
    srand((unsigned)time(NULL));

    printf("Size:(%d*%d), nnz: %d\n", row, row, row_nnz);

    sp.nnz = row_nnz * row;
    sp.row_size = row;
    sp.col_size = row;
    allocSparseMatrix(&sp);

    m.row_size = row;
    m.col_size = 100;
    allocDenseMatrix(&m);

    result.matrix.row_size = row;
    result.matrix.col_size = out_size;
    allocDenseMatrix(&result.matrix);

    #ifdef SAME_DISTANCE
    for (i = 0; i < row; i++) {
        for (j = 0; j < row_nnz; j++) {
            sp.col_p[i*row_nnz+j] = j*(sp.col_size/row_nnz);
            sp.val[i*row_nnz+j] = 1.0F;
        }
    }
    #endif

    #ifndef SAME_DISTANCE
    int acc = 0;
    for (i = 0; i < sp.row_size; i++) {
        int start = i - (i%10);
        if ((start - row_nnz) < 0) {
            for (j = 0; j < row_nnz*2; j+=2) {
                sp.col_p[acc] = j;
                sp.val[acc] = 1.0F;
                acc++;
            }
        } else if ((start - row_nnz) >= 0 && (start + row_nnz < sp.row_size)) {
            start = i - (i%10) - row_nnz;
            for (j = start; j < start + row_nnz*2; j+=2) {
                sp.col_p[acc] = j;
                sp.val[acc] = 1.0F;
                acc++;
            }
        } else {
            for (j = sp.row_size - row_nnz*2; j < sp.row_size; j+=2) {
                sp.col_p[acc] = j;
                sp.val[acc] = 1.0F;
                acc++;
            }
        }
    }
    #endif

    sp.row_p[0] = 0;
    for (i = 1; i < row; i++) {
        sp.row_p[i] = sp.row_p[i - 1] + row_nnz;
    }
    sp.row_p[row] = sp.nnz;

    for (i = 0; i < row * out_size; i++) {
        if (i % 2)
            m.val[i] = 0.0F;
        else
            m.val[i] = 1.0F;
    }

    #if defined(EMAX6) || defined(EMAX7)
        IMAXSparseMatrix imax_sp;
        IMAXDenseMatrix imax_m, imax_r, imax_r2, imax_r3;
        timespec_get(&t0, TIME_UTC);
        timespec_get(&t1, TIME_UTC);
        imax_sparse_format_init(&imax_sp, row, row, SPMM_H, 8 * NCHIP);
        convert_imax_sparse_format(&imax_sp, &sp);
        timespec_get(&t2, TIME_UTC);
        printf("Convert to IMAX_SpM: %lf usec.\n", cal_time(&t2, &t1));
        imax_dense_format_init_from_sparse(&imax_m, &imax_sp, out_size, 8 * NCHIP);
        imax_dense_format_init(&imax_r, row, out_size, imax_sp.row_padded_size, imax_m.col_padded_size, imax_sp.row_blk_size, imax_m.col_blk_size);
        imax_dense_format_init(&imax_r2, row, out_size, imax_sp.row_padded_size, imax_m.col_padded_size, imax_sp.row_blk_size, imax_m.col_blk_size);
        imax_dense_format_init(&imax_r3, row, out_size, imax_sp.row_padded_size, imax_m.col_padded_size, imax_sp.row_blk_size, imax_m.col_blk_size);
        imax_gcn_allocation(&imax_sp, &imax_m, &imax_r, &imax_r2, &imax_r3);
        timespec_get(&t1, TIME_UTC);
        convert_imax_dense_format(&imax_m, &m);
        timespec_get(&t2, TIME_UTC);
        printf("Convert to IMAX_MM: %lf usec.\n", cal_time(&t2, &t1));
        spmm(&imax_r, &imax_sp, &imax_m);
        convert_dense_format(&result.matrix, &imax_r);
    #else
        timespec_get(&t0, TIME_UTC);
        #ifdef USE_CUDA
            sendSparseMatrixToGPU(&sp);
            sendDenseMatrixToGPU(&m);
            createCusparse();
        #endif
        spmm(&result, &sp, &m);
        #ifdef USE_CUDA
            sendDenseMatrixToCPU(&result);
            destroyCusparse();
        #endif
        timespec_get(&t2, TIME_UTC);
    #endif

    print_weight(&result);
    printf("nnz val: %d\n", row_nnz);
    printf("nnz total: %d\n", sp.nnz);
    #if defined(EMAX6) || defined(EMAX7)
        printf("SpMM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[SPMM][0]/1000),
            (Uint)(all_nanosec[SPMM][1]/1000),
            (Uint)(all_nanosec[SPMM][2]/1000),
            (Uint)(all_nanosec[SPMM][3]/1000),
            (Uint)(all_nanosec[SPMM][4]/1000),
            (Uint)(all_nanosec[SPMM][5]/1000),
            (Uint)(all_nanosec[SPMM][6]/1000),
            (Uint)(all_nanosec[SPMM][7]/1000));
    #endif

    return 0;
}