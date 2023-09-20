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
    int i, j, k;
    #ifdef EMAX6
        Uchar *membase = NULL;
    #endif

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

    result.row_size = row;
    result.col_size = out_size;
    allocDenseMatrix(&result);

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

    #ifdef EMAX6
        IMAXSparseMatrix imax_sp;
        IMAXDenseMatrix imax_m, imax_r, imax_r2, imax_r3;
        timespec_get(&t0, TIME_UTC);
        timespec_get(&t1, TIME_UTC);
        imax_sparse_format_init(&imax_sp, row, row, 46, 8);
        convert_imax_sparse_format(&imax_sp, &sp);
        timespec_get(&t2, TIME_UTC);
        printf("Convert to IMAX_SpMM: %lf usec.\n", cal_time(&t2, &t1));
        imax_dense_format_init_from_sparse(&imax_m, &imax_sp, out_size, 8);
        imax_dense_format_init(&imax_r, row, out_size, imax_sp.row_padded_size, imax_m.col_padded_size, imax_sp.row_blk_size, imax_m.col_blk_size);
        imax_dense_format_init(&imax_r2, out_size, out_size, imax_m.row_padded_size, imax_m.col_padded_size, imax_m.row_blk_size, imax_m.col_blk_size);
        imax_dense_format_init(&imax_r3, row, out_size, imax_sp.row_padded_size, imax_m.col_padded_size, imax_m.row_blk_size, imax_m.col_blk_size);
        imax_sparse_allocation(&membase, &imax_sp);
        imax_dense_allocation(&membase, &imax_m);
        imax_dense_allocation(&membase, &imax_r);
        imax_dense_allocation(&membase, &imax_r2);
        imax_dense_allocation(&membase, &imax_r3);
        timespec_get(&t1, TIME_UTC);
        convert_imax_dense_format(&imax_m, &m);
        timespec_get(&t2, TIME_UTC);
        printf("Convert to IMAX_MM: %lf usec.\n", cal_time(&t2, &t1));
        printf("<<<IMAX>>>\n");
        reset_nanosec();
        spmm(&imax_r, &imax_sp, &imax_m);
        get_nanosec(0);
        show_nanosec();
        timespec_get(&t1, TIME_UTC);
        convert_dense_format(&result, &imax_r);
        timespec_get(&t2, TIME_UTC);
        printf("Convert to MM: %lf usec.\n", cal_time(&t2, &t1));
        imax_mm_allocation(&membase, &imax_r, &imax_r2, &imax_r3, 1, 1, 1);
        printf("<<<IMAX>>>\n");
        reset_nanosec();
        mm(&imax_r3, &imax_r, &imax_r2, 0);
        get_nanosec(0);
        show_nanosec();
    #else
        timespec_get(&t0, TIME_UTC);
        #ifdef USE_CUDA
            sendSparseMatrixToGPU(&sp);
            sendDenseMatrixToGPU(&m);
        #endif
        spmm(&result, &sp, &m);
        #ifdef USE_CUDA
            sendDenseMatrixToCPU(&result);
        #endif
        timespec_get(&t2, TIME_UTC);
    #endif

    print_weight(&result);
    printf("nnz val: %d\n", row_nnz);
    printf("nnz total: %d\n", sp.nnz);
    printf("SpMM: %lf usec.\n", cal_time(&t2, &t0));

    return 0;
}