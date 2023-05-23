#include "../include/layer.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    SparseMatrix sp;
    SparseMatrixParams sp_params;
    HiddenLayer result;
    struct timespec t1, t2;
    float *m;
    int i, j;
    int row_mul = 1;
    Uchar *membase = NULL;

    if (argc < 3) {
        printf("Usage: %s row_size row_nnz (row_mul)\n", argv[0]);
        return 1;
    }

    if (argc > 3) row_mul = atoi(argv[3]);
    
    int row_nnz = atoi(argv[2]);
    int row = atoi(argv[1]) * row_mul;

    printf("Size:(%d*%d), nnz: %d\n", row, row, row_nnz);

    sp.nnz = row_nnz * row / 2;
    sp.row_size = row;
    sp.row_p = (int*) malloc(sizeof(int) * (row + 1));
    sp.col_size = row;
    sp.col_p = (int*) malloc(sizeof(int) * sp.nnz);
    sp.val = (float*) malloc(sizeof(float) * sp.nnz);

    m = (float*) malloc(sizeof(float) * row * row);

    result.weight = (float*) malloc(sizeof(float) * row * row);
    memset(result.weight, 0, sizeof(float) * row * row);
    result.dim_in = row;
    result.dim_out = row;
    
    sp.row_p[0] = 0;
    for (i = 1; i < row; i++) {
        if (i % 2) sp.row_p[i] = sp.row_p[i-1] + row_nnz;
        else sp.row_p[i] = sp.row_p[i-1];
    }
    sp.row_p[row] = sp.nnz;

    for (i = 0; i < row; i++) {
        int k = 0;
        for (j = sp.row_p[i]; j < sp.row_p[i+1]; j++) {
            sp.col_p[j] = k;
            sp.val[j] = 1.0F;
            k += 2;
        }
    }

    for (i = 0; i < row*row; i++) {
        if (i % 2) m[i] = 0.0F;
        else m[i] = 1.0F;
    }

    #ifdef USE_IMAX2
    IMAXSparseMatrix imax_sp;
    IMAXDenseMatrix imax_m, imax_r;
    imax_sparse_format_init(&imax_sp, row, row, 46, 8);
    convert_imax_sparse_format(&imax_sp, &sp);
    imax_dense_format_init_from_sparse(&imax_m, &imax_sp, row, 8);
    imax_dense_format_init(&imax_r, row, row, imax_sp.row_padded_size, imax_m.col_padded_size, imax_sp.row_blk_size, imax_m.col_blk_size);
    imax_allocation(membase, &imax_sp, &imax_m, &imax_r);
    convert_imax_dense_format(&imax_m, m);
    timespec_get(&t1, TIME_UTC);
    printf("<<<IMAX>>>\n");
    reset_nanosec();
    spmm(&imax_r, &imax_sp, &imax_m);
    get_nanosec(0);
    timespec_get(&t2, TIME_UTC);
    convert_dense_format(result.weight, &imax_r);
    #else
    timespec_get(&t1, TIME_UTC);
    spmm(result.weight, &sp, &sp_params, m, row); 
    timespec_get(&t2, TIME_UTC);
    #endif

    print_weight(&result);
    printf("nnz val: %d\n", row_nnz);
    printf("SpMM: %lf sec.\n", cal_time(&t2, &t1));
    #ifdef USE_IMAX2
    show_nanosec();
    #endif

    return 0;
}