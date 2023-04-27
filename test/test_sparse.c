#include "../include/layer.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define ROW (46*8)

void trans_matrix(float *dst, float *org, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            dst[i*col + j] = org[(j/2)*(row*2)+(i*2)+(j%2)];
        }
    }
}

int main(int argc, char **argv) {
    SparseMatrix sp;
    SparseMatrixParams sp_params;
    HiddenLayer result;
    struct timespec t1, t2;
    float *m, *tm;
    int i, j;

    if (argc < 2) {
        printf("Usage: %s row_nnz\n");
        return 1;
    }
    
    int row_nnz = atoi(argv[1]);
    sp.nnz = row_nnz * ROW / 2;
    sp.row_size = ROW;
    sp.row_p = (int*) malloc(sizeof(int) * (ROW + 1));
    sp.col_size = ROW;
    sp.col_p = (int*) malloc(sizeof(int) * sp.nnz);
    sp.val = (float*) malloc(sizeof(float) * sp.nnz);

    m = (float*) malloc(sizeof(float) * ROW * ROW);
    #ifdef USE_IMAX2
    tm = (float*) malloc(sizeof(float) * ROW * ROW);
    #endif

    result.weight = (float*) malloc(sizeof(float) * ROW * ROW);
    memset(result.weight, 0, sizeof(float) * ROW * ROW);
    result.dim_in = ROW;
    result.dim_out = ROW;
    
    sp.row_p[0] = 0;
    for (i = 1; i < ROW; i++) {
        if (i % 2) sp.row_p[i] = sp.row_p[i-1] + row_nnz;
        else sp.row_p[i] = sp.row_p[i-1];
    }
    sp.row_p[ROW] = sp.nnz;

    for (i = 0; i < ROW; i++) {
        int k = 0;
        for (j = sp.row_p[i]; j < sp.row_p[i+1]; j++) {
            sp.col_p[j] = k;
            sp.val[j] = 1.0F;
            k += 2;
        }
    }

    for (i = 0; i < ROW*ROW; i++) {
        m[i] = 3.0F;
    }

    printf("%f %f\n", sp.val[100], m[100]);

    #ifdef USE_IMAX2
    IMAXSparseMatrix imax_sp;
    trans_imax_format(&imax_sp, &sp);
    timespec_get(&t1, TIME_UTC);
    spmm(result.weight, &imax_sp, m, ROW);
    timespec_get(&t2, TIME_UTC);
    trans_matrix(tm, result.weight, ROW, ROW);
    free(result.weight);
    result.weight = tm;
    #else
    timespec_get(&t1, TIME_UTC);
    spmm(result.weight, &sp, &sp_params, m, ROW); 
    timespec_get(&t2, TIME_UTC);
    #endif

    int nonzero = 0;
    for (i = 0; i < ROW*ROW; i++) {
        if (result.weight[i] != 0) nonzero++;
    }
    printf("\n");

    printf("%d\n", nonzero);

    print_weight(&result);
    printf("%d\n", row_nnz*3);
    printf("SpMM: %lf sec.\n", cal_time(&t2, &t1));

    return 0;
}