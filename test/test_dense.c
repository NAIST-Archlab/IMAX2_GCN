// EMAX6/7 GCN Test Program            //
// test_dense.c                        //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/layer.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    DenseMatrix m1;
    DenseMatrix m2;
    HiddenLayer result;
    struct timespec t0, t1;
    int i, j;

    if (argc < 3) {
        printf("Usage: %s row_size out_size\n", argv[0]);
        return 1;
    }

    int out_size = atoi(argv[2]);

    int row = atoi(argv[1]);
    srand((unsigned)time(NULL));

    printf("Size:(%d*%d)\n", row, row);

    m1.row_size = row;
    m1.col_size = row;
    allocDenseMatrix(&m1);

    m2.row_size = row;
    m2.col_size = out_size;
    allocDenseMatrix(&m2);

    result.matrix.row_size = row;
    result.matrix.col_size = out_size;
    allocDenseMatrix(&result.matrix);

    for (i = 0; i < row; i++) {
        for (j = 0; j < row; j++) {
            if (i != j)
                m1.val[i*row + j] = 0.0F;
            else
                m1.val[i*row + j] = 2.0F;
        }
    }

    for (i = 0; i < row; i++) {
        for (j = 0; j < out_size; j++) {
            m2.val[i*out_size + j] = 1.0F;
        }
    }
    #if defined(EMAX6) || defined(EMAX7)
        IMAXDenseMatrix imax_m1, imax_m2, imax_r;
        imax_dense_format_init(&imax_m1, row, row, row, row, row, 16);
        imax_dense_format_init(&imax_m2, row, out_size, row, out_size, 16, 16);
        imax_dense_format_init(&imax_r, row, out_size, row, out_size, row, 16);
        imax_m1.val = (Uint*) malloc(sizeof(Uint) * row * row);
        imax_m2.val = (Uint*) malloc(sizeof(Uint) * row * out_size);
        imax_r.val = (Uint*) malloc(sizeof(Uint) * row * out_size);
        convert_imax_dense_format(&imax_m1, &m1);
        convert_imax_dense_format(&imax_m2, &m2);
        convert_imax_dense_format(&imax_r, &result.matrix);
        timespec_get(&t0, TIME_UTC);
        mm(&imax_r, &imax_m1, &imax_m2);
        timespec_get(&t1, TIME_UTC);
        convert_dense_format(&result.matrix, &imax_r);
        convert_dense_format(&m1, &imax_m1);
        convert_dense_format(&m2, &imax_m2);
    #else
        #ifdef USE_CUDA
            createCublas();
            sendDenseMatrixToGPU(&m1);
            sendDenseMatrixToGPU(&m2);
        #endif
        timespec_get(&t0, TIME_UTC);
        mm(&result.matrix, &m1, &m2);
        timespec_get(&t1, TIME_UTC);
        #ifdef USE_CUDA
            sendDenseMatrixToCPU(&result.matrix);
            destroyCublas();
        #endif
    #endif

    print_weight(&result);
    printf("MM: %lf usec.\n", cal_time(&t1, &t0));

    return 0;
}