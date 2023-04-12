#include "../include/sparse.h"
#include "../include/layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define ROW 15000

int main() {
    SparseMatrix sp;
    SparseMatrixParams sp_params;
    HiddenLayer result;
    float *m;
    int i, j;

    sp.nnz = (int) round(ROW * ROW * 0.05);
    sp.row_size = ROW;
    sp.row_p = (int*) malloc(sizeof(int) * (ROW + 1));
    sp.col_size = ROW;
    sp.col_p = (int*) malloc(sizeof(int) * sp.nnz);
    sp.val = (float*) malloc(sizeof(float) * sp.nnz);

    m = (float*) malloc(sizeof(float) * ROW * ROW);
    result.weight = (float*) malloc(sizeof(float) * ROW * ROW);
    result.dim_in = ROW;
    result.dim_out = ROW;
    
    sp.row_p[0] = 0;
    for (i = 1; i < ROW; i++) {
        sp.row_p[i] = sp.row_p[i-1] + (sp.nnz / ROW);
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
        m[i] = 1.0F;
    }

    printf("%f %f\n", sp.val[100], m[100]);

    spmm(result.weight, &sp, &sp_params, m, ROW);

    int nonzero = 0;
    for (i = 0; i < ROW*ROW; i++) {
        if (result.weight[i] != 0) nonzero++;
    }

    printf("%d\n", nonzero);

    print_weight(&result);

    return 0;
}