// EMAX6/7 GCN Test Program            //
// layer.c                             //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/layer.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_weight(HiddenLayer *result) {
    Ull i, j;
    char ry_row = 0;

    printf("Size: (%d,%d)\n", result->row_size, result->col_size);
    printf("[\n");
    for (i = 0; i < result->row_size; i++) {
        if (i > 100 && (i < result->row_size - 30) && ry_row != 1) {
            printf("\t.\n\t.\n\t.\n");
            ry_row = 1;
        } else if (i > (result->row_size - 30) || ry_row == 0) {
            char ry_col = 0;
            printf("\t[ ");
            for (j = 0; j < result->col_size; j++) {
                if (j > 2 && j < (result->col_size - 3) && ry_col != 1) {
                    printf("... ");
                    ry_col = 1;
                } else if (j > (result->col_size - 3) || ry_col == 0) {
                    printf("%10.6f ", result->val[i * result->col_size + j]);
                }
            }
            printf("]\n");
        }
    }
    printf("]\n");
}