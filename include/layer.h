// EMAX6/7 GCN Test Program            //
// layer.h                             //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __LAYER_H__
#define __LAYER_H__
#include "sparse.h"

typedef struct sparse_graph {
    SparseMatrix          matrix;
    #if defined(EMAX6) || defined(EMAX7)
    IMAXSparseMatrix imax_matrix;
    #endif
} SparseGraph;

typedef DenseMatrix HiddenLayer;

void print_weight(HiddenLayer *result);

#endif