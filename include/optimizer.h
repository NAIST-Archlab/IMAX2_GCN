// EMAX6/7 GCN Test Program            //
// optimizer.h                         //
//         Copyright (C) 2024 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__
#include "sparse.h"
#include <math.h>

typedef struct optimizer_option {
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;
} OptimizerOption;

void adam(DenseMatrix *value, DenseMatrix *grad, OptimizerOption *option);

#endif