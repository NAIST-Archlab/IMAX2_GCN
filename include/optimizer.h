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