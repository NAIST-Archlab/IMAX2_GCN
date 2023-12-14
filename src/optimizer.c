#include "../include/optimizer.h"
#include <stdlib.h>

void adam(DenseMatrix *value, DenseMatrix *grad, OptimizerOption *option) {
    float beta1 = option->beta1;
    float beta2 = option->beta2;
    float epsilon = option->epsilon;
    float learning_rate = option->lr;
    float *m = (float *) malloc(sizeof(float) * value->row_size * value->col_size);
    float *v = (float *) malloc(sizeof(float) * value->row_size * value->col_size);
    int t = option->t;
    int size = value->row_size * value->col_size;

    for (int i = 0; i < size; i++) {
        m[i] = beta1 * m[i] + (1 - beta1) * grad->val[i];
        v[i] = beta2 * v[i] + (1 - beta2) * grad->val[i] * grad->val[i];
        float m_hat = m[i] / (1 - pow(beta1, t));
        float v_hat = v[i] / (1 - pow(beta2, t));
        float delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon);
        if (isnan(delta)) {
            delta = 0;
        }
        value->val[i] -= delta;
    }

    free(m);free(v);
}