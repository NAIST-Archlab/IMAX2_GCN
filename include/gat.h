
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include "linalg.h"
#include "sparse.h"


#define ALPHA 0.2         

// typedef struct vector {
//     int   col_size;  // column size
//     float      *val; // values
//     float *cuda_val; // values for CUDA
// } Vector;


typedef struct gat_graph{
    SparseMatrix *neighbor;
    SparseMatrix *features;
    DenseMatrix *newfeatures;
    DenseMatrix *concatfeature;
    int nfeature;
    
} GATGraph;

typedef struct parameters{
    int in_feature;
    int out_feature;
    DenseMatrix *weights;    
    DenseMatrix *linear;
    DenseMatrix *a_l;
    DenseMatrix *a_r;
    SparseMatrix *attention;
} Param;

typedef struct gat_layer{
    int num_heads;       
    Param *params;   
    DenseMatrix *merged_weight; // B
    DenseMatrix *merged_linear; // C
    DenseMatrix *separated_linear // C
} GATLayer;

void logsoftmax(DenseMatrix *A);
void matrix_elu(DenseMatrix *A);
void transpose(DenseMatrix *A, DenseMatrix *B);
void attention_coeff(DenseMatrix *linear_lr, DenseMatrix *linear, DenseMatrix *self_attn);
void EleWiseSum(DenseMatrix *a, DenseMatrix *linear_l, DenseMatrix *linear_r);
void mask(SparseMatrix *c, DenseMatrix *a, SparseMatrix *adj_matrix);
void matrix_lrelu(DenseMatrix *A);
void s_softmax(SparseMatrix *A);
void concat(DenseMatrix *source, DenseMatrix *destination, int num_heads);
