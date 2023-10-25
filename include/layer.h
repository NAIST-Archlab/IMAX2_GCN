#ifndef __LAYER_H__
#define __LAYER_H__
#if defined(EMAX7)
#include "../conv-c2d/emax7.h"
#elif defined(EMAX6)
#include "../conv-c2c/emax6.h"
#endif
#include "sparse.h"

typedef struct sparse_graph {
    SparseMatrix          matrix;
    #if defined(EMAX6) || defined(EMAX7)
    IMAXSparseMatrix imax_matrix;
    #endif
} SparseGraph;

typedef struct hidden_layer {
    DenseMatrix matrix;
} HiddenLayer;

typedef struct gcn_layer {
    HiddenLayer    hidden_layer; 
    HiddenLayer  latent_vectors;
    HiddenLayer    result_layer;
    struct gcn_layer      *prev;
    struct gcn_layer      *next;
} GCNLayer;

typedef struct gcn_network {
    int     num_layers;
    SparseGraph *graph;
    GCNLayer   *layers;
} GCNNetwork;

void print_weight(HiddenLayer *result);
SparseGraph* spia(SparseGraph *graph);
void print_layers(GCNNetwork *network);
void add_gcn_layer(GCNNetwork *network, DenseMatrix weight, DenseMatrix vectors);
void propagation(GCNNetwork *network);
void softmax(HiddenLayer *end_vectors);
float max_in_array(float *array, int size);

#endif