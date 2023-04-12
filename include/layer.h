#ifndef __LAYER_H__
#define __LAYER_H__
#include "options.h"
#include "emax6.h"
#include "sparse.h"

typedef struct sparse_graph {
    SparseMatrix matrix;
    SparseMatrixParams params;
    #ifdef USE_IMAX2
    IMAXSparseMatrix imax_matrix;
    #endif
} SparseGraph;

typedef struct hidden_layer {
    int dim_in;
    int dim_out;
    float *weight;
} HiddenLayer;

typedef struct gcn_layer {
    HiddenLayer hidden_layer; 
    HiddenLayer latent_vectors;
    struct gcn_layer *prev;
    struct gcn_layer *next;
} GCNLayer;

typedef struct gcn_network {
    int num_layers;
    SparseGraph *graph;
    GCNLayer *layers;
} GCNNetwork;

void print_weight(HiddenLayer *result);
SparseGraph* spia(SparseGraph *graph);
void print_layers(GCNNetwork *network);
void add_gcn_layer(GCNNetwork *network, float *weight, float *vectors, int dim_in, int dim_out);
float* make_weight(int dim_in, int dim_out);
HiddenLayer* propagation(GCNNetwork *network);
HiddenLayer* softmax(HiddenLayer *end_vectors);
float max_in_array(float *array, int size);

#endif