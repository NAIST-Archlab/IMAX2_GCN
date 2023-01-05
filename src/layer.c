#include "../include/layer.h"
#include <stdlib.h>

void add_gcn_layer(GCNNetwork *network, float *weight, float *vectors, int dim_in, int dim_out) {
    int i;
    GCNLayer *p = network->layers;
    GCNLayer *layer = (GCNLayer*) malloc(sizeof(GCNLayer));

    if (p != NULL) {
        while (p->next != NULL) {
            p = p->next;
        }
        p->next = layer;
        layer->prev = p;
    } else {
        network->layers = layer;
    }

    layer->next = NULL;
    layer->hidden_layer.dim_in = dim_in;
    layer->hidden_layer.dim_out = dim_out;
    layer->hidden_layer.weight = weight;
    layer->latent_vectors.dim_in = network->graph.matrix.row_size;
    layer->latent_vectors.dim_out = dim_in;
    layer->hidden_layer.weight = vectors;
}

float* make_weight(int dim_in, int dim_out) {
    return (float*) malloc(sizeof(float)*dim_in*dim_out);
}

HiddenLayer* propagation(GCNNetwork *network) {
    return NULL;
}