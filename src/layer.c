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
    GCNLayer *p = network->layers;
    HiddenLayer *result;

    while (p != NULL) {
        Uint *tmp = (Uint *)malloc(sizeof(Uint) * network->graph.matrix.row_size * p->latent_vectors.dim_out);
        int out_size = p->latent_vectors.dim_out * p->hidden_layer.dim_out;
        Uint *tmp2 = (Uint *)malloc(sizeof(Uint) * out_size);
        spmm(tmp, &network->graph.matrix, &network->graph.params, p->hidden_layer.weight, network->graph.matrix.row_size, p->latent_vectors.dim_out);
        mm(tmp2, tmp, p->hidden_layer.weight, p->latent_vectors.dim_in, p->hidden_layer.dim_in, p->hidden_layer.dim_out);
        if (p->next != NULL) {
            relu(&p->next->latent_vectors.weight, tmp2, out_size);
        } else {
            result = (Uint*) malloc(sizeof(Uint) * out_size);
            relu(result, tmp2, out_size);
        }
        p = p->next;
    }

    return result;
}