#include "../include/layer.h"
#include <stdlib.h>
#include <stdio.h>

void print_layers(GCNNetwork *network) {
    GCNLayer *p = network->layers;

    while (p != NULL) {
        printf("Vectors: %d * %d\n", p->latent_vectors.dim_in, p->latent_vectors.dim_out);
        printf("Weight: %d * %d\n", p->hidden_layer.dim_in, p->hidden_layer.dim_out);
        p = p->next;
    }
}

void add_gcn_layer(GCNNetwork *network, float *weight, float *vectors, int dim_in, int dim_out) {
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
    layer->latent_vectors.dim_in = network->graph->matrix.col_size;
    layer->latent_vectors.dim_out = dim_in;
    layer->latent_vectors.weight = vectors;
}

float* make_weight(int dim_in, int dim_out) {
    return (float*) malloc(sizeof(float)*dim_in*dim_out);
}

HiddenLayer* propagation(GCNNetwork *network) {
    GCNLayer *p = network->layers;
    HiddenLayer *result = NULL;

    while (p != NULL) {
        Uint *tmp = (Uint *)malloc(sizeof(Uint) * network->graph->matrix.row_size * p->latent_vectors.dim_out);
        int out_size = network->graph->matrix.row_size * p->hidden_layer.dim_out;
        Uint *tmp2 = (Uint *)malloc(sizeof(Uint) * out_size);
        spmm(tmp, &network->graph->matrix, &network->graph->params, (Uint*)p->latent_vectors.weight, p->latent_vectors.dim_out);
        mm(tmp2, tmp, (Uint*)p->hidden_layer.weight, network->graph->matrix.row_size, p->hidden_layer.dim_in, p->hidden_layer.dim_out);
        if (p->next != NULL) {
            relu((Uint*)p->next->latent_vectors.weight, tmp2, out_size);
        } else {
            result = (HiddenLayer*) malloc(sizeof(HiddenLayer));
            result->dim_in = network->graph->matrix.row_size;
            result->dim_out = p->hidden_layer.dim_out;
            result->weight = make_weight(network->graph->matrix.row_size, p->hidden_layer.dim_out);
            relu((Uint*)result->weight, tmp2, out_size);
        }
        free(tmp);
        free(tmp2);
        p = p->next;
    }

    return result;
}