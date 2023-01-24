#include "../include/layer.h"
#include <stdlib.h>
#include <stdio.h>

SparseGraph* spia(SparseGraph *graph) {
    SparseMatrix *sp_matrix = &graph->matrix;
    //SparseMatrixParams *sp_params = &graph->params;
    SparseGraph *result = (SparseGraph*) malloc(sizeof(SparseGraph));
    int nnz = sp_matrix->nnz;
    float *new_val;
    int *new_col_p, *new_row_p;
    int i, j, k = 0;
    char is_added = 0;

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (i = 0; i < sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        for (j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
            int col_index = sp_matrix->col_p[j];
            if (col_index < i && (j+1) >= sp_matrix->row_p[i+1]) {
                nnz++;
                break;
            }
            int col_next_index = sp_matrix->col_p[j+1];
            if (col_index < i && col_next_index > i) {
                nnz++;
                break;
            }  else if (col_index == i) {
                break;
            }
        }
    }

    new_col_p = (int*) malloc(sizeof(int) * nnz);
    new_row_p = (int*) malloc(sizeof(int) * (sp_matrix->row_size + 1));
    new_val = (float*) malloc(sizeof(float) * nnz);
    result->matrix.nnz = nnz;
    result->matrix.col_p = new_col_p;
    result->matrix.col_size = nnz;
    result->matrix.row_p = new_row_p;
    result->matrix.row_size = sp_matrix->row_size;
    result->matrix.val = new_val;
    if (nnz != 0) new_row_p[0] = 0;

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (i = 0; i <= sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        int sub = sp_matrix->row_p[i] - sp_matrix->row_p[i-1];
        if (i > 0) {
            if (is_added)
                new_row_p[i] = new_row_p[i-1] + sub + 1;
            else 
                new_row_p[i] = new_row_p[i-1] + sub;
            is_added = 0;
        }
       for (j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
            int col_index = sp_matrix->col_p[j];
            if (col_index < i && (j+1) >= sp_matrix->row_p[i+1]) {
                new_col_p[k] = col_index;
                new_val[k] = sp_matrix->val[j];
                new_col_p[++k] = i;
                new_val[k] = 1;
                is_added = 1;
            }

            if ((j+1) >= sp_matrix->col_size) break;

            int col_next_index = sp_matrix->col_p[j+1]; 
            if (col_index < i && col_next_index > i) {
                new_col_p[k] = col_index;
                new_val[k] = sp_matrix->val[j];
                new_col_p[++k] = i;
                new_val[k] = 1;
                is_added = 1;
            }  else if (col_index == i) {
                new_col_p[k] = col_index;
                new_val[k] = sp_matrix->val[j] + 1;
            } else {
                new_col_p[k] = col_index;
                new_val[k] = sp_matrix->val[j];
            }
            k++;
        }
    }

    return result;
}


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
        float *tmp = (float *)malloc(sizeof(float) * network->graph->matrix.row_size * p->latent_vectors.dim_out);
        int out_size = network->graph->matrix.row_size * p->hidden_layer.dim_out;
        float *tmp2 = (float *)malloc(sizeof(float) * out_size);
        spmm(tmp, &network->graph->matrix, &network->graph->params, p->latent_vectors.weight, p->latent_vectors.dim_out);
        mm(tmp2, tmp, p->hidden_layer.weight, network->graph->matrix.row_size, p->hidden_layer.dim_in, p->hidden_layer.dim_out);
        if (p->next != NULL) {
            relu(p->next->latent_vectors.weight, tmp2, out_size);
        } else {
            result = (HiddenLayer*) malloc(sizeof(HiddenLayer));
            result->dim_in = network->graph->matrix.row_size;
            result->dim_out = p->hidden_layer.dim_out;
            result->weight = make_weight(network->graph->matrix.row_size, p->hidden_layer.dim_out);
            relu(result->weight, tmp2, out_size);
        }
        free(tmp);
        free(tmp2);
        p = p->next;
    }

    return result;
}