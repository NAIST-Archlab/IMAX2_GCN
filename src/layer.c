#include "../include/layer.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_weight(HiddenLayer *result) {
    Ull i, j;
    char ry_row = 0;

    printf("[\n");
    for (i = 0; i < result->dim_in; i++) {
        if (i > 100 && (i < result->dim_in - 50) && ry_row != 1) {
            printf("\t.\n\t.\n\t.\n");
            ry_row = 1;
        } else if (i > (result->dim_in - 30) || ry_row == 0) {
            char ry_col = 0;
            printf("\t[ ");
            for (j = 0; j < result->dim_out; j++) {
                if (j > 2 && j < (result->dim_out - 3) && ry_col != 1) {
                    printf("... ");
                    ry_col = 1;
                } else if (j > (result->dim_out - 3) || ry_col == 0) {
                    if (result->weight[i * result->dim_out + j] != 0)
                        printf("%f ", result->weight[i * result->dim_out + j]);
                    else
                        printf("0 ");
                }
            }
            printf("]\n");
        }
    }
    printf("]\n");
}

SparseGraph *spia(SparseGraph *graph) {
    SparseMatrix *sp_matrix = &graph->matrix;
    SparseGraph *result = (SparseGraph *)malloc(sizeof(SparseGraph));
    int nnz = sp_matrix->nnz;
    float *new_val;
    int *new_col_p, *new_row_p;
    int k = 0;
    char is_added = 0;

    for (int i = 0; i < sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        is_added = 0;
        for (int j = col_index_of_index; j < sp_matrix->row_p[i+1]; j++) {
            int col_index = sp_matrix->col_p[j];
            if (col_index == i) {
                is_added = 1;
                break;
            }
        }

        if (!is_added) nnz++;
    }

    new_col_p = (int *)malloc(sizeof(int) * nnz);
    new_row_p = (int *)malloc(sizeof(int) * (sp_matrix->row_size + 1));
    new_val = (float *)malloc(sizeof(float) * nnz);
    memset(new_val, 0, sizeof(float) * nnz);
    result->matrix.nnz = nnz;
    result->matrix.col_p = new_col_p;
    result->matrix.col_size = sp_matrix->col_size;
    result->matrix.row_p = new_row_p;
    result->matrix.row_size = sp_matrix->row_size;
    result->matrix.val = new_val;
    if (nnz != 0)
        new_row_p[0] = 0;

    for (int i = 0; i < sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        int sub = sp_matrix->row_p[i+1] - sp_matrix->row_p[i];
        is_added = 0;
        for (int j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
            int col_index = sp_matrix->col_p[j];
            new_col_p[k++] = col_index;
            if (col_index == i) {
                is_added = 1;
                break;
            }
        }

        if (!is_added) {new_row_p[i+1] = new_row_p[i] + sub + 1;new_col_p[k++]=i;}
        else new_row_p[i+1] = new_row_p[i] + sub;
        is_added = 0;
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
    GCNLayer *layer = (GCNLayer *)malloc(sizeof(GCNLayer));

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

float *make_weight(int dim_in, int dim_out) {
    return (float *)malloc(sizeof(float) * dim_in * dim_out);
}

HiddenLayer *propagation(GCNNetwork *network) {
    GCNLayer *p = network->layers;
    HiddenLayer *result = NULL, *end_vectors = NULL;
    double spmm_time = 0, mm_time = 0, relu_time = 0;
    struct timespec t1, t2;
    #ifdef EMAX6
        IMAXDenseMatrix h, w, tmp_dh, tmp_r;
        Uchar *membase = NULL;
    #endif
    #ifdef USE_CUDA
        float *g_tmp, *g_tmp2;
    #endif

    while (p != NULL) {
        #ifdef EMAX6
            imax_dense_format_init_from_sparse(&h, &network->graph->imax_matrix, p->hidden_layer.dim_out, 8);
            imax_dense_format_init(&tmp_dh, network->graph->imax_matrix.row_size, h.col_size, network->graph->imax_matrix.row_padded_size, h.col_padded_size, network->graph->imax_matrix.row_blk_size, h.col_blk_size);
            imax_spmm_allocation(membase, &network->graph->imax_matrix, &h, &tmp_dh);
        #endif
        int out_size = network->graph->matrix.row_size * p->hidden_layer.dim_out;
        float *tmp = (float *)malloc(sizeof(float) * network->graph->matrix.row_size * p->latent_vectors.dim_out);
        float *tmp2 = (float *)malloc(sizeof(float) * out_size);

        timespec_get(&t1, TIME_UTC);
        #ifdef EMAX6
            convert_imax_dense_format(&h, p->latent_vectors.weight);
            spmm(&tmp_dh, &network->graph->imax_matrix, &h);
        #else
            #ifdef USE_CUDA
                float *gpu_h;
                sendDenseMatrixToGPU(&gpu_h, p->latent_vectors.weight, network->graph->matrix.row_size, p->latent_vectors.dim_out);
                spmm(&g_tmp, &network->graph->matrix, gpu_h, p->latent_vectors.dim_out);
                freeGPUDenseMatrix(gpu_h);
            #else
                spmm(tmp, &network->graph->matrix, p->latent_vectors.weight, p->latent_vectors.dim_out);
            #endif
        #endif
        timespec_get(&t2, TIME_UTC);
        spmm_time += cal_time(&t2, &t1);

        timespec_get(&t1, TIME_UTC);
        #ifdef EMAX6
            convert_imax_dense_format(&w, p->hidden_layer.weight);
            imax_dense_format_init(&tmp_r, h.row_size, w.col_size, h.row_padded_size, w.col_padded_size, h.row_blk_size, w.col_blk_size);
            mm(&tmp_r, &tmp_dh, &w, 1);
        #else
            #ifdef USE_CUDA
                float *gpu_w;
                sendDenseMatrixToGPU(&gpu_w, p->hidden_layer.weight, p->hidden_layer.dim_in, p->hidden_layer.dim_out);
                mm(&g_tmp2, g_tmp, gpu_w, network->graph->matrix.row_size, p->hidden_layer.dim_in, p->hidden_layer.dim_out);
                freeGPUDenseMatrix(gpu_w);
                freeGPUDenseMatrix(g_tmp);
            #else
                mm(tmp2, tmp, p->hidden_layer.weight, network->graph->matrix.row_size, p->hidden_layer.dim_in, p->hidden_layer.dim_out);
            #endif
        #endif
        timespec_get(&t2, TIME_UTC);
        mm_time += cal_time(&t2, &t1);

        if (p->next != NULL) {
            timespec_get(&t1, TIME_UTC);
            #ifdef EMAX6
                convert_dense_format(p->next->latent_vectors.weight, &tmp_r);
            #else
                #ifdef USE_CUDA
                    float *gpu_r;
                    relu(&gpu_r, g_tmp2, out_size);
                    freeGPUDenseMatrix(g_tmp2);
                    sendDenseMatrixToCPU(p->next->latent_vectors.weight, gpu_r, network->graph->matrix.row_size, p->hidden_layer.dim_out);
                #else
                    relu(p->next->latent_vectors.weight, tmp2, out_size);
                #endif
            #endif
            timespec_get(&t2, TIME_UTC);
            relu_time += cal_time(&t2, &t1);
        } else {
            end_vectors = (HiddenLayer *)malloc(sizeof(HiddenLayer));
            end_vectors->dim_in = network->graph->matrix.row_size;
            end_vectors->dim_out = p->hidden_layer.dim_out;
            end_vectors->weight = make_weight(network->graph->matrix.row_size, p->hidden_layer.dim_out);
            timespec_get(&t1, TIME_UTC);
            #ifdef EMAX6
                convert_dense_format(end_vectors->weight, &tmp_dh);
            #else
                #ifdef USE_CUDA
                    float *gpu_r;
                    relu(&gpu_r, g_tmp2, out_size);
                    freeGPUDenseMatrix(g_tmp2);
                    sendDenseMatrixToCPU(end_vectors->weight, gpu_r, network->graph->matrix.row_size, p->hidden_layer.dim_out);
                #else
                    relu(end_vectors->weight, tmp2, out_size);
                #endif
            #endif
            timespec_get(&t2, TIME_UTC);
            relu_time += cal_time(&t2, &t1);
        }
        free(tmp);
        free(tmp2);
        p = p->next;
    }

    timespec_get(&t1, TIME_UTC);
    result = softmax(end_vectors);
    timespec_get(&t2, TIME_UTC);
    double softmax_time = cal_time(&t2, &t1);

    printf("SpMM: %lf, MM: %lf, ReLu: %lf, Softmax: %lf usec.\n", spmm_time, mm_time, relu_time, softmax_time);
    return result;
}

HiddenLayer *softmax(HiddenLayer *end_vectors) {
    HiddenLayer *result;
    result = (HiddenLayer *)malloc(sizeof(HiddenLayer));
    result->weight = (float *)malloc(sizeof(float) * end_vectors->dim_in * end_vectors->dim_out);
    result->dim_in = end_vectors->dim_in;
    result->dim_out = end_vectors->dim_out;

    #ifdef USE_MP
    #pragma omp parallel
    #endif
    for (int i = 0; i < end_vectors->dim_in; i++) {
        float log_max = log(max_in_array(&end_vectors->weight[i], end_vectors->dim_out));
        float sum = 0;
        for (int j = 0; j < end_vectors->dim_out; j++) {
            sum += exp(end_vectors->weight[i * end_vectors->dim_out + j] + log_max);
        }
        for (int j = 0; j < end_vectors->dim_out; j++) {
            result->weight[i * end_vectors->dim_out + j] = exp(end_vectors->weight[i * end_vectors->dim_out + j] + log_max) / sum;
        }
    }

    return result;
}

float max_in_array(float *array, int size) {
    int i;
    float max = -1;

    for (i = 0; i < size; i++) {
        if (max < array[i])
            max = array[i];
    }

    return max;
}