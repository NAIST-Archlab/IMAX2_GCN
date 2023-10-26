// EMAX6/7 GCN Test Program            //
// layer.c                             //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/layer.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_weight(HiddenLayer *result) {
    Ull i, j;
    char ry_row = 0;

    printf("Size: (%d,%d)\n", result->row_size, result->col_size);
    printf("[\n");
    for (i = 0; i < result->row_size; i++) {
        if (i > 100 && (i < result->row_size - 30) && ry_row != 1) {
            printf("\t.\n\t.\n\t.\n");
            ry_row = 1;
        } else if (i > (result->row_size - 30) || ry_row == 0) {
            char ry_col = 0;
            printf("\t[ ");
            for (j = 0; j < result->col_size; j++) {
                if (j > 2 && j < (result->col_size - 3) && ry_col != 1) {
                    printf("... ");
                    ry_col = 1;
                } else if (j > (result->col_size - 3) || ry_col == 0) {
                    printf("%10.6f ", result->val[i * result->col_size + j]);
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

    result->matrix.nnz = nnz;
    result->matrix.col_size = sp_matrix->col_size;
    result->matrix.row_size = sp_matrix->row_size;
    allocSparseMatrix(&(result->matrix));

    if (nnz != 0)
        result->matrix.row_p[0] = 0;

    for (int i = 0; i < sp_matrix->row_size; i++) {
        int col_index_of_index = sp_matrix->row_p[i];
        int sub = sp_matrix->row_p[i+1] - sp_matrix->row_p[i];
        is_added = 0;
        for (int j = col_index_of_index; j < sp_matrix->row_p[i + 1]; j++) {
            int col_index = sp_matrix->col_p[j];
            result->matrix.col_p[k++] = col_index;
            if (col_index == i) {
                is_added = 1;
                break;
            }
        }

        if (!is_added) {result->matrix.row_p[i+1] = result->matrix.row_p[i] + sub + 1;result->matrix.col_p[k++]=i;}
        else result->matrix.row_p[i+1] = result->matrix.row_p[i] + sub;
        is_added = 0;
    }

    return result;
}

void print_layers(GCNNetwork *network) {
    GCNLayer *p = network->layers;

    while (p != NULL) {
        printf("Vectors: %d * %d\n", p->latent_vectors.row_size, p->latent_vectors.col_size);
        printf("Weight: %d * %d\n", p->hidden_layer.row_size, p->hidden_layer.col_size);
        p = p->next;
    }
}

void add_gcn_layer(GCNNetwork *network, DenseMatrix weight, DenseMatrix vectors) {
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
    layer->hidden_layer.row_size = weight.row_size;
    layer->hidden_layer.col_size = weight.col_size;
    layer->hidden_layer.val = weight.val;
    layer->hidden_layer.cuda_val = weight.cuda_val;
    layer->latent_vectors.row_size = vectors.row_size;
    layer->latent_vectors.col_size = vectors.col_size;
    layer->latent_vectors.val = vectors.val;
    layer->latent_vectors.cuda_val = vectors.cuda_val;
}

void propagation(GCNNetwork *network) {
    GCNLayer *p = network->layers;
    DenseMatrix r_spmm, r_mm, *last_weight, tmp_h, tmp_w;
    double spmm_time = 0, mm_time = 0, relu_time = 0;
    int out_size;
    struct timespec t1, t2;
    #if defined(EMAX6) || defined(EMAX7)
        IMAXDenseMatrix h, w, imax_r_spmm, imax_r_mm, imax_dump;
    #endif

    printf("Propagation...\n");

    #ifdef USE_CUDA
        timespec_get(&t1, TIME_UTC);
        createCusparse();
        timespec_get(&t2, TIME_UTC);
        spmm_time += cal_time(&t2, &t1);
        timespec_get(&t1, TIME_UTC);
        createCublas();
        timespec_get(&t2, TIME_UTC);
        mm_time += cal_time(&t2, &t1);
    #endif
    int layer_cnt = 0;
    while (p != NULL) {
        out_size = network->graph->matrix.row_size * p->hidden_layer.col_size;
        r_spmm.row_size = network->graph->matrix.row_size;
        r_spmm.col_size = p->latent_vectors.col_size;
        r_mm.row_size = p->latent_vectors.row_size;
        r_mm.col_size = p->hidden_layer.col_size;
        allocDenseMatrix(&r_spmm); allocDenseMatrix(&r_mm);
        
        #if defined(EMAX6) || defined(EMAX7)
            imax_dense_format_init_from_sparse(&h, &network->graph->imax_matrix, p->latent_vectors.col_size, 8);
            imax_dense_format_init(&imax_r_spmm, h.row_size, h.col_size, h.row_padded_size, h.col_padded_size, h.blk_row_size, h.blk_col_size);
            imax_dense_format_init(&w, imax_r_spmm.col_size, p->hidden_layer.col_size, imax_r_spmm.col_padded_size, p->hidden_layer.col_size + imax_r_spmm.blk_col_size - (p->hidden_layer.col_size%imax_r_spmm.blk_col_size), MM_H, imax_r_spmm.blk_col_size);
            imax_dense_format_init(&imax_r_mm, imax_r_spmm.row_size, w.col_size, imax_r_spmm.row_padded_size, w.col_padded_size, imax_r_spmm.blk_row_size, w.blk_col_size);
            imax_gcn_allocation(&network->graph->imax_matrix, &h, &imax_r_spmm, &w, &imax_r_mm);
            convert_imax_dense_format(&h, &(p->latent_vectors));
            convert_imax_dense_format(&w, &(p->hidden_layer));
        #endif

        printf("Layer %d: SpMM\n", ++layer_cnt);
        HiddenLayer t;
        timespec_get(&t1, TIME_UTC);
        #if defined(EMAX6) || defined(EMAX7)
            spmm(&imax_r_spmm, &(network->graph->imax_matrix), &h);
        #else
            spmm(&r_spmm, &(network->graph->matrix), &(p->latent_vectors));
        #endif
        timespec_get(&t2, TIME_UTC);
        spmm_time += cal_time(&t2, &t1);

        #if defined(EMAX6) || defined(EMAX7)
            convert_dense_format(&r_spmm, &imax_r_spmm);
        #elif defined(USE_CUDA)
            sendDenseMatrixToCPU(&r_spmm);
        #endif
        print_weight(&r_spmm);

        printf("Layer %d: MM\n", layer_cnt);
        timespec_get(&t1, TIME_UTC);
        #if defined(EMAX6) || defined(EMAX7)
            mm(&imax_r_mm, &imax_r_spmm, &w);
        #else
            mm(&r_mm, &r_spmm, &(p->hidden_layer));
        #endif
        timespec_get(&t2, TIME_UTC);
        mm_time += cal_time(&t2, &t1);

        #if defined(EMAX6) || defined(EMAX7)
            convert_dense_format(&r_mm, &imax_r_mm);
        #elif defined(USE_CUDA)
            sendDenseMatrixToCPU(&r_mm);
        #endif
        print_weight(&r_mm);


        if (p->next == NULL) last_weight = &(network->layers->result_layer);
        else last_weight = &(p->next->latent_vectors);

        printf("Layer %d: ReLU\n", layer_cnt);
        timespec_get(&t1, TIME_UTC);
        relu(last_weight, &r_mm);
        timespec_get(&t2, TIME_UTC);
        #ifdef USE_CUDA
            if (p->next == NULL) sendDenseMatrixToCPU(last_weight);
        #endif
        relu_time += cal_time(&t2, &t1);

        freeDenseMatrix(&r_spmm);freeDenseMatrix(&r_mm);
        p = p->next;
    }

    printf("Softmax\n");
    timespec_get(&t1, TIME_UTC);
    softmax(&(network->layers->result_layer));
    timespec_get(&t2, TIME_UTC);
    double softmax_time = cal_time(&t2, &t1);
    #ifdef USE_CUDA
        destroyCusparse();
        destroyCublas();
    #endif

    printf("SpMM: %lf, MM: %lf, ReLu: %lf, Softmax: %lf usec.\n", spmm_time, mm_time, relu_time, softmax_time);
    printf("Propagation: %lf usec.\n", spmm_time + mm_time + relu_time + softmax_time);
    #if !(defined(EMAX6) || defined(EMAX7))
        all_nanosec[SPMM] += (Ull) spmm_time*1000;
        all_nanosec[MM] += (Ull) mm_time*1000;
        all_nanosec[RELU] += (Ull) relu_time*1000;
        all_nanosec[SOFTMAX] += (Ull) softmax_time*1000;
    #else
        all_nanosec[RELU][0] += (Ull) relu_time*1000;
        all_nanosec[RELU][7] += (Ull) relu_time*1000;
        all_nanosec[SOFTMAX][0] += (Ull) softmax_time*1000;
        all_nanosec[SOFTMAX][7] += (Ull) softmax_time*1000;
    #endif
}

void softmax(HiddenLayer *result) {
    for (int i = 0; i < result->row_size; i++) {
        float max = max_in_array(&(result->val[i * result->col_size]), result->col_size);
        float log_max = log(max);
        float sum = 0;

        if (max <= 1) log_max = 0;
        for (int j = 0; j < result->col_size; j++) {
            sum += exp(result->val[i * result->col_size + j] + log_max);
        }
        for (int j = 0; j < result->col_size; j++) {
            result->val[i * result->col_size + j] = exp(result->val[i * result->col_size + j] + log_max) / sum;
        }
    }
}

float max_in_array(float *array, int size) {
    int i;
    float max = -INFINITY;

    for (i = 0; i < size; i++) {
        if (max < array[i])
            max = array[i];
    }

    return max;
}
