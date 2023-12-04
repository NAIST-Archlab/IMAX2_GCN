// EMAX6/7 GCN Test Program            //
// layer.c                             //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/gcn.h"
#include "../include/layer.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_gcn_layers(GCNNetwork *network) {
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

void gcn_propagation(GCNNetwork *network) {
    GCNLayer *p = network->layers;
    DenseMatrix r_spmm, r_mm, *last_weight, tmp_h, tmp_w;
    double spmm_time = 0, mm_time = 0, relu_time = 0;
    int out_size;
    struct timespec t1, t2;
    #if defined(EMAX6) || defined(EMAX7)
        IMAXDenseMatrix h, w, imax_r_spmm, imax_r_mm, imax_dump;
    #elif defined(USE_CUDA)
        double cusparse_load = 0;
        double cublas_load = 0;
    #endif

    printf("Propagation...\n");

    #ifdef USE_CUDA
        timespec_get(&t1, TIME_UTC);
        createCusparse();
        timespec_get(&t2, TIME_UTC);
        all_nanosec[SPMM][1] += (Ull) cal_time(&t2, &t1)*1000;
        spmm_time += cal_time(&t2, &t1);
        cusparse_load = cal_time(&t2, &t1);
        timespec_get(&t1, TIME_UTC);
        createCublas();
        timespec_get(&t2, TIME_UTC);
        all_nanosec[MM][1] += (Ull) cal_time(&t2, &t1)*1000;
        mm_time += cal_time(&t2, &t1);
        cublas_load = cal_time(&t2, &t1);
    #endif
    int layer_cnt = 0;
    while (p != NULL) {
        if (p->next == NULL) last_weight = &(network->layers->result_layer);
        else last_weight = &(p->next->latent_vectors);
        if (p->latent_vectors.col_size > p->hidden_layer.col_size) {
            out_size = network->graph->matrix.row_size * p->hidden_layer.col_size;
            r_spmm.row_size = network->graph->matrix.row_size;
            r_spmm.col_size = p->hidden_layer.col_size;
            r_mm.row_size = p->latent_vectors.row_size;
            r_mm.col_size = p->hidden_layer.col_size;
            allocDenseMatrix(&r_spmm); allocDenseMatrix(&r_mm);
            
            #if defined(EMAX6) || defined(EMAX7)
                imax_dense_format_init_from_sparse(&h, &network->graph->imax_matrix, p->latent_vectors.col_size, 8);
                imax_dense_format_init(&w, h.col_size, p->hidden_layer.col_size, h.col_padded_size, p->hidden_layer.col_size + h.blk_col_size - (p->hidden_layer.col_size%h.blk_col_size), MM_H, h.blk_col_size);
                imax_dense_format_init(&imax_r_mm, h.row_size, w.col_size, h.row_padded_size, w.col_padded_size, h.blk_row_size, w.blk_col_size);
                imax_dense_format_init(&imax_r_spmm, imax_r_mm.row_size, imax_r_mm.col_size, imax_r_mm.row_padded_size, imax_r_mm.col_padded_size, imax_r_mm.blk_row_size, imax_r_mm.blk_col_size);
                imax_gcn_allocation(&network->graph->imax_matrix, &h, &imax_r_spmm, &w, &imax_r_mm);
                convert_imax_dense_format(&h, &(p->latent_vectors));
                convert_imax_dense_format(&w, &(p->hidden_layer));
            #endif

            HiddenLayer t;
            printf("Layer %d: MM\n", ++layer_cnt);
            timespec_get(&t1, TIME_UTC);
            #if defined(EMAX6) || defined(EMAX7)
                mm(&imax_r_mm, &h, &w);
            #else
                mm(&r_mm, &(p->latent_vectors), &(p->hidden_layer));
            #endif
            timespec_get(&t2, TIME_UTC);
            mm_time += cal_time(&t2, &t1);

            #if defined(EMAX6) || defined(EMAX7)
                convert_dense_format(&r_mm, &imax_r_mm);
            #elif defined(USE_CUDA)
                sendDenseMatrixToCPU(&r_mm);
            #endif
            print_weight(&r_mm);

            printf("Layer %d: SpMM\n", layer_cnt);
            timespec_get(&t1, TIME_UTC);
            #if defined(EMAX6) || defined(EMAX7)
                spmm(&imax_r_spmm, &(network->graph->imax_matrix), &imax_r_mm);
            #else
                spmm(&r_spmm, &(network->graph->matrix), &r_mm);
            #endif
            timespec_get(&t2, TIME_UTC);
            spmm_time += cal_time(&t2, &t1);

            #if defined(EMAX6) || defined(EMAX7)
                convert_dense_format(&r_spmm, &imax_r_spmm);
            #elif defined(USE_CUDA)
                sendDenseMatrixToCPU(&r_spmm);
            #endif
            print_weight(&r_spmm);

            printf("Layer %d: ReLU\n", layer_cnt);
            timespec_get(&t1, TIME_UTC);
            relu(last_weight, &r_spmm);
            timespec_get(&t2, TIME_UTC);
            #ifdef USE_CUDA
                if (p->next == NULL) sendDenseMatrixToCPU(last_weight);
            #endif
            relu_time += cal_time(&t2, &t1);
        } else {
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

            printf("Layer %d: ReLU\n", layer_cnt);
            timespec_get(&t1, TIME_UTC);
            relu(last_weight, &r_mm);
            timespec_get(&t2, TIME_UTC);
            #ifdef USE_CUDA
                if (p->next == NULL) sendDenseMatrixToCPU(last_weight);
            #endif
            relu_time += cal_time(&t2, &t1);
        }

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
    #if !(defined(EMAX6) || defined(EMAX7) || defined(USE_CUDA))
        all_nanosec[SPMM] += (Ull) spmm_time*1000;
        all_nanosec[MM] += (Ull) mm_time*1000;
        all_nanosec[RELU] += (Ull) relu_time*1000;
        all_nanosec[SOFTMAX] += (Ull) softmax_time*1000;
    #elif defined(USE_CUDA)
        all_nanosec[SPMM][2] += (cusparse_load + spmm_time)*1000;
        all_nanosec[MM][2] += (cublas_load + mm_time)*1000;
        all_nanosec[RELU][0] += (Ull) relu_time*1000;
        all_nanosec[SOFTMAX][0] += (Ull) softmax_time*1000;
        all_nanosec[RELU][2] += (Ull) relu_time*1000;
        all_nanosec[SOFTMAX][2] += (Ull) softmax_time*1000;
    #else
        all_nanosec[RELU][0] += (Ull) relu_time*1000;
        all_nanosec[RELU][7] += (Ull) relu_time*1000;
        all_nanosec[SOFTMAX][0] += (Ull) softmax_time*1000;
        all_nanosec[SOFTMAX][7] += (Ull) softmax_time*1000;
    #endif
}