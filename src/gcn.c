// EMAX6/7 GCN Test Program            //
// gcn.c                               //
//         Copyright (C) 2024 by NAIST //
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
    IMAXDenseMatrix imax_h, imax_w, imax_r_spmm, imax_r_mm, imax_dump;
    imax_sparse_allocation(&network->graph->imax_matrix);
#elif defined(USE_CUDA)
    double cusparse_load = 0;
    double cublas_load = 0;
#endif

    printf("Propagation...\n");

#ifdef USE_CUDA
    timespec_get(&t1, TIME_UTC);
    createCusparse();
    timespec_get(&t2, TIME_UTC);
    all_nanosec[SPMM][1] += (Ull)cal_time(&t2, &t1) * 1000;
    spmm_time += cal_time(&t2, &t1);
    cusparse_load = cal_time(&t2, &t1);
    timespec_get(&t1, TIME_UTC);
    createCublas();
    timespec_get(&t2, TIME_UTC);
    all_nanosec[MM][1] += (Ull)cal_time(&t2, &t1) * 1000;
    mm_time += cal_time(&t2, &t1);
    cublas_load = cal_time(&t2, &t1);
#endif
    int layer_cnt = 0;
    while (p != NULL) {
        if (p->next == NULL)
            last_weight = &(network->layers->result_layer);
        else
            last_weight = &(p->next->latent_vectors);
#if defined(EMAX6) || defined(EMAX7)
        imax_h.row_size = p->latent_vectors.row_size;
        imax_h.col_size = p->latent_vectors.col_size;
        imax_w.row_size = p->hidden_layer.row_size;
        imax_w.col_size = p->hidden_layer.col_size;
#endif
        if (p->latent_vectors.col_size > p->hidden_layer.col_size) {
            out_size = network->graph->matrix.row_size * p->hidden_layer.col_size;
            r_spmm.row_size = network->graph->matrix.row_size;
            r_spmm.col_size = p->hidden_layer.col_size;
            r_mm.row_size = p->latent_vectors.row_size;
            r_mm.col_size = p->hidden_layer.col_size;
            allocDenseMatrix(&r_spmm);
            allocDenseMatrix(&r_mm);

#if defined(EMAX6) || defined(EMAX7)
            imax_h.blk_row_size = network->graph->imax_matrix.blk_row_size;
            imax_matrix_init_mm(&imax_r_mm, &imax_h, &imax_w, FIT_TO_DENSE);
            imax_matrix_init_spmm(&imax_r_spmm, &(network->graph->imax_matrix), &imax_r_mm, FIT_TO_DENSE);
            imax_dense_allocation(&imax_h);imax_dense_allocation(&imax_w);
            imax_dense_allocation(&imax_r_mm);imax_dense_allocation(&imax_r_spmm);
            convert_imax_dense_format(&imax_h, &(p->latent_vectors));
            convert_imax_dense_format(&imax_w, &(p->hidden_layer));
#endif

            HiddenLayer t;
            printf("Layer %d: MM\n", ++layer_cnt);
            timespec_get(&t1, TIME_UTC);
#if defined(EMAX6) || defined(EMAX7)
            mm(&imax_r_mm, &imax_h, &imax_w);
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
            if (p->next == NULL)
                sendDenseMatrixToCPU(last_weight);
#endif
            relu_time += cal_time(&t2, &t1);
        } else {
            out_size = network->graph->matrix.row_size * p->hidden_layer.col_size;
            r_spmm.row_size = network->graph->matrix.row_size;
            r_spmm.col_size = p->latent_vectors.col_size;
            r_mm.row_size = p->latent_vectors.row_size;
            r_mm.col_size = p->hidden_layer.col_size;
            allocDenseMatrix(&r_spmm);
            allocDenseMatrix(&r_mm);

#if defined(EMAX6) || defined(EMAX7)
            imax_matrix_init_spmm(&imax_r_spmm, &(network->graph->imax_matrix), &imax_h, FIT_TO_SPARSE);
            imax_matrix_init_mm(&imax_r_mm, &imax_r_spmm, &imax_w, FIT_TO_SPARSE);
            imax_dense_allocation(&imax_h);imax_dense_allocation(&imax_w);
            imax_dense_allocation(&imax_r_mm);imax_dense_allocation(&imax_r_spmm);
            convert_imax_dense_format(&imax_h, &(p->latent_vectors));
            convert_imax_dense_format(&imax_w, &(p->hidden_layer));
#endif

            printf("Layer %d: SpMM\n", ++layer_cnt);
            HiddenLayer t;
            timespec_get(&t1, TIME_UTC);
#if defined(EMAX6) || defined(EMAX7)
            spmm(&imax_r_spmm, &(network->graph->imax_matrix), &imax_h);
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
            mm(&imax_r_mm, &imax_r_spmm, &imax_w);
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
            if (p->next == NULL)
                sendDenseMatrixToCPU(last_weight);
#endif
            relu_time += cal_time(&t2, &t1);
        }
#if defined(EMAX6) || defined(EMAX7)
        imax_dense_deallocation(&imax_h);
        imax_dense_deallocation(&imax_w);
        imax_dense_deallocation(&imax_r_mm);
        imax_dense_deallocation(&imax_r_spmm);
#endif
        freeDenseMatrix(&r_spmm);
        freeDenseMatrix(&r_mm);
        p = p->next;
    }
#if defined(EMAX6) || defined(EMAX7)
    imax_sparse_deallocation(&network->graph->imax_matrix);
#endif

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
    all_nanosec[SPMM] += (Ull)spmm_time * 1000;
    all_nanosec[MM] += (Ull)mm_time * 1000;
    all_nanosec[RELU] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX] += (Ull)softmax_time * 1000;
#elif defined(USE_CUDA)
    all_nanosec[SPMM][2] += (cusparse_load + spmm_time) * 1000;
    all_nanosec[MM][2] += (cublas_load + mm_time) * 1000;
    all_nanosec[RELU][0] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX][0] += (Ull)softmax_time * 1000;
    all_nanosec[RELU][2] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX][2] += (Ull)softmax_time * 1000;
#else
    all_nanosec[RELU][0] += (Ull)relu_time * 1000;
    all_nanosec[RELU][7] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX][0] += (Ull)softmax_time * 1000;
    all_nanosec[SOFTMAX][7] += (Ull)softmax_time * 1000;
#endif
}

void gcn_backpropagation(GCNNetwork *network, DenseMatrix *labels, void (*optimizer)(DenseMatrix *value, DenseMatrix *grad, OptimizerOption *option), OptimizerOption *option) {
    GCNLayer *p = network->layers;
    DenseMatrix r_spmm, r_mm, *last_weight, tmp_h, tmp_w, w_error, h_error, h_next_error, h_transpose, w_transpose;
    double spmm_time = 0, mm_time = 0, relu_time = 0;
    int out_size;
    struct timespec t1, t2;
#if defined(EMAX6) || defined(EMAX7)
    IMAXDenseMatrix imax_r_spmm, imax_r_mm, imax_dump, imax_w_error, imax_h_error, imax_h_next_error, imax_h_transpose, imax_w_transpose;
    imax_sparse_allocation(&network->graph->imax_matrix);
#elif defined(USE_CUDA)
    double cusparse_load = 0;
    double cublas_load = 0;
#endif

    printf("Backpropagation...\n");

#ifdef USE_CUDA
    timespec_get(&t1, TIME_UTC);
    createCusparse();
    timespec_get(&t2, TIME_UTC);
    all_nanosec[SPMM][1] += (Ull)cal_time(&t2, &t1) * 1000;
    spmm_time += cal_time(&t2, &t1);
    cusparse_load = cal_time(&t2, &t1);
    timespec_get(&t1, TIME_UTC);
    createCublas();
    timespec_get(&t2, TIME_UTC);
    all_nanosec[MM][1] += (Ull)cal_time(&t2, &t1) * 1000;
    mm_time += cal_time(&t2, &t1);
    cublas_load = cal_time(&t2, &t1);
#endif
    int layer_cnt = 1;
    while (p->next != NULL) {
        p = p->next;
        layer_cnt++;
    }
    h_error.row_size = network->layers->result_layer.row_size;
    h_error.col_size = network->layers->result_layer.col_size;
    allocDenseMatrix(&h_error);
    msub(&h_error, labels, &(network->layers->result_layer));
#ifdef USE_CUDA
    sendDenseMatrixToCPU(&h_error);
#endif
    printf("Learning Error :%f\n", mmeans(&h_error));
    printf("dSoftmax...\n");
    timespec_get(&t1, TIME_UTC);
    d_softmax(&h_error);
    timespec_get(&t2, TIME_UTC);
    double softmax_time = cal_time(&t2, &t1);
    double optimizer_time = 0;
    while (p != NULL) {
        w_error.row_size = p->latent_vectors.row_size;
        w_error.col_size = p->hidden_layer.col_size;
        allocDenseMatrix(&w_error);
        printf("Layer %d: dReLU\n", layer_cnt);
        timespec_get(&t1, TIME_UTC);
        d_relu(&w_error, &h_error);
        timespec_get(&t2, TIME_UTC);
        relu_time += cal_time(&t2, &t1);
#ifdef USE_CUDA
        if (p->next == NULL)
            sendDenseMatrixToCPU(&w_error);
#endif
        memcpy(h_error.val, w_error.val, sizeof(w_error.row_size * w_error.col_size * sizeof(float)));
        freeDenseMatrix(&w_error);
        w_error.row_size = p->hidden_layer.row_size;
        w_error.col_size = p->hidden_layer.col_size;
        allocDenseMatrix(&w_error);

        out_size = network->graph->matrix.row_size * p->hidden_layer.col_size;
        r_spmm.row_size = network->graph->matrix.row_size;
        r_spmm.col_size = p->hidden_layer.col_size;
        h_next_error.row_size = p->latent_vectors.row_size;
        h_next_error.col_size = p->latent_vectors.col_size;
        allocDenseMatrix(&r_spmm);
        allocDenseMatrix(&h_next_error);

        h_transpose.row_size = p->latent_vectors.col_size;
        h_transpose.col_size = p->latent_vectors.row_size;
        allocDenseMatrix(&h_transpose);
        transpose(&h_transpose, &(p->latent_vectors));
        w_transpose.row_size = p->hidden_layer.col_size;
        w_transpose.col_size = p->hidden_layer.row_size;
        allocDenseMatrix(&w_transpose);
        transpose(&w_transpose, &(p->hidden_layer));

#if defined(EMAX6) || defined(EMAX7)
        imax_h_error.row_size = h_error.row_size;
        imax_h_error.col_size = h_error.col_size;
        imax_h_transpose.row_size = h_transpose.row_size;
        imax_h_transpose.col_size = h_transpose.col_size;
        imax_h_transpose.blk_row_size = network->graph->imax_matrix.blk_row_size;
        imax_w_transpose.row_size = w_transpose.row_size;
        imax_w_transpose.col_size = w_transpose.col_size;
        imax_matrix_init_spmm(&imax_r_spmm, &(network->graph->imax_matrix), &imax_h_error, FIT_TO_SPARSE);
        imax_matrix_init_mm(&imax_w_error, &imax_h_transpose, &imax_r_spmm, FIT_TO_DENSE);
        imax_matrix_init_mm(&imax_h_next_error, &imax_r_spmm, &imax_w_transpose, FIT_TO_SPARSE);
        imax_dense_allocation(&imax_h_transpose);imax_dense_allocation(&imax_r_spmm);
        imax_dense_allocation(&imax_w_transpose);imax_dense_allocation(&imax_h_error);
        imax_dense_allocation(&imax_h_next_error);imax_dense_allocation(&imax_w_error);
        convert_imax_dense_format(&imax_h_transpose, &h_transpose);convert_imax_dense_format(&imax_w_transpose, &w_transpose);
        convert_imax_dense_format(&imax_h_error, &h_error);convert_imax_dense_format(&imax_h_next_error, &h_next_error);
        convert_imax_dense_format(&imax_w_error, &w_error);
#endif

        printf("Layer %d: dSpMM\n", layer_cnt);
        HiddenLayer t;
        timespec_get(&t1, TIME_UTC);
#if defined(EMAX6) || defined(EMAX7)
        spmm(&imax_r_spmm, &(network->graph->imax_matrix), &imax_h_error);
#else
        spmm(&r_spmm, &(network->graph->matrix), &h_error);
#endif
        timespec_get(&t2, TIME_UTC);
        spmm_time += cal_time(&t2, &t1);

        printf("Layer %d: dMM\n", layer_cnt);
        timespec_get(&t1, TIME_UTC);
#if defined(EMAX6) || defined(EMAX7)
        mm(&imax_w_error, &imax_h_transpose, &imax_r_spmm);
        mm(&imax_h_next_error, &imax_r_spmm, &imax_w_transpose);
#else
        mm(&w_error, &h_transpose, &r_spmm);
        mm(&h_next_error, &r_spmm, &w_transpose);
#endif
        timespec_get(&t2, TIME_UTC);
        mm_time += cal_time(&t2, &t1);

#if defined(EMAX6) || defined(EMAX7)
        convert_dense_format(&w_error, &imax_w_error);
        convert_dense_format(&h_next_error, &imax_h_error);
#elif defined(USE_CUDA)
        sendDenseMatrixToCPU(&w_error);
        sendDenseMatrixToCPU(&h_next_error);
#endif
        printf("Layer %d: Optimizer\n", layer_cnt--);
        timespec_get(&t1, TIME_UTC);
        optimizer(&(p->hidden_layer), &w_error, option);
        timespec_get(&t2, TIME_UTC);
        print_weight(&(p->hidden_layer));
#ifdef USE_CUDA
        sendDenseMatrixToGPU(&(p->hidden_layer));
#endif
        optimizer_time += cal_time(&t2, &t1);
        freeDenseMatrix(&h_error);
        h_error = h_next_error;

#if defined(EMAX6) || defined(EMAX7)
        imax_dense_deallocation(&imax_h_transpose);
        imax_dense_deallocation(&imax_r_spmm);
        imax_dense_deallocation(&imax_w_transpose);
        imax_dense_deallocation(&imax_h_error);
        imax_dense_deallocation(&imax_h_next_error);
        imax_dense_deallocation(&imax_w_error);
#endif
        freeDenseMatrix(&r_spmm);
        freeDenseMatrix(&w_error);
        p = p->prev;
    }
    freeDenseMatrix(&h_error);
#if defined(EMAX6) || defined(EMAX7)
    imax_sparse_deallocation(&network->graph->imax_matrix);
#endif

#ifdef USE_CUDA
    destroyCusparse();
    destroyCublas();
#endif

    printf("SpMM: %lf, MM: %lf, ReLu: %lf, Softmax: %lf Optimizer: %lf usec.\n", spmm_time, mm_time, relu_time, softmax_time, optimizer_time);
    printf("Backpropagation: %lf usec.\n", spmm_time + mm_time + relu_time + softmax_time);
#if !(defined(EMAX6) || defined(EMAX7) || defined(USE_CUDA))
    all_nanosec[SPMM] += (Ull)spmm_time * 1000;
    all_nanosec[MM] += (Ull)mm_time * 1000;
    all_nanosec[RELU] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX] += (Ull)softmax_time * 1000;
#elif defined(USE_CUDA)
    all_nanosec[SPMM][2] += (cusparse_load + spmm_time) * 1000;
    all_nanosec[MM][2] += (cublas_load + mm_time) * 1000;
    all_nanosec[RELU][0] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX][0] += (Ull)softmax_time * 1000;
    all_nanosec[RELU][2] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX][2] += (Ull)softmax_time * 1000;
#else
    all_nanosec[RELU][0] += (Ull)relu_time * 1000;
    all_nanosec[RELU][7] += (Ull)relu_time * 1000;
    all_nanosec[SOFTMAX][0] += (Ull)softmax_time * 1000;
    all_nanosec[SOFTMAX][7] += (Ull)softmax_time * 1000;
#endif
}