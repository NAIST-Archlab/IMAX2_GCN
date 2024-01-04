// EMAX6/7 GCN Test Program            //
// main.c                              //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "./include/gcn.h"
#include "./include/utils.h"
#include "./include/sparse.h"
#include "./include/linalg.h"
#include "./include/reader.h"
#include "./include/optimizer.h"
#include "./include/imax.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef USE_MP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    GCNNetwork network;
    SparseGraph graph;
    SparseGraph *new_graph = &graph;
    FILE *fp_graph;
    int num_layers, dim_in, dim_out;
    DenseMatrix tmp_weight, tmp_vectors;
    char tmp_filename[100], graph_type;
    Ull version, sizeEdgeTy, nv, f_dim_in, ne;
    Uint f_dim_out;
    int from, to, iter=1;
    Uint *vertices;

    struct timespec t1, t2;

    if (argc < 5) {
        printf("Usage: %s weight graph from to (iter)\n", argv[0]);
        return 1;
    }

    if((from = atoi(argv[3])) < 0) {
        return 1;
    }

    to = atoi(argv[4]);

    if (argc > 5) {
        if((iter = atoi(argv[5])) < 1) iter = 1;
    }

    memset(tmp_filename, 0, 100);
    strcat(tmp_filename, argv[2]);
    strcat(tmp_filename, ".csgr");
    printf("Reading Graph now...\n");
    if (!(fp_graph = fopen(tmp_filename, "rb"))) {
        graph_type = 0;
        read_graph_bin(&graph, argv[2], from, to);
    } else {
        fclose(fp_graph);
        graph_type = 1;
        read_graph_csgr(&graph, argv[2], from, to);
    }

    printf("Caculating A + I\n");
    new_graph = (SparseGraph *)malloc(sizeof(SparseGraph));
    timespec_get(&t1, TIME_UTC);
    spia(&(new_graph->matrix), &(graph.matrix));

    printf("Calculating D^-1AD^-1\n");
    gcn_preprocessing(&(new_graph->matrix));

    timespec_get(&t2, TIME_UTC);
    freeSparseMatrix(&(graph.matrix));
    printf("Preprocessing: %lf usec.\n", cal_time(&t2, &t1));

    network.graph = new_graph;
    network.layers = NULL;

    printf("Reading Weight now...\n");
    read_gcn_weight(&network, argv[1]);

    printf("Reading Features now...\n");
    if (graph_type) read_gcn_feature_csgr(&network, argv[2], from, to);
    else read_gcn_feature_bin(&network, argv[2], from, to);

    #if defined(EMAX6) || defined(EMAX7)
        printf("Transform to IMAX Format..\n");
        timespec_get(&t1, TIME_UTC);
        imax_sparse_format_init(&(network.graph->imax_matrix), network.graph->matrix.row_size, network.graph->matrix.col_size, SPMM_H, 8 * NCHIP);
        convert_imax_sparse_format(&(network.graph->imax_matrix), &(network.graph->matrix));
        timespec_get(&t2, TIME_UTC);
        printf("Transform %lf usec.\n", cal_time(&t2, &t1));
    #endif

    for (int i = 0; i < iter; i++) {gcn_propagation(&network);}
    //OptimizerOption opt;
    //opt.lr = 0.01;
    //opt.beta1 = 0.9;
    //opt.beta2 = 0.999;
    //opt.epsilon = 1e-8;
    //opt.t = 1;
    //Uchar *vlabels = (Uchar *)malloc(sizeof(Uchar) * network.graph->matrix.row_size);
    //read_graph_bin_vlabels(vlabels, argv[2], 0, network.graph->matrix.row_size - 1);
    //DenseMatrix labels;
    //labels.row_size = network.layers->result_layer.row_size;
    //labels.col_size = network.layers->result_layer.col_size;
    //allocDenseMatrix(&labels);
    //expand_labels(&labels, vlabels);
    //#ifdef USE_CUDA
        //sendDenseMatrixToGPU(&labels);
    //#endif
    //for (int i = 0; i < iter; i++) {
        //gcn_propagation(&network);
        //gcn_backpropagation(&network, &labels, &adam, &opt);
    //}

    printf("Result\n");
    print_weight(&(network.layers->result_layer));
    printf("Propagation Done\n");
    #if defined(EMAX6) || defined(EMAX7)
        printf("SpMM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[SPMM][0]/1000/iter),
            (Uint)(all_nanosec[SPMM][1]/1000/iter),
            (Uint)(all_nanosec[SPMM][2]/1000/iter),
            (Uint)(all_nanosec[SPMM][3]/1000/iter),
            (Uint)(all_nanosec[SPMM][4]/1000/iter),
            (Uint)(all_nanosec[SPMM][5]/1000/iter),
            (Uint)(all_nanosec[SPMM][6]/1000/iter),
            (Uint)(all_nanosec[SPMM][7]/1000/iter));
        printf("MM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[MM][0]/1000/iter),
            (Uint)(all_nanosec[MM][1]/1000/iter),
            (Uint)(all_nanosec[MM][2]/1000/iter),
            (Uint)(all_nanosec[MM][3]/1000/iter),
            (Uint)(all_nanosec[MM][4]/1000/iter),
            (Uint)(all_nanosec[MM][5]/1000/iter),
            (Uint)(all_nanosec[MM][6]/1000/iter),
            (Uint)(all_nanosec[MM][7]/1000/iter));
        printf("ReLU usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[RELU][0]/1000/iter),
            (Uint)(all_nanosec[RELU][1]/1000/iter),
            (Uint)(all_nanosec[RELU][2]/1000/iter),
            (Uint)(all_nanosec[RELU][3]/1000/iter),
            (Uint)(all_nanosec[RELU][4]/1000/iter),
            (Uint)(all_nanosec[RELU][5]/1000/iter),
            (Uint)(all_nanosec[RELU][6]/1000/iter),
            (Uint)(all_nanosec[RELU][7]/1000/iter));
        printf("Softmax usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[SOFTMAX][0]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][1]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][2]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][3]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][4]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][5]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][6]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][7]/1000/iter));
    #elif defined(USE_CUDA)
        all_nanosec[SPMM][2] = all_nanosec[SPMM][0] + all_nanosec[SPMM][1];
        all_nanosec[MM][2] = all_nanosec[MM][0] + all_nanosec[MM][1];
        printf("SpMM usec: EXEC:%d CONF:%d total:%d\n",
            (Uint)(all_nanosec[SPMM][0]/1000/iter),
            (Uint)(all_nanosec[SPMM][1]/1000/iter),
            (Uint)(all_nanosec[SPMM][2]/1000/iter));
        printf("MM usec: EXEC:%d CONF:%d total:%d\n",
            (Uint)(all_nanosec[MM][0]/1000/iter),
            (Uint)(all_nanosec[MM][1]/1000/iter),
            (Uint)(all_nanosec[MM][2]/1000/iter));
        printf("ReLU usec: EXEC:%d CONF:%d total:%d\n",
            (Uint)(all_nanosec[RELU][0]/1000/iter),
            (Uint)(all_nanosec[RELU][1]/1000/iter),
            (Uint)(all_nanosec[RELU][2]/1000/iter));
        printf("Softmax usec: EXEC:%d CONF:%d total:%d\n",
            (Uint)(all_nanosec[SOFTMAX][0]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][1]/1000/iter),
            (Uint)(all_nanosec[SOFTMAX][2]/1000/iter));
    #else
        printf("SpMM usec: total:%d\n", (Uint)(all_nanosec[SPMM]/1000/iter));
        printf("MM usec: total:%d\n", (Uint)(all_nanosec[MM]/1000/iter));
        printf("ReLU usec: total:%d\n", (Uint)(all_nanosec[RELU]/1000/iter));
        printf("Softmax usec: total:%d\n", (Uint)(all_nanosec[SOFTMAX]/1000/iter));
    #endif

    return 0;
}
