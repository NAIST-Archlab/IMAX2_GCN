#include "../include/emax6.h"
#include "./include/layer.h"
#include "./include/utils.h"
#include "./include/sparse.h"
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
    FILE *fp_weight, *fp_graph, *fp_feats, *fp_dims, *fp_mask;
    int num_layers, dim_in, dim_out;
    DenseMatrix tmp_weight, tmp_vectors;
    char tmp_filename[100];
    Ull version, sizeEdgeTy, nv, f_dim_in, ne;
    Uint f_dim_out;
    int from, to;
    Uint *vertices;

    struct timespec t1, t2;

    if (argc < 5) {
        printf("Usage: %s weight graph from to\n", argv[0]);
        return 1;
    }

    if((from = atoi(argv[3])) < 0) {
        return 1;
    }

    if((to = atoi(argv[4])) < 0) {
        return 1;
    }

    if (!(fp_weight = fopen(argv[1], "rb"))) {
        return 1;
    }

    memset(tmp_filename, 0, 100);
    strcat(tmp_filename, argv[2]);
    strcat(tmp_filename, ".csgr");
    if (!(fp_graph = fopen(tmp_filename, "rb"))) {
        return 1;
    }

    memset(tmp_filename, 0, 100);
    strcat(tmp_filename, argv[2]);
    strcat(tmp_filename, "-feats.bin");
    if (!(fp_feats = fopen(tmp_filename, "rb"))) {
        return 1;
    }

    memset(tmp_filename, 0, 100);
    strcat(tmp_filename, argv[2]);
    strcat(tmp_filename, "-dims.txt");
    if (!(fp_dims = fopen(tmp_filename, "r"))) {
        return 1;
    }

    memset(tmp_filename, 0, 100);
    strcat(tmp_filename, argv[2]);
    strcat(tmp_filename, "-val_mask.txt");
    if (!(fp_mask = fopen(tmp_filename, "r"))) {
        return 1;
    }

    printf("Reading Graph now...\n");
    fread(&version, sizeof(Ull), 1, fp_graph);
    fread(&sizeEdgeTy, sizeof(Ull), 1, fp_graph);
    fread(&nv, sizeof(Ull), 1, fp_graph);
    fread(&ne, sizeof(Ull), 1, fp_graph);

    int new_nv = to - from;
    Ull *vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
    fread(vertices_tmp, sizeof(Ull), (nv + 1), fp_graph);
    vertices = (Uint *)malloc(sizeof(Uint) * (new_nv + 1));
    vertices[0] = 0;

    Uint *edges_tmp = (Uint *)malloc(sizeof(Uint)*vertices_tmp[nv]);
    Uint *edges_tmp2 = (Uint *)malloc(sizeof(Uint)*vertices_tmp[nv]);
    fread(edges_tmp, sizeof(Uint), ne, fp_graph);

    int cnt = 0;
    for (int i = from; i < to; i++) {
        int row_nnz = 0;
        for (int j = vertices_tmp[i]; j < vertices_tmp[i+1]; j++) {
            if ((edges_tmp[j] < to) && (edges_tmp[j] >=from)) {
                edges_tmp2[cnt] = edges_tmp[j] - from;
                cnt++;
                row_nnz++;
            }
        }
        if (row_nnz) {
            vertices[i+1-from] = vertices[i-from] + row_nnz;
        } else {
            vertices[i+1-from] = vertices[i-from];
        }
    }

    free(edges_tmp);
    free(vertices_tmp);

    graph.matrix.row_size = new_nv;
    graph.matrix.col_size = new_nv;
    graph.matrix.nnz = vertices[new_nv];
    allocSparseMatrix(&(graph.matrix));

    memcpy(graph.matrix.row_p, vertices, sizeof(Uint)*(new_nv+1));
    memcpy(graph.matrix.col_p, edges_tmp2, sizeof(Uint)*vertices[new_nv]);
    memset(graph.matrix.val, 0, sizeof(float)*vertices[new_nv]);

    printf("|V|=%d, |E|=%d\n", new_nv, vertices[new_nv]);
    free(edges_tmp2);
    free(vertices);

    printf("Caculating A + I\n");
    timespec_get(&t1, TIME_UTC);
    new_graph = spia(&graph);
    freeSparseMatrix(&(graph.matrix));

    // D^-1*A*D^-1
    printf("Calculating D^-1AD^-1\n");
    for (int i = 0; i < new_graph->matrix.row_size; i++) {
        for (int j = new_graph->matrix.row_p[i]; j < new_graph->matrix.row_p[i+1]; j++) {
            int col = new_graph->matrix.col_p[j];
            float d_row = 1 / sqrt(new_graph->matrix.row_p[i + 1] - new_graph->matrix.row_p[i] + 1);
            float d_col = 1 / sqrt(new_graph->matrix.row_p[col + 1] - new_graph->matrix.row_p[col] + 1);
            new_graph->matrix.val[j] = d_row * d_col;
        }
    }
    #ifdef USE_CUDA
        sendSparseMatrixToGPU(&(new_graph->matrix));
    #endif

    timespec_get(&t2, TIME_UTC);
    printf("Preprocessing: %lf usec.\n", cal_time(&t2, &t1));

    network.graph = new_graph;
    network.layers = NULL;

    printf("Reading Weight now...\n");
    fread(&num_layers, sizeof(Uint), 1, fp_weight);
    network.num_layers = num_layers;
    for (int i = 0; i < num_layers; i++) {
        fread(&dim_in, sizeof(Uint), 1, fp_weight);
        fread(&dim_out, sizeof(Uint), 1, fp_weight);
        tmp_weight.row_size = dim_in;
        tmp_weight.col_size = dim_out;
        allocDenseMatrix(&tmp_weight);
        fseek(fp_weight, sizeof(float) * dim_out * from, SEEK_CUR);
        fread(tmp_weight.val, sizeof(float), dim_in * dim_out, fp_weight);
        #ifdef USE_CUDA
            sendDenseMatrixToGPU(&tmp_weight);
        #endif
        tmp_vectors.row_size = new_nv;
        tmp_vectors.col_size = dim_in;
        allocDenseMatrix(&tmp_vectors);
        add_gcn_layer(&network, tmp_weight, tmp_vectors);
    }
    network.layers->result_layer.matrix.row_size = new_nv;
    network.layers->result_layer.matrix.col_size = dim_out;
    allocDenseMatrix(&(network.layers->result_layer.matrix));
    print_layers(&network);

    printf("Reading Features now...\n");
    fscanf(fp_dims, "%lld %d\n", &f_dim_in, &f_dim_out);
    fseek(fp_feats, sizeof(float) * f_dim_out * from, SEEK_CUR);
    fread(network.layers->latent_vectors.matrix.val, sizeof(float), new_nv * f_dim_out, fp_feats);
    #ifdef USE_CUDA
        sendDenseMatrixToGPU(&(network.layers->latent_vectors.matrix));
    #endif
    fclose(fp_feats);

    #ifdef EMAX6
        printf("Transform to IMAX Format..\n");
        timespec_get(&t1, TIME_UTC);
        imax_sparse_format_init(&(network.graph->imax_matrix), network.graph->matrix.row_size, network.graph->matrix.col_size, 46, 8 * NCHIP);
        convert_imax_sparse_format(&(network.graph->imax_matrix), &(network.graph->matrix));
        timespec_get(&t2, TIME_UTC);
        printf("Transform %lf usec.\n", cal_time(&t2, &t1));
    #endif

    propagation(&network);

    printf("Result\n");
    print_weight(&(network.layers->result_layer));
    printf("Propagation Done\n");
    #ifdef EMAX6
        printf("SpMM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[SPMM][0]/1000),
            (Uint)(all_nanosec[SPMM][1]/1000),
            (Uint)(all_nanosec[SPMM][2]/1000),
            (Uint)(all_nanosec[SPMM][3]/1000),
            (Uint)(all_nanosec[SPMM][4]/1000),
            (Uint)(all_nanosec[SPMM][5]/1000),
            (Uint)(all_nanosec[SPMM][6]/1000),
            (Uint)(all_nanosec[SPMM][7]/1000));
        printf("MM usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[MM][0]/1000),
            (Uint)(all_nanosec[MM][1]/1000),
            (Uint)(all_nanosec[MM][2]/1000),
            (Uint)(all_nanosec[MM][3]/1000),
            (Uint)(all_nanosec[MM][4]/1000),
            (Uint)(all_nanosec[MM][5]/1000),
            (Uint)(all_nanosec[MM][6]/1000),
            (Uint)(all_nanosec[MM][7]/1000));
        printf("ReLU usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[RELU][0]/1000),
            (Uint)(all_nanosec[RELU][1]/1000),
            (Uint)(all_nanosec[RELU][2]/1000),
            (Uint)(all_nanosec[RELU][3]/1000),
            (Uint)(all_nanosec[RELU][4]/1000),
            (Uint)(all_nanosec[RELU][5]/1000),
            (Uint)(all_nanosec[RELU][6]/1000),
            (Uint)(all_nanosec[RELU][7]/1000));
        printf("Softmax usec: ARM:%d DRAIN:%d CONF:%d REGV:%d RANGE:%d LOAD:%d EXEC:%d total:%d\n",
            (Uint)(all_nanosec[SOFTMAX][0]/1000),
            (Uint)(all_nanosec[SOFTMAX][1]/1000),
            (Uint)(all_nanosec[SOFTMAX][2]/1000),
            (Uint)(all_nanosec[SOFTMAX][3]/1000),
            (Uint)(all_nanosec[SOFTMAX][4]/1000),
            (Uint)(all_nanosec[SOFTMAX][5]/1000),
            (Uint)(all_nanosec[SOFTMAX][6]/1000),
            (Uint)(all_nanosec[SOFTMAX][7]/1000));
    #else
        printf("SpMM usec: total:%d\n", (Uint)(all_nanosec[SPMM]/1000));
        printf("MM usec: total:%d\n", (Uint)(all_nanosec[MM]/1000));
        printf("ReLU usec: total:%d\n", (Uint)(all_nanosec[RELU]/1000));
        printf("Softmax usec: total:%d\n", (Uint)(all_nanosec[SOFTMAX]/1000));
    #endif

    fclose(fp_graph);
    fclose(fp_weight);

    return 0;
}