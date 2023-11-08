// EMAX6/7 GCN Test Program            //
// main.c                              //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#if defined(EMAX7)
#include "../conv-c2d/emax7.h"
#elif defined(EMAX6)
#include "../conv-c2c/emax6.h"
#endif
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
    FILE *fp_weight, *fp_graph, *fp_feats, *fp_dims, *fp_mask, *fp_vertices, *fp_edges, *fp_meta;
    int num_layers, dim_in, dim_out;
    DenseMatrix tmp_weight, tmp_vectors;
    char tmp_filename[100];
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

    if (!(fp_weight = fopen(argv[1], "rb"))) {
        return 1;
    }

    memset(tmp_filename, 0, 100);
    strcat(tmp_filename, argv[2]);
    strcat(tmp_filename, ".csgr");
    if (!(fp_graph = fopen(tmp_filename, "rb"))) {
        memset(tmp_filename, 0, 100);
        strcat(tmp_filename, argv[2]);
        strcat(tmp_filename, "/graph.vertex.bin");
        if (!(fp_vertices = fopen(tmp_filename, "rb"))) {
            return 1;
        }
        memset(tmp_filename, 0, 100);
        strcat(tmp_filename, argv[2]);
        strcat(tmp_filename, "/graph.edge.bin");
        if (!(fp_edges = fopen(tmp_filename, "rb"))) {
            return 1;
        }
        memset(tmp_filename, 0, 100);
        strcat(tmp_filename, argv[2]);
        strcat(tmp_filename, "/graph.meta.txt");
        if (!(fp_meta = fopen(tmp_filename, "r"))) {
            return 1;
        }

        memset(tmp_filename, 0, 100);
        strcat(tmp_filename, argv[2]);
        strcat(tmp_filename, "/graph.feats.bin");
        if (!(fp_feats = fopen(tmp_filename, "rb"))) {
            return 1;
        }

        memset(tmp_filename, 0, 100);
        strcat(tmp_filename, argv[2]);
        strcat(tmp_filename, "/val.masks.bin");
        if (!(fp_mask = fopen(tmp_filename, "rb"))) {
            return 1;
        }
    } else {
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
    } 

    printf("Reading Graph now...\n");
    int new_nv;
    Uint *edges_tmp2;
    if (fp_graph) {
        fread(&version, sizeof(Ull), 1, fp_graph);
        fread(&sizeEdgeTy, sizeof(Ull), 1, fp_graph);
        fread(&nv, sizeof(Ull), 1, fp_graph);
        fread(&ne, sizeof(Ull), 1, fp_graph);
    } else {
        fscanf(fp_meta, "%lld\n", &nv);
        fscanf(fp_meta, "%lld\n", &ne);
        printf("%lld %lld\n", nv, ne);
    }
    if (to < from) {
        new_nv = nv;
        Ull *vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
        if (fp_graph) {
            fread(vertices_tmp, sizeof(Ull), (nv + 1), fp_graph);
        } else {
            fread(vertices_tmp, sizeof(Ull), (nv + 1), fp_vertices);
        }

        vertices = (Uint *) malloc(sizeof(Uint) * (nv + 1));
        for (int i = 0; i < nv + 1; i++) {
            vertices[i] = (Uint) vertices_tmp[i];
        }

        edges_tmp2 = (Uint *) malloc(sizeof(Uint) * vertices[nv]);
        if (fp_graph) {
            fread(edges_tmp2, sizeof(Uint), ne, fp_graph);
        } else {
            fread(edges_tmp2, sizeof(Uint), ne, fp_edges);
        }
        free(vertices_tmp);
    } else {
        if (fp_graph) {
            new_nv = to - from;
            Ull *vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
            fread(vertices_tmp, sizeof(Ull), (nv + 1), fp_graph);
            vertices = (Uint *)malloc(sizeof(Uint) * (new_nv + 1));
            vertices[0] = 0;

            Uint *edges_tmp = (Uint *)malloc(sizeof(Uint)*vertices_tmp[nv]);
            edges_tmp2 = (Uint *)malloc(sizeof(Uint)*vertices_tmp[nv]);
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
        } else {
            new_nv = to - from;
            Ull *vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
            fread(vertices_tmp, sizeof(Ull), (nv + 1), fp_vertices);
            vertices = (Uint *)malloc(sizeof(Uint) * (new_nv + 1));
            vertices[0] = 0;

            Uint *edges_tmp = (Uint *)malloc(sizeof(Uint)*vertices_tmp[nv]);
            edges_tmp2 = (Uint *)malloc(sizeof(Uint)*vertices_tmp[nv]);
            fread(edges_tmp, sizeof(Uint), ne, fp_edges);

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
        }
    }

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
    network.layers->result_layer.row_size = new_nv;
    network.layers->result_layer.col_size = dim_out;
    allocDenseMatrix(&(network.layers->result_layer));
    print_layers(&network);

    printf("Reading Features now...\n");
    if (fp_graph) {
        fscanf(fp_dims, "%lld %d\n", &f_dim_in, &f_dim_out);
    } else {
        dim_in = new_nv;
        fscanf(fp_meta, "%% %% %% %% %% %lld %% %%\n", &f_dim_out);
    }
    //fseek(fp_feats, sizeof(float) * f_dim_out * from, SEEK_CUR);
    //fread(network.layers->latent_vectors.val, sizeof(float), new_nv * f_dim_out, fp_feats);
    if (to > from) {
        fseek(fp_feats, sizeof(float) * network.layers->latent_vectors.col_size * from, SEEK_CUR);
    }
    fread(network.layers->latent_vectors.val, sizeof(float), new_nv * network.layers->latent_vectors.col_size, fp_feats);
    #ifdef USE_CUDA
        sendDenseMatrixToGPU(&(network.layers->latent_vectors));
    #endif
    fclose(fp_feats);

    #if defined(EMAX6) || defined(EMAX7)
        printf("Transform to IMAX Format..\n");
        timespec_get(&t1, TIME_UTC);
        imax_sparse_format_init(&(network.graph->imax_matrix), network.graph->matrix.row_size, network.graph->matrix.col_size, SPMM_H, 8 * NCHIP);
        convert_imax_sparse_format(&(network.graph->imax_matrix), &(network.graph->matrix));
        timespec_get(&t2, TIME_UTC);
        printf("Transform %lf usec.\n", cal_time(&t2, &t1));
    #endif

    for (int i = 0; i < iter; i++) {propagation(&network);}

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

    if (fp_graph) fclose(fp_graph);
    if (fp_weight) fclose(fp_weight);

    return 0;
}