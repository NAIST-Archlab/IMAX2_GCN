#include "../include/emax6.h"
#include "./include/layer.h"
#include "./include/utils.h"
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
    HiddenLayer *result;
    FILE *fp_weight, *fp_graph, *fp_feats, *fp_dims, *fp_mask;
    int num_layers, dim_in, dim_out;
    float *tmp_weight, *tmp_vectors, *edges_val;
    char tmp_filename[100];
    Ull version, sizeEdgeTy, nv, f_dim_in, ne;
    Uint f_dim_out;
    Uint *edges, *vertices, *mask;

    struct timespec t1, t2;

    if (argc < 3) {
        printf("Usage: %s weight graph\n", argv[0]);
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

    fscanf(fp_dims, "%lld %d\n", &f_dim_in, &f_dim_out);

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

    printf("Reading Mask now...\n");
    Uint mv, anv;
    fscanf(fp_mask, "%ld %ld", &mv, &anv);
    mask = (Uint *) malloc(sizeof(Uint) * nv);
    for (int i = 0; i < nv; i++) {
        fscanf(fp_mask, "%ld", &mask[i]);
    }

    Ull *vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
    fread(vertices_tmp, sizeof(Ull), (nv + 1), fp_graph);
    vertices = (Uint *)malloc(sizeof(Uint) * (nv + 1));
    vertices[0] = 0;

    Uint *edges_tmp = (Uint *)malloc(sizeof(Uint) * vertices_tmp[nv]);
    Uint *edges_tmp2 = (Uint *)malloc(sizeof(Uint) * vertices_tmp[nv]);
    fread(edges_tmp, sizeof(Uint), ne, fp_graph);

    int cnt = 0;
    for (int i = 0; i < nv; i++) {
        int row_nnz = 0;
        for (int j = vertices_tmp[i]; j < vertices_tmp[i+1]; j++) {
            if ((mask[edges_tmp[j]] != 0) && (mask[i] != 0)) {
                edges_tmp2[cnt] = edges_tmp[j];
                cnt++;
                row_nnz++;
            }
        }
        if (row_nnz) {
            vertices[i+1] = vertices[i] + row_nnz;
        } else {
            vertices[i+1] = vertices[i];
        }
    }

    for (int i = 0; i < nv; i++) {
        if (vertices[i+1] < vertices[i]) {
            printf("%d %d\n", vertices[i+1], vertices[i]);
        }
    }

    free(edges_tmp);
    free(vertices_tmp);
    
    edges = (Uint *)malloc(sizeof(Uint) * vertices[nv]);
    edges_val = (float *)malloc(sizeof(float) * vertices[nv]);
    memcpy(edges, edges_tmp2, sizeof(Uint) * vertices[nv]);
    memset(edges_val, 0, sizeof(float) * vertices[nv]);
    free(edges_tmp2);

    graph.matrix.nnz = vertices[nv];
    graph.matrix.col_size = nv;
    graph.matrix.row_size = nv;
    graph.matrix.row_p = (int *)vertices;
    graph.matrix.col_p = (int *)edges;
    graph.matrix.val = edges_val;

    printf("|V|=%d, |E|=%d\n", nv, vertices[nv]);

    printf("Caculating A + I\n");
    timespec_get(&t1, TIME_UTC);
    new_graph = spia(&graph);

    // D^-1*A*D^-1
    printf("Calculating D^-1AD^-1\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < new_graph->matrix.row_size; i++) {
        for (int j = new_graph->matrix.row_p[i]; j < new_graph->matrix.row_p[i+1]; j++) {
            int col = new_graph->matrix.col_p[j];
            float d_row = 1 / sqrt(new_graph->matrix.row_p[i + 1] - new_graph->matrix.row_p[i] + 1);
            float d_col = 1 / sqrt(new_graph->matrix.row_p[col + 1] - new_graph->matrix.row_p[col] + 1);
            new_graph->matrix.val[j] = d_row * d_col;
        }
    }
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
        tmp_weight = make_weight(dim_in, dim_out);
        fread(tmp_weight, sizeof(float), dim_in * dim_out, fp_weight);
        tmp_vectors = make_weight(nv, dim_in);
        add_gcn_layer(&network, tmp_weight, tmp_vectors, dim_in, dim_out);
    }
    print_layers(&network);

    printf("Reading Features now...\n");
    fread(network.layers->latent_vectors.weight, sizeof(float), f_dim_in * f_dim_out, fp_feats);

    #ifdef EMAX6
        printf("Transform to IMAX Format..\n");
        timespec_get(&t1, TIME_UTC);
        imax_sparse_format_init(&network.graph->imax_matrix, network.graph->matrix.row_size, network.graph->matrix.col_size, 46, 8 * NCHIP);
        convert_imax_sparse_format(&network.graph->imax_matrix, &network.graph->matrix);
        timespec_get(&t2, TIME_UTC);
        printf("Transform %lf usec.\n", cal_time(&t2, &t1));
    #endif

    printf("Propagation...\n");
    timespec_get(&t1, TIME_UTC);
    #ifdef USE_CUDA
        sendSparseMatrixToGPU(&(network.graph->matrix));
    #endif
    result = propagation(&network);
    timespec_get(&t2, TIME_UTC);
    printf("Propagation: %lf usec.\n", cal_time(&t2, &t1));
    printf("Result\n");
    print_weight(result);
    printf("Propagation Done\n");

    fclose(fp_graph);
    fclose(fp_weight);

    free(vertices);
    free(edges);
    free(edges_val);

    if (result != NULL)
        free(result);

    return 0;
}