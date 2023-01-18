#include "../include/layer.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#ifdef USE_MP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    GCNNetwork network;
    SparseGraph graph;
    HiddenLayer *result;
    FILE *fp_weight, *fp_graph;
    int num_layers, dim_in, dim_out;
    float *tmp_weight, *tmp_vectors;
    Ull version, sizeEdgeTy, nv, i, j;
    Ull *vertices;
    Uint *edges, *vertices_int;
    Uint *edges_val;

    if (argc < 3) {
        printf("Usage: %s weight graph\n", argv[0]);
        return 1;
    }

    if (!(fp_weight = fopen(argv[1], "rb"))) {
        return 1;
    }

    if (!(fp_graph = fopen(argv[2], "rb"))) {
        return 1;
    }

    printf("Reading Graph now...\n");
    fread(&version, sizeof(Ull), 1, fp_graph);
    fread(&sizeEdgeTy, sizeof(Ull), 1, fp_graph);
    fread(&nv, sizeof(Ull), 1, fp_graph);
    vertices = (Ull*) malloc(sizeof(Ull)*(nv+1));
    vertices_int = (Uint*) malloc(sizeof(Uint)*(nv+1));
    fread(vertices, sizeof(Ull), (nv+1), fp_graph);
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(i = 0; i <= nv; i++) {
        vertices_int[i] = (Uint) vertices[i];
    }
    edges = (Uint*) malloc(sizeof(Uint)*vertices[nv]);
    edges_val = (Uint*) malloc(sizeof(Uint)*vertices[nv]);
    fread(edges, sizeof(Uint), vertices[nv], fp_graph);

    // D^-1*A*D^-1
    printf("Calculating D^-1AD^-1\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(i = 0; i < nv; i++) {
        for (j = vertices[i]; j < vertices[i+1]; j++) {
            int col = edges[j];
            float d_row = 1/sqrt(vertices[i+1] - vertices[i]);
            float d_col = 1/sqrt(vertices[col+1] - vertices[col]);
            float d_row_col = d_row * d_col;
            edges_val[j] = *(Uint*)&d_row_col;
        }
    }

    graph.matrix.nnz = vertices[nv];
    graph.matrix.col_size = nv;
    graph.matrix.row_size = nv;
    graph.matrix.row_p = (int*)vertices_int;
    graph.matrix.col_p = (int*) edges;
    graph.matrix.val = edges_val;

    network.graph = &graph;
    network.layers = NULL;

    printf("Reading Weight now...\n");
    fread(&num_layers, sizeof(Uint), 1, fp_weight);
    network.num_layers = num_layers;
    for (i = 0; i< num_layers; i++) {
        fread(&dim_in, sizeof(Uint), 1, fp_weight);
        fread(&dim_out, sizeof(Uint), 1, fp_weight);
        tmp_weight = make_weight(dim_in, dim_out);
        fread(tmp_weight, sizeof(float), dim_in*dim_out, fp_weight);
        tmp_vectors = make_weight(nv, dim_in);
        add_gcn_layer(&network, tmp_weight, tmp_vectors, dim_in, dim_out);
    }

    printf("Propagation...\n");
    result = propagation(&network);
    printf("Propagation Done\n");

    fclose(fp_graph);
    fclose(fp_weight);

    free(vertices);
    free(edges);
    free(edges_val);

    if (result != NULL) free(result);

    return 0;
}