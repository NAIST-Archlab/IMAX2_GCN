#include "../include/layer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#ifdef USE_MP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    GCNNetwork network;
    SparseGraph graph;
    FILE *fp_weight, *fp_graph;
    int num_layers, dim_in, dim_out, i, j;
    float *tmp_weight, *tmp_vectors;
    Ull version, sizeEdgeTy, nv;
    Ull *vertices;
    Uint *vertices_int;
    Uint *edges;
    Uint *edges_val;

    if (argc < 3) {
        printf("Usage: %s weight graph\n", argv[0]);
        return 1;
    }

    if (!(fp_weight = fopen(argv[1], 'rb'))) {
        return 1;
    }

    if (!(fp_graph = fopen(argv[2], 'rb'))) {
        return 1;
    }

    fgets(&version, 8, fp_graph);
    fgets(&sizeEdgeTy, 8, fp_graph);
    fgets(&nv, 8, fp_graph);
    vertices = (Ull*) malloc(sizeof(Ull)*(nv+1));
    fgets(vertices, sizeof(Ull)*(nv+1), fp_graph);
    vertices_int = (Uint*) malloc(sizeof(Uint)*(nv+1));
    for (i = 0; i <= nv; i++) { 
        vertices_int[i] = (Uint) vertices[i];
    }
    edges = (Uint*) malloc(sizeof(Uint)*vertices[nv]);
    edges_val = (Uint*) malloc(sizeof(Uint)*vertices[nv]);
    fgets(edges, sizeof(Uint)*vertices[nv], fp_graph);

    // D^-1*A*D^-1
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
    graph.matrix.col_p = (int*)edges;
    graph.matrix.val = edges_val;

    fgets(&num_layers, 4, fp_weight);
    for (i = 0; i< num_layers; i++) {
        fgets(&dim_in, 4, fp_weight);
        fgets(&dim_out, 4, fp_weight);
        tmp_weight = make_weight(dim_in, dim_out);
        fgets(tmp_weight, dim_in*dim_out*sizeof(float), fp_weight);
        tmp_vectors = make_weight(nv, dim_in);
        add_gcn_layer(&network, tmp_weight, tmp_vectors, dim_in, dim_out);
    }

    propagation(&network);

    fclose(fp_graph);
    fclose(fp_weight);

    free(vertices);
    free(vertices_int);
    free(edges);
    free(edges_val);

    return 0;
}