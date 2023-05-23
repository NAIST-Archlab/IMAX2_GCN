#include "./include/layer.h"
#include "./include/utils.h"
#include "../include/emax6.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef USE_MP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    GCNNetwork network;
    SparseGraph graph;
    SparseGraph *new_graph = &graph;
    HiddenLayer *result;
    FILE *fp_weight, *fp_graph, *fp_feats, *fp_dims;
    int num_layers, dim_in, dim_out;
    float *tmp_weight, *tmp_vectors, *edges_val;
    char tmp_filename[100];
    Ull version, sizeEdgeTy, nv, f_dim_in, ne;
    Uint f_dim_out;
    Ull *vertices;
    Uint *edges, *vertices_int;

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

    printf("Reading Graph now...\n");
    fread(&version, sizeof(Ull), 1, fp_graph);
    fread(&sizeEdgeTy, sizeof(Ull), 1, fp_graph);
    fread(&nv, sizeof(Ull), 1, fp_graph);
    fread(&ne, sizeof(Ull), 1, fp_graph);
    vertices = (Ull*) malloc(sizeof(Ull)*(nv+1));
    vertices_int = (Uint*) malloc(sizeof(Uint)*(nv+1));
    fread(vertices, sizeof(Ull), (nv+1), fp_graph);

    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i <= nv; i++) vertices_int[i] = (Uint) vertices[i];

    edges = (Uint*) malloc(sizeof(Uint)*vertices[nv]);
    edges_val = (float*) malloc(sizeof(float)*vertices[nv]);
    memset(edges_val, 0, sizeof(float)*vertices[nv]);
    fread(edges, sizeof(Uint), ne, fp_graph);
    graph.matrix.nnz = vertices[nv];
    graph.matrix.col_size = nv;
    graph.matrix.row_size = nv;
    graph.matrix.row_p = (int*)vertices_int;
    graph.matrix.col_p = (int*)edges;
    graph.matrix.val = edges_val;

    printf("Caculating A + I\n");
    timespec_get(&t1, TIME_UTC);
    new_graph = spia(&graph);

    // D^-1*A*D^-1
    printf("Calculating D^-1AD^-1\n");
    #ifdef USE_MP
    #pragma omp parallel for
    #endif
    for(int i = 0; i <= new_graph->matrix.row_size; i++) {
        for (int j = new_graph->matrix.row_p[i]; j < new_graph->matrix.row_p[i+1]; j++) {
            int col = new_graph->matrix.col_p[j];
            float d_row = 1/sqrt(new_graph->matrix.row_p[i+1] - new_graph->matrix.row_p[i] + 1);
            float d_col = 1/sqrt(new_graph->matrix.row_p[col+1] - new_graph->matrix.row_p[col] + 1);
            new_graph->matrix.val[j] = d_row * d_col;
        }
    }
    timespec_get(&t2, TIME_UTC);
    printf("Preprocessing: %lf sec.\n", cal_time(&t2, &t1));

    network.graph = new_graph;
    network.layers = NULL;

    printf("Reading Weight now...\n");
    fread(&num_layers, sizeof(Uint), 1, fp_weight);
    network.num_layers = num_layers;
    for (int i = 0; i< num_layers; i++) {
        fread(&dim_in, sizeof(Uint), 1, fp_weight);
        fread(&dim_out, sizeof(Uint), 1, fp_weight);
        tmp_weight = make_weight(dim_in, dim_out);
        fread(tmp_weight, sizeof(float), dim_in*dim_out, fp_weight);
        tmp_vectors = make_weight(nv, dim_in);
        add_gcn_layer(&network, tmp_weight, tmp_vectors, dim_in, dim_out);
    }
    print_layers(&network);

    printf("Reading Features now...\n");
    fread(network.layers->latent_vectors.weight, sizeof(float), f_dim_in*f_dim_out, fp_feats);

    #ifdef USE_IMAX2
    printf("Transform to IMAX Format..\n");
    timespec_get(&t1, TIME_UTC);
    int graph_vec_unit_lcm = 46*8 / gcd(46, 8);
    int padded_graph_size = network.graph->matrix.col_size + ((network.graph->matrix.col_size % graph_vec_unit_lcm) ? (graph_vec_unit_lcm - (network.graph->matrix.col_size % graph_vec_unit_lcm)) : 0);
    imax_sparse_format_init(&network.graph->imax_matrix, network.graph->matrix.row_size, network.graph->matrix.col_size, 46, 8*NCHIP);
    convert_imax_sparse_format(&network.graph->imax_matrix, &network.graph->matrix);
    timespec_get(&t2, TIME_UTC);
    printf("Transform %lf sec.\n", cal_time(&t2, &t1));
    #endif

    printf("Propagation...\n");
    timespec_get(&t1, TIME_UTC);
    result = propagation(&network);
    timespec_get(&t2, TIME_UTC);
    printf("Propagation: %lf sec.\n", cal_time(&t2, &t1));
    printf("Result\n");
    print_weight(result);
    printf("Propagation Done\n");

    fclose(fp_graph);
    fclose(fp_weight);

    free(vertices);
    free(edges);
    free(edges_val);

    if (result != NULL) free(result);

    return 0;
}