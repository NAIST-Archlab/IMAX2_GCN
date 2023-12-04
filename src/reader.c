// EMAX6/7 GCN Test Program            //
// reader.c                            //
//         Copyright (C) 2023 by NAIST //
//          Primary writer: Dohyun Kim //
//          kim.dohyun.kg7@is.naist.jp //
#include "../include/reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void read_graph_csgr(SparseGraph *g, char *name, int from, int to) {
    char tmp_name[256];
    Ull version, sizeEdgeTy, nv, ne;
    Ull *vertices_tmp;
    Uint *vertices, *edges, *edges_tmp;
    int new_nv;
    FILE *fp;
    
    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s.csgr", name);
    if (!(fp = fopen(tmp_name, "rb"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }

    fread(&version, sizeof(Ull), 1, fp);
    fread(&sizeEdgeTy, sizeof(Ull), 1, fp);
    fread(&nv, sizeof(Ull), 1, fp);
    fread(&ne, sizeof(Ull), 1, fp);

    new_nv = to - from;
    if (new_nv < 0) {
        new_nv = nv;
    } else if (new_nv > nv) {
        fprintf(stderr, "Error: Invalid range of vertex\n");
        return;
    }

    vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
    fread(vertices_tmp, sizeof(Ull), nv + 1, fp);
    vertices = (Uint *)malloc(sizeof(Uint) * (new_nv + 1));
    edges = (Uint *)malloc(sizeof(Uint) * ne);
    if (new_nv == nv) {
        for (int i  = 0; i < nv + 1; i++) {
            vertices[i] = vertices_tmp[i];
        }
        fread(edges, sizeof(Uint), ne, fp);
    } else {
        int cnt = 0;
        vertices[0] = 0;
        edges_tmp = (Uint *)malloc(sizeof(Uint) * ne);
        fread(edges_tmp, sizeof(Uint), ne, fp);
        for (int i = from; i < to; i++) {
            int row_nnz = 0;
            for (int j = vertices_tmp[i]; j < vertices_tmp[i+1]; j++) {
                if ((edges_tmp[j] < to) && (edges_tmp[j] >= from)) {
                    edges[cnt] = edges_tmp[j] - from;
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
    }
    fclose(fp);

    g->matrix.row_size = new_nv;
    g->matrix.col_size = new_nv;
    g->matrix.nnz = vertices[new_nv];
    allocSparseMatrix(&(g->matrix));

    memcpy(g->matrix.row_p, vertices, sizeof(Uint) * (new_nv + 1));
    memcpy(g->matrix.col_p, edges, sizeof(Uint) * g->matrix.nnz);
    memset(g->matrix.val, 0, sizeof(float) * g->matrix.nnz);
    free(vertices_tmp);
    free(vertices);
    free(edges);

    printf("|V|=%d, |E|=%d\n", g->matrix.row_size, g->matrix.nnz);
}

void read_graph_bin(SparseGraph *g, char *name, int from, int to) {
    char tmp_name[256];
    Ull version, sizeEdgeTy, nv, ne;
    Ull *vertices_tmp;
    Uint *vertices, *edges, *edges_tmp;
    int new_nv;
    FILE *fp_meta, *fp_vertices, *fp_edges;
    
    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s/graph.meta.txt", name);
    if (!(fp_meta = fopen(tmp_name, "r"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }

    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s/graph.vertex.bin", name);
    if (!(fp_vertices = fopen(tmp_name, "rb"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }

    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s/graph.edge.bin", name);
    if (!(fp_edges = fopen(tmp_name, "rb"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }

    fscanf(fp_meta, "%llu\n", &nv);
    fscanf(fp_meta, "%llu\n", &ne);

    new_nv = to - from;
    if (new_nv < 0) {
        new_nv = nv;
    } else if (new_nv > nv) {
        fprintf(stderr, "Error: Invalid range of vertex\n");
        return;
    }

    vertices_tmp = (Ull *)malloc(sizeof(Ull) * (nv + 1));
    fread(vertices_tmp, sizeof(Ull), nv + 1, fp_vertices);
    vertices = (Uint *)malloc(sizeof(Uint) * (new_nv + 1));
    edges = (Uint *)malloc(sizeof(Uint) * ne);
    if (new_nv == nv) {
        for (int i  = 0; i < nv + 1; i++) {
            vertices[i] = vertices_tmp[i];
        }
        fread(edges, sizeof(Uint), ne, fp_edges);
    } else {
        int cnt = 0;
        vertices[0] = 0;
        edges_tmp = (Uint *)malloc(sizeof(Uint) * ne);
        fread(edges_tmp, sizeof(Uint), ne, fp_edges);
        for (int i = from; i < to; i++) {
            int row_nnz = 0;
            for (int j = vertices_tmp[i]; j < vertices_tmp[i+1]; j++) {
                if ((edges_tmp[j] < to) && (edges_tmp[j] >= from)) {
                    edges[cnt] = edges_tmp[j] - from;
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
    }
    fclose(fp_meta);
    fclose(fp_vertices);
    fclose(fp_edges);

    g->matrix.row_size = new_nv;
    g->matrix.col_size = new_nv;
    g->matrix.nnz = vertices[new_nv];
    allocSparseMatrix(&(g->matrix));

    memcpy(g->matrix.row_p, vertices, sizeof(Uint) * (new_nv + 1));
    memcpy(g->matrix.col_p, edges, sizeof(Uint) * g->matrix.nnz);
    memset(g->matrix.val, 0, sizeof(float) * g->matrix.nnz);
    free(vertices_tmp);
    free(vertices);
    free(edges);

    printf("|V|=%d, |E|=%d\n", g->matrix.row_size, g->matrix.nnz);
}

void read_gcn_weight(GCNNetwork *n, char *path) {
    FILE *fp;
    DenseMatrix weight, vectors;
    Uint dim_in, dim_out, num_layers;

    if (!(fp = fopen(path, "rb"))) {
        fprintf(stderr, "Error: Cannot open %s\n", path);
        return;
    }

    fread(&num_layers, sizeof(Uint), 1, fp);
    n->num_layers = num_layers;
    for (int i = 0; i < num_layers; i++) {
        fread(&dim_in, sizeof(Uint), 1, fp);
        fread(&dim_out, sizeof(Uint), 1, fp);
        vectors.row_size = n->graph->matrix.col_size;
        vectors.col_size = dim_in;
        weight.row_size = dim_in;
        weight.col_size = dim_out;
        allocDenseMatrix(&vectors);
        allocDenseMatrix(&weight);
        fread(weight.val, sizeof(float), dim_in * dim_out, fp);
        #ifdef USE_CUDA
            sendDenseMatrixToGPU(&weight);
        #endif
        add_gcn_layer(n, weight, vectors);
    }
    fclose(fp);

    n->layers->result_layer.row_size = n->graph->matrix.row_size;
    n->layers->result_layer.col_size = dim_out;
    allocDenseMatrix(&(n->layers->result_layer));
    print_gcn_layers(n);
}

void read_gcn_feature_bin(GCNNetwork *n, char *name, int from, int to) {
    char tmp_name[256];
    FILE *fp_feats, *fp_meta;
    Uint dim_in, dim_out;

    sprintf(tmp_name, "%s/graph.meta.txt", name);
    if (!(fp_meta = fopen(tmp_name, "r"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }
    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s/graph.feats.bin", name);
    if (!(fp_feats = fopen(tmp_name, "rb"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }

    dim_in = n->layers->latent_vectors.row_size;

    fscanf(fp_meta, "%*u\n");
    fscanf(fp_meta, "%*u\n");
    fscanf(fp_meta, "%*u %*u %*u %*u\n");
    fscanf(fp_meta, "%*u\n");
    fscanf(fp_meta, "%u\n", &dim_out);

    fseek(fp_feats, sizeof(float) * from * dim_out, SEEK_SET);
    fread(n->layers->latent_vectors.val, sizeof(float), dim_in * dim_out, fp_feats);
    #ifdef USE_CUDA
        sendDenseMatrixToGPU(&(n->layers->latent_vectors));
    #endif
    fclose(fp_feats);
    fclose(fp_meta);
}

void read_gcn_feature_csgr(GCNNetwork *n, char *name, int from, int to) {
    char tmp_name[256];
    FILE *fp_feats, *fp_dims;
    Uint dim_in, dim_out;

    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s-dims.txt", name);
    if (!(fp_dims = fopen(tmp_name, "r"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }
    memset(tmp_name, 0, sizeof(tmp_name));
    sprintf(tmp_name, "%s-feats.bin", name);
    if (!(fp_feats = fopen(tmp_name, "rb"))) {
        fprintf(stderr, "Error: Cannot open %s\n", tmp_name);
        return;
    }

    dim_in = n->layers->latent_vectors.row_size;
    fscanf(fp_dims, "%*u %u\n", &dim_out);

    fseek(fp_feats, sizeof(float) * from * dim_out, SEEK_SET);
    fread(n->layers->latent_vectors.val, sizeof(float), dim_in * dim_out, fp_feats);
    #ifdef USE_CUDA
        sendDenseMatrixToGPU(&(n->layers->latent_vectors));
    #endif
    fclose(fp_feats);
    fclose(fp_dims);
}