#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#if defined(USE_CUDA)
#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#endif

#include "../include/linalg.h"
#include "../include/sparse.h"
#include "../include/reader_gat.h"
#include "../include/layer.h"

void printDMatrix(DenseMatrix *A, char *s){
    printf("row: %d\ncol: %d\n", A->row_size, A->col_size);
    int col =  A->col_size;
    char name[500];
    sprintf(name, "output/%s.txt",s);
    FILE *out_file = fopen(name, "w+");
    char tmp[500];
    for (int i=0; i<A->row_size; i++){
        for (int j=0; j<col; j++){
            sprintf(tmp, "%f ", A->val[i*col + j]);
            fputs(tmp, out_file);
        }
        fputs("\n", out_file);
    }
    fclose(out_file);
}

void printVector(Vector *A, char *s) {
    int col = A->col_size;
    char name[500];
    sprintf(name, "output/%s.txt",s);
    FILE *out_file = fopen(name, "w+");
    char tmp[50];
    for (int i=0; i<A->col_size; i++) {
        sprintf(tmp, "%f ", A->val[i]);
        fputs(tmp, out_file);
    }
    fclose(out_file);
}

void printSpMatrix(SparseMatrix *A, char *s) {
    char name[500];
    sprintf(name, "output/%s.txt",s);
    FILE *file = fopen(name, "w+");

    float sparsity = (float)A->nnz / (float)(A->row_size * A->col_size);
    fprintf(file, "Sparsity: %f, Non-Zero Elements: %d, Total Elements: %d\n\n", sparsity, A->nnz, A->row_size * A->col_size);

    int nz_idx = 0;
    for (int i = 0; i < A->row_size; i++) {
        for (int j = 0; j < A->col_size; j++) {
            if (nz_idx < A->row_p[i + 1] && j == A->col_p[nz_idx]) {
                fprintf(file, "%.6f ", A->val[nz_idx]);
                nz_idx++;
            } else {
                fprintf(file, "0.000000 ");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void init_csr(SparseMatrix *matrix, int row, int col, int nz) {
    matrix->row_size = row;
    matrix->col_size = col;
    matrix->nnz = nz;
    
    matrix->row_p = calloc(row+1, sizeof(int));
    matrix->col_p = calloc(nz, sizeof(int));
    matrix->val = calloc(nz, sizeof(float));

# if defined(USE_CUDA)
    CHECK(cudaMalloc((void**) &matrix->cuda_row_p, sizeof(int)*(matrix->row_size+1)));
    CHECK(cudaMalloc((void**) &matrix->cuda_col_p, sizeof(int)*(matrix->nnz)));
    CHECK(cudaMalloc((void**) &matrix->cuda_val,   sizeof(float)*(matrix->nnz)));
#endif
}

void init_dense(DenseMatrix *matrix, int row, int col) {
    matrix->val = calloc(row * col, sizeof(float));
    matrix->row_size = row;
    matrix->col_size = col;
# if defined(USE_CUDA)
    CHECK(cudaMalloc((void**) &matrix->cuda_val, sizeof(float)*(matrix->row_size*matrix->col_size)));
#endif
}

void init_vector(Vector *vector, int col) {
    vector->val = calloc(1 * col, sizeof(float));
    vector->col_size = col;
}

void init_graph(GATGraph *g, int nnode, int nedge, int nfeature, int nz_nei, int nz_fea, bool inductive) {

    g->out_feature = calloc(nnode * nfeature, sizeof(float));

    if (!inductive) {
        g->s_neighbor = malloc(sizeof(SparseMatrix));
        init_csr(g->s_neighbor, nnode, nnode, nz_nei);

        g->s_features = malloc(sizeof(SparseMatrix)); // features
        init_csr(g->s_features, nnode, nfeature, nz_fea);
    }

    else {
        g->d_neighbor = malloc(sizeof(DenseMatrix));
        init_dense(g->d_neighbor, nnode, nnode);

        g->d_features = malloc(sizeof(DenseMatrix)); // features
        init_dense(g->d_features, nnode, nfeature);
    }

}

void read_csr(SparseMatrix *matrix, FILE *infile) {
    fread(matrix->row_p, sizeof(int), matrix->row_size + 1, infile);
    fread(matrix->col_p, sizeof(int), matrix->nnz, infile);
    fread(matrix->val, sizeof(float), matrix->nnz, infile);    
}

void read_vector(Vector *v, FILE *infile) {
    for (int i=0; i<v->col_size; i++) {
            float temp_val;
            fread(&temp_val, sizeof(float), 1, infile);
            v->val[i] = (float)temp_val;
    }
}

void read_dense(DenseMatrix *a, FILE *infile) {
    for (int i = 0; i < a->row_size; i++) {
        for (int j = 0; j < a->col_size; j++) {
            float temp_val;
            fread(&temp_val, sizeof(float), 1, infile);
            a->val[i * a->col_size + j] = (float)temp_val;
        }
    }
}

void read_metadata(FILE *infile) {

}

void read_graph(GATGraph *g, char *path, bool inductive) {
    char allpath[256];
    sprintf(allpath, "datasets/%s.bin",path);
    // printf("%s", allpath);

    FILE *infile = fopen(allpath, "rb");
    
    if (!infile) {
        perror("Error opening graph.bin");
    }

    // read meta data
    int nnode, nedge, nfeature, nz_nei, nz_fea;
    fread(&nnode, sizeof(int), 1, infile);
    fread(&nedge, sizeof(int), 1, infile);
    fread(&nfeature, sizeof(int), 1, infile);

    if (!inductive) {
        fread(&nz_nei, sizeof(int), 1, infile);
        fread(&nz_fea, sizeof(int), 1, infile);
    }

    printf("nnode:%d nedge:%d nfeature:%d nz_nei:%d nz_fea:%d", nnode, nedge, nfeature, nz_nei, nz_fea);

    // graph init
    init_graph(g, nnode, nedge, nfeature, nz_nei, nz_fea, inductive);

    if (!inductive) {
        read_csr(g->s_neighbor, infile);
        read_csr(g->s_features, infile);
    }

    else {
        read_dense(g->d_neighbor, infile);
        read_dense(g->d_features, infile);
    }

}

Param *param_init(int in, int out, int nnode) {
    Param *param = malloc(sizeof(Param));
    param->in_feature = in;
    param->out_feature = out;

    param->weights = malloc(sizeof(DenseMatrix)); 
    init_dense(param->weights, in, out);

    param->linear = malloc(sizeof(DenseMatrix)); 
    init_dense(param->linear, nnode, out);

    param->a_l = malloc(sizeof(Vector)); 
    init_vector(param->a_l, out);

    param->a_r = malloc(sizeof(Vector)); 
    init_vector(param->a_r, out);

    return param;
}   

void read_layer(GATLayer *layer, int nnode, char *path) {
    char allpath[256];
    sprintf(allpath, "datasets/%s.bin", path);

    FILE *infile = fopen(allpath, "rb");

    if (!infile) {
        perror("Error opening layer");
        return;
    }

    int nhead, in_feature, out_feature;
    fread(&nhead, sizeof(int), 1, infile);
    fread(&in_feature, sizeof(int), 1, infile);
    fread(&out_feature, sizeof(int), 1, infile);
    printf("in_feat:%d\nout_feat:%d\nnnode:%d\nnhead:%d", in_feature, out_feature, nnode, nhead);
    layer->num_heads = nhead;
    Param **params = malloc(sizeof(Param *) * nhead);
    layer->params = params;
    for (int hid = 0; hid < nhead; hid++) { // for nheads
        params[hid] = param_init(in_feature, out_feature, nnode);
    }
    for (int hid = 0; hid < nhead; hid++) { // for nheads
        read_vector(params[hid]->a_l, infile); // weights of attn L
    }
    for (int hid = 0; hid < nhead; hid++) { // for nheads
        read_vector(params[hid]->a_r, infile); // weights of attn R
    }
    for (int hid = 0; hid < nhead; hid++) { // for nheads
        read_dense(params[hid]->weights, infile); // weights of linear
    }
    
}