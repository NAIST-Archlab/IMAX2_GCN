#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "../include/linalg.h"
#include "../include/sparse.h"
#include "../include/reader_gat.h"
#include "../include/layer.h"


int get_index (int row_size, int col_size, int row, int col, int depth) {
    int idx = depth * (row_size * col_size) + row * col_size + col;
    return idx;
}

void printDMatrix(DenseMatrix *A, char *s){
    printf("%s: \ndepth: %d\nrow: %d\ncol: %d\n--------\n",s, A->depth_size, A->row_size, A->col_size);
    int col =  A->col_size;
    char name[500];
    sprintf(name, "output/%s.txt",s);
    FILE *out_file = fopen(name, "w+");
    char tmp[500];
    for (int d=0; d<A->depth_size; d++){
        fprintf(out_file, "[%d]\n", d);
        for (int i=0; i<A->row_size; i++){
            for (int j=0; j<col; j++){
                int idx = get_index(A->row_size, A->col_size, i, j, d);
                sprintf(tmp, "%f ", A->val[idx]);
                fputs(tmp, out_file);
            }
            fputs("\n", out_file);
        }
    }
    fclose(out_file);
}

// void printVector(Vector *A, char *s) {
//     int col = A->col_size;
//     char name[500];
//     sprintf(name, "output/%s.txt",s);
//     FILE *out_file = fopen(name, "w+");
//     char tmp[50];
//     for (int i=0; i<A->col_size; i++) {
//         sprintf(tmp, "%f ", A->val[i]);
//         fputs(tmp, out_file);
//     }
//     fclose(out_file);
// }

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

void init_csr(SparseMatrix *matrix, int depth, int row, int col, int nz) {
    matrix->depth_size = depth;
    matrix->row_size = row;
    matrix->col_size = col;
    matrix->nnz = nz;

    matrix->row_p = calloc((row*depth)+1, sizeof(int));
    matrix->col_p = calloc(nz, sizeof(int));
    matrix->val = calloc(nz, sizeof(float));
    matrix->depth_p = calloc(depth+1, sizeof(int));
}

void init_dense(DenseMatrix *matrix,int depth, int row, int col) {
    matrix->val = calloc(depth * row * col, sizeof(float));
    matrix->depth_size = depth;
    matrix->row_size = row;
    matrix->col_size = col;
}

// void init_vector(Vector *vector, int col) {
//     vector->val = calloc(1 * col, sizeof(float));
//     vector->col_size = col;
// }

void init_graph(GATGraph *g, int nnode, int nedge, int nfeature, int nz_nei, int nz_fea) {
    g->concatfeature = calloc(nnode * nfeature, sizeof(float));

    g->neighbor = malloc(sizeof(SparseMatrix));
    init_csr(g->neighbor, 1, nnode, nnode, nz_nei);

    g->features = malloc(sizeof(SparseMatrix)); // features
    init_csr(g->features, 1, nnode, nfeature, nz_fea);

}

void read_csr(SparseMatrix *matrix, FILE *infile) {
    fread(matrix->row_p, sizeof(int), matrix->row_size + 1, infile);
    fread(matrix->col_p, sizeof(int), matrix->nnz, infile);
    fread(matrix->val, sizeof(float), matrix->nnz, infile);    
}

// void read_vector(Vector *v, FILE *infile) {
//     for (int i=0; i<v->col_size; i++) {
//             float temp_val;
//             fread(&temp_val, sizeof(float), 1, infile);
//             v->val[i] = (float)temp_val;
//     }
// }

void read_dense(DenseMatrix *a, FILE *infile) {
    for (int d = 0; d < a->depth_size; d++) {
        for (int i = 0; i < a->row_size; i++) {
            for (int j = 0; j < a->col_size; j++) {
                float temp_val;
                int idx = get_index(a->row_size, a->col_size, i, j, d);
                fread(&temp_val, sizeof(float), 1, infile);
                a->val[idx] = (float)temp_val;
            }
        }
    }
}

void read_metadata(FILE *infile) {

}

void read_graph(GATGraph *g, char *path) {
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
    fread(&nz_nei, sizeof(int), 1, infile);
    fread(&nz_fea, sizeof(int), 1, infile);
    // printf("nnode:%d nedge:%d nfeature:%d nz_nei:%d nz_fea:%d", nnode, nedge, nfeature, nz_nei, nz_fea);

    // graph init
    init_graph(g, nnode, nedge, nfeature, nz_nei, nz_fea);

    read_csr(g->neighbor, infile);
    read_csr(g->features, infile);
}

Param *param_init(int in, int out, int nnode, int nhead) {
    Param *param = malloc(sizeof(Param));
    param->in_feature = in;
    param->out_feature = out;

    // printf("in:%d, out:%d, nnode:%d",in,out,nnode);
    param->weights = malloc(sizeof(DenseMatrix)); 
    init_dense(param->weights, nhead, in, out);

    param->linear = malloc(sizeof(DenseMatrix)); 
    init_dense(param->linear, nhead, nnode, out);

    param->a_l = malloc(sizeof(DenseMatrix)); 
    init_dense(param->a_l, nhead, 1, out);

    param->a_r = malloc(sizeof(DenseMatrix)); 
    init_dense(param->a_r, nhead, 1, out);

    return param;
}   

void read_layer(GATLayer *layer, int nnode, char *path) {
    
    char allpath[256];
    sprintf(allpath, "datasets/%s.bin", path);
    // printf("%s", allpath);

    FILE *infile = fopen(allpath, "rb");
    
    if (!infile) {
        perror("Error opening layer");
        return;
    }

    int nhead, in_feature, out_feature;
    fread(&nhead, sizeof(int), 1, infile);
    fread(&in_feature, sizeof(int), 1, infile);
    fread(&out_feature, sizeof(int), 1, infile);
    layer->num_heads = nhead;
    Param *params = malloc(sizeof(Param *));
    layer->params = params;

    printf("in_feat:%d\nout_feat:%d\nnnode:%d\nnhead:%d", in_feature, out_feature, nnode, nhead);
    Param *param = param_init(in_feature, out_feature, nnode, nhead);
    read_dense(param->a_l, infile); // weights of attn L
    // printDMatrix(param->a_l, "a_l_sasd");
    read_dense(param->a_r, infile); // weights of attn R
    // printDMatrix(param->a_r, "a_r");
    read_dense(param->weights, infile); // weights of linear
    layer->params = param;

    

}