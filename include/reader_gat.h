#include "sparse.h"
#include "linalg.h"
#include "gat.h"

int get_index (int row_size, int col_size, int row, int col, int depth);
void printDMatrix(DenseMatrix *A, char *s);
// void printVector(Vector *A, char *s);
void printSpMatrix(SparseMatrix *A, char *s);
void init_csr(SparseMatrix *matrix, int depth, int row, int col, int nz);
void init_dense(DenseMatrix *matrix, int depth,int row, int col);
// void init_vector(Vector *vector, int col);
void new_graph(GATGraph *g, int nnode, int nedge, int nfeature, int nz_nei, int nz_fea);
void read_csr(SparseMatrix *matrix, FILE *infile);
void read_graph(GATGraph *g, char *path);
Param *param_init(int in, int out, int nnode, int nhead);
void read_layer(GATLayer *layer, int nnode, char *path);

