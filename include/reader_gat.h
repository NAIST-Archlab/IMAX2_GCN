#include "sparse.h"
#include "linalg.h"
#include "gat.h"

void printDMatrix(DenseMatrix *A, char *s);
void printVector(Vector *A, char *s);
void printSpMatrix(SparseMatrix *A, char *s);
void init_csr(SparseMatrix *matrix, int row, int col, int nz);
void init_dense(DenseMatrix *matrix, int row, int col);
void init_vector(Vector *vector, int col);
void new_graph(GATGraph *g, int nnode, int nedge, int nfeature, int nz_nei, int nz_fea, bool inductive);
void read_csr(SparseMatrix *matrix, FILE *infile);
void read_graph(GATGraph *g, char *path, bool inductive);
Param *param_init(int in, int out, int nnode);
void read_layer(GATLayer *layer, int nnode, char *path);

