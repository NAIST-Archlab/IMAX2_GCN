#include "../include/imax.h"
#include "../include/layer.h"
#include "../include/utils.h"
#include "../include/sparse.h"
#include "../include/linalg.h"
#include "../include/gat.h"
#include <time.h>

void logsoftmax(DenseMatrix *A) {
    #pragma omp parallel for
    for (int i = 0; i < A->row_size; i++) {
        float sum = 0.0;
        for (int k = 0; k < A->col_size; k++) {
            A->val[i * A->col_size + k] = exp(A->val[i * A->col_size + k]);
            sum += A->val[i * A->col_size + k];
        }
        for (int k = 0; k < A->col_size; k++) {
            A->val[i * A->col_size + k] = log(A->val[i * A->col_size + k] / sum);
        }
    }
}

// TODO: implement func of normalizing feature matrix

// void norm_feature(SparseMatrix *A) {
//     #pragma omp parallel for
//     //row wise sum
//     for (int i = 0; i < A->row_size; i++) {
//         for (int j = A->col_p[i]; j < A->col_p[i+1]) {
//         }
//     }
// }

void matrix_elu(DenseMatrix *A) {
    float alpha = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < A->row_size; i++) {
        for (int j = 0; j < A->col_size; j ++) {
            float x = A->val[i * A->col_size + j];
            A->val[i * A->col_size + j] = x <= 0.0 ? alpha * (exp(x) - 1.0) : x;
        }
    } 
}

// void transpose(DenseMatrix *A, DenseMatrix *B) {
//     init_dense(A, B->col_size, B->row_size);
//     for (int i = 0; i < B->row_size; i++) {
//         for (int j = 0; j < B->col_size; j++) {
//             A->val[j * B->row_size + i] = B->val[i * B->col_size + j];
//         }
//     }
// }

// mv
void mv(Vector *linear_lr, DenseMatrix *linear, Vector *self_attn) {
    #pragma omp parallel for
    for (int nid = 0; nid < linear->row_size; nid++) {
        float left = 0;
        for (int fid = 0; fid < linear->col_size; fid++) {
            left += self_attn->val[fid] * linear->val[nid * linear->col_size + fid];
        }
        linear_lr->val[nid] = left;
    }
}

void elewise_sum(DenseMatrix *a, Vector *linear_l, Vector *linear_r){ 
    #pragma omp parallel for
    for (int i = 0; i < linear_l->col_size; i++) {
        for (int j = 0; j < linear_l->col_size; j++) {
            a->val[i * linear_l->col_size + j] = linear_l->val[i] + linear_r->val[j];
        }
    }
}

void mask(SparseMatrix *c, DenseMatrix *a, SparseMatrix *adj_matrix) {
    c->row_p = adj_matrix->row_p;
    c->col_p = adj_matrix->col_p;
    #pragma omp parallel for
    for (int i = 0; i < adj_matrix->row_size; i++) {
        for (int k = adj_matrix->row_p[i]; k < adj_matrix->row_p[i + 1]; k++) {
            int j = adj_matrix->col_p[k];
            c->val[k] = a->val[i * a->col_size + j];
        }
    }
}


void matrix_lrelu(DenseMatrix *A) {
    #pragma omp parallel for

    for (int i = 0; i < A->row_size; i++) {
        for (int j = 0; j < A->col_size; j ++) {
            float x = A->val[i * A->col_size + j];
            A->val[i * A->col_size + j] = x < 0 ? ALPHA * x : x;
        }
    } 
}

void s_softmax(SparseMatrix *A) {
    #pragma omp parallel for
    
    for (int i = 0; i < A->row_size; i++) {
        float sum = 0.0;
        for (int k = A->row_p[i]; k < A->row_p[i + 1]; k++) {
            A->val[k] = exp(A->val[k]);
            sum += A->val[k];
        }

        for (int k = A->row_p[i]; k < A->row_p[i + 1]; k++) {
            A->val[k] /= sum;
        }
    }
}

// source of matrices, destination matrix
void concat(DenseMatrix **source, DenseMatrix *destination, int num_heads) {
    init_dense(destination, source[0]->row_size, num_heads * source[0]->col_size);
    #pragma omp parallel for
    for (int i = 0; i < source[0]->row_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < source[j]->col_size; k++) {
                destination->val[i * destination->col_size + j * source[j]->col_size + k] = source[j]->val[i * source[j]->col_size + k]; //
            }
        }
    }
}

void split(DenseMatrix *source, DenseMatrix **destination, int num) {
    int col_per_dest = source->col_size / num;

    for (int i = 0; i < num; i++) {
        init_dense(destination[i], source->row_size, col_per_dest);
    }

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < source->row_size; j++) {
            for (int k = 0; k < col_per_dest; k++) {
                int sourceCol = i * col_per_dest + k;
                destination[i]->val[j * col_per_dest + k] = source->val[j * source->col_size + sourceCol];
            }
        }
    }
}




void concat_average(GATGraph *g, int num_heads, int out) {
    g->concatfeature = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    init_dense(g->concatfeature, g->features->row_size, out);
    printf("concat : %d %d ", g->features->row_size, out);
    
    #pragma omp parallel for
    for (int i = 0; i < g->newfeatures[0]->row_size; i++) {
        for (int j = 0; j < out; j++) {
            float sum = 0.0;
            for (int k = 0; k < num_heads; k++) {
                int col_size = g->newfeatures[k]->col_size;
                sum += g->newfeatures[k]->val[i * col_size + j];  
            }

            for (int k = 0; k < num_heads; k++) {
                g->concatfeature->val[i * out + j] = sum / num_heads; 
            }

        }
    }
}

void freeVector(Vector *A) {
    free(A->val);
}

#if defined(EMAX6) || defined(EMAX7)
void spmm_imax(DenseMatrix *C, SparseMatrix *A, DenseMatrix *B) {
        IMAXSparseMatrix imax_sp;
        IMAXDenseMatrix imax_m, imax_r;
        imax_m.row_size = B->row_size;
        imax_m.col_size = B->col_size;
        imax_r.row_size = C->row_size;
        imax_r.col_size = C->col_size;

        imax_sparse_format_init(&imax_sp, A->row_size, A->row_size, SPMM_H, 8 * NCHIP);
        convert_imax_sparse_format(&imax_sp, A);
        imax_sparse_allocation(&imax_sp);

        imax_matrix_init_spmm(&imax_r, &imax_sp, &imax_m, 0);
        imax_dense_allocation(&imax_m);
        imax_dense_allocation(&imax_r);
        convert_imax_dense_format(&imax_m, B);

        spmm(&imax_r, &imax_sp, &imax_m);

        convert_dense_format(C, &imax_r);
        
        imax_sparse_deallocation(&imax_sp);
        imax_dense_deallocation(&imax_m);
        imax_dense_deallocation(&imax_r);
}

void mm_imax(DenseMatrix *C, DenseMatrix *A, DenseMatrix *B) {
        IMAXDenseMatrix imax_m1, imax_m2, imax_r;
        imax_m1.row_size = A->row_size;
        imax_m1.col_size = A->col_size;
        imax_m1.blk_row_size = (imax_m1.row_size < 1024) ? imax_m1.row_size + (MM_H - imax_m1.row_size%MM_H) : 1024;
        imax_m2.row_size = B->row_size;
        imax_m2.col_size = B->col_size;
        imax_r.row_size = C->row_size;
        imax_r.col_size = C->col_size;
        imax_matrix_init_mm(&imax_r, &imax_m1, &imax_m2, FIT_TO_DENSE);
        imax_dense_allocation(&imax_r);
        imax_dense_allocation(&imax_m1);
        imax_dense_allocation(&imax_m2);
        convert_imax_dense_format(&imax_m1, A);
        convert_imax_dense_format(&imax_m2, B);
        convert_imax_dense_format(&imax_r, C);
        mm(&imax_r, &imax_m1, &imax_m2);
        convert_dense_format(C, &imax_r);
        convert_dense_format(A, &imax_m1);
        convert_dense_format(B, &imax_m2);
        imax_dense_deallocation(&imax_r);
        imax_dense_deallocation(&imax_m1);
        imax_dense_deallocation(&imax_m2);
}
#endif


void GATLayer_forward(GATLayer *L, GATGraph *g, bool cat, bool issparse) {
    int nnode = g->neighbor->row_size;
    int nhead = L->num_heads;
    SparseMatrix *adj_matrix = g->neighbor;
    Param **params = L->params;

    int out = L->params[0]->out_feature;
    struct timespec t1, t2;

    L->merged_weight = malloc(sizeof(DenseMatrix *));
    init_dense(L->merged_weight, L->params[0]->weights->row_size, L->params[0]->weights->col_size * L->num_heads);

    L->merged_linear = malloc(sizeof(DenseMatrix *));
    init_dense(L->merged_linear, L->params[0]->linear->row_size, L->params[0]->weights->col_size * L->num_heads);

    DenseMatrix **temp_merge = malloc(sizeof(DenseMatrix *) * L->num_heads);

    // init for n heads
    L->separated_linear = malloc(sizeof(DenseMatrix **) * L->num_heads); // C
    for (int i = 0; i < nhead ; i++) {
        temp_merge[i] = L->params[i]->weights;
        L->separated_linear[i] = (DenseMatrix *)malloc(sizeof(DenseMatrix));
        init_dense(L->separated_linear[i], L->params[0]->linear->row_size, L->params[0]->linear->col_size);
    }

    concat(temp_merge, L->merged_weight, L->num_heads);

    Vector *linear_l = (Vector *)malloc(sizeof(Vector));
    init_vector(linear_l, nnode);

    Vector *linear_r = (Vector *)malloc(sizeof(Vector));
    init_vector(linear_r, nnode);

    Vector *tmp_sum = (Vector *)malloc(sizeof(Vector));
    init_vector(tmp_sum, nnode);

    SparseMatrix *sfeatures; // A
    DenseMatrix *dfeatures; // A

    // SpMM(concatenated)
    DenseMatrix *weights = L->merged_weight; // B
    DenseMatrix *merged_linear = L->merged_linear; // C

    if (issparse) {
        sfeatures = g->features;
#if defined(EMAX6) || defined(EMAX7)
        spmm_imax(merged_linear, sfeatures, weights);          

#elif defined(USE_MP)
        timespec_get(&t1, TIME_UTC);
        spmm(merged_linear, sfeatures, weights);
        timespec_get(&t2, TIME_UTC);
        all_nanosec[SPMM] += (Ull)cal_time(&t2, &t1) * 1000;

#elif defined(USE_CUDA)
        sendSparseMatrixToGPU(sfeatures);
        sendDenseMatrixToGPU(weights);
        createCusparse();
        spmm(merged_linear, sfeatures, weights);
        sendDenseMatrixToCPU(merged_linear);
        destroyCusparse();
#endif
    }

    else {
        dfeatures = g->concatfeature;
#if defined(EMAX6) || defined(EMAX7)
        mm_imax(merged_linear, dfeatures, weights);
        
#elif defined(USE_MP)
        timespec_get(&t1, TIME_UTC);
        mm(merged_linear, dfeatures, weights);
        timespec_get(&t2, TIME_UTC);
        all_nanosec[MM] += (Ull)cal_time(&t2, &t1) * 1000;

#elif defined(USE_CUDA)
        timespec_get(&t1, TIME_UTC);
        createCublas();
        sendDenseMatrixToGPU(dfeatures);
        sendDenseMatrixToGPU(weights);
        timespec_get(&t1, TIME_UTC);
        mm(merged_linear, dfeatures, weights);
        timespec_get(&t2, TIME_UTC);
        sendDenseMatrixToCPU(merged_linear);
        // printDMatrix(merged_linear, "newfeat");
        destroyCublas();
#endif
    }
    freeDenseMatrix(weights);

    g->newfeatures = (DenseMatrix **)malloc(nhead * sizeof(DenseMatrix));

    DenseMatrix **linear = L->separated_linear;
    split(merged_linear, linear, nhead);
    freeDenseMatrix(merged_linear);
    
    // Attention section
    for (int headid = 0; headid < nhead; headid++) {
        
        g->newfeatures[headid] = (DenseMatrix *)malloc(sizeof(DenseMatrix));
        init_dense(g->newfeatures[headid], nnode, L->params[0]->out_feature);

        Vector *a_l = params[headid]->a_l;
        Vector *a_r = params[headid]->a_r;

        timespec_get(&t1, TIME_UTC);
        mv(linear_l, linear[headid], a_l);
        mv(linear_r, linear[headid], a_r);

        DenseMatrix *Linear_lr = (DenseMatrix *)malloc(sizeof(DenseMatrix));
        init_dense(Linear_lr, linear_l->col_size, linear_r->col_size);

        elewise_sum(Linear_lr, linear_l, linear_r);
        matrix_lrelu(Linear_lr);

        SparseMatrix *masked_Linear_lr = (SparseMatrix *)malloc(sizeof(SparseMatrix));
        init_csr(masked_Linear_lr, adj_matrix->row_size, adj_matrix->col_size, adj_matrix->nnz);
        mask(masked_Linear_lr, Linear_lr, adj_matrix);
        s_softmax(masked_Linear_lr);    
        timespec_get(&t2, TIME_UTC);
        freeDenseMatrix(Linear_lr);
        printf("Attention: %lf", cal_time(&t2, &t1));

#if defined(EMAX6) || defined(EMAX7)
        all_nanosec[ATTENTION][0] += (Ull)cal_time(&t2, &t1) * 1000;
#elif defined(USE_MP)
        all_nanosec[ATTENTION] += (Ull)cal_time(&t2, &t1) * 1000;
#endif
        
#if defined(EMAX6) || defined(EMAX7)
        spmm_imax(g->newfeatures[headid], masked_Linear_lr, linear[headid]);
    
#elif defined(USE_MP) 
        timespec_get(&t1, TIME_UTC);
        spmm(g->newfeatures[headid], masked_Linear_lr, linear[headid]);
        timespec_get(&t2, TIME_UTC);
        all_nanosec[SPMM] += (Ull)cal_time(&t2, &t1) * 1000;
        // printDMatrix(g->newfeatures[headid], "MatMul2");

#elif defined(USE_CUDA)
        sendSparseMatrixToGPU(masked_Linear_lr);
        sendDenseMatrixToGPU(linear[headid]);
        createCusparse();
        spmm(g->newfeatures[headid], masked_Linear_lr, linear[headid]);
        sendDenseMatrixToCPU(g->newfeatures[headid]);
        destroyCusparse();
#endif
        freeDenseMatrix(linear[headid]);
        // printf("headid: %d", headid);
        
    }
    if(cat) {
        concat(g->newfeatures, g->concatfeature, nhead);
    }

}