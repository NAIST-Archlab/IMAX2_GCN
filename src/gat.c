#include "../include/imax.h"
#include "../include/layer.h"
#include "../include/utils.h"
#include "../include/sparse.h"
#include "../include/linalg.h"
#include "../include/gat.h"
#include <time.h>

void logsoftmax(DenseMatrix *A) {
    #pragma omp parallel for
    for (int i=0; i < A->depth_size; i++){
        for (int j = 0; j < A->row_size; j++) {
            float sum = 0.0;
            for (int k = 0; k < A->col_size; k++) {
                int idx;
                idx = get_index(A->row_size, A->col_size, j, k, 0);
                A->val[idx] = exp(A->val[idx]);
                sum += A->val[idx];
            }
            for (int k = 0; k < A->col_size; k++) {
                int idx;
                idx = get_index(A->row_size, A->col_size, j, k, 0);
                A->val[idx] = log(A->val[idx] / sum);
            }
        }
    }
}

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
    for (int k = 0; k < A->depth_size; k++){
        for (int i = 0; i < A->row_size; i++) {
            for (int j = 0; j < A->col_size; j++) {
                int idx = get_index(A->row_size, A->col_size, i, j, k);
                float x = A->val[idx];
                A->val[idx] = x <= 0.0 ? alpha * (exp(x) - 1.0) : x;
            }
        } 
    }
}

void attention_coeff(DenseMatrix *linear_lr, DenseMatrix *linear, DenseMatrix *self_attn) {
    #pragma omp parallel for
    for (int depth = 0; depth < linear->depth_size; depth++) {
        for (int nid = 0; nid < linear->row_size; nid++) {
            float left = 0;
            for (int fid = 0; fid < linear->col_size; fid++) {
                int idx0 = get_index(self_attn->row_size, self_attn->col_size, 0, fid, depth);
                int idx1 = get_index(linear->row_size, linear->col_size, nid, fid, depth);
                left += self_attn->val[idx0] * linear->val[idx1];
            }
            int idx2 = get_index(linear_lr->row_size, linear_lr->col_size, 0, nid, depth);
            linear_lr->val[idx2] = left;
        }
    }
}

void EleWiseSum(DenseMatrix *a, DenseMatrix *linear_l, DenseMatrix *linear_r){ 
    #pragma omp parallel for
    for (int d = 0; d < linear_l->depth_size; d++){
        for (int i = 0; i < linear_l->col_size; i++) {
            for (int j = 0; j < linear_l->col_size; j++) {
                int idx0 = get_index(a->row_size, a->col_size, i, j, d);
                int idx1 = get_index(linear_l->row_size, linear_l->col_size, 0, j, d);
                int idx2 = get_index(linear_r->row_size, linear_r->col_size, 0, j, d);
                a->val[idx0] = linear_l->val[idx1] + linear_r->val[idx2];
            }
        }
    }
}

void mask(SparseMatrix *c, DenseMatrix *a, SparseMatrix *adj_matrix) {
    c->row_p = adj_matrix->row_p;
    c->col_p = adj_matrix->col_p;
    #pragma omp parallel for
    for (int d = 0; d < a->depth_size; d++){
        for (int i = 0; i < adj_matrix->row_size; i++) {
            for (int k = adj_matrix->row_p[i]; k < adj_matrix->row_p[i + 1]; k++) {
                int j = adj_matrix->col_p[k];
                c->val[k] = a->val[i * a->col_size + j];
            }
        }
    }
}


void matrix_lrelu(DenseMatrix *A) {
    for (int d=0; d < A->depth_size; d++) {
        #pragma omp parallel for
        for (int i = 0; i < A->row_size; i++) {
            for (int j = 0; j < A->col_size; j ++) {
                int idx = get_index(A->row_size, A->col_size, i, j, d);
                float x = A->val[idx];
                A->val[idx] = x < 0 ? ALPHA * x : x;
            }
        } 
    }
}

void s_softmax(SparseMatrix *A) {
    #pragma omp parallel for
    for (int d=0; d < A->depth_size; d++) {
        for (int i = 0; i < A->row_size; i++) {
            float sum = 0.0;
            for (int k = A->row_p[i]; k < A->row_p[i + 1]; k++) {
                int idx = get_index(A->row_size, A->col_size, i, k, d);
                A->val[k] = exp(A->val[k]);
                sum += A->val[k];
            }

            for (int k = A->row_p[i]; k < A->row_p[i + 1]; k++) {
                A->val[k] /= sum;
            }
        }
    }
}

void concat(DenseMatrix *source, DenseMatrix *destination, int num_heads) {
    init_dense(destination, 1, source->row_size, num_heads * source->col_size);
    #pragma omp parallel for
    for (int i = 0; i < source->row_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < source->col_size; k++) {
                int idx0 = get_index(destination->row_size, destination->col_size, i, (j * source->col_size) + k, 0);
                int idx1 = get_index(source->row_size, source->col_size, i, k, j);
                destination->val[idx0] = source->val[idx1];
            }
        }
    }
}

void split(DenseMatrix *source, DenseMatrix *destination, int num) {
    init_dense(destination, num, source->row_size, source->col_size / num);
    for (int i = 0; i < num; i++) {
        for (int j = 0; j < source->row_size; j++) {
            for (int k = 0; k < source->col_size / num; k++) {
                int idx0 = get_index(source->row_size, source->col_size, j, k + (i * (source->col_size / num)), 0);
                int idx1 = get_index(destination->row_size, destination->col_size, j, k, i);
                destination->val[idx1] = source->val[idx0];
            }
        }
    }
}

void multiple_split(DenseMatrix *source, DenseMatrix **destination, int num) {
    int col_per_dest = source->col_size / num;

    for (int i = 0; i < num; i++) {
        init_dense(destination[i], 1, source->row_size, col_per_dest);
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

void multiple_concat(DenseMatrix **source, DenseMatrix *destination, int num_heads) {
    init_dense(destination, 1, source[0]->row_size, num_heads * source[0]->col_size);
    #pragma omp parallel for
    for (int i = 0; i < source[0]->row_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < source[j]->col_size; k++) {
                destination->val[i * destination->col_size + j * source[j]->col_size + k] = source[j]->val[i * source[j]->col_size + k]; //
            }
        }
    }
}


// void concat_average(GATGraph *g, int num_heads, int out) {
//     g->concatfeature = (DenseMatrix *)malloc(sizeof(DenseMatrix));
//     init_dense(g->concatfeature, 1, g->features->row_size, out);
//     printf("concat : %d %d ", g->features->row_size, out);
    
//     #pragma omp parallel for
//     for (int i = 0; i < g->newfeatures->row_size; i++) {
//         for (int j = 0; j < out; j++) {
//             float sum = 0.0;
//             for (int k = 0; k < num_heads; k++) {
//                 int col_size = g->newfeatures[k]->col_size;
//                 sum += g->newfeatures[k]->val[i * col_size + j];  
//             }

//             for (int k = 0; k < num_heads; k++) {
//                 g->concatfeature->val[i * out + j] = sum / num_heads; 
//             }

//         }
//     }
// }

void freeDenseMatrix(DenseMatrix *A) {
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
}
#endif

void GATLayer_forward(GATLayer *L, GATGraph *g, bool cat, bool issparse) {
    int nnode = g->neighbor->row_size;
    int nhead = L->num_heads;
    SparseMatrix *adj_matrix = g->neighbor;
    Param *params = L->params;

    int out = L->params->out_feature;
    struct timespec t1, t2;

    L->merged_weight = malloc(sizeof(DenseMatrix *));
    init_dense(L->merged_weight, 1, L->params->weights->row_size, L->params->weights->col_size * L->num_heads);

    L->merged_linear = malloc(sizeof(DenseMatrix *));
    init_dense(L->merged_linear, 1, L->params->linear->row_size, L->params->weights->col_size * L->num_heads);

    DenseMatrix *temp_merge = malloc(sizeof(DenseMatrix *));

    L->separated_linear = malloc(sizeof(DenseMatrix *)); // C
    temp_merge = L->params->weights;
    L->separated_linear = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    init_dense(L->separated_linear, L->params->linear->depth_size, L->params->linear->row_size, L->params->linear->col_size);


    concat(temp_merge, L->merged_weight, L->num_heads);
    // printDMatrix(L->merged_weight,"merged");
    // printDMatrix(temp_merge,"unmerged");

    DenseMatrix *linear_l = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    init_dense(linear_l, nhead, 1, nnode);

    DenseMatrix *linear_r = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    init_dense(linear_r, nhead, 1, nnode);

    // DenseMatrix *tmp_sum = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    // init_dense(tmp_sum, nnode);

    SparseMatrix *sfeatures; // A
    DenseMatrix *dfeatures; // A

    // SpMM1(concat)
    DenseMatrix *weights = L->merged_weight; // B
    DenseMatrix *merged_linear = L->merged_linear; // C

    // SpMM1
    if (issparse) {
        sfeatures = g->features;
#if defined(EMAX6) || defined(EMAX7)
        spmm_imax(merged_linear, sfeatures, weights);  

#elif defined(USE_MP)

        // printf("spmm1 row: %d col: %d", weights->row_size, weights->col_size);
        timespec_get(&t1, TIME_UTC);
        spmm(merged_linear, sfeatures, weights);
        timespec_get(&t2, TIME_UTC);
        all_nanosec[SPMM] += (Ull)cal_time(&t2, &t1) * 1000;

// TODO: spmm for gpu

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
#endif

    }

    g->newfeatures = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    init_dense(g->newfeatures, nhead, nnode, L->params->out_feature);
    DenseMatrix *linear = L->separated_linear;
    split(merged_linear, linear, nhead);

// Attention section
    DenseMatrix *a_l = params->a_l;
    DenseMatrix *a_r = params->a_r;
    // printDenseMatrix(a_l, "aaa");

    timespec_get(&t1, TIME_UTC);
    attention_coeff(linear_l, linear, a_l);
    attention_coeff(linear_r, linear, a_r);

    DenseMatrix *Linear_lr = (DenseMatrix *)malloc(sizeof(DenseMatrix));
    init_dense(Linear_lr, nhead, linear_l->col_size, linear_r->col_size);
    EleWiseSum(Linear_lr, linear_l, linear_r);
    matrix_lrelu(Linear_lr);

    SparseMatrix *masked_Linear_lr = (SparseMatrix *)malloc(sizeof(SparseMatrix));
    init_csr(masked_Linear_lr, nhead, adj_matrix->row_size, adj_matrix->col_size, adj_matrix->nnz);
    mask(masked_Linear_lr, Linear_lr, adj_matrix);
    s_softmax(masked_Linear_lr);
    timespec_get(&t2, TIME_UTC);

#if defined(EMAX6) || defined(EMAX7)
    all_nanosec[ATTENTION][0] += (Ull)cal_time(&t2, &t1) * 1000;
#elif defined(USE_MP)
    all_nanosec[ATTENTION] += (Ull)cal_time(&t2, &t1) * 1000;
#endif
    
//MatMul2(batched spmm) for IMAX

    /*  
        depthwisesplit_csr(): masked_linear_lr(x,y,z) -> masked_linear_lr[x](y,z)
        depthwisesplit_dense(): linear(x,y,z) -> linear[x](y,z)
        for (heads):
            spmm(a[],b[],c[])
        multiple_concat(): result[x](y,z) -> result(x,y,z)
    */

    for (int headid=0; headid < nhead; headid++){
#if defined(EMAX6) || defined(EMAX7)

        spmm_imax(g->newfeatures[headid], masked_Linear_lr, linear[headid]);

// MatMul2(batched spmm) for CPU     
#elif defined(USE_MP) 
        timespec_get(&t1, TIME_UTC);    
        spmm(g->newfeatures, masked_Linear_lr, linear); 
        timespec_get(&t2, TIME_UTC);    
        all_nanosec[SPMM] += (Ull)cal_time(&t2, &t1) * 1000;    
#endif
// TODO: batched spmm for gpu
    }
    
    if(cat) {
        concat(g->newfeatures, g->concatfeature, nhead);
    }
}